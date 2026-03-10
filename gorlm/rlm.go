package myrlm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"
)

func NewRLM(client *OpenAIClient, opts ...RLMOption) *RLM {
	r := &RLM{
		maxDepth:  DefaultMaxDepth,
		maxBudget: DefaultTokenBudget,
		client:    client,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

func WithMaxDepth(d int) RLMOption            { return func(r *RLM) { r.maxDepth = d } }
func WithMaxTimeout(t float64) RLMOption      { return func(r *RLM) { r.maxTimeout = &t } }
func WithMaxTokens(n int64) RLMOption         { return func(r *RLM) { r.maxTokens = &n } }
func WithMaxBudget(n int64) RLMOption         { return func(r *RLM) { r.maxBudget = n } }
func WithCustomTools(tools []Tool) RLMOption  { return func(r *RLM) { r.customTools = tools } }
func WithDockerConfig(cfg DockerConfig) RLMOption {
	return func(r *RLM) { r.dockerConfig = cfg }
}

// Completion runs the truly-recursive single-shot RLM: prompt the LLM once,
// execute every code block from the response, and return the final answer.
// If the model needs more reasoning, its code calls rlm_query() which
// recursively invokes another Completion — the recursion is model-driven,
// not system-driven.
func (r *RLM) Completion(ctx context.Context, rlmCtx Context, query Query) (string, error) {
	return r.completion(ctx, rlmCtx, query, true)
}

func (r *RLM) completion(ctx context.Context, rlmCtx Context, query Query, startServer bool) (string, error) {
	if r.client == nil {
		return "", errors.New("nil OpenAI client")
	}

	if r.maxTimeout != nil && *r.maxTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(*r.maxTimeout*float64(time.Second)))
		defer cancel()
	}

	if startServer {
		srv := NewREPLServer(r.client, r)
		if err := srv.Start(); err != nil {
			return "", fmt.Errorf("start repl server: %w", err)
		}
		defer srv.Shutdown(ctx)
	}

	replCfg := r.dockerConfig
	replCfg.Depth = r.maxDepth
	repl, err := NewDockerREPL(replCfg)
	if err != nil {
		return "", fmt.Errorf("create docker repl: %w", err)
	}
	defer repl.Close()

	if err := repl.LoadContext(rlmCtx.Content); err != nil {
		return "", fmt.Errorf("load context: %w", err)
	}

	systemPrompt := GetSystemPromptWithCustomTools(r.customTools)
	initialMessages := BuildInitialMessages(systemPrompt, rlmCtx.Metadata)
	userPrompt := BuildUserPrompt(query)

	chat := r.client.NewChatWithInstructions(systemPrompt)
	if r.maxTokens != nil {
		chat.SetMaxOutputTokens(*r.maxTokens)
	}

	userMessages := []Prompt{initialMessages[1], userPrompt}
	log.Printf("[RLM] calling LLM (depth=%d, ~%d chars)...", r.maxDepth, promptChars(userMessages))
	callStart := time.Now()

	result, err := chat.SendMessages(ctx, userMessages)
	if err != nil {
		return "", fmt.Errorf("LLM call: %w", err)
	}

	response := result.Text
	codeBlocks := findCodeBlocks(response)

	preview := response
	if len(preview) > 300 {
		preview = preview[:300] + "..."
	}
	log.Printf("[RLM depth=%d] LLM responded in %s (%d chars, %d code blocks, %d tokens)\n  Response: %s",
		r.maxDepth, time.Since(callStart).Round(time.Millisecond), len(response), len(codeBlocks),
		result.Stats.Tokens.TotalTokens, preview)

	// Execute all code blocks from the single response.
	// The model's code may call rlm_query() which triggers recursive
	// Completion calls via the HTTP server — that's where recursion lives.
	for i, code := range codeBlocks {
		log.Printf("[RLM depth=%d] executing code block %d/%d (%d chars)...", r.maxDepth, i+1, len(codeBlocks), len(code))
		replResult, execErr := repl.ExecuteCode(code)

		if execErr != nil {
			log.Printf("[RLM depth=%d] code block %d exec error: %v", r.maxDepth, i+1, execErr)
		}
		if replResult != nil && strings.TrimSpace(replResult.Stderr) != "" {
			log.Printf("[RLM depth=%d] code block %d stderr: %.200s", r.maxDepth, i+1, replResult.Stderr)
		}

		if replResult != nil && replResult.FinalAnswer != "" {
			log.Printf("[RLM depth=%d] FINAL answer from code block %d: %.100s", r.maxDepth, i+1, replResult.FinalAnswer)
			return replResult.FinalAnswer, nil
		}

		if replResult != nil {
			formatted := FormatREPLResult(replResult)
			if len(formatted) > 200 {
				formatted = formatted[:200] + "..."
			}
			log.Printf("[RLM depth=%d] code block %d output: %s", r.maxDepth, i+1, formatted)
		}
	}

	// Check for FINAL()/FINAL_VAR() in the LLM's text (outside code blocks)
	if answer, findErr := findFinalAnswer(response); findErr == nil {
		log.Printf("[RLM depth=%d] FINAL() found in response text: %.100s", r.maxDepth, answer)
		return answer, nil
	}
	if varName, ok := findFinalVar(response); ok {
		log.Printf("[RLM depth=%d] FINAL_VAR(%s) found in response text, resolving...", r.maxDepth, varName)
		if res, execErr := repl.ExecuteCode(fmt.Sprintf("FINAL_VAR(%q)", varName)); execErr == nil && res != nil && res.FinalAnswer != "" {
			return res.FinalAnswer, nil
		}
	}

	// Fallback: return the raw LLM response if it has content
	if trimmed := strings.TrimSpace(response); trimmed != "" {
		log.Printf("[RLM depth=%d] no FINAL found — returning raw LLM response as fallback", r.maxDepth)
		return trimmed, nil
	}

	return "", errors.New("completion produced no answer")
}

func promptChars(msgs []Prompt) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
	}
	return n
}
