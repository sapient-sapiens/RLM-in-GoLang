package myrlm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"
)

const (
	DefaultMaxIterations = 10
	DefaultMaxDepth      = 1
)

type RLM struct {
	maxIterations int
	maxDepth      int
	maxTimeout    *float64
	maxTokens     *int64
	maxBudget     *int64
	maxErrors     *int
	customTools   []Tool
	dockerConfig  DockerConfig
	client        *OpenAIClient
}

func NewRLM(client *OpenAIClient, opts ...RLMOption) *RLM {
	r := &RLM{
		maxIterations: DefaultMaxIterations,
		maxDepth:      DefaultMaxDepth,
		client:        client,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

type RLMOption func(*RLM)

func WithMaxIterations(n int) RLMOption      { return func(r *RLM) { r.maxIterations = n } }
func WithMaxDepth(d int) RLMOption           { return func(r *RLM) { r.maxDepth = d } }
func WithMaxTimeout(t float64) RLMOption     { return func(r *RLM) { r.maxTimeout = &t } }
func WithMaxTokens(n int64) RLMOption        { return func(r *RLM) { r.maxTokens = &n } }
func WithMaxBudget(n int64) RLMOption        { return func(r *RLM) { r.maxBudget = &n } }
func WithMaxErrors(e int) RLMOption          { return func(r *RLM) { r.maxErrors = &e } }
func WithCustomTools(tools []Tool) RLMOption { return func(r *RLM) { r.customTools = tools } }
func WithDockerConfig(cfg DockerConfig) RLMOption {
	return func(r *RLM) { r.dockerConfig = cfg }
}

// Completion runs the RLM iteration loop: prompt the root LLM (via Go OpenAI client),
// parse code blocks, execute them in a Docker REPL, feed output back, repeat until
// a FINAL answer is found.
func (r *RLM) Completion(ctx context.Context, rlmCtx Context, query Query) (string, error) {
	return r.completion(ctx, rlmCtx, query, true)
}

// completion is the internal implementation for both root and child RLM calls.
// Child calls reuse the already-running REPL server and set startServer=false.
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
	chat := r.client.NewChatWithInstructions(systemPrompt)
	if r.maxTokens != nil {
		chat.SetMaxOutputTokens(*r.maxTokens)
	}

	pendingUserMessages := make([]Prompt, 0, 1)
	pendingUserMessages = append(pendingUserMessages, initialMessages[1])

	var (
		bestPartialAnswer string
		totalTokensUsed   int64
	)
	consecutiveErrors := 0

	for i := 0; i < r.maxIterations; i++ {
		userPrompt := BuildUserPrompt(query, i)

		currentInputs := append([]Prompt(nil), pendingUserMessages...)
		currentInputs = append(currentInputs, userPrompt)
		log.Printf("[RLM] iter %d/%d — calling LLM (%d messages, ~%d chars)...",
			i+1, r.maxIterations, len(currentInputs), promptChars(currentInputs))
		iterStart := time.Now()

		result, err := chat.SendMessages(ctx, currentInputs)
		if err != nil {
			return "", fmt.Errorf("root LLM call (iteration %d): %w", i, err)
		}
		pendingUserMessages = pendingUserMessages[:0]

		totalTokensUsed += result.Stats.Tokens.TotalTokens
		if r.maxBudget != nil && totalTokensUsed > *r.maxBudget {
			log.Printf("[RLM] iter %d — token budget exhausted (%d/%d)", i+1, totalTokensUsed, *r.maxBudget)
			if bestPartialAnswer != "" {
				return bestPartialAnswer, nil
			}
			return result.Text, nil
		}

		response := result.Text
		codeBlocks := findCodeBlocks(response)

		preview := response
		if len(preview) > 300 {
			preview = preview[:300] + "..."
		}
		log.Printf("[RLM] iter %d/%d — LLM responded in %s (%d chars, %d code blocks)\n  Response: %s",
			i+1, r.maxIterations, time.Since(iterStart).Round(time.Millisecond), len(response), len(codeBlocks), preview)

		var newUserMessages []Prompt
		iterationHadError := false

		for j, code := range codeBlocks {
			log.Printf("[RLM] iter %d — executing code block %d/%d (%d chars)...",
				i+1, j+1, len(codeBlocks), len(code))
			replResult, execErr := repl.ExecuteCode(code)
			if execErr != nil || (replResult != nil && strings.TrimSpace(replResult.Stderr) != "") {
				iterationHadError = true
				if execErr != nil {
					log.Printf("[RLM] iter %d — code block %d exec error: %v", i+1, j+1, execErr)
				}
				if replResult != nil && strings.TrimSpace(replResult.Stderr) != "" {
					log.Printf("[RLM] iter %d — code block %d stderr: %.200s", i+1, j+1, replResult.Stderr)
				}
			}

			if replResult != nil && replResult.FinalAnswer != "" {
				log.Printf("[RLM] iter %d — FINAL answer from REPL: %.100s", i+1, replResult.FinalAnswer)
				return replResult.FinalAnswer, nil
			}

			formatted := FormatREPLResult(replResult)
			if len(formatted) > 20000 {
				formatted = formatted[:20000] + fmt.Sprintf("... + [%d chars...]", len(formatted)-20000)
			}
			log.Printf("[RLM] iter %d — code block %d output: %.200s", i+1, j+1, formatted)
			newUserMessages = append(newUserMessages, Prompt{
				Role:    "user",
				Content: fmt.Sprintf("Code executed:\n```python\n%s\n```\n\nREPL output:\n%s", code, formatted),
			})
		}

		if finalAnswer, finalErr := findFinalAnswer(response); finalErr == nil {
			log.Printf("[RLM] iter %d — FINAL() found in response: %.100s", i+1, finalAnswer)
			return finalAnswer, nil
		}

		if varName, ok := findFinalVar(response); ok {
			log.Printf("[RLM] iter %d — FINAL_VAR(%s) found, resolving...", i+1, varName)
			if res, err := repl.ExecuteCode(fmt.Sprintf("FINAL_VAR(%q)", varName)); err == nil && res != nil && res.FinalAnswer != "" {
				return res.FinalAnswer, nil
			}
		}

		if iterationHadError {
			consecutiveErrors++
			if r.maxErrors != nil && consecutiveErrors >= *r.maxErrors {
				if bestPartialAnswer != "" {
					return bestPartialAnswer, nil
				}
				return "", fmt.Errorf("error threshold exceeded: %d consecutive errors", consecutiveErrors)
			}
		} else {
			consecutiveErrors = 0
		}

		pendingUserMessages = append(pendingUserMessages, newUserMessages...)

		if strings.TrimSpace(response) != "" {
			bestPartialAnswer = response
		}
	}

	log.Printf("[RLM] exhausted %d iterations without FINAL — requesting fallback answer...", r.maxIterations)
	fallbackInputs := append([]Prompt(nil), pendingUserMessages...)
	fallbackInputs = append(fallbackInputs, Prompt{
		Role:    "user",
		Content: "You have run out of iterations. Based on everything above, provide your final answer to the original query now. Output ONLY the answer, nothing else.",
	})
	if fallbackResult, err := chat.SendMessages(ctx, fallbackInputs); err == nil && strings.TrimSpace(fallbackResult.Text) != "" {
		return fallbackResult.Text, nil
	}

	if bestPartialAnswer != "" {
		return bestPartialAnswer, nil
	}
	return "", errors.New("max iterations reached without final answer")
}

func promptChars(msgs []Prompt) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
	}
	return n
}
