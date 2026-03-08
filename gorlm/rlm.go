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
	DefaultMaxDepth          = 3
	DefaultMaxTurns          = 50
	DefaultMaxConsecErrors   = 5
	DefaultTokenBudget int64 = 500_000
)

type RLM struct {
	maxDepth    int
	maxTurns    int
	maxTimeout  *float64
	maxTokens   *int64
	maxBudget   int64
	maxErrors   int
	customTools []Tool
	dockerConfig DockerConfig
	client       *OpenAIClient
}

func NewRLM(client *OpenAIClient, opts ...RLMOption) *RLM {
	r := &RLM{
		maxDepth:  DefaultMaxDepth,
		maxTurns:  DefaultMaxTurns,
		maxBudget: DefaultTokenBudget,
		maxErrors: DefaultMaxConsecErrors,
		client:    client,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

type RLMOption func(*RLM)

func WithMaxDepth(d int) RLMOption            { return func(r *RLM) { r.maxDepth = d } }
func WithMaxTurns(n int) RLMOption            { return func(r *RLM) { r.maxTurns = n } }
func WithMaxTimeout(t float64) RLMOption      { return func(r *RLM) { r.maxTimeout = &t } }
func WithMaxTokens(n int64) RLMOption         { return func(r *RLM) { r.maxTokens = &n } }
func WithMaxBudget(n int64) RLMOption         { return func(r *RLM) { r.maxBudget = n } }
func WithMaxErrors(e int) RLMOption           { return func(r *RLM) { r.maxErrors = e } }
func WithCustomTools(tools []Tool) RLMOption  { return func(r *RLM) { r.customTools = tools } }
func WithDockerConfig(cfg DockerConfig) RLMOption {
	return func(r *RLM) { r.dockerConfig = cfg }
}

// Completion runs the truly-recursive RLM loop: prompt the LLM, parse code
// blocks, execute them in a Docker REPL, feed output back — repeating until
// the model produces a FINAL answer on its own. There is no fixed iteration
// cap; the model decides when it's done. Safety guards (token budget, timeout,
// consecutive error limit, max turns) prevent runaway loops.
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
	chat := r.client.NewChatWithInstructions(systemPrompt)
	if r.maxTokens != nil {
		chat.SetMaxOutputTokens(*r.maxTokens)
	}

	pendingUserMessages := make([]Prompt, 0, 1)
	pendingUserMessages = append(pendingUserMessages, initialMessages[1])

	var (
		bestPartialAnswer string
		totalTokensUsed   int64
		consecutiveErrors int
		turn              int
	)

	for {
		turn++

		if r.maxTurns > 0 && turn > r.maxTurns {
			log.Printf("[RLM] safety: max turns (%d) reached — requesting fallback", r.maxTurns)
			break
		}

		if err := ctx.Err(); err != nil {
			log.Printf("[RLM] context cancelled/timed out at turn %d: %v", turn, err)
			break
		}

		userPrompt := BuildUserPrompt(query, turn-1)

		currentInputs := append([]Prompt(nil), pendingUserMessages...)
		currentInputs = append(currentInputs, userPrompt)
		log.Printf("[RLM] turn %d — calling LLM (%d messages, ~%d chars)...",
			turn, len(currentInputs), promptChars(currentInputs))
		iterStart := time.Now()

		result, err := chat.SendMessages(ctx, currentInputs)
		if err != nil {
			return "", fmt.Errorf("LLM call (turn %d): %w", turn, err)
		}
		pendingUserMessages = pendingUserMessages[:0]

		totalTokensUsed += result.Stats.Tokens.TotalTokens
		if r.maxBudget > 0 && totalTokensUsed > r.maxBudget {
			log.Printf("[RLM] turn %d — token budget exhausted (%d/%d)", turn, totalTokensUsed, r.maxBudget)
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
		log.Printf("[RLM] turn %d — LLM responded in %s (%d chars, %d code blocks)\n  Response: %s",
			turn, time.Since(iterStart).Round(time.Millisecond), len(response), len(codeBlocks), preview)

		var newUserMessages []Prompt
		iterationHadError := false

		for j, code := range codeBlocks {
			log.Printf("[RLM] turn %d — executing code block %d/%d (%d chars)...",
				turn, j+1, len(codeBlocks), len(code))
			replResult, execErr := repl.ExecuteCode(code)
			if execErr != nil || (replResult != nil && strings.TrimSpace(replResult.Stderr) != "") {
				iterationHadError = true
				if execErr != nil {
					log.Printf("[RLM] turn %d — code block %d exec error: %v", turn, j+1, execErr)
				}
				if replResult != nil && strings.TrimSpace(replResult.Stderr) != "" {
					log.Printf("[RLM] turn %d — code block %d stderr: %.200s", turn, j+1, replResult.Stderr)
				}
			}

			if replResult != nil && replResult.FinalAnswer != "" {
				log.Printf("[RLM] turn %d — FINAL answer from REPL: %.100s", turn, replResult.FinalAnswer)
				return replResult.FinalAnswer, nil
			}

			formatted := FormatREPLResult(replResult)
			if len(formatted) > 20000 {
				formatted = formatted[:20000] + fmt.Sprintf("... + [%d chars...]", len(formatted)-20000)
			}
			log.Printf("[RLM] turn %d — code block %d output: %.200s", turn, j+1, formatted)
			newUserMessages = append(newUserMessages, Prompt{
				Role:    "user",
				Content: fmt.Sprintf("Code executed:\n```python\n%s\n```\n\nREPL output:\n%s", code, formatted),
			})
		}

		if finalAnswer, finalErr := findFinalAnswer(response); finalErr == nil {
			log.Printf("[RLM] turn %d — FINAL() found in response: %.100s", turn, finalAnswer)
			return finalAnswer, nil
		}

		if varName, ok := findFinalVar(response); ok {
			log.Printf("[RLM] turn %d — FINAL_VAR(%s) found, resolving...", turn, varName)
			if res, err := repl.ExecuteCode(fmt.Sprintf("FINAL_VAR(%q)", varName)); err == nil && res != nil && res.FinalAnswer != "" {
				return res.FinalAnswer, nil
			}
		}

		if iterationHadError {
			consecutiveErrors++
			if r.maxErrors > 0 && consecutiveErrors >= r.maxErrors {
				log.Printf("[RLM] turn %d — consecutive error limit (%d) reached", turn, r.maxErrors)
				if bestPartialAnswer != "" {
					return bestPartialAnswer, nil
				}
				break
			}
		} else {
			consecutiveErrors = 0
		}

		pendingUserMessages = append(pendingUserMessages, newUserMessages...)

		if strings.TrimSpace(response) != "" {
			bestPartialAnswer = response
		}
	}

	log.Printf("[RLM] loop ended after %d turns without FINAL — requesting fallback answer...", turn)
	fallbackInputs := append([]Prompt(nil), pendingUserMessages...)
	fallbackInputs = append(fallbackInputs, Prompt{
		Role:    "user",
		Content: "You have used many turns. Based on everything above, provide your final answer to the original query now. Output ONLY the answer, nothing else.",
	})
	if fallbackResult, err := chat.SendMessages(ctx, fallbackInputs); err == nil && strings.TrimSpace(fallbackResult.Text) != "" {
		return fallbackResult.Text, nil
	}

	if bestPartialAnswer != "" {
		return bestPartialAnswer, nil
	}
	return "", errors.New("RLM loop ended without final answer")
}

func promptChars(msgs []Prompt) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
	}
	return n
}
