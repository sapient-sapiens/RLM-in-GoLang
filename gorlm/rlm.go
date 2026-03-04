package myrlm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	DefaultMaxIterations = 10
	DefaultMaxDepth      = 1
	DefaultDepth         = 0
)

type RLM struct {
	depth         int
	maxIterations int
	maxDepth      int
	maxBudget     *float64
	maxTimeout    *float64
	maxTokens     *int
	maxErrors     *int
	systemPrompt  string
	customTools   []Tool
	dockerConfig  DockerConfig
	client        *OpenAIClient
}

func NewRLM(client *OpenAIClient, opts ...RLMOption) *RLM {
	r := &RLM{
		depth:         DefaultDepth,
		maxIterations: DefaultMaxIterations,
		maxDepth:      DefaultMaxDepth,
		systemPrompt:  DEFAULT_SYSTEM_PROMPT.Content,
		client:        client,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

type RLMOption func(*RLM)

func WithDepth(d int) RLMOption              { return func(r *RLM) { r.depth = d } }
func WithMaxIterations(n int) RLMOption      { return func(r *RLM) { r.maxIterations = n } }
func WithMaxDepth(d int) RLMOption           { return func(r *RLM) { r.maxDepth = d } }
func WithMaxBudget(b float64) RLMOption      { return func(r *RLM) { r.maxBudget = &b } }
func WithMaxTimeout(t float64) RLMOption     { return func(r *RLM) { r.maxTimeout = &t } }
func WithMaxTokens(t int) RLMOption          { return func(r *RLM) { r.maxTokens = &t } }
func WithMaxErrors(e int) RLMOption          { return func(r *RLM) { r.maxErrors = &e } }
func WithSystemPrompt(p string) RLMOption    { return func(r *RLM) { r.systemPrompt = p } }
func WithCustomTools(tools []Tool) RLMOption { return func(r *RLM) { r.customTools = tools } }
func WithDockerConfig(cfg DockerConfig) RLMOption {
	return func(r *RLM) { r.dockerConfig = cfg }
}

// Completion runs the RLM iteration loop: prompt the root LLM (via Go OpenAI client),
// parse code blocks, execute them in a Docker REPL, feed output back, repeat until
// a FINAL answer is found.
func (r *RLM) Completion(ctx context.Context, rlmCtx Context, query Query) (string, error) {
	if r.client == nil {
		return "", errors.New("nil OpenAI client")
	}

	if r.maxTimeout != nil && *r.maxTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(*r.maxTimeout*float64(time.Second)))
		defer cancel()
	}
	srv := NewREPLServer(r.client, r)
	if err := srv.Start(); err != nil {
		return "", fmt.Errorf("start repl server: %w", err)
	}
	defer srv.Shutdown(ctx)

	repl, err := NewDockerREPL(r.dockerConfig)
	if err != nil {
		return "", fmt.Errorf("create docker repl: %w", err)
	}
	defer repl.Close()

	if err := repl.LoadContext(rlmCtx.Content); err != nil {
		return "", fmt.Errorf("load context: %w", err)
	}

	systemPrompt := GetSystemPromptWithCustomToolsFrom(r.systemPrompt, r.customTools)
	messageHistory := BuildInitialMessages(systemPrompt, rlmCtx.Metadata)

	var bestPartialAnswer string
	consecutiveErrors := 0

	for i := 0; i < r.maxIterations; i++ {
		userPrompt := BuildUserPrompt(query, i)

		currentPrompt := make([]Prompt, len(messageHistory))
		copy(currentPrompt, messageHistory)
		currentPrompt = append(currentPrompt, userPrompt)

		result, err := r.client.QueryMessages(ctx, currentPrompt)
		if err != nil {
			return "", fmt.Errorf("root LLM call (iteration %d): %w", i, err)
		}
		response := result.Text
		codeBlocks := findCodeBlocks(response)

		var newMessages []Prompt
		newMessages = append(newMessages, Prompt{Role: "assistant", Content: response})
		iterationHadError := false

		for _, code := range codeBlocks {
			replResult, execErr := repl.ExecuteCode(code)
			if execErr != nil || (replResult != nil && strings.TrimSpace(replResult.Stderr) != "") {
				iterationHadError = true
			}

			if replResult != nil && replResult.FinalAnswer != "" {
				return replResult.FinalAnswer, nil
			}

			formatted := FormatREPLResult(replResult)
			if len(formatted) > 20000 {
				formatted = formatted[:20000] + fmt.Sprintf("... + [%d chars...]", len(formatted)-20000)
			}
			newMessages = append(newMessages, Prompt{
				Role:    "user",
				Content: fmt.Sprintf("Code executed:\n```python\n%s\n```\n\nREPL output:\n%s", code, formatted),
			})
		}

		// Check FINAL(...) after REPL execution.
		if finalAnswer, finalErr := findFinalAnswer(response); finalErr == nil {
			return finalAnswer, nil
		}

		// Check FINAL_VAR(name) in raw text — resolve by executing in the REPL.
		if varName, ok := findFinalVar(response); ok {
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

		messageHistory = append(messageHistory, newMessages...)

		if strings.TrimSpace(response) != "" {
			bestPartialAnswer = response
		}
	}

	if bestPartialAnswer != "" {
		return bestPartialAnswer, nil
	}
	// Let it error out if no answer retrieved after max iterations
	return "", errors.New("max iterations reached without final answer")
}
