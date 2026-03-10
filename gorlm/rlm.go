package myrlm

import (
	"context"
	"encoding/base64"
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

// PipelineMode selects the RLM pipeline strategy.
type PipelineMode int

const (
	PipelineStandard PipelineMode = iota
	PipelineCompact
	PipelineEager
	PipelineLean
	PipelineEagerLean
)

type RLM struct {
	maxIterations  int
	maxDepth       int
	maxTimeout     *float64
	maxTokens      *int64
	maxBudget      *int64
	maxErrors      *int
	customTools    []Tool
	dockerConfig   DockerConfig
	client         *OpenAIClient
	compactHistory bool
	pipelineMode   PipelineMode
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
func WithCompactHistory(enabled bool) RLMOption {
	return func(r *RLM) { r.compactHistory = enabled }
}
func WithPipelineMode(mode PipelineMode) RLMOption {
	return func(r *RLM) { r.pipelineMode = mode }
}

// Completion runs the RLM iteration loop: prompt the root LLM (via Go OpenAI client),
// parse code blocks, execute them in a Docker REPL, feed output back, repeat until
// a FINAL answer is found.
func (r *RLM) Completion(ctx context.Context, rlmCtx Context, query Query) (string, error) {
	switch r.pipelineMode {
	case PipelineEager:
		return r.completionEager(ctx, rlmCtx, query, true)
	case PipelineLean:
		return r.completionLean(ctx, rlmCtx, query, true)
	case PipelineEagerLean:
		return r.completionEagerLean(ctx, rlmCtx, query, true)
	case PipelineCompact:
		return r.completionCompacted(ctx, rlmCtx, query, true)
	default:
		if r.compactHistory {
			return r.completionCompacted(ctx, rlmCtx, query, true)
		}
		return r.completion(ctx, rlmCtx, query, true)
	}
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

// turnRecord stores what happened in a single RLM turn for the context ledger.
type turnRecord struct {
	Turn         int
	CodeBlocks   int
	HadError     bool
	VarsCreated  []string
	OutputPreview string // first ~200 chars of combined REPL output
	AssistantLen int    // length of the full assistant response
}

// buildLedger creates a compact summary of all past turns for inclusion in the prompt.
func buildLedger(records []turnRecord) string {
	if len(records) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("## Context Ledger — History of Previous Turns\n")
	b.WriteString("Each turn's full assistant response and REPL output are stored in REPL variables.\n")
	b.WriteString("To recall details: `print(_turn_N_response)` or `print(_turn_N_output)`\n\n")
	for _, rec := range records {
		status := "OK"
		if rec.HadError {
			status = "ERROR"
		}
		fmt.Fprintf(&b, "- **Turn %d** [%s]: %d code blocks executed, assistant response=%d chars",
			rec.Turn, status, rec.CodeBlocks, rec.AssistantLen)
		if len(rec.VarsCreated) > 0 {
			fmt.Fprintf(&b, ", vars: %s", strings.Join(rec.VarsCreated, ", "))
		}
		b.WriteString("\n")
		if rec.OutputPreview != "" {
			fmt.Fprintf(&b, "  Output preview: %s\n", rec.OutputPreview)
		}
	}
	return b.String()
}

// storeInREPL saves a value into a REPL variable via base64 to avoid escaping issues.
func storeInREPL(repl *DockerREPL, varName, value string) {
	encoded := b64encode(value)
	code := fmt.Sprintf("import base64 as _b64; %s = _b64.b64decode(%q).decode('utf-8', errors='replace')", varName, encoded)
	if _, err := repl.ExecuteCode(code); err != nil {
		log.Printf("[RLM-compact] warning: failed to store %s in REPL: %v", varName, err)
	}
}

func b64encode(s string) string {
	return base64.StdEncoding.EncodeToString([]byte(s))
}

// completionCompacted is the context-compacted variant of the RLM loop.
// Instead of letting the full conversation history accumulate via PreviousResponseID,
// it maintains an explicit "context ledger" — a compact summary of what happened
// each turn — and stores full responses in REPL variables for on-demand recall.
func (r *RLM) completionCompacted(ctx context.Context, rlmCtx Context, query Query, startServer bool) (string, error) {
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

	systemPrompt := GetCompactedSystemPrompt(r.customTools)
	contextInfo := fmt.Sprintf(
		"Your context is a %s with %d total characters, "+
			"loaded into the `context` variable in the REPL. "+
			"Use ```repl``` code blocks to work with it.",
		rlmCtx.Metadata.Type, rlmCtx.Metadata.Length)

	var (
		bestPartialAnswer string
		totalTokensUsed   int64
		consecutiveErrors int
		ledger            []turnRecord
	)

	var lastTurnOutput string // full REPL output from the most recent turn

	for i := 0; i < r.maxIterations; i++ {
		// Build messages from scratch each turn: system + context info + ledger + last output + user prompt
		messages := []Prompt{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: contextInfo},
		}

		// Add the context ledger for all but the most recent turn (compacted)
		if len(ledger) > 1 {
			messages = append(messages, Prompt{
				Role:    "user",
				Content: buildLedger(ledger[:len(ledger)-1]),
			})
		}

		// Include the most recent turn's full REPL output so the model has
		// immediate context of what just happened (not compacted)
		if len(ledger) > 0 {
			lastRec := ledger[len(ledger)-1]
			recentMsg := fmt.Sprintf("[Turn %d completed] Your code produced this output:\n\n%s",
				lastRec.Turn, lastTurnOutput)
			if len(recentMsg) > 20000 {
				recentMsg = recentMsg[:20000] + "... [truncated]"
			}
			messages = append(messages, Prompt{
				Role:    "user",
				Content: recentMsg,
			})
		}

		// Add the user prompt for this turn
		userPrompt := BuildUserPrompt(query, i)
		messages = append(messages, userPrompt)

		log.Printf("[RLM-compact] iter %d/%d — calling LLM (%d messages, ~%d chars)...",
			i+1, r.maxIterations, len(messages), promptChars(messages))
		iterStart := time.Now()

		// Use QueryMessages directly — no PreviousResponseID chaining
		result, err := r.client.QueryMessages(ctx, messages)
		if err != nil {
			return "", fmt.Errorf("root LLM call (iteration %d): %w", i, err)
		}

		totalTokensUsed += result.Stats.Tokens.TotalTokens
		if r.maxBudget != nil && totalTokensUsed > *r.maxBudget {
			log.Printf("[RLM-compact] iter %d — token budget exhausted (%d/%d)", i+1, totalTokensUsed, *r.maxBudget)
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
		log.Printf("[RLM-compact] iter %d/%d — LLM responded in %s (%d chars, %d code blocks)\n  Response: %s",
			i+1, r.maxIterations, time.Since(iterStart).Round(time.Millisecond), len(response), len(codeBlocks), preview)

		// Store the full assistant response in a REPL variable
		storeInREPL(repl, fmt.Sprintf("_turn_%d_response", i+1), response)

		rec := turnRecord{
			Turn:        i + 1,
			CodeBlocks:  len(codeBlocks),
			AssistantLen: len(response),
		}

		var combinedOutput strings.Builder
		iterationHadError := false

		for j, code := range codeBlocks {
			log.Printf("[RLM-compact] iter %d — executing code block %d/%d (%d chars)...",
				i+1, j+1, len(codeBlocks), len(code))
			replResult, execErr := repl.ExecuteCode(code)
			if execErr != nil || (replResult != nil && strings.TrimSpace(replResult.Stderr) != "") {
				iterationHadError = true
				rec.HadError = true
				if execErr != nil {
					log.Printf("[RLM-compact] iter %d — code block %d exec error: %v", i+1, j+1, execErr)
				}
				if replResult != nil && strings.TrimSpace(replResult.Stderr) != "" {
					log.Printf("[RLM-compact] iter %d — code block %d stderr: %.200s", i+1, j+1, replResult.Stderr)
				}
			}

			if replResult != nil && replResult.FinalAnswer != "" {
				log.Printf("[RLM-compact] iter %d — FINAL answer from REPL: %.100s", i+1, replResult.FinalAnswer)
				return replResult.FinalAnswer, nil
			}

			formatted := FormatREPLResult(replResult)
			if len(formatted) > 20000 {
				formatted = formatted[:20000] + fmt.Sprintf("... + [%d chars...]", len(formatted)-20000)
			}
			log.Printf("[RLM-compact] iter %d — code block %d output: %.200s", i+1, j+1, formatted)
			combinedOutput.WriteString(formatted)
			combinedOutput.WriteString("\n")
		}

		// Store combined REPL output in a variable too
		outputStr := combinedOutput.String()
		storeInREPL(repl, fmt.Sprintf("_turn_%d_output", i+1), outputStr)
		lastTurnOutput = outputStr

		// Build the output preview for the ledger (used for older turns)
		outputPreview := strings.TrimSpace(outputStr)
		if len(outputPreview) > 300 {
			outputPreview = outputPreview[:300] + "..."
		}
		rec.OutputPreview = outputPreview

		// Detect variables created by parsing SHOW_VARS or checking output
		rec.VarsCreated = extractNewVars(response)

		if finalAnswer, finalErr := findFinalAnswer(response); finalErr == nil {
			log.Printf("[RLM-compact] iter %d — FINAL() found in response: %.100s", i+1, finalAnswer)
			return finalAnswer, nil
		}

		if varName, ok := findFinalVar(response); ok {
			log.Printf("[RLM-compact] iter %d — FINAL_VAR(%s) found, resolving...", i+1, varName)
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

		// Append this turn's record to the ledger
		ledger = append(ledger, rec)

		if strings.TrimSpace(response) != "" {
			bestPartialAnswer = response
		}
	}

	log.Printf("[RLM-compact] exhausted %d iterations without FINAL — requesting fallback answer...", r.maxIterations)
	fallbackMessages := []Prompt{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: contextInfo},
	}
	if len(ledger) > 0 {
		fallbackMessages = append(fallbackMessages, Prompt{
			Role:    "user",
			Content: buildLedger(ledger),
		})
		if lastTurnOutput != "" {
			recentMsg := fmt.Sprintf("[Most recent output from Turn %d]:\n%s", ledger[len(ledger)-1].Turn, lastTurnOutput)
			if len(recentMsg) > 20000 {
				recentMsg = recentMsg[:20000] + "... [truncated]"
			}
			fallbackMessages = append(fallbackMessages, Prompt{
				Role:    "user",
				Content: recentMsg,
			})
		}
	}
	fallbackMessages = append(fallbackMessages, Prompt{
		Role:    "user",
		Content: "You have run out of iterations. Based on everything above, provide your final answer to the original query now. Output ONLY the answer, nothing else.",
	})
	if fallbackResult, err := r.client.QueryMessages(ctx, fallbackMessages); err == nil && strings.TrimSpace(fallbackResult.Text) != "" {
		return fallbackResult.Text, nil
	}

	if bestPartialAnswer != "" {
		return bestPartialAnswer, nil
	}
	return "", errors.New("max iterations reached without final answer")
}

// extractNewVars looks for Python variable assignments in the response code blocks.
func extractNewVars(response string) []string {
	blocks := findCodeBlocks(response)
	seen := make(map[string]bool)
	var vars []string
	for _, block := range blocks {
		for _, line := range strings.Split(block, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "#") || line == "" {
				continue
			}
			if idx := strings.Index(line, "="); idx > 0 {
				candidate := strings.TrimSpace(line[:idx])
				// Skip comparisons, augmented assignments, etc.
				if strings.ContainsAny(candidate, " +-*/<>!") {
					continue
				}
				if strings.HasPrefix(candidate, "_turn_") {
					continue
				}
				if !seen[candidate] && isValidPythonIdent(candidate) {
					seen[candidate] = true
					vars = append(vars, candidate)
				}
			}
		}
	}
	return vars
}

func isValidPythonIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, c := range s {
		if i == 0 {
			if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
				return false
			}
		} else {
			if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
				return false
			}
		}
	}
	return true
}
