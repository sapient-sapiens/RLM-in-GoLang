package myrlm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"
)

// completionLean implements the "Lean Context" strategy:
//
//  1. Sends the full query only on turn 1; subsequent turns get a short reminder.
//  2. Strips code echo from feedback — never re-sends code the model wrote.
//  3. Compresses REPL output: collapses whitespace, caps per-block output, and
//     replaces purely diagnostic output with a one-line summary.
//  4. On errors: includes only the traceback + the offending code, not the full
//     output history.
//  5. Tracks what variables exist in the REPL and tells the model on each turn,
//     so it doesn't waste time checking.
func (r *RLM) completionLean(ctx context.Context, rlmCtx Context, query Query, startServer bool) (string, error) {
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

	systemPrompt := GetLeanSystemPrompt(r.customTools)
	chat := r.client.NewChatWithInstructions(systemPrompt)
	if r.maxTokens != nil {
		chat.SetMaxOutputTokens(*r.maxTokens)
	}

	initialMessages := BuildInitialMessages(systemPrompt, rlmCtx.Metadata)
	pendingUserMessages := []Prompt{initialMessages[1]}

	var (
		bestPartialAnswer string
		totalTokensUsed   int64
		consecutiveErrors int
		knownVars         []string
	)

	for i := 0; i < r.maxIterations; i++ {
		// --- Lean Change 1: Short query reminder after turn 1 ---
		userPrompt := buildLeanUserPrompt(query, i, knownVars)

		currentInputs := append([]Prompt(nil), pendingUserMessages...)
		currentInputs = append(currentInputs, userPrompt)
		log.Printf("[RLM-lean] iter %d/%d — calling LLM (%d messages, ~%d chars)...",
			i+1, r.maxIterations, len(currentInputs), promptChars(currentInputs))
		iterStart := time.Now()

		result, err := chat.SendMessages(ctx, currentInputs)
		if err != nil {
			return "", fmt.Errorf("root LLM call (iteration %d): %w", i, err)
		}
		pendingUserMessages = pendingUserMessages[:0]

		totalTokensUsed += result.Stats.Tokens.TotalTokens
		if r.maxBudget != nil && totalTokensUsed > *r.maxBudget {
			log.Printf("[RLM-lean] iter %d — token budget exhausted (%d/%d)", i+1, totalTokensUsed, *r.maxBudget)
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
		log.Printf("[RLM-lean] iter %d/%d — LLM responded in %s (%d chars, %d code blocks)\n  Response: %s",
			i+1, r.maxIterations, time.Since(iterStart).Round(time.Millisecond), len(response), len(codeBlocks), preview)

		// Track new variables from this turn's code
		newVars := extractNewVars(response)
		for _, v := range newVars {
			found := false
			for _, kv := range knownVars {
				if kv == v {
					found = true
					break
				}
			}
			if !found {
				knownVars = append(knownVars, v)
			}
		}

		var outputParts []string
		iterationHadError := false

		for j, code := range codeBlocks {
			log.Printf("[RLM-lean] iter %d — executing code block %d/%d (%d chars)...",
				i+1, j+1, len(codeBlocks), len(code))
			replResult, execErr := repl.ExecuteCode(code)

			hasError := execErr != nil || (replResult != nil && strings.TrimSpace(replResult.Stderr) != "")
			if hasError {
				iterationHadError = true
				if execErr != nil {
					log.Printf("[RLM-lean] iter %d — code block %d exec error: %v", i+1, j+1, execErr)
				}
				if replResult != nil && strings.TrimSpace(replResult.Stderr) != "" {
					log.Printf("[RLM-lean] iter %d — code block %d stderr: %.200s", i+1, j+1, replResult.Stderr)
				}
			}

			if replResult != nil && replResult.FinalAnswer != "" {
				log.Printf("[RLM-lean] iter %d — FINAL answer from REPL: %.100s", i+1, replResult.FinalAnswer)
				return replResult.FinalAnswer, nil
			}

			// --- Lean Change 2: Compress and structure output ---
			formatted := leanFormatOutput(replResult, hasError, j+1, len(codeBlocks), code)
			log.Printf("[RLM-lean] iter %d — code block %d output: %.200s", i+1, j+1, formatted)
			outputParts = append(outputParts, formatted)
		}

		if finalAnswer, finalErr := findFinalAnswer(response); finalErr == nil {
			log.Printf("[RLM-lean] iter %d — FINAL() found in response: %.100s", i+1, finalAnswer)
			return finalAnswer, nil
		}

		if varName, ok := findFinalVar(response); ok {
			log.Printf("[RLM-lean] iter %d — FINAL_VAR(%s) found, resolving...", i+1, varName)
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

		// --- Lean Change 3: Single merged output message, no code echo ---
		if len(outputParts) > 0 {
			merged := strings.Join(outputParts, "\n---\n")
			if len(merged) > 15000 {
				merged = merged[:15000] + "\n... [output truncated at 15K chars]"
			}
			pendingUserMessages = append(pendingUserMessages, Prompt{
				Role:    "user",
				Content: merged,
			})
		}

		if strings.TrimSpace(response) != "" {
			bestPartialAnswer = response
		}
	}

	log.Printf("[RLM-lean] exhausted %d iterations without FINAL — requesting fallback answer...", r.maxIterations)
	fallbackInputs := append([]Prompt(nil), pendingUserMessages...)
	fallbackInputs = append(fallbackInputs, Prompt{
		Role:    "user",
		Content: "You have run out of iterations. Based on everything above, provide your final answer now. Output ONLY the answer.",
	})
	if fallbackResult, err := chat.SendMessages(ctx, fallbackInputs); err == nil && strings.TrimSpace(fallbackResult.Text) != "" {
		return fallbackResult.Text, nil
	}

	if bestPartialAnswer != "" {
		return bestPartialAnswer, nil
	}
	return "", errors.New("max iterations reached without final answer")
}

// buildLeanUserPrompt creates a lean prompt:
//   - Turn 1: full query with instructions
//   - Turn 2+: full query (necessary for complex constraints) + variable inventory
func buildLeanUserPrompt(query Query, turn int, knownVars []string) Prompt {
	turnInfo := fmt.Sprintf("[Turn %d] ", turn+1)

	if turn == 0 {
		content := turnInfo +
			"Task:\n\"" + string(query) + "\"\n\n" +
			"Delegate semantic work to llm_query_batched. REPL is for parsing/computation only. Begin:"
		return Prompt{Role: "user", Content: content}
	}

	var b strings.Builder
	b.WriteString(turnInfo)
	b.WriteString("Continue on: \"")
	b.WriteString(string(query))
	b.WriteString("\"\n")
	if len(knownVars) > 0 {
		b.WriteString("REPL vars: ")
		b.WriteString(strings.Join(knownVars, ", "))
		b.WriteString("\n")
	}
	b.WriteString("When done, call FINAL_VAR(variable_name).")
	return Prompt{Role: "user", Content: b.String()}
}

// leanFormatOutput formats REPL output with compression and structure.
func leanFormatOutput(r *REPLResult, hadError bool, blockIdx, blockCount int, code string) string {
	if r == nil {
		return fmt.Sprintf("[Block %d/%d] No output", blockIdx, blockCount)
	}

	var b strings.Builder

	if hadError && r.Stderr != "" {
		// On error: show the error clearly, include only the failing code line if identifiable
		fmt.Fprintf(&b, "[Block %d/%d ERROR]\n", blockIdx, blockCount)
		stderr := strings.TrimSpace(r.Stderr)
		// Extract just the last traceback line (the actual error)
		lines := strings.Split(stderr, "\n")
		if len(lines) > 3 {
			// Show last 3 lines of traceback (usually the most informative)
			b.WriteString(strings.Join(lines[len(lines)-3:], "\n"))
		} else {
			b.WriteString(stderr)
		}
		if r.Stdout != "" {
			stdout := strings.TrimSpace(r.Stdout)
			if len(stdout) > 500 {
				stdout = stdout[:500] + "..."
			}
			fmt.Fprintf(&b, "\nPartial stdout: %s", stdout)
		}
		return b.String()
	}

	stdout := strings.TrimSpace(r.Stdout)
	if stdout == "" {
		return fmt.Sprintf("[Block %d/%d] OK (no output)", blockIdx, blockCount)
	}

	// Compress: collapse multiple blank lines, cap length
	stdout = compressOutput(stdout)

	if blockCount == 1 {
		return stdout
	}
	fmt.Fprintf(&b, "[Block %d/%d]\n%s", blockIdx, blockCount, stdout)
	return b.String()
}

// compressOutput reduces whitespace and caps output length.
func compressOutput(s string) string {
	lines := strings.Split(s, "\n")
	var compressed []string
	prevBlank := false
	for _, line := range lines {
		trimmed := strings.TrimRight(line, " \t")
		if trimmed == "" {
			if !prevBlank {
				compressed = append(compressed, "")
				prevBlank = true
			}
			continue
		}
		prevBlank = false
		compressed = append(compressed, trimmed)
	}
	result := strings.Join(compressed, "\n")
	if len(result) > 15000 {
		result = result[:15000] + fmt.Sprintf("\n... [+%d chars truncated]", len(result)-15000)
	}
	return result
}
