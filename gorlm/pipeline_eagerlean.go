package myrlm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"
)

// completionEagerLean combines the best of Eager and Lean:
//   - Pre-seeds the REPL with parsed context + helper module (from Eager)
//   - Strips code echo from feedback (from Lean)
//   - Merges multi-block output into a single message (from Lean)
//   - Uses the eager system prompt (skip inspection)
//   - Adds a "verify classification counts" instruction to catch misclassifications
func (r *RLM) completionEagerLean(ctx context.Context, rlmCtx Context, query Query, startServer bool) (string, error) {
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

	// Write helper module
	helperCode := `from collections import defaultdict

VALID_LABELS = {'abbreviation', 'entity', 'location', 'description and abstract concept', 'human being', 'numeric value'}

def norm_label(s):
    s = s.strip().lower()
    for vl in VALID_LABELS:
        if vl in s:
            return vl
    return s

def parse_classification_results(results):
    user_labels = defaultdict(list)
    for batch in results:
        for line in batch.strip().splitlines():
            line = line.strip()
            if '|' not in line:
                continue
            parts = line.split('|', 1)
            try:
                uid = int(parts[0].strip())
            except ValueError:
                continue
            label = norm_label(parts[1])
            if label in VALID_LABELS:
                user_labels[uid].append(label)
    return dict(user_labels)
`
	repl.WriteTempFile("rlm_helpers.py", helperCode)

	// Pre-seed REPL
	preseedCode := `
import re, json

_lines = context.strip().split('\n')
_header = _lines[0] if _lines else ""
_data_lines = [l for l in _lines if '||' in l]
_sample = _data_lines[:3] if _data_lines else _lines[:3]

_pattern = re.compile(r"User:\s*(\d+)\s*\|\|.*?Instance:\s*(.+)")
records = []
for l in _data_lines:
    m = _pattern.search(l)
    if m:
        records.append((int(m.group(1)), m.group(2).strip()))
_user_ids = sorted(set(uid for uid, _ in records))

_context_info = {
    "total_chars": len(context),
    "total_lines": len(_lines),
    "data_lines": len(_data_lines),
    "parsed_records": len(records),
    "unique_users": len(_user_ids),
    "header": _header[:200],
    "sample_lines": [l[:200] for l in _sample],
}
print(json.dumps(_context_info, indent=2))
`
	preseedResult, preseedErr := repl.ExecuteCode(preseedCode)
	preseedOutput := "Pre-parsed context info:\n"
	if preseedErr != nil {
		preseedOutput += fmt.Sprintf("(preseed error: %v)", preseedErr)
	} else if preseedResult != nil {
		preseedOutput += FormatREPLResult(preseedResult)
	}

	systemPrompt := GetEagerLeanSystemPrompt(r.customTools)
	chat := r.client.NewChatWithInstructions(systemPrompt)
	if r.maxTokens != nil {
		chat.SetMaxOutputTokens(*r.maxTokens)
	}

	pendingUserMessages := []Prompt{
		{Role: "user", Content: fmt.Sprintf(
			"Your context is a %s with %d total characters, loaded into `context` in the REPL.\n"+
				"Records pre-parsed into `records` (list of (user_id, question) tuples).\n"+
				"Helper: `from rlm_helpers import parse_classification_results, norm_label`\n\n%s",
			rlmCtx.Metadata.Type, rlmCtx.Metadata.Length, preseedOutput)},
	}

	var (
		bestPartialAnswer string
		totalTokensUsed   int64
		consecutiveErrors int
	)

	for i := 0; i < r.maxIterations; i++ {
		userPrompt := BuildUserPrompt(query, i)

		currentInputs := append([]Prompt(nil), pendingUserMessages...)
		currentInputs = append(currentInputs, userPrompt)
		log.Printf("[RLM-eagerlean] iter %d/%d — calling LLM (%d messages, ~%d chars)...",
			i+1, r.maxIterations, len(currentInputs), promptChars(currentInputs))
		iterStart := time.Now()

		result, err := chat.SendMessages(ctx, currentInputs)
		if err != nil {
			return "", fmt.Errorf("root LLM call (iteration %d): %w", i, err)
		}
		pendingUserMessages = pendingUserMessages[:0]

		totalTokensUsed += result.Stats.Tokens.TotalTokens
		if r.maxBudget != nil && totalTokensUsed > *r.maxBudget {
			log.Printf("[RLM-eagerlean] iter %d — token budget exhausted (%d/%d)", i+1, totalTokensUsed, *r.maxBudget)
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
		log.Printf("[RLM-eagerlean] iter %d/%d — LLM responded in %s (%d chars, %d code blocks)\n  Response: %s",
			i+1, r.maxIterations, time.Since(iterStart).Round(time.Millisecond), len(response), len(codeBlocks), preview)

		var outputParts []string
		iterationHadError := false

		for j, code := range codeBlocks {
			log.Printf("[RLM-eagerlean] iter %d — executing code block %d/%d (%d chars)...",
				i+1, j+1, len(codeBlocks), len(code))
			replResult, execErr := repl.ExecuteCode(code)

			hasError := execErr != nil || (replResult != nil && strings.TrimSpace(replResult.Stderr) != "")
			if hasError {
				iterationHadError = true
				if execErr != nil {
					log.Printf("[RLM-eagerlean] iter %d — code block %d exec error: %v", i+1, j+1, execErr)
				}
			}

			if replResult != nil && replResult.FinalAnswer != "" {
				log.Printf("[RLM-eagerlean] iter %d — FINAL answer: %.100s", i+1, replResult.FinalAnswer)
				return replResult.FinalAnswer, nil
			}

			formatted := leanFormatOutput(replResult, hasError, j+1, len(codeBlocks), code)
			log.Printf("[RLM-eagerlean] iter %d — code block %d output: %.200s", i+1, j+1, formatted)
			outputParts = append(outputParts, formatted)
		}

		if finalAnswer, finalErr := findFinalAnswer(response); finalErr == nil {
			log.Printf("[RLM-eagerlean] iter %d — FINAL() found: %.100s", i+1, finalAnswer)
			return finalAnswer, nil
		}

		if varName, ok := findFinalVar(response); ok {
			log.Printf("[RLM-eagerlean] iter %d — FINAL_VAR(%s) found", i+1, varName)
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

		// Single merged output message, no code echo
		if len(outputParts) > 0 {
			merged := strings.Join(outputParts, "\n---\n")
			if len(merged) > 15000 {
				merged = merged[:15000] + "\n... [truncated]"
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

	log.Printf("[RLM-eagerlean] exhausted %d iterations — requesting fallback...", r.maxIterations)
	fallbackInputs := append([]Prompt(nil), pendingUserMessages...)
	fallbackInputs = append(fallbackInputs, Prompt{
		Role:    "user",
		Content: "You have run out of iterations. Provide your final answer now. Output ONLY the answer.",
	})
	if fallbackResult, err := chat.SendMessages(ctx, fallbackInputs); err == nil && strings.TrimSpace(fallbackResult.Text) != "" {
		return fallbackResult.Text, nil
	}

	if bestPartialAnswer != "" {
		return bestPartialAnswer, nil
	}
	return "", errors.New("max iterations reached without final answer")
}
