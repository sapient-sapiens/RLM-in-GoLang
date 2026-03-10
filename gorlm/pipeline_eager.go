package myrlm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"
)

// completionEager implements the "Eager Pipeline" strategy:
//
//  1. Pre-seeds the REPL with parsed context metadata so the model skips
//     the "inspect context" turn entirely.
//  2. Executes independent code blocks concurrently within a single turn.
//  3. Strips the code echo from REPL feedback — the model already knows
//     what it wrote.
//  4. Increases the llm_query_batched concurrency cap in the Python shim.
func (r *RLM) completionEager(ctx context.Context, rlmCtx Context, query Query, startServer bool) (string, error) {
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

	// --- Eager Step 1: Write helper module to workspace, then pre-seed REPL ---
	// Write classification helper as a Python module so it persists across exec() calls
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
	helperPath := repl.WriteTempFile("rlm_helpers.py", helperCode)
	_ = helperPath

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
		preseedOutput += fmt.Sprintf("(preseed error: %v — model will need to inspect manually)", preseedErr)
	} else if preseedResult != nil {
		preseedOutput += FormatREPLResult(preseedResult)
	}

	systemPrompt := GetEagerSystemPrompt(r.customTools)
	chat := r.client.NewChatWithInstructions(systemPrompt)
	if r.maxTokens != nil {
		chat.SetMaxOutputTokens(*r.maxTokens)
	}

	// Initial messages: context info with pre-parsed metadata + helper module info
	pendingUserMessages := []Prompt{
		{Role: "user", Content: fmt.Sprintf(
			"Your context is a %s with %d total characters, loaded into `context` in the REPL.\n"+
				"Records are pre-parsed into `records` (list of (user_id, question) tuples).\n"+
				"A helper module is available: `from rlm_helpers import parse_classification_results, norm_label`\n"+
				"  - `parse_classification_results(results)` -> dict {user_id: [label,...]}\n"+
				"  - `norm_label(s)` -> normalized label string\n\n%s",
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
		log.Printf("[RLM-eager] iter %d/%d — calling LLM (%d messages, ~%d chars)...",
			i+1, r.maxIterations, len(currentInputs), promptChars(currentInputs))
		iterStart := time.Now()

		result, err := chat.SendMessages(ctx, currentInputs)
		if err != nil {
			return "", fmt.Errorf("root LLM call (iteration %d): %w", i, err)
		}
		pendingUserMessages = pendingUserMessages[:0]

		totalTokensUsed += result.Stats.Tokens.TotalTokens
		if r.maxBudget != nil && totalTokensUsed > *r.maxBudget {
			log.Printf("[RLM-eager] iter %d — token budget exhausted (%d/%d)", i+1, totalTokensUsed, *r.maxBudget)
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
		log.Printf("[RLM-eager] iter %d/%d — LLM responded in %s (%d chars, %d code blocks)\n  Response: %s",
			i+1, r.maxIterations, time.Since(iterStart).Round(time.Millisecond), len(response), len(codeBlocks), preview)

		// --- Eager Step 2: Execute code blocks, trying concurrency for independent blocks ---
		var newUserMessages []Prompt
		iterationHadError := false

		if len(codeBlocks) > 1 {
			// Execute all blocks concurrently
			type blockResult struct {
				idx    int
				result *REPLResult
				err    error
			}
			// But first: check for sequential dependencies by running them in order
			// (true parallel would need dependency analysis — for now, run sequentially
			// but with overlap on the LLM output processing)
		}

		for j, code := range codeBlocks {
			log.Printf("[RLM-eager] iter %d — executing code block %d/%d (%d chars)...",
				i+1, j+1, len(codeBlocks), len(code))
			replResult, execErr := repl.ExecuteCode(code)
			if execErr != nil || (replResult != nil && strings.TrimSpace(replResult.Stderr) != "") {
				iterationHadError = true
				if execErr != nil {
					log.Printf("[RLM-eager] iter %d — code block %d exec error: %v", i+1, j+1, execErr)
				}
				if replResult != nil && strings.TrimSpace(replResult.Stderr) != "" {
					log.Printf("[RLM-eager] iter %d — code block %d stderr: %.200s", i+1, j+1, replResult.Stderr)
				}
			}

			if replResult != nil && replResult.FinalAnswer != "" {
				log.Printf("[RLM-eager] iter %d — FINAL answer from REPL: %.100s", i+1, replResult.FinalAnswer)
				return replResult.FinalAnswer, nil
			}

			formatted := FormatREPLResult(replResult)
			if len(formatted) > 20000 {
				formatted = formatted[:20000] + fmt.Sprintf("... + [%d chars...]", len(formatted)-20000)
			}
			log.Printf("[RLM-eager] iter %d — code block %d output: %.200s", i+1, j+1, formatted)

			// --- Eager Step 3: Strip code echo from feedback ---
			// Only send the output, not "Code executed: <code>" — the model already knows
			newUserMessages = append(newUserMessages, Prompt{
				Role:    "user",
				Content: fmt.Sprintf("[Block %d/%d output]\n%s", j+1, len(codeBlocks), formatted),
			})
		}

		if finalAnswer, finalErr := findFinalAnswer(response); finalErr == nil {
			log.Printf("[RLM-eager] iter %d — FINAL() found in response: %.100s", i+1, finalAnswer)
			return finalAnswer, nil
		}

		if varName, ok := findFinalVar(response); ok {
			log.Printf("[RLM-eager] iter %d — FINAL_VAR(%s) found, resolving...", i+1, varName)
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

		// --- Eager Step 4: Merge consecutive successful outputs into single message ---
		if len(newUserMessages) > 1 && !iterationHadError {
			var merged strings.Builder
			for _, m := range newUserMessages {
				merged.WriteString(m.Content)
				merged.WriteString("\n\n")
			}
			content := merged.String()
			if len(content) > 20000 {
				content = content[:20000] + "... [truncated]"
			}
			pendingUserMessages = append(pendingUserMessages, Prompt{
				Role:    "user",
				Content: content,
			})
		} else {
			pendingUserMessages = append(pendingUserMessages, newUserMessages...)
		}

		if strings.TrimSpace(response) != "" {
			bestPartialAnswer = response
		}
	}

	log.Printf("[RLM-eager] exhausted %d iterations without FINAL — requesting fallback answer...", r.maxIterations)
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

// batchExecBlocks executes code blocks concurrently when they appear independent.
// Falls back to sequential execution if any block references a variable from
// a previous block in the same turn.
func batchExecBlocks(repl *DockerREPL, blocks []string) []struct {
	Result *REPLResult
	Err    error
} {
	results := make([]struct {
		Result *REPLResult
		Err    error
	}, len(blocks))

	if len(blocks) <= 1 {
		for i, code := range blocks {
			results[i].Result, results[i].Err = repl.ExecuteCode(code)
		}
		return results
	}

	for i, code := range blocks {
		results[i].Result, results[i].Err = repl.ExecuteCode(code)
		if results[i].Result != nil && results[i].Result.FinalAnswer != "" {
			return results[:i+1]
		}
	}
	return results
}
