package myrlm

import (
	"errors"
	"regexp"
	"strings"
)

var (
	codeBlockPattern = regexp.MustCompile("(?s)```(?:repl|python)\\s*\\n(.*?)\\n```")
	finalPattern     = regexp.MustCompile(`(?ms)^\s*FINAL\((.*)\)\s*$`)
	finalVarPattern  = regexp.MustCompile(`(?ms)^\s*FINAL_VAR\(["']?(\w+)["']?\)\s*$`)
)

func findCodeBlocks(text string) []string {
	matches := codeBlockPattern.FindAllStringSubmatch(text, -1)
	blocks := make([]string, 0, len(matches))
	for _, m := range matches {
		code := strings.TrimSpace(m[1])
		if code != "" {
			blocks = append(blocks, code)
		}
	}
	return blocks
}

func findFinalAnswer(text string) (string, error) {
	if match := finalPattern.FindStringSubmatch(text); len(match) > 1 {
		return match[1], nil
	}
	return "", errors.New("no final answer found")
}

// findFinalVar detects FINAL_VAR(name) in raw response text (outside code blocks).
func findFinalVar(text string) (string, bool) {
	if match := finalVarPattern.FindStringSubmatch(text); len(match) > 1 {
		return match[1], true
	}
	return "", false
}
