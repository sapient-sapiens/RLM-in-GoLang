package myrlm

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

type TokenUsage struct {
	InputTokens      int64
	OutputTokens     int64
	TotalTokens      int64
	CachedTokens     int64
	ReasoningTokens  int64
}

type RequestStats struct {
	Model           string
	StartTime       time.Time
	EndTime         time.Time
	Duration        time.Duration
	TimeToFirstTok  time.Duration
	Streamed        bool
	Tokens          TokenUsage
	ResponseID      string
}

func (s RequestStats) TokensPerSecond() float64 {
	secs := s.Duration.Seconds()
	if secs == 0 {
		return 0
	}
	return float64(s.Tokens.OutputTokens) / secs
}

func (s RequestStats) InputTokensPerSecond() float64 {
	secs := s.Duration.Seconds()
	if secs == 0 {
		return 0
	}
	return float64(s.Tokens.InputTokens) / secs
}

func (s RequestStats) String() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("── Request Stats (%s) ──\n", s.ResponseID))
	b.WriteString(fmt.Sprintf("  Model:             %s\n", s.Model))
	b.WriteString(fmt.Sprintf("  Duration:          %s\n", s.Duration.Round(time.Millisecond)))
	if s.Streamed && s.TimeToFirstTok > 0 {
		b.WriteString(fmt.Sprintf("  Time to 1st token: %s\n", s.TimeToFirstTok.Round(time.Millisecond)))
	}
	b.WriteString(fmt.Sprintf("  Input tokens:      %d", s.Tokens.InputTokens))
	if s.Tokens.CachedTokens > 0 {
		b.WriteString(fmt.Sprintf(" (%d cached)", s.Tokens.CachedTokens))
	}
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  Output tokens:     %d", s.Tokens.OutputTokens))
	if s.Tokens.ReasoningTokens > 0 {
		b.WriteString(fmt.Sprintf(" (%d reasoning)", s.Tokens.ReasoningTokens))
	}
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  Total tokens:      %d\n", s.Tokens.TotalTokens))
	b.WriteString(fmt.Sprintf("  Output tok/s:      %.1f\n", s.TokensPerSecond()))
	return b.String()
}

type UsageTracker struct {
	mu       sync.Mutex
	requests []RequestStats
}

func NewUsageTracker() *UsageTracker {
	return &UsageTracker{}
}

func (t *UsageTracker) Record(stats RequestStats) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.requests = append(t.requests, stats)
}

func (t *UsageTracker) Requests() []RequestStats {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]RequestStats, len(t.requests))
	copy(out, t.requests)
	return out
}

func (t *UsageTracker) Summary() SessionSummary {
	t.mu.Lock()
	defer t.mu.Unlock()

	s := SessionSummary{TotalRequests: len(t.requests)}
	if len(t.requests) == 0 {
		return s
	}

	for _, r := range t.requests {
		s.TotalDuration += r.Duration
		s.TotalInputTokens += r.Tokens.InputTokens
		s.TotalOutputTokens += r.Tokens.OutputTokens
		s.TotalTokens += r.Tokens.TotalTokens
		s.TotalCachedTokens += r.Tokens.CachedTokens
		s.TotalReasoningTokens += r.Tokens.ReasoningTokens
	}

	if secs := s.TotalDuration.Seconds(); secs > 0 {
		s.AvgOutputTokPerSec = float64(s.TotalOutputTokens) / secs
	}
	s.AvgLatency = s.TotalDuration / time.Duration(s.TotalRequests)

	return s
}

type SessionSummary struct {
	TotalRequests        int
	TotalDuration        time.Duration
	AvgLatency           time.Duration
	TotalInputTokens     int64
	TotalOutputTokens    int64
	TotalTokens          int64
	TotalCachedTokens    int64
	TotalReasoningTokens int64
	AvgOutputTokPerSec   float64
}

func (s SessionSummary) String() string {
	var b strings.Builder
	b.WriteString("══ Session Summary ══\n")
	b.WriteString(fmt.Sprintf("  Requests:          %d\n", s.TotalRequests))
	b.WriteString(fmt.Sprintf("  Total duration:    %s\n", s.TotalDuration.Round(time.Millisecond)))
	b.WriteString(fmt.Sprintf("  Avg latency:       %s\n", s.AvgLatency.Round(time.Millisecond)))
	b.WriteString(fmt.Sprintf("  Input tokens:      %d", s.TotalInputTokens))
	if s.TotalCachedTokens > 0 {
		b.WriteString(fmt.Sprintf(" (%d cached)", s.TotalCachedTokens))
	}
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  Output tokens:     %d", s.TotalOutputTokens))
	if s.TotalReasoningTokens > 0 {
		b.WriteString(fmt.Sprintf(" (%d reasoning)", s.TotalReasoningTokens))
	}
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  Total tokens:      %d\n", s.TotalTokens))
	b.WriteString(fmt.Sprintf("  Avg output tok/s:  %.1f\n", s.AvgOutputTokPerSec))
	return b.String()
}
