package myrlm

import (
	"net/http"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// ─── RLM Defaults ───────────────────────────────────────────────────────────

const (
	DefaultMaxDepth          = 3
	DefaultTokenBudget int64 = 500_000
)

// ─── Core Types ─────────────────────────────────────────────────────────────

type Query string

type Prompt struct {
	Role    string
	Content string
}

type Tool struct {
	Name        string
	Description string
	Examples    []string
}

type Context struct {
	Content  any
	Path     string
	Metadata ContextMetadata
}

type ContextMetadata struct {
	Name         string
	Description  string
	Length       int
	ChunkLengths []int
	Source       string
	Depth        int
	Type         string
}

// ─── RLM ────────────────────────────────────────────────────────────────────

type RLM struct {
	maxDepth     int
	maxTimeout   *float64
	maxTokens    *int64
	maxBudget    int64
	reuseDocker  bool
	reusePool    *dockerReusePool
	customTools  []Tool
	dockerConfig DockerConfig
	client       *OpenAIClient
}

type RLMOption func(*RLM)

type dockerReusePool struct {
	mu   sync.Mutex
	repl *DockerREPL
}

// ─── Docker / REPL ──────────────────────────────────────────────────────────

type DockerConfig struct {
	Image        string
	Model        string
	Depth        int
	BatchWorkers int
}

// DockerREPL manages a Docker container that runs Python code and can call
// back into the host Go server via llm_query()/rlm_query().
type DockerREPL struct {
	image        string
	containerID  string
	tempDir      string
	rlmDepth     int
	batchWorkers int
	closed       bool
}

// REPLResult holds the captured output from a single code execution
// inside the Docker container.
type REPLResult struct {
	Stdout        string
	Stderr        string
	Locals        map[string]string
	ExecutionTime time.Duration
	FinalAnswer   string
}

// REPLServer is the HTTP server that the Docker container calls back into
// for llm_query and rlm_query.
type REPLServer struct {
	server *http.Server
	client *OpenAIClient
	rlm    *RLM
}

// ─── OpenAI Client ──────────────────────────────────────────────────────────

var DefaultModel shared.ResponsesModel = openai.ChatModelGPT5

type ClientConfig struct {
	APIKey       string
	Model        shared.ResponsesModel
	ReasoningEff *shared.ReasoningEffort
	Temperature  *float64
	MaxTokens    *int64
	Instructions string
	// CompactionThreshold controls when Responses API context compaction
	// triggers. If nil, a conservative default is used. Set to <= 0 to
	// disable explicit compaction configuration.
	CompactionThreshold *int64
	// DisableAutoTruncation disables Responses API truncation="auto"
	// behavior. By default truncation stays enabled so oversized contexts
	// don't hard-fail.
	DisableAutoTruncation bool
}

type OpenAIClient struct {
	raw    openai.Client
	model  shared.ResponsesModel
	config ClientConfig
	Usage  *UsageTracker
}

type QueryResult struct {
	Text  string
	Stats RequestStats
}

type StreamCallback func(delta string) error

type StreamResult struct {
	Text  string
	Stats RequestStats
}

// ─── Chat Sessions ──────────────────────────────────────────────────────────

type Chat struct {
	client          *OpenAIClient
	lastResponseID  string
	instructions    string
	model           shared.ResponsesModel
	maxOutputTokens *int64
	turns           int
}

// ─── Usage Tracking ─────────────────────────────────────────────────────────

type TokenUsage struct {
	InputTokens     int64
	OutputTokens    int64
	TotalTokens     int64
	CachedTokens    int64
	ReasoningTokens int64
}

type RequestStats struct {
	Model          string
	StartTime      time.Time
	EndTime        time.Time
	Duration       time.Duration
	TimeToFirstTok time.Duration
	Streamed       bool
	Tokens         TokenUsage
	ResponseID     string
}

type UsageTracker struct {
	mu       sync.Mutex
	requests []RequestStats
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
