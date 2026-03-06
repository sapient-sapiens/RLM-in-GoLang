package myrlm

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

var DefaultModel shared.ResponsesModel = openai.ChatModelGPT5

const (
	defaultCompactionThreshold int64 = 12000
)

func SharedModel(name string) shared.ResponsesModel {
	return shared.ResponsesModel(name)
}

type ClientConfig struct {
	APIKey       string
	Model        shared.ResponsesModel
	ReasoningEff *shared.ReasoningEffort
	Temperature  *float64
	MaxTokens    *int64
	Instructions string
	// CompactionThreshold controls when Responses API context compaction triggers.
	// If nil, a conservative default is used.
	// Set to <= 0 to disable explicit compaction configuration.
	CompactionThreshold *int64
	// DisableAutoTruncation disables Responses API truncation="auto" behavior.
	// By default truncation stays enabled so oversized contexts don't hard-fail.
	DisableAutoTruncation bool
}

type OpenAIClient struct {
	raw    openai.Client
	model  shared.ResponsesModel
	config ClientConfig
	Usage  *UsageTracker
}

func NewOpenAIClient(cfg ClientConfig) *OpenAIClient {
	_ = godotenv.Load()

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}

	raw := openai.NewClient(option.WithAPIKey(apiKey))

	return &OpenAIClient{
		raw:    raw,
		model:  model,
		config: cfg,
		Usage:  NewUsageTracker(),
	}
}

func (c *OpenAIClient) baseParams(input responses.ResponseNewParamsInputUnion) responses.ResponseNewParams {
	p := responses.ResponseNewParams{
		Model: c.model,
		Input: input,
	}
	if c.config.Temperature != nil {
		p.Temperature = openai.Float(*c.config.Temperature)
	}
	if c.config.MaxTokens != nil {
		p.MaxOutputTokens = openai.Int(*c.config.MaxTokens)
	}
	if c.config.ReasoningEff != nil {
		p.Reasoning = shared.ReasoningParam{
			Effort: *c.config.ReasoningEff,
		}
	}
	if c.config.Instructions != "" {
		p.Instructions = openai.String(c.config.Instructions)
	}
	if !c.config.DisableAutoTruncation {
		p.Truncation = responses.ResponseNewParamsTruncationAuto
	}
	if threshold := c.compactionThreshold(); threshold > 0 {
		p.ContextManagement = []responses.ResponseNewParamsContextManagement{
			{
				Type:             "compaction",
				CompactThreshold: openai.Int(threshold),
			},
		}
	}
	return p
}

func (c *OpenAIClient) compactionThreshold() int64 {
	if c.config.CompactionThreshold != nil {
		return *c.config.CompactionThreshold
	}
	return defaultCompactionThreshold
}

func (c *OpenAIClient) extractUsage(resp *responses.Response, start time.Time, streamed bool, ttft time.Duration) RequestStats {
	stats := RequestStats{
		Model:      string(resp.Model),
		StartTime:  start,
		EndTime:    time.Now(),
		Duration:   time.Since(start),
		Streamed:   streamed,
		ResponseID: resp.ID,
	}
	if streamed {
		stats.TimeToFirstTok = ttft
	}
	stats.Tokens = TokenUsage{
		InputTokens:     resp.Usage.InputTokens,
		OutputTokens:    resp.Usage.OutputTokens,
		TotalTokens:     resp.Usage.TotalTokens,
		CachedTokens:    resp.Usage.InputTokensDetails.CachedTokens,
		ReasoningTokens: resp.Usage.OutputTokensDetails.ReasoningTokens,
	}
	c.Usage.Record(stats)
	return stats
}

type QueryResult struct {
	Text  string
	Stats RequestStats
}

func (c *OpenAIClient) Query(ctx context.Context, prompt string) (*QueryResult, error) {
	params := c.baseParams(responses.ResponseNewParamsInputUnion{
		OfString: openai.String(prompt),
	})

	start := time.Now()
	resp, err := c.raw.Responses.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai query: %w", err)
	}

	stats := c.extractUsage(resp, start, false, 0)
	return &QueryResult{Text: resp.OutputText(), Stats: stats}, nil
}

// QueryMessages sends a multi-message prompt (system + user + assistant turns)
// and returns the response. Used by the RLM loop for the root LLM call.
func (c *OpenAIClient) QueryMessages(ctx context.Context, messages []Prompt) (*QueryResult, error) {
	var parts []responses.ResponseInputItemUnionParam
	for _, m := range messages {
		parts = append(parts, responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role: responses.EasyInputMessageRole(m.Role),
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: openai.String(m.Content),
				},
			},
		})
	}

	params := c.baseParams(responses.ResponseNewParamsInputUnion{
		OfInputItemList: parts,
	})

	start := time.Now()
	resp, err := c.raw.Responses.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai query messages: %w", err)
	}

	stats := c.extractUsage(resp, start, false, 0)
	return &QueryResult{Text: resp.OutputText(), Stats: stats}, nil
}

type StreamCallback func(delta string) error

type StreamResult struct {
	Text  string
	Stats RequestStats
}

func (c *OpenAIClient) QueryStream(ctx context.Context, prompt string, onDelta StreamCallback) (*StreamResult, error) {
	params := c.baseParams(responses.ResponseNewParamsInputUnion{
		OfString: openai.String(prompt),
	})
	return c.doStream(ctx, params, onDelta)
}

func (c *OpenAIClient) doStream(ctx context.Context, params responses.ResponseNewParams, onDelta StreamCallback) (*StreamResult, error) {
	start := time.Now()
	stream := c.raw.Responses.NewStreaming(ctx, params)

	var (
		fullText  string
		ttft      time.Duration
		firstSeen bool
		finalResp *responses.Response
	)

	for stream.Next() {
		event := stream.Current()

		switch event.Type {
		case "response.output_text.delta":
			fullText += event.Delta
			if !firstSeen {
				ttft = time.Since(start)
				firstSeen = true
			}
			if onDelta != nil {
				if err := onDelta(event.Delta); err != nil {
					return nil, fmt.Errorf("stream callback: %w", err)
				}
			}

		case "response.completed":
			completed := event.AsResponseCompleted()
			finalResp = &completed.Response
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("openai stream: %w", err)
	}

	if finalResp == nil {
		return &StreamResult{Text: fullText}, nil
	}

	fullText = finalResp.OutputText()
	stats := c.extractUsage(finalResp, start, true, ttft)
	return &StreamResult{Text: fullText, Stats: stats}, nil
}

func PrintStreamCallback(w io.Writer) StreamCallback {
	return func(delta string) error {
		_, err := fmt.Fprint(w, delta)
		return err
	}
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

func (c *OpenAIClient) NewChat() *Chat {
	return &Chat{
		client:       c,
		model:        c.model,
		instructions: c.config.Instructions,
	}
}

func (c *OpenAIClient) NewChatWithInstructions(instructions string) *Chat {
	ch := c.NewChat()
	ch.instructions = instructions
	return ch
}

func (ch *Chat) Turns() int             { return ch.turns }
func (ch *Chat) LastResponseID() string { return ch.lastResponseID }

func (ch *Chat) SetMaxOutputTokens(n int64) { ch.maxOutputTokens = &n }

func (ch *Chat) buildParamsForMessages(messages []Prompt) responses.ResponseNewParams {
	var parts []responses.ResponseInputItemUnionParam
	for _, m := range messages {
		parts = append(parts, responses.ResponseInputItemUnionParam{
			OfMessage: &responses.EasyInputMessageParam{
				Role: responses.EasyInputMessageRole(m.Role),
				Content: responses.EasyInputMessageContentUnionParam{
					OfString: openai.String(m.Content),
				},
			},
		})
	}

	p := ch.client.baseParams(responses.ResponseNewParamsInputUnion{
		OfInputItemList: parts,
	})
	p.Store = openai.Bool(true)
	if ch.instructions != "" {
		p.Instructions = openai.String(ch.instructions)
	}
	if ch.lastResponseID != "" {
		p.PreviousResponseID = openai.String(ch.lastResponseID)
	}
	if ch.maxOutputTokens != nil {
		p.MaxOutputTokens = openai.Int(*ch.maxOutputTokens)
	}
	return p
}

func (ch *Chat) buildParams(prompt string) responses.ResponseNewParams {
	return ch.buildParamsForMessages([]Prompt{{Role: "user", Content: prompt}})
}

func (ch *Chat) Send(ctx context.Context, prompt string) (*QueryResult, error) {
	params := ch.buildParams(prompt)

	start := time.Now()
	resp, err := ch.client.raw.Responses.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat send: %w", err)
	}

	ch.lastResponseID = resp.ID
	ch.turns++

	stats := ch.client.extractUsage(resp, start, false, 0)
	return &QueryResult{Text: resp.OutputText(), Stats: stats}, nil
}

func (ch *Chat) SendMessages(ctx context.Context, messages []Prompt) (*QueryResult, error) {
	params := ch.buildParamsForMessages(messages)

	start := time.Now()
	resp, err := ch.client.raw.Responses.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat send messages: %w", err)
	}

	ch.lastResponseID = resp.ID
	ch.turns++

	stats := ch.client.extractUsage(resp, start, false, 0)
	return &QueryResult{Text: resp.OutputText(), Stats: stats}, nil
}

func (ch *Chat) SendStream(ctx context.Context, prompt string, onDelta StreamCallback) (*StreamResult, error) {
	params := ch.buildParams(prompt)

	start := time.Now()
	stream := ch.client.raw.Responses.NewStreaming(ctx, params)

	var (
		fullText  string
		ttft      time.Duration
		firstSeen bool
		finalResp *responses.Response
	)

	for stream.Next() {
		event := stream.Current()

		switch event.Type {
		case "response.output_text.delta":
			fullText += event.Delta
			if !firstSeen {
				ttft = time.Since(start)
				firstSeen = true
			}
			if onDelta != nil {
				if err := onDelta(event.Delta); err != nil {
					return nil, fmt.Errorf("chat stream callback: %w", err)
				}
			}

		case "response.completed":
			completed := event.AsResponseCompleted()
			finalResp = &completed.Response
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("chat stream: %w", err)
	}

	if finalResp != nil {
		fullText = finalResp.OutputText()
		ch.lastResponseID = finalResp.ID
	}
	ch.turns++

	if finalResp == nil {
		return &StreamResult{Text: fullText}, nil
	}

	stats := ch.client.extractUsage(finalResp, start, true, ttft)
	return &StreamResult{Text: fullText, Stats: stats}, nil
}
