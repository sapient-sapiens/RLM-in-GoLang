package myrlm


type Tool struct {
	Name        string
	Description string
	Examples    []string
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

type Context struct {
	// Context can be anything - a document, string, etc.
	Content  any
	path     string
	Metadata ContextMetadata
}

type Query string

// Support for model by model usage tracking
type ModelUsage struct {
	Model        string
	inputTokens  int
	outputTokens int
	totalCalls   int
	totalCost    float64
}

type TotalUsage struct {
	totalCost         float64
	totalInputTokens  int
	totalOutputTokens int
	totalCalls        int
	modelUsage        []ModelUsage
}
