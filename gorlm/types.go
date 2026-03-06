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
	Content  any
	Path     string
	Metadata ContextMetadata
}

type Query string

