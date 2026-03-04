package myrlm

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"

	"github.com/openai/openai-go/v3/shared"
)

// REPLServer is the HTTP server that the Docker container calls back into
// for llm_query and rlm_query.
type REPLServer struct {
	server *http.Server
	client *OpenAIClient
	rlm    *RLM
}

func NewREPLServer(client *OpenAIClient, rlm *RLM) *REPLServer {
	s := &REPLServer{client: client, rlm: rlm}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /llm_query", s.handleLLMQuery)
	mux.HandleFunc("POST /rlm_query", s.handleRLMQuery)

	port := os.Getenv("RLM_SERVER_PORT")
	if port == "" {
		port = "9712"
	}

	s.server = &http.Server{
		Addr:    ":" + port,
		Handler: mux,
	}
	return s
}

// Start begins listening in a background goroutine. Returns once the
// listener is bound so the caller knows the port is ready.
func (s *REPLServer) Start() error {
	ln, err := net.Listen("tcp", s.server.Addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.server.Addr, err)
	}
	go s.server.Serve(ln)
	return nil
}

func (s *REPLServer) Shutdown(ctx context.Context) error {
	return s.server.Shutdown(ctx)
}

// --- request/response types ---

type llmQueryRequest struct {
	Messages []Prompt `json:"messages"`
	Model    *string  `json:"model"`
}

type rlmQueryRequest struct {
	Prompt string  `json:"prompt"`
	Model  *string `json:"model"`
}

type queryResponse struct {
	Text  string `json:"text,omitempty"`
	Error string `json:"error,omitempty"`
}

// --- handlers ---

func (s *REPLServer) handleLLMQuery(w http.ResponseWriter, r *http.Request) {
	var req llmQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, queryResponse{Error: err.Error()})
		return
	}

	client := s.clientForModel(req.Model)
	result, err := client.QueryMessages(r.Context(), req.Messages)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, queryResponse{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, queryResponse{Text: result.Text})
}

func (s *REPLServer) handleRLMQuery(w http.ResponseWriter, r *http.Request) {
	var req rlmQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, queryResponse{Error: err.Error()})
		return
	}

	client := s.clientForModel(req.Model)
	result, err := client.Query(r.Context(), req.Prompt)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, queryResponse{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, queryResponse{Text: result.Text})
}

// --- helpers ---

// clientForModel returns a shallow copy of the client with the requested model
// override, or the original client if model is nil/empty.
func (s *REPLServer) clientForModel(model *string) *OpenAIClient {
	if model == nil || *model == "" {
		return s.client
	}
	clone := *s.client
	clone.model = shared.ResponsesModel(*model)
	return &clone
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
