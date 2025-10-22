package api

import (
	"encoding/json"
	"log"
	"net/http"
)

type PredictionRequest struct {
	Input []float64 `json:"input"`
}

type PredictionResponse struct {
	Sum     float64 `json:"sum"`
	Count   int     `json:"count"`
	Average float64 `json:"average"`
}

type errorResponse struct {
	Error string `json:"error"`
}

type healthResponse struct {
	Status string `json:"status"`
}

type Server struct {
	logger *log.Logger
}

func NewServer(logger *log.Logger) *Server {
	return &Server{logger: logger}
}

func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/predict", s.handlePredict)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if err := respondJSON(w, http.StatusOK, healthResponse{Status: "ok"}); err != nil {
		s.logger.Printf("failed to encode health response: %v", err)
	}
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		if err := respondJSON(w, http.StatusMethodNotAllowed, errorResponse{Error: "only POST is supported"}); err != nil {
			s.logger.Printf("failed to encode method not allowed response: %v", err)
		}
		return
	}

	defer r.Body.Close()
	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.logger.Printf("failed to decode payload: %v", err)
		if err := respondJSON(w, http.StatusBadRequest, errorResponse{Error: "invalid payload"}); err != nil {
			s.logger.Printf("failed to encode error response: %v", err)
		}
		return
	}

	var sum float64
	for _, value := range req.Input {
		sum += value
	}

	resp := PredictionResponse{
		Sum:   sum,
		Count: len(req.Input),
	}
	if resp.Count > 0 {
		resp.Average = resp.Sum / float64(resp.Count)
	}

	if err := respondJSON(w, http.StatusOK, &resp); err != nil {
		s.logger.Printf("failed to encode response: %v", err)
	}
}

func respondJSON(w http.ResponseWriter, status int, payload interface{}) error {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	return json.NewEncoder(w).Encode(payload)
}
