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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	defer r.Body.Close()
	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.logger.Printf("failed to decode payload: %v", err)
		http.Error(w, "invalid payload", http.StatusBadRequest)
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

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&resp); err != nil {
		s.logger.Printf("failed to encode response: %v", err)
	}
}
