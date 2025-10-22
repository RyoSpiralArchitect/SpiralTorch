package api

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
)

type PredictionRequest struct {
	Input []float64 `json:"input"`
}

type PredictionResponse struct {
	Sum     float64  `json:"sum"`
	Count   int      `json:"count"`
	Average float64  `json:"average"`
	Minimum *float64 `json:"min,omitempty"`
	Maximum *float64 `json:"max,omitempty"`
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

	resp, err := summarize(req.Input)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&resp); err != nil {
		s.logger.Printf("failed to encode response: %v", err)
	}
}

// summarize calculates aggregate statistics for the provided values.
func summarize(values []float64) (PredictionResponse, error) {
	count := len(values)
	if count == 0 {
		return PredictionResponse{}, errors.New("input must contain at least one value")
	}

	var sum float64
	min := values[0]
	max := values[0]
	for _, value := range values {
		sum += value
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}

	average := sum / float64(count)
	minCopy := min
	maxCopy := max
	return PredictionResponse{
		Sum:     sum,
		Count:   count,
		Average: average,
		Minimum: &minCopy,
		Maximum: &maxCopy,
	}, nil
}
