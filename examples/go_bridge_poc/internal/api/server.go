package api

import (
	"encoding/json"
	"errors"
	"io"
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

const (
	maxPayloadBytes int64 = 256 * 1024 // 256KiB protects server resources while remaining generous for demos
	maxPayloadError       = "payload exceeds 256KiB limit"
)

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

	limitedBody := http.MaxBytesReader(w, r.Body, maxPayloadBytes)
	defer limitedBody.Close()
	r.Body = limitedBody
	var req PredictionRequest
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		if maxErr := new(http.MaxBytesError); errors.As(err, &maxErr) {
			s.logger.Printf("payload exceeds limit: %v", err)
			if err := respondJSON(w, http.StatusRequestEntityTooLarge, errorResponse{Error: maxPayloadError}); err != nil {
				s.logger.Printf("failed to encode size error response: %v", err)
			}
			return
		}

		s.logger.Printf("failed to decode payload: %v", err)
		if err := respondJSON(w, http.StatusBadRequest, errorResponse{Error: "invalid payload"}); err != nil {
			s.logger.Printf("failed to encode error response: %v", err)
		}
		return
	}

	if err := ensureEOF(decoder); err != nil {
		s.logger.Printf("unexpected trailing data: %v", err)
		if err := respondJSON(w, http.StatusBadRequest, errorResponse{Error: "invalid payload"}); err != nil {
			s.logger.Printf("failed to encode error response: %v", err)
		}
		return
	}

	if len(req.Input) == 0 {
		if err := respondJSON(w, http.StatusBadRequest, errorResponse{Error: "input must include at least one value"}); err != nil {
			s.logger.Printf("failed to encode validation response: %v", err)
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
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	if _, err := w.Write(append(body, '\n')); err != nil {
		return err
	}

	return nil
}

func ensureEOF(decoder *json.Decoder) error {
	if decoder == nil {
		return errors.New("decoder is nil")
	}

	if err := decoder.Decode(&struct{}{}); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		return err
	}

	return errors.New("extra data after JSON payload")
}
