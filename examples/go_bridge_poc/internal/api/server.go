package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"

	bridgespec "go_bridge_poc/api"
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
	mux.HandleFunc("/openapi.json", s.handleOpenAPI)
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
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
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

func (s *Server) handleOpenAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", fmt.Sprintf("%s, %s", http.MethodGet, http.MethodHead))
		if err := respondJSON(w, http.StatusMethodNotAllowed, errorResponse{Error: "only GET is supported"}); err != nil {
			s.logger.Printf("failed to encode method not allowed response: %v", err)
		}
		return
	}

	if len(bridgespec.OpenAPISpec) == 0 {
		s.logger.Println("openapi spec is empty")
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "public, max-age=60")
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodHead {
		return
	}

	if _, err := w.Write(bridgespec.OpenAPISpec); err != nil {
		s.logger.Printf("failed to write openapi spec: %v", err)
	}
	if bridgespec.OpenAPISpec[len(bridgespec.OpenAPISpec)-1] != '\n' {
		if _, err := w.Write([]byte{'\n'}); err != nil {
			s.logger.Printf("failed to write trailing newline: %v", err)
		}
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
