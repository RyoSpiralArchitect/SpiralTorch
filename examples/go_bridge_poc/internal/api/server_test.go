package api

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestServer(t *testing.T) *Server {
	t.Helper()
	logger := log.New(io.Discard, "", log.LstdFlags)
	return NewServer(logger)
}

func TestHealthEndpoint(t *testing.T) {
	server := newTestServer(t)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("expected json content type, got %q", got)
	}
}

func TestPredictEndpoint(t *testing.T) {
	server := newTestServer(t)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)

	body := PredictionRequest{Input: []float64{1, 2, 3, 4}}
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var resp PredictionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if resp.Sum != 10 {
		t.Fatalf("unexpected sum: %v", resp.Sum)
	}
	if resp.Count != 4 {
		t.Fatalf("unexpected count: %v", resp.Count)
	}
	if resp.Average != 2.5 {
		t.Fatalf("unexpected average: %v", resp.Average)
	}
	if resp.Minimum == nil || *resp.Minimum != 1 {
		t.Fatalf("unexpected min: %v", resp.Minimum)
	}
	if resp.Maximum == nil || *resp.Maximum != 4 {
		t.Fatalf("unexpected max: %v", resp.Maximum)
	}
}

func TestPredictEndpointRejectsEmptyInput(t *testing.T) {
	server := newTestServer(t)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)

	body := PredictionRequest{Input: []float64{}}
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(payload))
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", rec.Code)
	}
}

func TestPredictEndpointRejectsWrongMethod(t *testing.T) {
	server := newTestServer(t)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)

	req := httptest.NewRequest(http.MethodGet, "/predict", nil)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected status 405, got %d", rec.Code)
	}
}
