package api

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestServer() *Server {
	logger := log.New(io.Discard, "", 0)
	return NewServer(logger)
}

func TestHandleHealth(t *testing.T) {
	srv := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()

	srv.handleHealth(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, res.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body["status"] != "ok" {
		t.Fatalf("expected status ok, got %q", body["status"])
	}
}

func TestHandlePredictSuccess(t *testing.T) {
	srv := newTestServer()
	payload := PredictionRequest{Input: []float64{1, 2, 3}}
	buf, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal payload: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(buf))
	w := httptest.NewRecorder()

	srv.handlePredict(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status %d, got %d", http.StatusOK, res.StatusCode)
	}

	var body PredictionResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body.Sum != 6 || body.Count != 3 {
		t.Fatalf("unexpected aggregation: %+v", body)
	}

	if body.Average != 2 {
		t.Fatalf("expected average 2, got %v", body.Average)
	}
}

func TestHandlePredictValidation(t *testing.T) {
	srv := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader([]byte("not-json")))
	w := httptest.NewRecorder()

	srv.handlePredict(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, res.StatusCode)
	}

	var body errorResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body.Error == "" {
		t.Fatal("expected error message, got empty string")
	}
}

func TestHandlePredictRejectsEmptyInput(t *testing.T) {
	srv := newTestServer()
	payload := PredictionRequest{Input: []float64{}}
	buf, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("failed to marshal payload: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(buf))
	w := httptest.NewRecorder()

	srv.handlePredict(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, res.StatusCode)
	}

	var body errorResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body.Error != "input must include at least one value" {
		t.Fatalf("unexpected error message: %q", body.Error)
	}
}

func TestHandlePredictRejectsUnknownFields(t *testing.T) {
	srv := newTestServer()
	buf := []byte(`{"input":[1],"extra":true}`)

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(buf))
	w := httptest.NewRecorder()

	srv.handlePredict(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, res.StatusCode)
	}

	var body errorResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body.Error != "invalid payload" {
		t.Fatalf("unexpected error message: %q", body.Error)
	}
}

func TestHandlePredictRejectsTrailingData(t *testing.T) {
	srv := newTestServer()
	buf := []byte(`{"input":[1]} {}`)

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(buf))
	w := httptest.NewRecorder()

	srv.handlePredict(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, res.StatusCode)
	}

	var body errorResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body.Error != "invalid payload" {
		t.Fatalf("unexpected error message: %q", body.Error)
	}
}

func TestHandlePredictMethodNotAllowed(t *testing.T) {
	srv := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/predict", nil)
	w := httptest.NewRecorder()

	srv.handlePredict(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("expected status %d, got %d", http.StatusMethodNotAllowed, res.StatusCode)
	}

	if got := res.Header.Get("Allow"); got != http.MethodPost {
		t.Fatalf("expected Allow header %s, got %s", http.MethodPost, got)
	}
}

func TestHandlePredictRejectsOversizedPayload(t *testing.T) {
	srv := newTestServer()

	// Build a JSON payload whose encoded size is guaranteed to exceed the handler's 256KiB cap.
	var builder strings.Builder
	builder.Grow(int(maxPayloadBytes) + 1024)
	builder.WriteString(`{"input":[0`)
	repeats := int(maxPayloadBytes/2) + 1
	for i := 0; i < repeats; i++ {
		builder.WriteString(",0")
	}
	builder.WriteString("]}")

	buf := []byte(builder.String())
	if int64(len(buf)) <= maxPayloadBytes {
		t.Fatalf("test payload length %d does not exceed limit %d", len(buf), maxPayloadBytes)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(buf))
	w := httptest.NewRecorder()

	srv.handlePredict(w, req)

	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected status %d, got %d", http.StatusRequestEntityTooLarge, res.StatusCode)
	}

	var body errorResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if body.Error != "payload exceeds 256KiB limit" {
		t.Fatalf("unexpected error message: %q", body.Error)
	}
}
