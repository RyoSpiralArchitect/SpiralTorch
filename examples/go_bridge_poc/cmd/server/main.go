package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"go_bridge_poc/internal/api"
)

func main() {
	logger := log.New(os.Stdout, "go-bridge-poc ", log.LstdFlags)

	mux := http.NewServeMux()
	server := api.NewServer(logger)
	server.RegisterRoutes(mux)

	address := ":8080"
	srv := &http.Server{
		Addr:              address,
		Handler:           mux,
		ReadHeaderTimeout: 2 * time.Second,
		ReadTimeout:       5 * time.Second,
		WriteTimeout:      5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	logger.Printf("listening on %s", address)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logger.Fatalf("server error: %v", err)
	}
}
