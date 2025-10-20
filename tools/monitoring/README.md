# Monitoring assets

This directory holds reusable assets for the SpiralTorch observability stack:

- `grafana.json` – Starter dashboard that visualises runtime latency, reward, and performance gauges. Import it into Grafana after provisioning the Prometheus datasource with UID `prometheus`.

Pair these assets with the Docker Compose stack in `../ops/observability` to launch a full Prometheus + Grafana environment locally.
