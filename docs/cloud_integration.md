# Cloud Integration Guide

SpiralTorch can be deployed on managed infrastructure while still leaning on its
Rust-first runtime. This guide outlines reference patterns for Azure and AWS so
you can ship Z-space workloads without abandoning the workflow described in the
root `README.md`.

## Common prerequisites

- Build release artefacts up front to keep cluster images lean:
  - `just stack` for the full Rust stack.
  - `maturin build -m bindings/st-py/Cargo.toml --release --features wgpu`
    when a Python wheel is required.
- Produce a container image that bakes the chosen backend features and the
  `spiraltorch` CLI entrypoints:
  ```Dockerfile
  FROM ghcr.io/ryospiralarchitect/spiraltorch:base
  COPY target/release /opt/spiraltorch/bin
  COPY bindings/st-py/target/wheels/*.whl /tmp/
  RUN pip install /tmp/spiraltorch-*.whl && rm -rf /tmp/*.whl
  ENV SPIRALTORCH_HOME=/opt/spiraltorch
  ENTRYPOINT ["/opt/spiraltorch/bin/st-runner"]
  ```
- Enable telemetry forwarding by exporting `SPIRALTORCH_TRACE=info` and pointing
  `SPIRALTORCH_TRACE_ENDPOINT` at your observability stack (Azure Monitor,
  CloudWatch, or OpenTelemetry collector).
- Persist Z-space checkpoints by directing `SPIRALTORCH_CACHE` to a mounted
  volume (Azure Files, Azure Blob fuse, Amazon EFS, or S3 via s3fs).

## Azure reference deployment

1. **Provision registries and identity**
   - Create an Azure Container Registry (ACR) and grant a managed identity pull
     rights (`AcrPull`) for the target compute surface (AKS, Azure Container
     Apps, or Azure Machine Learning).
   - Store secrets for downstream services (Blob SAS, Event Hubs) inside Azure
     Key Vault and mount them at runtime via CSI secrets drivers.
2. **Define an AKS workload**
   - Use a GPU-enabled node pool if you need CUDA or DirectML; otherwise the
     default WGPU+CPU profile suffices. Attach the ACR with `az aks update \
     --attach-acr`.
   - Deploy a `Deployment` manifest that mounts persistent volumes and injects
     backend toggles:
     ```yaml
     apiVersion: apps/v1
     kind: Deployment
     metadata:
       name: spiraltorch-zspace
     spec:
       replicas: 2
       template:
         spec:
           containers:
             - name: trainer
               image: <acr>.azurecr.io/spiraltorch:latest
               env:
                 - name: SPIRALTORCH_BACKEND
                   value: "wgpu"
                 - name: SPIRALTORCH_TRACE_ENDPOINT
                   value: "http://otel-collector:4317"
               volumeMounts:
                 - name: checkpoint-cache
                   mountPath: /opt/spiraltorch/cache
           volumes:
             - name: checkpoint-cache
               azureFile:
                 secretName: spiraltorch-storage-secret
                 shareName: checkpoints
     ```
3. **Wire Azure ML orchestration**
   - Register the same container image as an Azure ML environment.
   - Define a `command` job that mounts Blob storage and streams telemetry back
     to Application Insights using `SPIRALTORCH_TRACE_ENDPOINT`.
   - Configure `distribution` with `type: pytorch` to reuse the library's
     roundtable trainer across nodes.
4. **Telemetry and scaling**
   - Export metrics to Azure Monitor or Log Analytics by enabling the
     Container Insights addon or deploying an OpenTelemetry collector with
     Azure Monitor exporters.
   - Scale with `kubectl scale` or AKS Horizontal Pod Autoscaler; SpiralTorch's
     event-driven trainer remains stateless if checkpoints live on Azure Files
     or Blob.

## AWS reference deployment

1. **Prepare registries and IAM roles**
   - Push the built image to Amazon ECR. Grant the target EKS node role or
     ECS task role permissions to pull from that repository.
   - Store credentials (S3 access keys, Kinesis endpoints) in AWS Secrets
     Manager or AWS Systems Manager Parameter Store.
2. **Run on Amazon EKS**
   - Provision GPU node groups (p4d, g5) when CUDA is required; otherwise
     Graviton or x86 nodes work with the WGPU backend.
   - Deploy a `Deployment` or `Job` resource referencing the ECR image and mount
     persistent volumes through Amazon EFS or S3 CSI drivers. Example:
     ```yaml
     apiVersion: batch/v1
     kind: Job
     metadata:
       name: spiraltorch-trainer
     spec:
       template:
         spec:
           containers:
             - name: trainer
               image: <account>.dkr.ecr.<region>.amazonaws.com/spiraltorch:latest
               env:
                 - name: SPIRALTORCH_BACKEND
                   value: "cuda"
                 - name: SPIRALTORCH_TRACE_ENDPOINT
                   value: "http://aws-otel-collector:4317"
               volumeMounts:
                 - name: checkpoints
                   mountPath: /opt/spiraltorch/cache
           restartPolicy: OnFailure
           volumes:
             - name: checkpoints
               persistentVolumeClaim:
                 claimName: spiraltorch-checkpoints
     ```
3. **Integrate with SageMaker**
   - Create a custom SageMaker training image using the same base Dockerfile and
     register it with SageMaker.
   - Define a training job with `distribution` set to `mpi` so SpiralTorch's
     distributed trainer can fan out across hosts while sharing the Z-space
     cache via FSx for Lustre or EFS.
   - Stream metrics into Amazon CloudWatch by configuring the OpenTelemetry
     collector sidecar or the CloudWatch agent to scrape SpiralTorch spans.
4. **Observability and automation**
   - Enable autoscaling using Karpenter (EKS) or SageMaker managed spot
     training. SpiralTorch tolerates instance churn as long as checkpoints
     persist in S3/EFS/FSx.
   - Wire AWS Step Functions or EventBridge to trigger retraining pipelines when
     new data lands in S3, calling into SpiralTorch CLI entrypoints inside your
     container image.

## Next steps

- Track backend compatibility in [`docs/backend_matrix.md`](backend_matrix.md).
- Consult [`docs/compatibility_strategy.md`](compatibility_strategy.md) when
  migrating from existing PyTorch/JAX/TensorFlow stacks.
- File issues or discussion threads if your Azure/AWS topology requires new
  runtime feature flags or scheduler hooks.
