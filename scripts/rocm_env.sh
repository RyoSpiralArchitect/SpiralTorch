#!/usr/bin/env bash
# Example ROCm container run (adjust image/tag for your environment)
# docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
#   --ipc=host --shm-size=8G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#   -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
#   -v $PWD:/work -w /work rocm/pytorch-training:latest bash
echo "This is a template. Uncomment and adjust for your ROCm container."
