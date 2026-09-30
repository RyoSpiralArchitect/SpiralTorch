"""Independent eager Torch reference for SpiralTorch's classifier checkpoint.

This deliberately matches the Rust architecture, not torchvision ConvNeXt:
the final affine norm spans flattened NCHW features before spatial pooling.
It is a test oracle, not a production execution route.
"""
import json

import torch
from torch import nn
from torch.nn import functional as F


class ConvNeXtReference(nn.Module):
    def __init__(self, payload, device="cpu"):
        super().__init__()
        state = json.loads(payload)
        if state["schema"] != "spiraltorch.convnext.classifier_plain_sgd_checkpoint.v1":
            raise ValueError("unsupported classifier checkpoint")
        self.config = state["backbone"]["config"]
        self.classes = state["classes"]
        records = state["backbone"]["parameters"] + state["head"]
        self.names = [p["name"] for p in records]
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate parameter names")
        self.values = nn.ParameterList([
            nn.Parameter(torch.tensor(p["values"], dtype=torch.float32, device=device).reshape(p["shape"]))
            for p in records
        ])
        self.lookup = dict(zip(self.names, self.values))
        # Match the f32 epsilon stored in the Rust inference operation.
        curvature = torch.tensor(-self.config["curvature"], dtype=torch.float32)
        self.epsilon = (torch.tensor(self.config["epsilon"], dtype=torch.float32)
                        * (1 + curvature.sqrt() * 0.1)).item()

    def affine(self, x, name):
        return x @ self.lookup[name + "::weight"] + self.lookup[name + "::bias"]

    def norm(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self.lookup[name + "_gamma"].flatten(),
                            self.lookup[name + "_beta"].flatten(), self.epsilon)

    def conv(self, x, name, kernel, stride, padding=0, depthwise=False):
        weights = self.lookup[name + "::weight"]
        groups = x.shape[1] if depthwise else 1
        weights = weights.reshape(weights.shape[0], x.shape[1] // groups, *kernel)
        return F.conv2d(x, weights, self.lookup[name + "::bias"].flatten(),
                        stride=stride, padding=padding, groups=groups)

    def forward(self, x):
        c = self.config
        if list(x.shape[1:]) != [c["input_channels"], *c["input_hw"]]:
            raise ValueError("reference NCHW input differs from checkpoint")
        x = self.conv(x, "convnext.stem", c["patch_size"], c["patch_size"])
        for stage, depth in enumerate(c["stage_depths"]):
            for block in range(depth):
                name = f"convnext.stage{stage}.block{block}"
                residual = x
                x = self.conv(x, name + ".dw", (7, 7), 1, 3, depthwise=True)
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x, name + ".ln")
                x = self.affine(x, name + ".fc1")
                x = F.gelu(x, approximate="tanh")
                x = self.affine(x, name + ".fc2").permute(0, 3, 1, 2)
                x = x + residual
            if stage + 1 < len(c["stage_dims"]):
                x = self.conv(x, f"convnext.stage{stage}.downsample", (2, 2), 2)
        shape = x.shape
        x = self.norm(x.reshape(shape[0], -1), "convnext.final_norm").reshape(shape)
        return self.affine(x.mean(dim=(2, 3)), "convnext.classifier")
