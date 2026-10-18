"""Single-process f32 HF Trainer integration for Rust-backed geometry adapters.

HF remains the optimizer/scheduler/data/RNG owner. Geometry math and portable
state validation remain Rust-owned. No new training loop is implemented here.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
from typing import Any

__all__ = ["hf_geometry_trainer_class"]
_ARTIFACT_FILE = "spiraltorch-geometry.json"
_MPS_RNG_FILE = "spiraltorch-mps-rng.pt"
_TRAINER_CLASS: type | None = None


def _write_artifact(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".geometry-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def hf_geometry_trainer_class() -> type:
    """Return a lazily imported Trainer class; requires Torch and Transformers.

    Construct with model=<frozen float32 causal LM>, geometry=<GeometryAdapterStack>,
    base_model_identity=<verified runtime ID>, and ordinary Trainer arguments.
    No DDP, AMP, quantization, activation checkpointing, compilation or Hub push.
    Checkpoint resumes require a separately retained geometry_checkpoint_id.
    The base model must be reloaded and verified by the surrounding HF executor.
    """
    global _TRAINER_CLASS
    if _TRAINER_CLASS is not None:
        return _TRAINER_CLASS
    try:
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError("install spiraltorch[hf-geometry] to use the geometry Trainer") from error
    from packaging.version import Version
    if transformers.__version__ != "4.57.6" or Version(torch.__version__) < Version("2.6"):
        raise RuntimeError("geometry Trainer currently requires transformers==4.57.6 and torch>=2.6")
    from transformers import Trainer
    from . import GeometryAdapterArtifact
    from .geometry_adapters import GeometryAdapterStack

    class GeometryModel(torch.nn.Module):
        def __init__(self, base: Any, geometry: Any) -> None:
            super().__init__()
            self.base = base
            self.geometry = geometry
            self.config = base.config
            self.main_input_name = getattr(base, "main_input_name", "input_ids")
            self.accepts_loss_kwargs = getattr(base, "accepts_loss_kwargs", any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(base.forward).parameters.values()))
            if hasattr(base, "generation_config"):
                self.generation_config = base.generation_config

        def _get_name(self) -> str:
            # HF uses this name to apply causal shifting during label smoothing.
            return self.base._get_name()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            if self.geometry._active_token is None:
                raise RuntimeError("use Trainer train/evaluate/predict, or attach geometry to base_model explicitly")
            return self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    class HfGeometryTrainer(Trainer):
        """Frozen-base geometry FT with ordinary HF checkpoint companions."""

        def __init__(self, *, model, geometry, base_model_identity: str, **kwargs):
            if not isinstance(model, torch.nn.Module) or type(geometry) is not GeometryAdapterStack:
                raise TypeError("provide an HF model and an exact GeometryAdapterStack")
            if kwargs.get("model_init") is not None:
                raise ValueError("geometry training requires an explicit pinned base, not model_init")
            if kwargs.get("compute_loss_func") is not None:
                raise ValueError("geometry Trainer currently requires the model's causal-LM loss")
            if any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError("freeze the base model and clear its gradients before geometry training")
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("geometry Trainer requires a float32 base model")
            if getattr(model, "is_quantized", False) or getattr(model, "hf_device_map", None):
                raise ValueError("quantized or device-mapped bases are unsupported")
            if any(hasattr(m, "_orig_mod") or getattr(m, "_compiled_call_impl", None) is not None
                   for m in model.modules()):
                raise ValueError("compiled base models are unsupported")
            GeometryAdapterStack._check_model_mode(model)
            if not any(p.requires_grad for p in geometry.parameters()):
                raise ValueError("geometry has no trainable parameters")
            artifact = geometry.export_artifact(base_model_identity=base_model_identity)
            self._geometry_base_id = base_model_identity
            self._geometry_layout_id = artifact.layout_id
            self._geometry_resume_id = None
            self.last_geometry_artifact_id = None
            self._validate_arguments(kwargs.get("args"))
            # Siblings: base registration stays untouched, while HF optimizer
            # discovery and grad clipping now see the actual geometry parameters.
            super().__init__(model=GeometryModel(model, geometry), **kwargs)
            self._validate_arguments(self.args)
            self.label_names = ["labels"]
            self._signature_columns = None
            if self.label_smoother is not None:
                # HF's stock smoother returns a microbatch mean; supply its
                # loss hook with valid-target weighting for unequal masks.
                self.compute_loss_func = self._causal_smoothed_loss
            if (self.is_deepspeed_enabled or self.is_fsdp_enabled
                    or self.accelerator.distributed_type.value != "NO"
                    or self.accelerator.mixed_precision != "no"
                    or self.accelerator.state.dynamo_plugin.backend.value != "NO"
                    or self.args.device.type not in ("cpu", "cuda", "mps")):
                raise ValueError("distributed/sharded/compiled/mixed-precision geometry training is unsupported")
            if self.optimizer is not None:
                self._validate_optimizer()

        @staticmethod
        def _validate_arguments(args):
            if args is None:
                raise ValueError("explicit TrainingArguments are required")
            if (args.world_size != 1 or args.n_gpu > 1 or args.deepspeed or args.fsdp
                    or args.gradient_checkpointing or args.torch_compile
                    or args.fp16 or args.bf16 or args.fp16_full_eval or args.bf16_full_eval
                    or args.neftune_noise_alpha is not None):
                raise ValueError("geometry Trainer requires single-process f32 without checkpointing/compile/NEFTune")
            if args.push_to_hub or args.load_best_model_at_end:
                raise ValueError("use explicit artifact selection/loading; automatic best-model load and Hub push are unsupported")
            if args.save_only_model:
                raise ValueError("geometry training requires full HF optimizer/scheduler/RNG checkpoints")
            if args.label_names is not None and args.label_names != ["labels"]:
                raise ValueError("geometry Trainer requires causal-LM labels")

        @property
        def geometry(self):
            return self.model.geometry

        @property
        def base_model(self):
            return self.model.base

        def _set_signature_columns_if_needed(self):
            if self._signature_columns is None:
                self._signature_columns = list(inspect.signature(self.base_model.forward).parameters)
                self._signature_columns += ["label", "label_ids", "labels"]

        def _validate_optimizer(self):
            expected = {id(p) for p in self.geometry.parameters() if p.requires_grad}
            actual = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError("optimizer must own exactly the trainable geometry parameters")

        def _validate_live_model(self):
            if any(p.requires_grad or p.grad is not None for p in self.base_model.parameters()):
                raise ValueError("the base must remain frozen with no gradients")
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in self.base_model.parameters()):
                raise ValueError("the base must remain float32")
            GeometryAdapterStack._check_model_mode(self.base_model)
            if self.optimizer is not None:
                self._validate_optimizer()

        def create_optimizer(self):
            result = super().create_optimizer()
            self._validate_optimizer()
            return result

        def _causal_smoothed_loss(self, outputs, labels, num_items_in_batch=None):
            loss = self.label_smoother(outputs, labels, shift_labels=True)
            if num_items_in_batch is not None:
                count = labels[..., 1:].ne(-100).sum().to(loss.device)
                total = torch.as_tensor(num_items_in_batch, device=loss.device)
                loss = loss * (count / total)
            return loss

        def _get_num_items_in_batch(self, batch_samples, device):
            # HF 4.57.6 counts every non-ignored label, although causal loss
            # never predicts labels[:, 0]. Reuse its counting/reduction logic
            # on the shifted targets, without changing the model's loss.
            shifted = [dict(batch, labels=batch["labels"][..., 1:])
                       if "labels" in batch else batch for batch in batch_samples]
            return super()._get_num_items_in_batch(shifted, device)

        @staticmethod
        def _check_batch(inputs):
            ids = inputs.get("input_ids")
            if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or min(ids.shape) == 0:
                raise ValueError("geometry training needs nonempty [batch, time] input_ids")
            mask = inputs.get("attention_mask")
            if mask is not None and (not isinstance(mask, torch.Tensor)
                                     or mask.shape != ids.shape or not bool((mask == 1).all())):
                raise ValueError("geometry training requires complete unpadded prefixes")
            if inputs.get("inputs_embeds") is not None:
                raise ValueError("provide input_ids, not inputs_embeds")
            if inputs.get("shift_labels") is not None:
                raise ValueError("provide ordinary causal labels, not pre-shifted targets")
            labels = inputs.get("labels")
            if labels is not None and (not isinstance(labels, torch.Tensor) or labels.shape != ids.shape
                                       or not bool(labels[..., 1:].ne(-100).any())):
                raise ValueError("causal labels need matching shape and at least one valid shifted target")

        def training_step(self, model, inputs, num_items_in_batch=None):
            self._validate_live_model()
            self._check_batch(inputs)
            with self.geometry.attach(self.base_model, require_all=True):
                return super().training_step(model, inputs, num_items_in_batch)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            self._check_batch(inputs)
            with self.geometry.attach(self.base_model, require_all=True):
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None,
                  *, geometry_checkpoint_id=None, **kwargs):
            self._validate_arguments(self.args)
            self._validate_live_model()
            artifact = self.geometry.export_artifact(base_model_identity=self._geometry_base_id)
            if artifact.layout_id != self._geometry_layout_id:
                raise ValueError("geometry recipe/layout changed before training")
            if bool(resume_from_checkpoint) != (geometry_checkpoint_id is not None):
                raise ValueError("resume requires a separately pinned geometry_checkpoint_id")
            if trial is not None:
                raise ValueError("use an explicit pinned run, not in-process HP search")
            if resume_from_checkpoint and self.args.ignore_data_skip:
                raise ValueError("geometry resume requires HF data-cursor replay, not ignore_data_skip")
            self._geometry_resume_id = geometry_checkpoint_id
            try:
                return super().train(resume_from_checkpoint=resume_from_checkpoint,
                                     ignore_keys_for_eval=ignore_keys_for_eval, **kwargs)
            finally:
                self._geometry_resume_id = None

        def _save(self, output_dir=None, state_dict=None):
            if state_dict is not None:
                raise ValueError("external state_dict replacement is unsupported for geometry checkpoints")
            artifact = self.geometry.export_artifact(base_model_identity=self._geometry_base_id)
            if artifact.layout_id != self._geometry_layout_id:
                raise ValueError("geometry recipe/layout changed during training")
            output = Path(output_dir or self.args.output_dir)
            _write_artifact(output / _ARTIFACT_FILE, artifact.to_json())
            self.last_geometry_artifact_id = artifact.artifact_id
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output)
            # HF still owns the trainer state, optimizer, scheduler and RNG.
            torch.save(self.args, output / "training_args.bin")

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            if self._geometry_resume_id is None:
                raise ValueError("missing pinned geometry checkpoint identity")
            if model is not None and model is not self.model:
                raise ValueError("wrapped/distributed checkpoint loading is unsupported")
            checkpoint = Path(resume_from_checkpoint)
            # HF otherwise tolerates absent optimizer/RNG files and silently
            # turns a requested continuation into a partially restarted run.
            required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
            if self.args.device.type == "mps":
                required += (_MPS_RNG_FILE,)
            missing = [name for name in required if not (checkpoint / name).is_file()]
            if missing:
                raise ValueError(f"incomplete HF checkpoint companions: {', '.join(missing)}")
            if self.args.device.type == "mps":
                self._read_mps_rng_state(checkpoint)
            path = checkpoint / _ARTIFACT_FILE
            with path.open("rb") as stream:
                data = stream.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError("geometry artifact exceeds byte budget")
            artifact = GeometryAdapterArtifact.from_json(data.decode("utf-8"), self._geometry_resume_id,
                                                        self._geometry_base_id)
            if artifact.layout_id != self._geometry_layout_id:
                raise ValueError("checkpoint geometry recipe/layout differs from this run")
            restored = GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=self._geometry_base_id)
            # Compatible state is fully validated before any live parameter copy.
            self.geometry.load_state_dict(restored.state_dict(), strict=True)

        @staticmethod
        def _read_mps_rng_state(checkpoint):
            state = torch.load(Path(checkpoint) / _MPS_RNG_FILE, map_location="cpu", weights_only=True)
            expected = torch.mps.get_rng_state()
            if not isinstance(state, torch.Tensor) or state.dtype != expected.dtype or state.shape != expected.shape:
                raise ValueError("invalid MPS RNG companion")
            return state

        def _save_rng_state(self, output_dir):
            super()._save_rng_state(output_dir)
            if self.args.device.type == "mps":
                # Transformers 4.57.6 omits this device generator. Keep the
                # additional Torch state alongside HF's ordinary RNG payload.
                torch.save(torch.mps.get_rng_state(), Path(output_dir) / _MPS_RNG_FILE)

        def _load_rng_state(self, checkpoint):
            mps = (self._read_mps_rng_state(checkpoint)
                   if checkpoint is not None and self.args.device.type == "mps" else None)
            super()._load_rng_state(checkpoint)
            if mps is not None:
                torch.mps.set_rng_state(mps)

    HfGeometryTrainer.__name__ = "HfGeometryTrainer"
    _TRAINER_CLASS = HfGeometryTrainer
    return HfGeometryTrainer
