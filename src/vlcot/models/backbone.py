from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from vlcot.config import ModelConfig


class OpenVLABackbone(nn.Module):
    def __init__(self, config: ModelConfig, device: str = "cuda", training: bool = False):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForVision2Seq, AutoProcessor

        source = config.backbone
        revision = None if Path(source).is_dir() else config.revision
        kwargs = dict(revision=revision, trust_remote_code=True, local_files_only=config.local_files_only)
        try:
            self.processor = AutoProcessor.from_pretrained(source, **kwargs)
            self.vla = AutoModelForVision2Seq.from_pretrained(
                source,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                **kwargs,
            )
        except OSError as error:
            raise RuntimeError(
                "OpenVLA artifacts are unavailable. Supply a complete local OpenVLA directory or cache the "
                "pinned revision before running. Automatic network downloads are disabled by default."
            ) from error
        self.vla.requires_grad_(False)
        self.vla.language_model = get_peft_model(
            self.vla.language_model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        self.hidden_size = self.vla.config.text_config.hidden_size
        self.to(device)
        self.train(training)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return self.vla.get_input_embeddings().weight.dtype

    def visual_embeddings(self, image: np.ndarray):
        processed = self.processor.image_processor(Image.fromarray(image), return_tensors="pt")
        pixels = processed["pixel_values"].to(device=self.device, dtype=self.dtype)
        return self.vla.projector(self.vla.vision_backbone(pixels))

    def text_embeddings(self, instruction: str):
        prompt = f"In: What action should the robot take to {instruction.strip().lower()}?\nOut:"
        encoded = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        return self.vla.get_input_embeddings()(encoded["input_ids"].to(self.device))

    def bos_embedding(self):
        token = self.processor.tokenizer.bos_token_id
        return self.vla.get_input_embeddings()(torch.tensor([[token]], device=self.device))

    def hidden(self, embeddings: torch.Tensor):
        decoder = self.vla.language_model.get_base_model().model
        result = decoder(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device),
            use_cache=False,
            return_dict=True,
        )
        return result.last_hidden_state

    def enable_gradient_checkpointing(self):
        self.vla.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "vla"):
            self.vla.vision_backbone.eval()
            self.vla.projector.eval()
        return self
