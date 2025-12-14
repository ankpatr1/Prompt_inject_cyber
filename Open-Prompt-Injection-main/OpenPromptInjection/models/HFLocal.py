# OpenPromptInjection/models/HFLocal.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from .Model import Model

class HFLocal(Model):
    """
    Simple local Hugging Face CausalLM adapter (no API key).
    Supports CPU, MPS (Apple), CUDA. Optional 4-bit on CUDA if bitsandbytes is installed.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model_id = self.name  # reuse name field as HF repo id
        params = config.get("params", {})
        self.max_output_tokens = int(params.get("max_output_tokens", 256))
        self.temperature = float(params.get("temperature", getattr(self, "temperature", 0.7)))

        # Device
        if torch.cuda.is_available():
            device = 0
            dtype = torch.float16
            device_map = "auto"
            load_in_4bit = params.get("load_in_4bit", False)
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
            device_map = None
            load_in_4bit = False  # 4-bit not supported on MPS
        else:
            device = "cpu"
            dtype = torch.float32
            device_map = None
            load_in_4bit = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            # most instruct models are decoder-only; set pad to eos to keep pipeline happy
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        model_kwargs = dict(
            trust_remote_code=True,
        )
        if device == 0:
            # CUDA
            model_kwargs.update(dict(
                device_map=device_map,
                torch_dtype=dtype,
            ))
            if load_in_4bit:
                # requires bitsandbytes
                model_kwargs.update(dict(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ))
        elif device == "mps":
            model_kwargs.update(dict(
                torch_dtype=dtype,
                device_map={"": "mps"},
            ))
        else:  # CPU
            model_kwargs.update(dict(
                torch_dtype=dtype,
            ))

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        # Pipeline
        self.pipe = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device if isinstance(device, int) else -1,
        )

    def set_API_key(self):
        # Not needed for local HF
        return

    def query(self, msg: str) -> str:
        # Minimal prompt formatting—works best with *instruct*-tuned models.
        prompt = msg.strip()
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_output_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # HF pipeline returns a list of dicts with 'generated_text'
        text = out[0]["generated_text"]
        # Return only the *new* portion after the prompt to avoid echo spam
        return text[len(prompt):].strip()
