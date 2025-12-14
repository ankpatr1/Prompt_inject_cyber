# OpenPromptInjection/models/Ollama.py
import json
import requests
from .Model import Model

class Ollama(Model):
    """
    Minimal adapter for local Ollama models via /api/generate (non-streaming).
    Maps repo's Model interface -> Ollama HTTP API.
    """

    def __init__(self, config):
        super().__init__(config)
        # read params
        p = config.get("params", {})
        self.base_url = p.get("ollama_url", "http://127.0.0.1:11434")
        # repo uses self.name as the model id; here it must match `ollama list` (e.g., "llama3:8b-instruct")
        # temperature/max tokens come from base class (already loaded); map in query()

        # quick sanity check (optional but helpful)
        try:
            r = requests.get(self.base_url, timeout=2)
            _ = r.status_code  # just touch it
        except Exception:
            pass  # don't hard-fail; main script can still run

    def set_API_key(self):
        # not used for Ollama
        return

    def query(self, msg: str) -> str:
        """
        Call /api/generate with a plain prompt.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.name,         # e.g., "llama3:8b-instruct"
            "prompt": msg,
            "stream": False,
            "options": {
                "temperature": float(getattr(self, "temperature", 0.2)),
                "num_predict": int(getattr(self, "max_output_tokens", 256)),
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # non-stream result has "response"
            out = data.get("response", "")
            return out if isinstance(out, str) else ""
        except Exception:
            return ""
