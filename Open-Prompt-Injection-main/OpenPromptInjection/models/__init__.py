# OpenPromptInjection/models/__init__.py

def create_model(config):
    """
    Factory method to create an LLM instance with LAZY imports so that
    only the selected provider's dependencies are loaded.
    """
    provider = config["model_info"]["provider"].lower()
    model_name = config["model_info"]["name"].lower()

    if provider == 'palm2':
        from .PaLM2 import PaLM2
        return PaLM2(config)

    elif provider == 'vicuna':
        from .Vicuna import Vicuna
        return Vicuna(config)

    elif provider == 'gpt':
        from .GPT import GPT
        return GPT(config)

    elif provider == 'gpt_azure':
        from .GPT import GPTAzure
        return GPTAzure(config)

    elif provider == 'llama':
        from .Llama import Llama
        return Llama(config)

    elif provider == 'flan':
        from .Flan import Flan
        return Flan(config)

    elif provider == 'internlm':
        from .Internlm import Internlm
        return Internlm(config)

    elif provider == 'deepseek':
        from .DeepSeek import (
            DeepSeekWrapper,
            DeepSeekR1DistillLLaMAWrapper,
            DeepSeekR1DistillQwenWrapper,
        )
        if 'distill' in model_name:
            if 'llama' in model_name:
                return DeepSeekR1DistillLLaMAWrapper(config)
            elif 'qwen' in model_name:
                return DeepSeekR1DistillQwenWrapper(config)
            else:
                raise ValueError("Unknown DeepSeek distill variant in model name")
        else:
            return DeepSeekWrapper(config)

    elif provider == 'ollama':
        from .Ollama import Ollama
        return Ollama(config)

    elif provider == 'hflocal':
        # Local Hugging Face Transformers (no API key)
        from .HFLocal import HFLocal
        return HFLocal(config)

    else:
        raise ValueError(f"ERROR: Unknown provider '{provider}'")


def create_qlora_model(config):
    """
    Factory method to create a QLoRA LLM (also using lazy imports).
    """
    provider = config["model_info"]["provider"].lower()

    if provider == 'mistral':
        from .QLoraModel import QLoraModel
        return QLoraModel(config)

    elif provider == 'llama':
        from .QLoraModel import QLoraModel
        return QLoraModel(config)

    elif provider == 'llama3':
        from .Llama3 import Llama3
        return Llama3(config)

    else:
        raise ValueError(f"ERROR: Unknown provider '{provider}'")
