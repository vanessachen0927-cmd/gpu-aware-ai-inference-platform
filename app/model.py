from typing import Protocol


class ModelBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


class MockBackend:
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return f"[mock inference] {prompt[:max_new_tokens]}"


class TransformersBackend:
    def __init__(self, model_id: str):
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline("text-generation", model=model_id, device=device)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        return result[0]["generated_text"]


def create_backend(name: str, model_id: str) -> ModelBackend:
    if name == "mock":
        return MockBackend()
    if name == "transformers":
        return TransformersBackend(model_id)
    raise ValueError(f"Unsupported MODEL_BACKEND: {name}")
