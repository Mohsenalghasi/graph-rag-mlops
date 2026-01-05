# core/generate.py
import os
from openai import AzureOpenAI

def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

def generate_answer(messages, temperature: float = 0.2, max_tokens: int = 600) -> str:
    """
    Uses your Azure OpenAI chat deployment.
    """
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("Missing AZURE_OPENAI_CHAT_DEPLOYMENT in .env (chat model deployment name).")

    resp = _client().chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
