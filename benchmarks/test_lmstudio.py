"""Verify connection and basic generation with LM Studio."""

import json
import urllib.request
import time
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.embeddings import Embeddings


class LMStudioChat(BaseChatModel):
    base_url: str = "http://127.0.0.1:1234/v1"
    model_name: str = "google/gemma-4-12b"
    temperature: float = 0.0
    timeout: int = 120

    @property
    def _llm_type(self) -> str:
        return "lmstudio"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload_messages = []
        for m in messages:
            role = "user"
            if m.type == "system":
                role = "system"
            elif m.type == "ai":
                role = "assistant"
            payload_messages.append({"role": role, "content": m.content})

        req_body = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_json = json.loads(response.read().decode("utf-8"))

        choice = res_json.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        # Handle reasoning models where output might be in reasoning_content or content
        if not content and "reasoning_content" in msg:
            content = msg["reasoning_content"]

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])


class LMStudioEmbeddings(Embeddings):
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "text-embedding-nomic-embed-text-v1.5"
    timeout: int = 60

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        req_body = {
            "model": self.model,
            "input": texts,
        }
        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            res_json = json.loads(response.read().decode("utf-8"))
        # Sort by index if returned out of order
        data_items = sorted(res_json.get("data", []), key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in data_items]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


if __name__ == "__main__":
    print("Testing LMStudioEmbeddings...")
    emb = LMStudioEmbeddings()
    t0 = time.time()
    vec = emb.embed_query("test query")
    print(f"Embedding success: dim={len(vec)}, took {(time.time() - t0)*1000:.1f}ms")

    print("\nTesting LMStudioChat with gemma-4-12b...")
    chat = LMStudioChat()
    t0 = time.time()
    res = chat.invoke([HumanMessage(content="Hello! In 5 words, what is OKF?")])
    print(f"Chat response (took {time.time() - t0:.2f}s):\n{res.content}")
