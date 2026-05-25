from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

from dotenv import load_dotenv

from app.models import Product, RecommendRequest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
QWEN_EMBEDDING_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
QWEN_EMBEDDING_MODEL = "text-embedding-v4"


class VectorRecallUnavailable(RuntimeError):
    """Raised when vector recall cannot be used and rule fallback should run."""


class HashEmbeddingClient:
    """Small deterministic local embedding fallback for offline development."""

    provider = "local_hash"
    model = "hashing-64"
    dimension = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = [token for token in _tokenize(text) if token]
        if not tokens:
            tokens = [text or "empty"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        return _normalize(vector)


class QwenEmbeddingClient:
    """Qwen/DashScope OpenAI-compatible embedding client."""

    provider = "qwen"
    model = QWEN_EMBEDDING_MODEL
    dimension = 1024

    def __init__(self, api_key: str, endpoint: str = QWEN_EMBEDDING_ENDPOINT):
        self.api_key = api_key
        self.endpoint = endpoint

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 10):
            embeddings.extend(self._embed_batch(texts[start:start + 10]))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps(
            {
                "model": self.model,
                "input": texts,
                "dimensions": self.dimension,
                "encoding_format": "float",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=20) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise VectorRecallUnavailable(f"Qwen embedding request failed: {exc}") from exc

        data = json.loads(raw)
        embeddings = [
            item["embedding"]
            for item in sorted(data.get("data", []), key=lambda item: item["index"])
        ]
        if len(embeddings) != len(texts):
            raise VectorRecallUnavailable("Qwen embedding response size mismatch")
        return embeddings


class ProductVectorStore:
    """Chroma-backed product recall with rule-based fallback outside this class."""

    def __init__(self):
        self.embedding_client = _build_embedding_client()
        self.collection_name = (
            f"products_{self.embedding_client.provider}_"
            f"{self.embedding_client.model.replace('-', '_')}_"
            f"{self.embedding_client.dimension}"
        )
        self._client = None
        self._collection = None
        self._indexed_fingerprint: str | None = None

    @property
    def backend_name(self) -> str:
        return f"chroma:{self.embedding_client.provider}:{self.embedding_client.model}"

    def recall(
        self,
        request_model: RecommendRequest,
        products: list[Product],
        limit: int,
    ) -> list[str]:
        if not products:
            return []
        self._ensure_collection(products)
        query_text = build_query_text(request_model)
        query_embedding = self.embedding_client.embed_texts([query_text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, len(products)),
            include=["metadatas"],
        )
        return list(results.get("ids", [[]])[0])

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "collection": self.collection_name,
            "persist_directory": str(CHROMA_DIR),
            "indexed_fingerprint": self._indexed_fingerprint,
        }

    def _ensure_collection(self, products: list[Product]) -> None:
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise VectorRecallUnavailable("chromadb is not installed") from exc

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "embedding_provider": self.embedding_client.provider,
                    "embedding_model": self.embedding_client.model,
                    "embedding_dimension": self.embedding_client.dimension,
                },
            )

        fingerprint = _products_fingerprint(products)
        if fingerprint == self._indexed_fingerprint:
            return

        existing_count = self._collection.count()
        if existing_count == len(products):
            self._indexed_fingerprint = fingerprint
            return

        ids = [product.product_id for product in products]
        documents = [build_product_document(product) for product in products]
        metadatas = [
            {
                "category": product.category,
                "brand": product.brand,
                "price": product.price,
                "stock": product.stock,
            }
            for product in products
        ]
        embeddings = self.embedding_client.embed_texts(documents)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._indexed_fingerprint = fingerprint


def build_product_document(product: Product) -> str:
    return " ".join(
        str(part)
        for part in [
            product.name,
            product.source_name or "",
            product.category,
            product.source_category or "",
            product.brand,
            " ".join(product.tags),
            f"价格 {product.price}",
            f"评分 {product.rating}" if product.rating is not None else "",
        ]
        if part
    )


def build_query_text(request_model: RecommendRequest) -> str:
    parts = [
        request_model.scene,
        " ".join(request_model.preferred_categories),
        " ".join(request_model.liked_brands),
        " ".join(request_model.preferred_tags),
    ]
    if request_model.budget_min is not None or request_model.budget_max is not None:
        parts.append(f"预算 {request_model.budget_min or 0} 到 {request_model.budget_max or '不限'}")
    return " ".join(part for part in parts if part).strip() or "电商商品推荐"


def _build_embedding_client() -> QwenEmbeddingClient | HashEmbeddingClient:
    load_dotenv(BASE_DIR / ".env")
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("ALIYUN_API_KEY")
    )
    provider = os.getenv("PRODUCT_VECTOR_EMBEDDING_PROVIDER", "auto").lower()
    if provider == "local":
        return HashEmbeddingClient()
    if provider in {"qwen", "dashscope", "auto"} and api_key and not _is_placeholder_key(api_key):
        return QwenEmbeddingClient(api_key=api_key)
    return HashEmbeddingClient()


def _is_placeholder_key(api_key: str) -> bool:
    lowered = api_key.strip().lower()
    return (
        not lowered
        or "your-" in lowered
        or lowered in {"sk-your-dashscope-api-key", "sk-xxx", "sk-placeholder"}
    )


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    current = []
    for char in lowered:
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _products_fingerprint(products: list[Product]) -> str:
    raw = "|".join(
        f"{product.product_id}:{product.name}:{product.brand}:{product.category}:{product.price}"
        for product in products
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_product_vector_store() -> ProductVectorStore:
    return ProductVectorStore()
