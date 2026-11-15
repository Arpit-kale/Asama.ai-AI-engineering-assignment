"""Text embedding providers.

* ``LocalEmbedder`` - fully offline. Hashed bag-of-words plus a business *concept* layer, so
  "Diwali 5% off on deliveries" lands near "festive discounts, card payment partnerships"
  even with little word overlap. Query-side expansion adds *complementary* concepts
  (a consumer promotion is relevant to payment and card partners).
* ``OpenAIEmbedder`` - dense embeddings (text-embedding-3-small) for production quality.

Both return L2-normalised float32 vectors, so a dot product equals cosine similarity.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np

# --------------------------------------------------------------------------- concepts
CONCEPT_LEXICON: dict[str, set[str]] = {
    "promotion": {"offer", "discount", "off", "sale", "deal", "promo", "coupon", "cashback",
                  "voucher", "rebate", "promotion", "flat", "bogo", "freebie", "co-marketing"},
    "festive": {"diwali", "holi", "festive", "festival", "eid", "christmas", "navratri",
                "dussehra", "pongal", "onam", "new year", "season", "big billion days"},
    "food_delivery": {"food", "restaurant", "meal", "swiggy", "zomato", "dining", "grocery",
                      "quick commerce", "instamart", "delivery order", "food delivery"},
    "ecommerce": {"e-commerce", "ecommerce", "shopping", "flipkart", "amazon", "cart",
                  "checkout", "seller", "marketplace", "online", "retail", "order", "shopper"},
    "payments": {"payment", "pay", "upi", "wallet", "transaction", "merchant", "phonepe",
                 "paytm", "gpay", "qr", "checkout", "cashback"},
    "cards_banking": {"card", "credit", "debit", "bank", "banking", "hdfc", "icici", "sbi",
                      "account", "reward", "points", "spend", "spending"},
    "lending": {"loan", "emi", "financing", "finance", "bnpl", "credit line", "interest rate",
                "no-cost emi"},
    "automotive": {"bike", "motorcycle", "scooter", "two-wheeler", "vehicle", "auto", "ev",
                   "electric vehicle", "dealer", "dealership", "fuel", "car", "showroom",
                   "test ride", "automobile", "bajaj"},
    "logistics": {"logistics", "shipping", "courier", "warehouse", "fulfilment",
                  "fulfillment", "last-mile", "last mile", "supply chain", "fleet", "rider"},
    "travel": {"travel", "flight", "hotel", "booking", "trip", "airline", "holiday", "railway"},
    "insurance": {"insurance", "policy", "premium", "cover", "claim"},
    "telecom": {"telecom", "recharge", "mobile plan", "data pack", "broadband", "5g", "sim"},
}

# If a post is about X, partners interested in Y may also care (query-side expansion).
COMPLEMENTS: dict[str, tuple[str, ...]] = {
    "promotion": ("payments", "cards_banking"),
    "festive": ("promotion", "cards_banking", "ecommerce"),
    "food_delivery": ("payments", "logistics"),
    "ecommerce": ("payments", "logistics", "cards_banking", "lending"),
    "payments": ("ecommerce", "cards_banking"),
    "cards_banking": ("payments", "lending"),
    "lending": ("automotive", "cards_banking"),
    "automotive": ("lending", "insurance"),
    "logistics": ("ecommerce", "food_delivery"),
    "travel": ("payments", "cards_banking", "insurance"),
    "insurance": ("automotive", "travel"),
    "telecom": ("payments",),
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "at", "by", "with", "from",
    "is", "are", "be", "this", "that", "we", "our", "all", "any", "during", "above", "below",
    "will", "can", "via", "as", "it", "its", "into", "about", "interested", "opportunity",
    "new", "get", "now", "more", "up", "per", "week", "when", "who", "you", "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9₹%]+(?:-[a-z0-9]+)*")


def _stem(token: str) -> str:
    """Tiny, predictable plural stemmer (deliveries->delivery, offers->offer)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "sses")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


_LEXICON_STEMMED: dict[str, set[str]] = {
    concept: {" ".join(tokenize(term)) for term in terms} for concept, terms in CONCEPT_LEXICON.items()
}


def extract_concepts(text: str) -> set[str]:
    tokens = tokenize(text)
    grams = set(tokens)
    grams.update(" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1))
    grams.update(" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2))
    found = {c for c, terms in _LEXICON_STEMMED.items() if grams & terms}
    if any("%" in t for t in tokens):
        found.add("promotion")
    return found


def expand_concepts(concepts: set[str]) -> set[str]:
    """Complementary concepts that are *not* already present."""
    out: set[str] = set()
    for c in concepts:
        out.update(COMPLEMENTS.get(c, ()))
    return out - concepts


# --------------------------------------------------------------------------- providers
class Embedder(Protocol):
    dim: int

    def embed_document(self, text: str) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


class LocalEmbedder:
    """Deterministic, dependency-free embedder (feature hashing)."""

    TOKEN_WEIGHT = 1.0
    CONCEPT_WEIGHT = 3.0
    COMPLEMENT_WEIGHT = 1.5

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _index(self, feature: str) -> int:
        return int(hashlib.md5(feature.encode()).hexdigest()[:8], 16) % self.dim

    def _vector(self, text: str, expand: bool) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            if tok not in STOPWORDS and len(tok) > 1:
                vec[self._index(f"t:{tok}")] += self.TOKEN_WEIGHT
        concepts = extract_concepts(text)
        for c in concepts:
            vec[self._index(f"c:{c}")] += self.CONCEPT_WEIGHT
        if expand:
            for c in expand_concepts(concepts):
                vec[self._index(f"c:{c}")] += self.COMPLEMENT_WEIGHT
        return _normalize(vec)

    def embed_document(self, text: str) -> np.ndarray:
        return self._vector(text, expand=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text, expand=True)


class OpenAIEmbedder:
    """Dense embeddings through the OpenAI (or any compatible) embeddings API."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.dim = 1536
        self._model = model
        self._client = httpx.Client(
            base_url=base_url, timeout=20, headers={"Authorization": f"Bearer {api_key}"}
        )

    def _embed(self, text: str) -> np.ndarray:
        resp = self._client.post("/embeddings", json={"model": self._model, "input": text})
        resp.raise_for_status()
        vec = np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)
        self.dim = vec.shape[0]
        return _normalize(vec)

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


def build_embedder(kind: str, api_key: str | None, model: str, base_url: str) -> Embedder:
    if kind == "openai":
        if not api_key:
            raise RuntimeError("AN_EMBEDDER=openai requires OPENAI_API_KEY")
        return OpenAIEmbedder(api_key, model, base_url)
    return LocalEmbedder()
