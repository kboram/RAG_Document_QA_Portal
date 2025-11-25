"""
문서 내용을 여러 조각(청크)으로 나눈 뒤,
질문과의 단어 겹침 정도를 점수로 계산해서
상위 몇 개 조각을 반환하는 간단한 RAG 준비용 유틸 모듈.

- 외부 라이브러리( sklearn, sentence-transformers ) 없이 동작하도록 작성
- UploadedDocument 모델의 doc.content(문서 전체 텍스트)를 사용
"""

from typing import List, Dict
import re
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

from chatbot.utils.vector_store import get_embedding_model

def _normalize(text: str) -> List[str]:
    """
    한글/영문/숫자만 남기고 소문자 + 공백 기준으로 토큰화.
    너무 정교하진 않지만 간단한 검색에는 충분.
    """
    if not text:
        return []

    text = text.lower()
    # 한글, 영문, 숫자, 공백만 남기기
    text = re.sub(r"[^0-9a-z가-힣\s]", " ", text)
    tokens = text.split()
    return tokens


def split_into_chunks(text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
    """
    긴 텍스트를 max_chars 길이로 잘라 여러 청크로 만든다.
    청크 사이에는 overlap 만큼 겹치게 해서 문장이 너무 잘리지 않게 함.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        # 다음 청크 시작 위치 (겹치는 부분만큼 뒤로 당김)
        start = max(0, end - overlap)

    return chunks


def _overlap_score(question: str, chunk: str) -> float:
    """
    질문과 청크 사이의 '단어 겹침 점수'를 계산.
    - 공통 토큰 개수를 기반으로 간단히 점수화
    - 너무 긴 청크가 유리하지 않도록 길이로 나눠서 정규화
    """
    q_tokens = _normalize(question)
    c_tokens = _normalize(chunk)

    if not q_tokens or not c_tokens:
        return 0.0

    q_counts = Counter(q_tokens)
    c_counts = Counter(c_tokens)

    common = 0
    for t in q_counts:
        if t in c_counts:
            common += min(q_counts[t], c_counts[t])

    # 길이 보정 (루트 길이 사용)
    denom = (len(q_tokens) ** 0.5) * (len(c_tokens) ** 0.5)
    if denom == 0:
        return 0.0

    return common / denom


def hybrid_chunk_search(document, query, top_k: int = 5, alpha: float = 0.6):
    """
    BM25 + 임베딩(semantic) 기반 하이브리드 검색 함수.

    - top_k : 상위 몇 개 청크를 가져올지 (기본 5개)
    - alpha : 의미 검색(임베딩) 비중 (0~1, 클수록 semantic 비중 ↑)
    """
    query = (query or "").strip()
    if not query:
        return []

    # 1) 이 문서의 청크들 가져오기
    chunks = list(document.chunks.all().order_by("chunk_index"))
    if not chunks:
        return []

    chunk_texts = [c.content for c in chunks]

    # -------------------- (A) BM25 점수 --------------------
    tokenized_chunks = [text.split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    # 질문도 간단히 띄어쓰기로 토큰화
    query_tokens = query.split()
    bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=float)

    bm25_min, bm25_max = float(bm25_scores.min()), float(bm25_scores.max())
    if bm25_max - bm25_min < 1e-9:
        # 모든 점수가 거의 동일하면 1로 통일
        bm25_norm = np.ones_like(bm25_scores)
    else:
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)

    # -------------------- (B) 의미(임베딩) 점수 --------------------
    model = get_embedding_model()
    chunk_embs = model.encode(chunk_texts, convert_to_numpy=True)
    query_emb = model.encode([query], convert_to_numpy=True)[0]

    chunk_norms = np.linalg.norm(chunk_embs, axis=1) + 1e-10
    query_norm = np.linalg.norm(query_emb) + 1e-10
    semantic_scores = chunk_embs.dot(query_emb) / (chunk_norms * query_norm)

    semantic_scores = semantic_scores.astype(float)
    sem_min, sem_max = float(semantic_scores.min()), float(semantic_scores.max())
    if sem_max - sem_min < 1e-9:
        sem_norm = np.ones_like(semantic_scores)
    else:
        sem_norm = (semantic_scores - sem_min) / (sem_max - sem_min)

    # -------------------- (C) 점수 결합 --------------------
    # alpha * semantic + (1 - alpha) * BM25
    final_scores = alpha * sem_norm + (1.0 - alpha) * bm25_norm

    # -------------------- (D) 상위 top_k 추출 --------------------
    sorted_idx = np.argsort(final_scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(sorted_idx, start=1):
        results.append(
            {
                "rank": rank,
                "score": float(final_scores[idx]),  # 0~1 사이 정도의 최종 점수
                "text": chunk_texts[idx],
            }
        )

    return results