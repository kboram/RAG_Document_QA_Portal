"""
문서 요약 관련 유틸 함수 모음

- build_summary_context(text):
    LLM 요약에 넣기 좋은 컨텍스트 청크 리스트 생성
- summarize_document_chunks(chunks, top_k):
    간단한 청크 요약용 (document_summary 뷰에서 사용 가능)
"""

from typing import List
from chatbot.utils.llm_client import generate_answer_with_context

def build_summary_context(raw_text: str, max_chars: int = 6000) -> List[str]:
    """
    LLM 요약에 사용할 컨텍스트 청크 리스트를 만들어준다.

    - 문서가 짧으면 전체를 한 덩어리로 반환
    - 길면 앞/뒤 중요한 부분을 잘라서 2~3개로 반환
    """
    if not raw_text:
        return []

    text = raw_text.strip()

    # 전체가 max_chars 이하면 그대로 하나만 반환
    if len(text) <= max_chars:
        return [text]

    # 너무 길면 앞/뒤를 잘라서 구성 (대략 절반씩)
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]

    # head / tail 이 너무 비슷하면 head만 쓰고, 아니면 둘 다 사용
    contexts = [head, tail] if head.strip() != tail.strip() else [head]
    return contexts


def summarize_document_chunks(chunks: List[str], top_k: int = 3) -> List[str]:
    """
    document_summary 뷰에서 사용할 수 있는
    '간단 버전' 요약용 함수.

    지금은 일단 '앞쪽 중요해 보이는 청크 top_k개' 만 골라서 반환.
    (원하면 여기 나중에 BM25/임베딩 기반으로 똑똑하게 바꿀 수 있음)
    """
    if not chunks:
        return []

    # 너무 짧은 청크 제거
    filtered = [c.strip() for c in chunks if c and len(c.strip()) > 20]

    if not filtered:
        return []

    return filtered[:top_k]

def summarize_document_with_gpt(document, max_lines=5):
    """
    OpenAI GPT 기반 문서 요약 함수
    document: UploadedDocument 인스턴스
    """

    # 문서 내용이 없는 경우
    if not document or not document.content:
        return "요약할 문서 내용이 없습니다."

    # 1) 문서 청크 가져오기
    chunks = [c.content for c in document.chunks.all().order_by("chunk_index")]

    if not chunks:
        return "이 문서에는 청크 데이터가 없습니다."

    # 2) GPT에게 전달할 요약 프롬프트 생성
    question = (
        f"다음 문서를 기반으로, 반드시 {max_lines}줄 이하로 한국어로 핵심만 요약해줘. "
        f"불필요한 설명 없이 중요한 내용만 정리해줘."
    )

    try:
        summary = generate_answer_with_context(
            question=question,
            context_chunks=chunks[:5]  # 너무 많으면 비용 크므로 상위 5개만 전달
        )
        return summary

    except Exception:
        return "GPT 요약 생성 중 오류가 발생했습니다."