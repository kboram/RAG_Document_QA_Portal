"""
문서 기반 RAG QA를 위한 OpenAI LLM 호출 모듈.

- 문서에서 추출한 청크(context)를 기반으로 OpenAI GPT에 질문
- 반드시 문서 안에서만 근거를 찾도록 강제하는 system 프롬프트 적용
- 잘못 추론하는 것을 막기 위해 temperature=0.2 적용
"""

import os
from typing import List
from openai import OpenAI, OpenAIError


# 전역 클라이언트 재사용
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """
    OpenAI 클라이언트를 생성하거나, 이미 있으면 재사용.
    """
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다. "
            ".env 파일 또는 OS 환경변수에서 확인해주세요."
        )

    _client = OpenAI(api_key=api_key)
    return _client


def generate_answer_with_context(
    question: str,
    context_chunks: List[str],
    model: str = "gpt-4o-mini",
) -> str:
    """
    문서에서 뽑은 청크들(context_chunks) + 사용자의 질문(question)을 바탕으로
    LLM에게 답변을 생성하게 한다.

    - context_chunks: 의미/하이브리드 검색으로 뽑은 상위 청크들
    - question: 사용자의 질문
    - model: 사용할 OpenAI 모델 이름 (gpt-4o-mini 등)
    """
    client = get_client()

    # 청크들을 보기 좋게 하나의 문자열로 합친다.
    # [근거 1], [근거 2] 이런 식으로 번호를 붙여주면 LLM이 근거를 기억하기 쉽다.
    context_text = "\n\n".join(
        f"[근거 {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    system_prompt = (
        "당신은 한국어로 답변하는 '문서 기반 질의응답 보조 AI'입니다. "
        "반드시 제공된 문서 내용(근거) 안에서만 답을 찾으려고 하세요. "
        "문서에 없는 내용은 절대 추측하지 말고, "
        "'이 문서에서 알 수 없습니다.' 라고 말해야 합니다. "
        "답변할 때는 가능한 한 [근거 N] 번호를 같이 언급해 주세요."
    )

    user_prompt = f"""
아래는 한 문서의 일부 내용(여러 청크)입니다. 이 내용을 바탕으로 사용자의 질문에 답변해 주세요.

[문서 내용]
{context_text}

[질문]
{question}

[요청]
1) 최대 5문장 이내로 간단하고 명확하게 답변해 주세요.
2) 반드시 위의 [근거 N] 안에서만 답을 찾으세요.
3) 만약 질문에 대한 정보가 문서에 충분히 없다면,
   무조건 "이 문서에서 알 수 없습니다."라고 답하세요.
4) 답변 마지막 줄에 괄호로 사용한 근거 번호를 적어 주세요. 예: (근거 1, 3)
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,  # 더 보수적으로 (추측 줄이기)
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except OpenAIError as e:
        # OpenAI 쪽에서 할당량 부족, 인증 오류 등
        return f"[LLM 호출 실패] {type(e).__name__}: {e}"

    except Exception as e:
        # 그 외 알 수 없는 예외 처리
        return f"[LLM 호출 중 알 수 없는 오류] {e}"
