from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Count, Max, Q
from django.db.models.functions import TruncDate

from .models import UploadedDocument, DocumentChunk, QuestionHistory
from .forms import DocumentUploadForm

from rank_bm25 import BM25Okapi
import numpy as np

# 유틸 함수들
from chatbot.utils.document_loader import extract_text_from_file
from chatbot.utils.chunking import split_into_chunks
from chatbot.utils.vector_store import get_embedding_model
from chatbot.utils.llm_client import generate_answer_with_context
from chatbot.utils.summarizer import summarize_document_chunks
from chatbot.utils.rag_pipeline import hybrid_chunk_search


# ------------------------------------------------
# 메인 홈 화면
# ------------------------------------------------
def home(request):
    """
    메인 페이지
    - 업로드된 문서 목록을 보여줌
    - 검색(q)으로 제목 + 내용 검색
    - 문서별 질문 개수 / 마지막 질문 시각 간단 통계 포함
    """

    # 1) 검색어(q) 가져오기
    q = request.GET.get("q", "").strip()

    # 2) 기본 쿼리셋
    base_qs = UploadedDocument.objects.all()

    # 3) 검색어가 있으면 제목 + 내용으로 필터
    if q:
        base_qs = base_qs.filter(
            Q(title__icontains=q) |
            Q(content__icontains=q)
        )

    # 4) 문서별 간단 통계 annotate
    docs = (
        base_qs
        .annotate(
            question_count=Count("questions", distinct=True),
            last_question_at=Max("questions__created_at"),
        )
        .order_by("-uploaded_at")
    )

    context = {
        "documents": docs,
        "search_query": q,
        "total_docs": base_qs.count(),   # 검색 후 기준 개수
    }
    return render(request, "chatbot/home.html", context)

# ------------------------------------------------
# 문서 업로드
# ------------------------------------------------
def upload_document(request):
    """
    문서 업로드 뷰
    1) 사용자가 PDF/TXT 파일과 제목을 업로드하면
    2) UploadedDocument 로 저장
    3) 파일에서 텍스트를 추출해서 content 필드에 저장
    4) 텍스트를 청크 단위로 분리하여 DocumentChunk 로 저장
    """
    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # 1) 문서 정보 + 파일 저장
            doc = form.save()

            # 2) 파일 경로 가져와서 텍스트 추출
            file_path = doc.file.path
            full_text = extract_text_from_file(file_path)

            # 3) 추출된 텍스트를 문서 content 에 저장
            doc.content = full_text or ""
            doc.save()

            # 4) 기존 청크가 있으면 삭제 (같은 문서 재업로드 대비)
            doc.chunks.all().delete()

            # 5) 텍스트를 청크로 분리
            chunks = split_into_chunks(doc.content, max_length=400)

            # 6) 청크들을 DocumentChunk로 저장
            for idx, chunk_text in enumerate(chunks):
                DocumentChunk.objects.create(
                    document=doc,
                    chunk_index=idx,
                    content=chunk_text,
                )

            # 7) 문서 상세 페이지로 이동
            return redirect("document_detail", pk=doc.pk)
    else:
        form = DocumentUploadForm()

    return render(request, "chatbot/upload.html", {"form": form})


# ------------------------------------------------
# 문서 상세 (요약 + 문서 기반 질문 + 히스토리)
# ------------------------------------------------
def document_detail(request, pk):
    """
    문서 상세 페이지:
    - GET  : 문서 내용 + 이전 질문 기록 보여줌
    - POST : mode 값에 따라 동작 분기
        * mode == 'summary' → 문서 요약 (GPT 사용)
        * mode == 'qa'      → 문서 기반 질문 (RAG + GPT)
        * mode == 'reset'   → 요약/답변/검색결과 초기화
    """
    # 1) 문서 객체 가져오기
    doc = get_object_or_404(UploadedDocument, pk=pk)

    # 2) 기본 값 초기화
    summary_text = None          # 문서 요약 결과
    question = None              # 사용자가 입력한 질문
    llm_answer = None            # GPT가 생성한 답변
    search_results = []          # RAG 검색 결과 리스트
    confidence = None            # 신뢰도 점수 (%)

    # 3) 기존 질문 히스토리 (항상 템플릿에 보내기)
    history = QuestionHistory.objects.filter(document=doc).order_by("-created_at")

    if request.method == "POST":
        mode = request.POST.get("mode", "qa")  # 기본값 'qa'

        # ---------- 문서 요약 모드 ----------
        if mode == "summary":
            if doc.content:
                # 이 문서에 해당하는 청크들 가져오기
                chunks_qs = doc.chunks.all().order_by("chunk_index")
                chunk_texts = [c.content for c in chunks_qs]

                if chunk_texts:
                    try:
                        # 요약에 쓸 대표 청크 선택 (임베딩 기반)
                        key_chunks = summarize_document_chunks(chunk_texts, top_k=5)
                    except Exception:
                        # summarizer 에러 시 앞부분 몇 개 청크만 사용
                        key_chunks = chunk_texts[:5]

                    try:
                        # OpenAI GPT에게 요약 요청
                        summary_text = generate_answer_with_context(
                            question=(
                                "다음 문서 내용을 보고, 대학생이 이해하기 쉽게 "
                                "5줄 이내의 한국어로 핵심만 요약해줘."
                            ),
                            context_chunks=key_chunks,
                        )
                    except Exception:
                        # GPT 호출 실패 시, 앞부분 잘라서 보여주는 fallback
                        text = doc.content.strip()
                        max_len = 600
                        if len(text) > max_len:
                            summary_text = (
                                text[:max_len] + "\n...\n(일부만 표시한 요약입니다)"
                            )
                        else:
                            summary_text = text
                else:
                    summary_text = "이 문서에는 청크 데이터가 없습니다."
            else:
                summary_text = "이 문서에는 저장된 텍스트 내용이 없습니다."

        # ---------- 문서 기반 질문 모드 ----------
        elif mode == "qa":
            question = (request.POST.get("question") or "").strip()

            if question and doc.content:
                # 1) 하이브리드 검색 (top_k=5, alpha=0.6은 rag_pipeline 쪽에서 기본값)
                search_results = hybrid_chunk_search(doc, question, top_k=5, alpha=0.6)

                if search_results:
                    # 2) 최상위 결과의 점수 기준으로 신뢰도 계산
                    top_score = search_results[0].get("score", 0.0)
                    confidence = int(round(float(top_score) * 100))

                    # 임계값(Threshold) 설정: 너무 낮으면 "모르겠다" 처리
                    threshold = 0.35  # 필요하면 0.3~0.4 사이에서 조절해 보기

                    if top_score < threshold:
                        # 문서와 질문이 거의 안 맞는 경우 → GPT 호출 대신 안내 메시지
                        llm_answer = (
                            "질문과 관련된 내용을 문서에서 충분히 찾지 못했습니다.\n"
                            "질문을 조금 더 구체적으로 바꾸거나, 다른 문서를 참고하는 것이 좋겠습니다."
                        )
                    else:
                        # 3) 검색된 청크들을 GPT 컨텍스트로 사용
                        context_chunks = [r["text"] for r in search_results]

                        # 필요하면 여기에서 문서 요약 등 추가 컨텍스트도 붙일 수 있음
                        # ex) context_chunks.append("[문서 전체 요약]\n" + some_summary)

                        llm_answer = generate_answer_with_context(
                            question=question,
                            context_chunks=context_chunks,
                        )

                    # 4) 질문/답변 기록 저장
                    QuestionHistory.objects.create(
                        document=doc,
                        question=question,
                        answer=llm_answer or "",
                        confidence=confidence,
                    )
                    # 방금 추가한 기록 포함해서 다시 조회
                    history = QuestionHistory.objects.filter(document=doc).order_by(
                        "-created_at"
                    )
                else:
                    llm_answer = "질문과 관련된 문단을 문서에서 찾지 못했습니다."
                    confidence = None
            else:
                llm_answer = "질문이 비어 있거나, 문서 내용이 없습니다."

        # ---------- 결과 초기화 모드 ----------
        elif mode == "reset":
            summary_text = None
            question = None
            llm_answer = None
            search_results = []
            confidence = None
            # history는 그대로 둠 (기록 삭제까지 할 필요는 없음)

    # 4) 템플릿에 넘길 컨텍스트
    context = {
        "document": doc,
        "summary_text": summary_text,
        "question": question,
        "llm_answer": llm_answer,
        "search_results": search_results,
        "confidence": confidence,
        "history": history,
    }
    return render(request, "chatbot/document_detail.html", context)

# ------------------------------------------------
# 별도 요약 페이지 (선택 기능, 안 쓰면 안 들어가도 됨)
# ------------------------------------------------
def document_summary(request, pk):
    """
    업로드된 문서의 '요약 페이지'를 보여주는 뷰.
    (간단 버전 – 앞쪽 몇 개 청크만 사용)
    """
    doc = get_object_or_404(UploadedDocument, pk=pk)
    chunks_qs = doc.chunks.all().order_by("chunk_index")
    chunk_texts = [c.content for c in chunks_qs]

    summary_chunks = summarize_document_chunks(chunk_texts, top_k=3) if chunk_texts else []
    summary_text = "\n\n".join(summary_chunks) if summary_chunks else ""

    context = {
        "document": doc,
        "summary_chunks": summary_chunks,
        "summary_text": summary_text,
    }
    return render(request, "chatbot/document_summary.html", context)


# ------------------------------------------------
# 대시보드: 전체 통계 / 상위 문서 / 최근 질문
# ------------------------------------------------
def dashboard(request):
    """
    전체 RAG 포털에 대한 간단 대시보드
    - 총 문서 수
    - 총 질문 수
    - 질문이 많은 문서 TOP5
    - 최근 질문 10개
    - 일자별 질문 수
    """
    total_docs = UploadedDocument.objects.count()
    total_questions = QuestionHistory.objects.count()

    # 문서별 질문 수 TOP5
    top_docs = (
        UploadedDocument.objects
        .annotate(question_count=Count("questions"))
        .order_by("-question_count", "-uploaded_at")[:5]
    )

    # 최근 질문 10개
    recent_questions = (
        QuestionHistory.objects
        .select_related("document")
        .order_by("-created_at")[:10]
    )

    # 일자별 질문 수 (최근 30일)
    daily_counts = (
        QuestionHistory.objects
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )

    context = {
        "total_docs": total_docs,
        "total_questions": total_questions,
        "top_docs": top_docs,
        "recent_questions": recent_questions,
        "daily_counts": daily_counts,
    }
    return render(request, "chatbot/dashboard.html", context)
