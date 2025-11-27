# **RAG 기반 문서 질의응답 포털(RAG Document QA Portal)**

## 프로젝트 소개

이 프로젝트는 PDF나 텍스트 문서를 업로드하면, 문서를 자동으로 청크 단위로 분석하고 의미 기반 검색을 통해 문서 기반 질문에 정확하게 답변할 수 있는 시스템이다.

GPT 모델과 의미 검색, BM25 검색을 조합해 질문과 가장 관련 있는 문단을 찾고, 그 내용을 기반으ㅗㄹ 답변을 제공한다.

최근 기업 및 학교 연구 환경에서 방대한 문서를 빠르게 이해해야 하는 일이 많아지고 있는데, 이 시스템은 문서를 직접 읽기 않고 핵심 내용을 빠르게 파악하도록 도와주는 도구를 목표로 한다.


## 주요 기능

* 문서 업로드 & 자동 텍스트 추출
* 문서 자동 요약(GPT 기반 핵심 요약)
* RAG 기반 문서 질문응답 (문서 내용을 기반으로 답변 생성)
* BM25 + 임베딩 하이브리드 검색
* 질문 기록 저장 및 신뢰도 점수 표시
* 대시보드에서 문서별 질문 횟수 통계
* 문서 검색 기능

## 기술 스택

* Backend - Python, Django
* NLP/AI - Sentence-Transformers, BM25, OpenAI GPT
* DB - SQLite
* Fromtend - HTML, Bootstrap
* ETC - RAG 구조 설계, Embedding 기반 검색

## 실행 화면 및 결과물

* 첫 페이지
* <img width="1918" height="907" alt="홈" src="https://github.com/user-attachments/assets/6863ae21-073b-459a-9f52-09422807598e" />

* 문서 업로드 페이지
* <img width="1918" height="907" alt="문서 업로드" src="https://github.com/user-attachments/assets/c08e3c81-5b01-4fce-96b6-7ff30250d6a5" />

* 문서 업로드 후 페이지
* <img width="1918" height="913" alt="문서 업로드 후 첫 화면" src="https://github.com/user-attachments/assets/0a6508e8-308c-496a-9ee2-45d49b1427a1" />

* 문서 요약 결과 화면
* <img width="1917" height="906" alt="문서 요약 생성" src="https://github.com/user-attachments/assets/810fb29b-bd1e-4f27-8904-e0509cc217e2" />

* RAG 질문하기 결과 화면
* <img width="1918" height="915" alt="문서 기반 질문 결과" src="https://github.com/user-attachments/assets/a08b6744-f292-455d-aa82-4f92b5dd52ac" />

* 대시보드 예시 캡처
* <img width="1918" height="922" alt="대시보드 화면" src="https://github.com/user-attachments/assets/2fa48de1-93b7-412f-9d01-62c4bc1eaf26" />


## 트러블슈팅 & 개선 사항
* 문서 기반 질문 정확도가 낮았던 문제
  * BM25 + Embedding 하이브리드 검색 도입, top_k 조절
* 요약 기능이 단순 앞부분 자르기였던 문제
  * GPT 기반 요약 모델 적용
* 질문 기록이 삭제되는 문제
  * 세션 방식에서 DB 저장 방식으로 개선
* 질문에 문맥이 맞지 않는 답변 출력
  * LLM 프롬프트 강화 및 temperature 조정
 
## 앞으로 개선할 내용

* 문서 길이에 따라 자동 요약 길이 조정
* 여러 문서 간 RAG 비교 기능 적용
* 대용량 문서 처리 속도 개선
* 문서 카테고리별 추천 질문 자동 생성 기능


## 프로젝트를 진행하며 느낀 점

단순한 GPT 응답이 아니라, 실제 문서를 기반으로 답변하는 시스템을 만들기 위해서는 검색 단계의 품질이 얼마나 중요한지 많이 느꼈다.

처음에는 정확도가 낮아서 답변이 엉뚱하게 나오는 경우가 많았는데, BM25와 Embedding을 조합하고 GPT 프롬프트를 개선하면서 점점 정확도가 올라갔다.

이번 프로젝트를 통해 RAG 구조의 핵심 원리와 실제 적용 과정을 직접 경험했고, AI를 활용한 서비스 개발의 가능성을 체감할 수 있었다.


