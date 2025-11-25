from django.db import models


class UploadedDocument(models.Model):
    title = models.CharField(max_length=200)
    file = models.FileField(upload_to="documents/")
    content = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class DocumentChunk(models.Model):
    document = models.ForeignKey(
        UploadedDocument,
        related_name="chunks",
        on_delete=models.CASCADE,
    )
    chunk_index = models.IntegerField()
    content = models.TextField()

    class Meta:
        ordering = ["chunk_index"]

    def __str__(self):
        return f"{self.document.title} - chunk {self.chunk_index}"


class QuestionHistory(models.Model):
    """
    문서별 Q&A 기록 저장용 모델
    - document : 어떤 문서에 대한 질문인지
    - question : 사용자가 한 질문
    - answer   : GPT가 생성한 답변
    - confidence : 하이브리드 검색 상위 청크의 스코어(%) (옵션)
    - created_at : 기록 시각
    """
    document = models.ForeignKey(
        UploadedDocument,
        related_name="questions",
        on_delete=models.CASCADE,
    )
    question = models.TextField()
    answer = models.TextField()
    confidence = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.document.title}] {self.question[:20]}..."
