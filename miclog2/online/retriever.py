from __future__ import annotations

import math
import re
from collections import Counter

from .types import LogRecord, RetrievedDemo


TOKEN_RE = re.compile(r"<[a-z_]+>|[A-Za-z_]+|\d+|[^\sA-Za-z_0-9]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(
        self,
        support_records: list[LogRecord],
        retrieval_field: str = "Content",
        exclude_same_content: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.support_records = support_records
        self.retrieval_field = retrieval_field
        self.exclude_same_content = exclude_same_content
        self.k1 = k1
        self.b = b

        self.documents = [tokenize(self._record_text(record)) for record in support_records]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(doc) for doc in self.documents]
        doc_freq: Counter[str] = Counter()
        for document in self.documents:
            doc_freq.update(set(document))
        doc_count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def _record_text(self, record: LogRecord) -> str:
        if self.retrieval_field == "Content":
            return record.content
        value = record.extra.get(self.retrieval_field)
        if value is None or str(value).strip() == "":
            return record.content
        return str(value)

    def _query_text(self, record: LogRecord) -> str:
        return self._record_text(record)

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        if not query_terms or not self.documents:
            return 0.0
        doc_length = self.doc_lengths[doc_index]
        if doc_length == 0 or self.avg_doc_length == 0.0:
            return 0.0
        term_freq = self.term_freqs[doc_index]
        score = 0.0
        for term in set(query_terms):
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (tf * (self.k1 + 1.0)) / denominator
        return score

    def retrieve(self, query_record: LogRecord, top_k: int) -> list[RetrievedDemo]:
        if top_k <= 0 or not self.support_records:
            return []
        query_terms = tokenize(self._query_text(query_record))
        scored: list[tuple[int, float]] = []
        for index, record in enumerate(self.support_records):
            if self.exclude_same_content and record.content == query_record.content:
                continue
            scored.append((index, self._score(query_terms, index)))
        scored.sort(
            key=lambda item: (
                -item[1],
                self.support_records[item[0]].line_id,
                self.support_records[item[0]].content,
            )
        )

        demos: list[RetrievedDemo] = []
        for rank, (doc_index, score) in enumerate(scored[:top_k], start=1):
            record = self.support_records[doc_index]
            if not record.event_template:
                continue
            demos.append(
                RetrievedDemo(
                    dataset=record.dataset,
                    line_id=record.line_id,
                    content=record.content,
                    template=record.event_template,
                    score=score,
                    rank=rank,
                )
            )
        return demos


def demos_for_prompt(demos: list[RetrievedDemo]) -> list[RetrievedDemo]:
    return list(reversed(demos))
