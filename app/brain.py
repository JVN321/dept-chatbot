from __future__ import annotations

import json
import re
import sys
from urllib import error as urlerror
from urllib import request as urlrequest
from pathlib import Path
from typing import Any

from openai import OpenAI
from rapidfuzz import fuzz

from app.config import settings

if sys.version_info < (3, 14):
    try:
        import chromadb
    except Exception:  # noqa: BLE001
        chromadb = None
else:
    chromadb = None


class DepartmentBrain:
    _OFF_TOPIC_MESSAGE = (
        "I can help only with department-related questions like faculty, subjects, cabins, "
        "semesters, and availability."
    )
    _UNKNOWN_MESSAGE = "I don't have that information right now."
    _ABUSIVE_MESSAGE = "Please use respectful language. I can help with department information."
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
        "what",
        "who",
        "where",
        "when",
        "which",
        "whom",
    }

    def __init__(self) -> None:
        self._faculty = self._load_faculty(settings.faculty_file)
        self._department_notes = self._load_department_notes(settings.department_notes_file)
        self._rag_enabled = chromadb is not None and sys.version_info < (3, 14)
        self._collection = None
        if self._rag_enabled:
            try:
                self._chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=settings.chroma_collection
                )
            except Exception:  # noqa: BLE001
                self._rag_enabled = False
                self._collection = None
        self._provider = settings.ai_provider
        self._openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self._gemini_key = settings.gemini_api_key
        self._domain_terms = self._build_domain_terms()

    def readiness(self) -> dict[str, bool]:
        return {
            "data_loaded": bool(self._faculty),
            "rag_enabled": self._rag_enabled and self._collection is not None,
            "llm_configured": self._llm_is_configured(),
        }

    def _llm_is_configured(self) -> bool:
        if self._provider == "gemini":
            return bool(self._gemini_key)
        return self._openai is not None

    @staticmethod
    def _load_faculty(path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_department_notes(path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        return {t for t in tokens if len(t) > 2 and t not in DepartmentBrain._STOPWORDS}

    def _build_domain_terms(self) -> set[str]:
        base = {
            "department",
            "faculty",
            "professor",
            "teacher",
            "subject",
            "course",
            "semester",
            "cabin",
            "office",
            "room",
            "availability",
            "timing",
            "timetable",
            "class",
            "lab",
        }
        for item in self._faculty:
            base.update(self._tokenize(item.get("name", "")))
            for subject in item.get("subjects", []):
                base.update(self._tokenize(subject))
        for note in self._department_notes:
            base.update(self._tokenize(note.get("title", "")))
            base.update(self._tokenize(note.get("content", "")))
        return base

    def _is_abusive(self, question: str) -> bool:
        bad_words = {"idiot", "stupid", "useless", "dumb", "hate", "fool", "moron"}
        return bool(self._tokenize(question) & bad_words)

    def _is_domain_question(self, question: str) -> bool:
        q_tokens = self._tokenize(question)
        if q_tokens & self._domain_terms:
            return True

        # Intent keywords should still count as in-domain even if subject tokens
        # are abbreviated or not present in the current dataset.
        intent_terms = {
            "who",
            "where",
            "when",
            "teach",
            "teaches",
            "teacher",
            "faculty",
            "subject",
            "subjects",
            "course",
            "cabin",
            "office",
            "room",
            "semester",
            "availability",
            "timing",
            "timetable",
            "lab",
            "class",
            "department",
        }
        if q_tokens & intent_terms:
            return True

        if self._best_faculty_match(question) is not None:
            return True
        if self._extract_subject(question) is not None:
            return True
        return False

    def _best_faculty_match(self, question: str) -> dict[str, Any] | None:
        best_score = 0
        best_item: dict[str, Any] | None = None
        q = question.lower()
        for item in self._faculty:
            score = fuzz.partial_ratio(item["name"].lower(), q)
            if score > best_score:
                best_score = score
                best_item = item
        return best_item if best_score >= 75 else None

    def _extract_subject(self, question: str) -> str | None:
        q_tokens = self._tokenize(question)
        for item in self._faculty:
            for subject in item.get("subjects", []):
                s_tokens = self._tokenize(subject)
                if s_tokens and (s_tokens.issubset(q_tokens) or q_tokens.issubset(s_tokens)):
                    return subject
                if fuzz.partial_ratio(subject.lower(), question.lower()) >= 85:
                    return subject
        return None

    @staticmethod
    def _extract_branch(question: str) -> str | None:
        q = question.lower()
        branch_aliases = {
            "cse": "CSE",
            "computer science": "CSE",
            "it": "IT",
            "information technology": "IT",
            "aiml": "AIML",
            "ai ml": "AIML",
            "ai&ml": "AIML",
            "ai/ml": "AIML",
            "artificial intelligence": "AIML",
            "ds": "DS",
            "data science": "DS",
            "cyber": "CYBER",
            "cyber security": "CYBER",
            "cybersecurity": "CYBER",
        }
        for alias, canonical in branch_aliases.items():
            if alias in q:
                return canonical
        return None

    @staticmethod
    def _extract_semester(question: str) -> int | None:
        q = question.lower()
        match = re.search(r"\b(?:sem|semester)\s*[-:]?\s*(\d)\b", q)
        if match:
            return int(match.group(1))
        num_match = re.search(r"\b([1-8])(?:st|nd|rd|th)?\s*(?:sem|semester)\b", q)
        if num_match:
            return int(num_match.group(1))
        return None

    def try_structured_lookup(self, question: str) -> tuple[str | None, list[dict[str, Any]]]:
        q = question.lower()
        faculty = self._best_faculty_match(question)

        if faculty and any(k in q for k in ["where", "cabin", "room", "office", "find"]):
            return (
                f'{faculty["name"]} is in cabin {faculty["cabin"]}. '
                f'Availability: {faculty["availability"]}.',
                [{"id": faculty["id"], "text": json.dumps(faculty), "metadata": {"source": "structured"}}],
            )

        if faculty and any(k in q for k in ["subject", "teach", "teaches", "course"]):
            subjects = ", ".join(faculty["subjects"])
            return (
                f'{faculty["name"]} teaches: {subjects}.',
                [{"id": faculty["id"], "text": json.dumps(faculty), "metadata": {"source": "structured"}}],
            )

        subject = self._extract_subject(question)
        if subject and any(k in q for k in ["who", "teacher", "faculty", "teach"]):
            teachers = [x for x in self._faculty if subject in x.get("subjects", [])]
            if teachers:
                branch = self._extract_branch(question)
                semester = self._extract_semester(question)

                filtered = teachers
                if branch:
                    filtered = [t for t in filtered if branch in t.get("branches", [])]
                if semester:
                    filtered = [t for t in filtered if semester in t.get("semesters", [])]

                if not filtered:
                    teacher_details = "; ".join(
                        f'{t["name"]} (branches: {", ".join(t.get("branches", []))}; '
                        f'semesters: {", ".join(str(s) for s in t.get("semesters", []))})'
                        for t in teachers
                    )
                    return (
                        f"No exact match found for {subject}"
                        f"{' in ' + branch if branch else ''}"
                        f"{' semester ' + str(semester) if semester else ''}. "
                        f"Available faculty for {subject}: {teacher_details}.",
                        [
                            {"id": t["id"], "text": json.dumps(t), "metadata": {"source": "structured"}}
                            for t in teachers
                        ],
                    )

                names = ", ".join(t["name"] for t in filtered)
                detail_suffix = ""
                if len(filtered) > 1:
                    detail_suffix = " " + " ".join(
                        f'{t["name"]} handles branches {", ".join(t.get("branches", []))} '
                        f'for semesters {", ".join(str(s) for s in t.get("semesters", []))}.'
                        for t in filtered
                    )

                serialized = [
                    {"id": t["id"], "text": json.dumps(t), "metadata": {"source": "structured"}}
                    for t in filtered
                ]
                qualifier = []
                if branch:
                    qualifier.append(f"branch {branch}")
                if semester:
                    qualifier.append(f"semester {semester}")
                qualifier_text = f" for {' and '.join(qualifier)}" if qualifier else ""
                return (f"{subject} is taught by {names}{qualifier_text}.{detail_suffix}", serialized)

        note_answer, note_sources = self._lookup_department_note(question)
        if note_answer:
            return note_answer, note_sources

        return (None, [])

    def _lookup_department_note(self, question: str) -> tuple[str | None, list[dict[str, Any]]]:
        if not self._department_notes:
            return (None, [])

        q = question.lower().strip()

        if any(k in q for k in ["head of department", "hod", "department head"]):
            for note in self._department_notes:
                if "head of department" in note.get("title", "").lower():
                    return (
                        note.get("content", self._UNKNOWN_MESSAGE),
                        [
                            {
                                "id": note.get("id", "dept-note"),
                                "text": note.get("content", ""),
                                "metadata": {
                                    "source": "structured_note",
                                    "title": note.get("title", ""),
                                },
                            }
                        ],
                    )

        best_score = 0.0
        best_note: dict[str, Any] | None = None
        for note in self._department_notes:
            title = note.get("title", "")
            content = note.get("content", "")
            score = max(
                fuzz.partial_ratio(title.lower(), q),
                fuzz.partial_ratio(content.lower(), q),
            )
            if score > best_score:
                best_score = score
                best_note = note

        if best_note and best_score >= 70:
            return (
                best_note.get("content", self._UNKNOWN_MESSAGE),
                [
                    {
                        "id": best_note.get("id", "dept-note"),
                        "text": best_note.get("content", ""),
                        "metadata": {
                            "source": "structured_note",
                            "title": best_note.get("title", ""),
                        },
                    }
                ],
            )

        return (None, [])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._provider == "gemini":
            return [self._gemini_embed_text(text) for text in texts]

        if not self._openai:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        result = self._openai.embeddings.create(model=settings.embedding_model, input=texts)
        return [d.embedding for d in result.data]

    def _gemini_embed_text(self, text: str) -> list[float]:
        if not self._gemini_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        model = settings.embedding_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
            f"?key={self._gemini_key}"
        )
        body = {"content": {"parts": [{"text": text}]}}
        req = urlrequest.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini embedding request failed: {detail}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc
        values = payload.get("embedding", {}).get("values")
        if not values:
            raise RuntimeError("Gemini embedding response missing values.")
        return values

    def ingest_faculty(self) -> int:
        if not self._rag_enabled or self._collection is None:
            raise RuntimeError("RAG backend is unavailable in this environment.")

        ids: list[str] = []
        docs: list[str] = []
        metadata: list[dict[str, Any]] = []

        for faculty in self._faculty:
            chunk = (
                f'Faculty: {faculty["name"]}. '
                f'Subjects: {", ".join(faculty["subjects"])}. '
                f'Semesters: {", ".join(str(s) for s in faculty.get("semesters", []))}. '
                f'Cabin: {faculty["cabin"]}. '
                f'Availability: {faculty["availability"]}.'
            )
            ids.append(faculty["id"])
            docs.append(chunk)
            metadata.append(
                {
                    "type": "faculty",
                    "name": faculty["name"],
                    "cabin": faculty["cabin"],
                }
            )

        embeddings = self._embed(docs)
        self._collection.upsert(ids=ids, documents=docs, metadatas=metadata, embeddings=embeddings)
        return len(ids)

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not self._rag_enabled or self._collection is None:
            return []

        k = top_k or settings.top_k
        query_embedding = self._embed([question])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        items: list[dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            items.append(
                {
                    "id": f"rag-{idx + 1}",
                    "text": doc,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "score": dists[idx] if idx < len(dists) else None,
                }
            )
        return items

    def generate_grounded_answer(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not self._llm_is_configured():
            return self._UNKNOWN_MESSAGE

        context_block = "\n\n".join(
            [f"[{i+1}] {item['text']}" for i, item in enumerate(contexts)]
        )
        system_prompt = (
            "You are a department assistant."
            #"Do not guess or add details not present in the context. "
            #f"If answer is not in context, say: {self._UNKNOWN_MESSAGE} "
            "Keep answer brief, factual, and polite."
        )
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
        if self._provider == "gemini":
            return self._gemini_generate(system_prompt, user_prompt)

        completion = self._openai.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            max_tokens=180,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content or self._UNKNOWN_MESSAGE

    def _gemini_generate(self, system_prompt: str, user_prompt: str) -> str:
        
        if not self._gemini_key:
            return self._UNKNOWN_MESSAGE
        
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent"
            f"?key={self._gemini_key}"
        )
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 180},
        }
        req = urlrequest.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini generation request failed: {detail}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"Gemini generation request failed: {exc}") from exc

        candidates = payload.get("candidates", [])
        if not candidates:
            return self._UNKNOWN_MESSAGE
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or self._UNKNOWN_MESSAGE

    def _has_relevant_context(self, contexts: list[dict[str, Any]]) -> bool:
        # Chroma distance is lower for better matches.
        good = [
            item
            for item in contexts
            if item.get("score") is not None and item["score"] <= settings.max_rag_distance
        ]
        return bool(good)

    def answer(self, question: str) -> tuple[str, str, list[dict[str, Any]]]:
        if self._is_abusive(question):
            return self._ABUSIVE_MESSAGE, "guardrail_abuse", []

        if not settings.allow_off_topic and not self._is_domain_question(question):
            return self._OFF_TOPIC_MESSAGE, "guardrail_domain", []

        structured_answer, structured_sources = self.try_structured_lookup(question)
        if structured_answer:
            return structured_answer, "structured", structured_sources

        rag_context = self.retrieve(question)
        
        # if not rag_context or not self._has_relevant_context(rag_context):
        #     return self._UNKNOWN_MESSAGE, "rag", rag_context

        answer = self.generate_grounded_answer(question, rag_context)
        return answer, "rag", rag_context


def ensure_storage_path() -> None:
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
