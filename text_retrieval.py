"""Low-memory, deterministic résumé retrieval for the shared 4 GB VPS.
The local FAISS option remains available; server deployment uses this index.
"""
import math
import re
from collections import Counter
from types import SimpleNamespace

class TextResumeIndex:
    def __init__(self, root):
        self.documents = []
        for name in ("resume.txt", "about_me.txt"):
            path = root / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for section in re.split(r"\n\s*\n", text):
                for start in range(0, len(section), 1800):
                    chunk = section[start:start + 1800].strip()
                    if chunk:
                        self.documents.append(SimpleNamespace(page_content=chunk, metadata={"source_name": name, "content_type": "profile"}))
        if not self.documents:
            raise ValueError("No résumé source documents found")
        self.tokens = [Counter(re.findall(r"[a-z0-9]+", d.page_content.lower())) for d in self.documents]

    def similarity_search(self, question, k=6):
        stop = {"what", "are", "your", "about", "tell", "the", "and", "you", "me", "do", "is", "of", "a", "to"}
        query = set(re.findall(r"[a-z0-9]+", question.lower())) - stop
        scores = []
        for i, counts in enumerate(self.tokens):
            score = sum((1 + math.log(counts[t])) * math.log(1 + len(self.tokens) / (1 + sum(t in c for c in self.tokens))) for t in query if counts[t])
            scores.append((score, -i, self.documents[i]))
        return [d for _, _, d in sorted(scores, key=lambda x: (x[0], x[1]), reverse=True)[:k]]
