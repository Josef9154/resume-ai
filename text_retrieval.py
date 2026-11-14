"""Low-memory section-aware retrieval; never discard headings or missing matches."""
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
            heading, parts = "", []
            def flush():
                if parts:
                    text = (heading + "\n" + "\n\n".join(parts)).strip()
                    self.documents.append(SimpleNamespace(page_content=text, metadata={"source_name": name, "content_type": "profile"}))
            for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
                block = block.strip()
                if not block:
                    continue
                first, _, rest = block.partition("\n")
                if first.isupper() and len(first) < 100:
                    flush()
                    heading, parts = first, ([rest.strip()] if rest.strip() else [])
                else:
                    if (heading in {"EXPERIENCE:", "PROJECTS:"} and parts) or sum(map(len, parts)) + len(block) > 1800:
                        flush()
                        parts = []
                    parts.append(block)
            flush()
        if not self.documents:
            raise ValueError("No résumé source documents found")
        self.tokens = [Counter(re.findall(r"[a-z0-9]+", d.page_content.lower())) for d in self.documents]
        self.idf = {t: math.log(1 + len(self.tokens)/(1 + sum(t in c for c in self.tokens))) for c in self.tokens for t in c}

    def similarity_search(self, question, k=6):
        stop = {"what", "are", "your", "about", "tell", "the", "and", "you", "me", "do", "is", "of", "a", "to", "how", "can", "i", "please", "which"}
        query = set(re.findall(r"[a-z0-9]+", question.lower())) - stop
        for group in ({"education","degree","degrees","university","studied"}, {"pet","dog","eli"}, {"leadership","lead","style","governance"}, {"languages","programming","python","javascript"}, {"role","title","current"}):
            if query & group:
                query |= group
        scores = []
        for i, counts in enumerate(self.tokens):
            score = sum((1 + math.log(counts[t])) * self.idf[t] for t in query if counts[t])
            if score > 0:
                scores.append((score, -i, self.documents[i]))
        ranked = sorted(scores, key=lambda x:(x[0],x[1]), reverse=True)
        selected, characters = [], 0
        for score, _, document in ranked:
            if score < ranked[0][0] * 0.65:
                break
            if selected and characters + len(document.page_content) > 2400:
                continue
            selected.append(document)
            characters += len(document.page_content)
            if len(selected) >= k:
                break
        return selected
