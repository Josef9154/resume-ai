"""Bounded exact-question cache and evidence-checked common answers."""
import hashlib
import json
import re
import threading
from collections import OrderedDict
from pathlib import Path

class ProfileAnswers:
    def __init__(self, root, capacity=128):
        self.root = Path(root)
        self.capacity = capacity
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def normalize(question):
        return re.sub(r"[^a-z0-9+#]+", " ", question.lower().replace("’", "'")).strip()

    def revision(self):
        digest = hashlib.sha256()
        for name in ("resume.txt", "about_me.txt", "profile_answers.json"):
            path = self.root / name
            digest.update(path.read_bytes() if path.exists() else b"")
        return digest.hexdigest()

    def direct(self, question):
        path = self.root / "profile_answers.json"
        if not path.exists():
            return None
        key = self.normalize(question)
        for item in json.loads(path.read_text(encoding="utf-8")):
            if key not in [self.normalize(q) for q in item["questions"]]:
                continue
            if item.get("evidence") and all(
                e["file"] in {"resume.txt", "about_me.txt"} and
                (self.root / e["file"]).exists() and
                e["quote"] in (self.root / e["file"]).read_text(encoding="utf-8")
                for e in item["evidence"]
            ):
                return item["answer"]
        return None

    def key(self, question, model):
        return (self.revision(), model, self.normalize(question))

    def get(self, key):
        with self.lock:
            value = self.entries.get(key)
            if value is not None:
                self.entries.move_to_end(key)
            return value

    def put(self, key, answer):
        if not answer.strip():
            return
        with self.lock:
            self.entries[key] = answer
            self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
