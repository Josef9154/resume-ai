import ast
import unittest
from pathlib import Path
from text_retrieval import TextResumeIndex

ROOT = Path(__file__).resolve().parent

class DeploymentTests(unittest.TestCase):
    def test_python_syntax(self):
        ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))

    def test_resume_facts_are_retrieved(self):
        docs = TextResumeIndex(ROOT).similarity_search("What is your current title and functional scope?", 6)
        self.assertTrue(any("Associate Director" in d.page_content for d in docs))

    def test_languages_retrieved(self):
        docs = TextResumeIndex(ROOT).similarity_search("What programming languages Python do you know?", 6)
        self.assertTrue(any("Python" in d.page_content for d in docs))

    def test_subpath_compatible_assets_and_chat(self):
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")
        self.assertIn('src="static/profile.png"', html)
        self.assertIn('fetch("chat",', html)
        self.assertNotIn('fetch("/chat",', html)

    def test_private_listener(self):
        service = (ROOT / "deploy/resume-ai.service").read_text()
        self.assertIn("--host 127.0.0.1 --port 8200", service)
        self.assertIn("MemoryMax=384M", service)

class StreamingTests(unittest.TestCase):
    def test_stream_emits_tokens_and_releases_slot(self):
        import json
        from unittest.mock import MagicMock, patch
        import main
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [b'{"message":{"content":"Joseph: "}}', b'{"message":{"content":"Hello"}, "done":true}']
        self.assertTrue(main.CHAT_SLOT.acquire(blocking=False))
        with patch.object(main.requests, "post", return_value=response):
            events = [json.loads(line) for line in main.stream_ollama("test")]
        self.assertEqual("".join(e.get("delta", "") for e in events), "Joseph: Hello")
        self.assertTrue(events[-1]["done"])
        self.assertTrue(main.CHAT_SLOT.acquire(blocking=False))
        main.CHAT_SLOT.release()

    def test_stream_error_releases_slot(self):
        import json
        from unittest.mock import patch
        import main
        self.assertTrue(main.CHAT_SLOT.acquire(blocking=False))
        with patch.object(main.requests, "post", side_effect=main.requests.Timeout):
            events = [json.loads(line) for line in main.stream_ollama("test")]
        self.assertIn("error", events[-1])
        self.assertTrue(main.CHAT_SLOT.acquire(blocking=False))
        main.CHAT_SLOT.release()

    def test_closed_stream_releases_slot(self):
        from unittest.mock import MagicMock, patch
        import main
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [b'{"message":{"content":"Hello"}}', b'{"done":true}']
        self.assertTrue(main.CHAT_SLOT.acquire(blocking=False))
        with patch.object(main.requests, "post", return_value=response):
            stream = main.stream_ollama("test")
            next(stream)
            stream.close()
        self.assertTrue(main.CHAT_SLOT.acquire(blocking=False))
        main.CHAT_SLOT.release()

if __name__ == "__main__":
    unittest.main()
