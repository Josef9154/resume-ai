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

if __name__ == "__main__":
    unittest.main()
