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

class ProfileCoverageTests(unittest.TestCase):
    def test_specific_degree_avoids_unrelated_career_sections(self):
        docs = TextResumeIndex(ROOT).similarity_search("Which university awarded your Executive MBA?", 3)
        self.assertIn("Bethel University", docs[0].page_content)
        self.assertLess(sum(len(d.page_content) for d in docs), 1500)

    def test_eli_is_retrieved(self):
        docs = TextResumeIndex(ROOT).similarity_search("Tell me about Eli", 3)
        self.assertTrue(any("Eli is my dog" in d.page_content for d in docs))

    def test_education_keeps_details(self):
        docs = TextResumeIndex(ROOT).similarity_search("What degrees have you earned?", 3)
        self.assertTrue(any("EDUCATION" in d.page_content and "Executive MBA" in d.page_content and "MS – Data Science" in d.page_content for d in docs))

    def test_unknown_topic_has_no_arbitrary_matches(self):
        self.assertEqual(TextResumeIndex(ROOT).similarity_search("zebrafish", 3), [])

    def test_all_common_answers_have_valid_evidence(self):
        import json
        from answer_cache import ProfileAnswers
        answers = ProfileAnswers(ROOT)
        for record in json.loads((ROOT / "profile_answers.json").read_text(encoding="utf-8")):
            self.assertEqual(answers.direct(record["questions"][0]), record["answer"])

    def test_dog_age_not_invented(self):
        from answer_cache import ProfileAnswers
        self.assertIsNone(ProfileAnswers(ROOT).direct("How old is Eli?"))

    def test_source_changes_invalidate_cache_and_direct_answer(self):
        import tempfile, json
        from answer_cache import ProfileAnswers
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "resume.txt").write_text("old fact")
            (root / "profile_answers.json").write_text(json.dumps([{"questions":["fact"],"answer":"old","evidence":[{"file":"resume.txt","quote":"old fact"}]}]))
            answers = ProfileAnswers(root, capacity=2)
            key = answers.key("fact", "model")
            answers.put(key, "old")
            self.assertEqual(answers.get(key), "old")
            (root / "resume.txt").write_text("new fact")
            self.assertIsNone(answers.get(answers.key("fact", "model")))
            self.assertIsNone(answers.direct("fact"))

    def test_cache_is_bounded_and_language_names_distinct(self):
        from answer_cache import ProfileAnswers
        answers = ProfileAnswers(ROOT, capacity=2)
        for i in range(3): answers.put(str(i), "answer")
        self.assertIsNone(answers.get("0"))
        self.assertNotEqual(answers.normalize("C++"), answers.normalize("C#"))

if __name__ == "__main__":
    unittest.main()
