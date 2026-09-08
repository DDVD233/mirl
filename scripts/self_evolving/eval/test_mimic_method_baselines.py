"""Offline contract tests for method orchestration and evidence handling."""

import argparse
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aiohttp
import mimic_method_baselines as baseline


class FakeChat:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.messages = []

    async def call(self, messages, trace, label, **kwargs):
        self.messages.append((label, copy.deepcopy(messages)))
        return next(self.answers)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    async def search(self, query, trace):
        self.queries.append(query)
        return [{"entry_id": query, "source": "textbook", "text": "Reference evidence."}]


class MethodTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            retries=1, queries=2, top_k=2, rounds=2, followups=1, aux_tokens=100, samples=3, seed=42, verifications=2
        )
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Clinical case"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                ],
            }
        ]

    async def test_cove_verification_cannot_see_draft(self):
        chat = FakeChat(["SECRET_DRAFT", '["Check one?", "Check two?"]', "Fact one", "Fact two", "Final"])
        answer, _ = await baseline.generate(self.messages, "cove", chat, None, self.args, str)
        self.assertEqual(answer, "Final")
        verification = [messages for label, messages in chat.messages if label == "verification"]
        self.assertEqual(len(verification), 2)
        for messages in verification:
            self.assertNotIn("SECRET_DRAFT", str(messages))
            self.assertEqual(messages[0], self.messages[0])
        self.assertIn("SECRET_DRAFT", str(chat.messages[-1]))

    async def test_imedrag_answers_followups_and_conditions_next_round(self):
        chat = FakeChat(['["First?"]', "Evidence one", '["Second?"]', "Evidence two", "Final"])
        retriever = FakeRetriever()
        answer, _ = await baseline.generate(self.messages, "imedrag", chat, retriever, self.args, str)
        self.assertEqual(answer, "Final")
        self.assertEqual(retriever.queries, ["First?", "Second?"])
        self.assertIn("Evidence one", str(chat.messages[2]))
        self.assertIn("Evidence two", str(chat.messages[-1]))
        self.assertEqual(chat.messages[-1][1][0], self.messages[0])

    async def test_direct_preserves_original_messages(self):
        chat = FakeChat(["Answer"])
        await baseline.generate(self.messages, "direct", chat, None, self.args, str)
        self.assertEqual(chat.messages[0][1], self.messages)

    def test_fusion_rewards_cross_query_agreement(self):
        doc = lambda key: {"entry_id": key, "source": "textbook", "text": key}
        result = baseline.fuse([[doc("a"), doc("b")], [doc("b"), doc("c")]], 2)
        self.assertEqual(result[0]["entry_id"], "b")
        repeated = baseline.fuse([[doc("a"), doc("a")]], 1)
        self.assertAlmostEqual(repeated[0]["rrf_score"], 1 / 61)

    def test_vote_ignores_empty_and_breaks_ties_by_sample_order(self):
        self.assertEqual(baseline.vote(["", "Disease A", "disease a."], str)[0], 1)
        self.assertEqual(baseline.vote(["Disease B", "Disease A"], str)[0], 0)

    def test_malformed_query_lists_fail(self):
        for text in ('{"questions": ["a"]}', '["a", "a"]', '["a", 2]', "not json"):
            with self.assertRaises(ValueError):
                baseline.parse_list(text, 2)

    def test_query_json_can_have_trailing_explanation(self):
        self.assertEqual(baseline.parse_list('["one?", "two?"]\nThese queries cover the case.', 2), ["one?", "two?"])

    def test_passage_filter_identifies_heading_only_retrieval(self):
        hits = [
            {
                "id": str(i),
                "distance": 0.9,
                "entity": {"entry_id": str(i), "source_dataset": "medrag_wiki", "text_content": "Symptoms"},
            }
            for i in range(32)
        ]
        passages, audit = baseline.select_passages(hits, 8, 1200)
        self.assertEqual(passages, [])
        self.assertEqual(audit["short_rejected"], 32)
        hits.append(
            {
                "id": "clinical",
                "distance": 0.8,
                "entity": {
                    "source_dataset": "medrag_textbook",
                    "text_content": "A useful clinical reference passage. " * 5,
                },
            }
        )
        passages, _ = baseline.select_passages(hits, 8, 1200)
        self.assertEqual(len(passages), 1)

    async def test_failed_planner_gets_bounded_retry(self):
        class RepairChat(FakeChat):
            async def call(self, messages, trace, label, **kwargs):
                self.options.append(kwargs)
                return await super().call(messages, trace, label, **kwargs)

        self.args.retries = 2
        self.args.rounds = 1
        chat = RepairChat(['["unterminated', '["Useful question?"]', "Medical fact", "Final"])
        chat.options = []
        answer, _ = await baseline.generate(self.messages, "imedrag", chat, FakeRetriever(), self.args, str)
        self.assertEqual(answer, "Final")
        self.assertEqual(chat.options[0]["tokens"], 1024)
        self.assertEqual(chat.options[1]["tokens"], 2048)
        self.assertEqual(chat.options[1]["json_max_chars"], 240)

    async def test_retrieval_deepens_after_heading_only_results(self):
        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            async def json(self):
                return {"data": [{"embedding": [1.0, 0.0]}]}

        class Session:
            def post(self, *args, **kwargs):
                return Response()

        class Client:
            def __init__(self):
                self.depths = []
                self.empty = False

            def search(self, **kwargs):
                self.depths.append(kwargs["limit"])
                text = "Symptoms" if self.empty or kwargs["limit"] == 32 else "Useful medical reference. " * 8
                return [
                    [{"id": "doc", "distance": 0.5, "entity": {"source_dataset": "medrag_wiki", "text_content": text}}]
                ]

        retriever = baseline.Retriever.__new__(baseline.Retriever)
        retriever.session = Session()
        retriever.client = Client()
        retriever.filter = "fixed source filter"
        retriever.args = argparse.Namespace(
            retries=1,
            embed_base="http://embedding",
            embed_model="embedding",
            collection="medical",
            top_k=8,
            passage_chars=1200,
        )
        trace = []
        result = await retriever.search("Medical question", trace)
        self.assertEqual(len(result), 1)
        self.assertEqual(retriever.client.depths, [32, 128])
        self.assertEqual(trace[0]["searches"][0]["short_rejected"], 1)
        retriever.client.empty = True
        retriever.client.depths = []
        trace = []
        self.assertEqual(await retriever.search("Medical question", trace), [])
        self.assertEqual(retriever.client.depths, [32, 128, 512])
        self.assertEqual(trace[0]["status"], "no_usable_passages")

        def outage(**kwargs):
            raise ConnectionError("synthetic Milvus outage")

        retriever.client.search = outage
        with self.assertRaises(ConnectionError):
            await retriever.search("Medical question", [])

    async def test_judge_outage_is_not_an_incorrect_answer(self):
        class FailedSession:
            def post(self, *args, **kwargs):
                raise aiohttp.ClientConnectionError("synthetic outage")

        args = argparse.Namespace(
            judge_model="fixed-judge", judge_base="http://invalid", judge_key_env="SYNTHETIC_KEY", retries=1
        )
        reward = argparse.Namespace(JUDGE_ACCURACY_LENIENT_PROMPT="Fixed rubric")
        with patch.dict(os.environ, {"SYNTHETIC_KEY": "test"}):
            with self.assertRaises(aiohttp.ClientConnectionError):
                await baseline.grade(
                    {"extracted_answer": "Diagnosis", "ground_truth": "Label"}, {}, FailedSession(), args, reward
                )

    def test_only_audited_image_failure_is_omitted_and_labels_are_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jpg"
            path.write_bytes(b"synthetic corrupt image")
            helper = argparse.Namespace(_read_image_b64=lambda *args, **kwargs: None)
            original = helper._read_image_b64

            def builder(entry, *args):
                self.assertNotIn("reward_model", entry)
                self.assertNotIn("extra_info", entry)
                helper._read_image_b64(str(path), max_pixels=65536)
                return {"messages": []}

            helper._build_openai_request = builder
            args = argparse.Namespace(model="test", max_pixels=65536, max_text_chars=9000)
            entry = {"prompt": [], "images": [], "reward_model": {"ground_truth": "SECRET"}, "extra_info": {}}
            with self.assertRaisesRegex(ValueError, "Unaudited"):
                baseline.build_request(entry, helper, args)
            self.assertIs(helper._read_image_b64, original)
            with patch.object(baseline, "KNOWN_CORRUPT_IMAGES", {baseline.file_digest(path)}):
                _, exceptions = baseline.build_request(entry, helper, args)
            self.assertEqual(exceptions[0]["sha256"], baseline.file_digest(path))
            self.assertIs(helper._read_image_b64, original)

    async def test_planner_requests_constrained_json(self):
        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            async def json(self):
                return {"choices": [{"message": {"content": '["Question?"]'}, "finish_reason": "stop"}]}

        class Session:
            def post(self, url, *, json, headers):
                self.payload = json
                return Response()

        session = Session()
        args = argparse.Namespace(
            model="test",
            max_tokens=1024,
            provider="vllm",
            retries=1,
            base_url="http://test",
            api_key_env="SYNTHETIC_KEY",
        )
        await baseline.Chat(session, args).call([], [], "plan", json_count=1)
        self.assertEqual(session.payload["structured_outputs"]["json"]["maxItems"], 1)


if __name__ == "__main__":
    unittest.main()
