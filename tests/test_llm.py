import importlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import httpx
from dotenv import load_dotenv

import llm


def configure(**settings):
    with patch.dict(os.environ, settings, clear=True), patch("dotenv.load_dotenv", return_value=False):
        return importlib.reload(llm)


class LLMTests(unittest.TestCase):
    def tearDown(self):
        configure()

    def test_dotenv_loads_settings_and_shell_overrides_them(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "LLM_BASE_URL=http://localhost:11434\n"
                "LLM_MODEL=local-model\n"
                "MIN_WORD_COUNT=250\n"
                "OUTPUT_DIR=custom-output\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LLM_MODEL": "shell-model"}, clear=True), patch(
                "dotenv.load_dotenv", side_effect=lambda _: load_dotenv(env_file),
            ):
                module = importlib.reload(llm)
                import main
                importlib.reload(main)
                self.assertEqual(module.MODEL, "shell-model")
                self.assertEqual(main.MIN_WORD_COUNT, 250)
                self.assertEqual(main.OUTPUT_DIR, "custom-output")

    def test_local_uses_ollama_and_its_context_options(self):
        module = configure()
        with patch.object(module.client, "chat", return_value={"message": {"content": "  notes  "}}) as chat:
            self.assertEqual(module._generate("chapter"), "notes")
        self.assertEqual(chat.call_args.kwargs["model"], "qwen3:30b-a3b-instruct-2507-q4_K_M")
        self.assertEqual(chat.call_args.kwargs["options"]["num_ctx"], 32768)
        self.assertEqual(module.CONCURRENCY, 1)

    def test_remote_uses_account_default_without_local_limits(self):
        module = configure(LLM_BASE_URL="https://openrouter.ai/api/v1", LLM_API_KEY="test-key")
        self.assertEqual(module.CONCURRENCY, 6)
        self.assertEqual(str(module.client.base_url), "https://openrouter.ai/api/v1/")
        with patch.object(module.client, "post", return_value=Mock(
            json=lambda: {"choices": [{"message": {"content": " notes "}, "finish_reason": "stop"}]},
        )) as post:
            self.assertEqual(module._generate("chapter"), "notes")
        self.assertEqual(post.call_args.args, ("chat/completions",))
        self.assertEqual(post.call_args.kwargs["json"], {
            "messages": [{"role": "user", "content": "chapter"}],
            "temperature": 0.7,
            "top_p": 0.8,
        })

    def test_remote_selected_model_and_base_url_without_api_path(self):
        module = configure(
            LLM_BASE_URL="https://openrouter.ai/",
            LLM_API_KEY="test-key",
            LLM_MODEL="provider/model",
            REMOTE_CONCURRENCY="3",
        )
        self.assertEqual(str(module.client.base_url), "https://openrouter.ai/api/v1/")
        self.assertEqual(module.CONCURRENCY, 3)
        with patch.object(module.client, "post", return_value=Mock(
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )) as post:
            module._generate("chapter")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "provider/model")

    def test_remote_request_sends_key_to_openrouter_endpoint(self):
        module = configure(LLM_BASE_URL="https://openrouter.ai/api/v1", LLM_API_KEY="test-key")
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            })

        original_client = module.client
        module.client = httpx.Client(
            base_url=original_client.base_url,
            headers=original_client.headers,
            transport=httpx.MockTransport(respond),
        )
        try:
            self.assertEqual(module._generate("chapter"), "ok")
        finally:
            module.client.close()
            module.client = original_client
        self.assertEqual(str(requests[0].url), "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test-key")

    def test_remote_errors_and_truncation_propagate(self):
        module = configure(LLM_BASE_URL="https://openrouter.ai/api/v1", LLM_API_KEY="test-key")
        for result in (
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        ):
            with self.subTest(result=result), patch.object(
                module.client, "post", return_value=Mock(json=lambda: result),
            ), self.assertRaises(ValueError):
                module._generate("chapter")
        response = httpx.Response(429, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
        with patch.object(module.client, "post", return_value=response), self.assertRaises(httpx.HTTPStatusError):
            module._generate("chapter")

    def test_remote_classification_runs_concurrently_and_preserves_order(self):
        module = configure(LLM_BASE_URL="https://openrouter.ai/api/v1", LLM_API_KEY="test-key", REMOTE_CONCURRENCY="2")
        barrier = threading.Barrier(2, timeout=3)

        def generate(prompt):
            barrier.wait()
            return "RELEVANT" if "first" in prompt else "IRRELEVANT"

        with patch.object(module, "_generate", side_effect=generate):
            self.assertEqual(module.classify_chapters_batch(["first", "second"]), ["first"])

    def test_notes_and_translations_keep_input_order(self):
        module = configure(LLM_BASE_URL="https://openrouter.ai/api/v1", LLM_API_KEY="test-key")
        with patch.object(module, "_generate", side_effect=lambda prompt: (
            "FIRST" if "first" in prompt else "SECOND"
        )):
            notes = module.extract_notes_batch(["first", "second"])
            translations = module.translate_text_batch(["first", "second"], "es")
        self.assertTrue(notes[0].startswith("FIRST"))
        self.assertTrue(notes[1].startswith("SECOND"))
        self.assertEqual(translations, ["FIRST", "SECOND"])

    def test_local_notes_do_not_prequeue_other_chapters(self):
        module = configure()
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return "notes"

        with patch.object(module, "_generate", side_effect=generate):
            module.extract_notes_batch(["first", "second"])
        self.assertEqual(len(calls), 12)
        self.assertTrue(all("first" in prompt for prompt in calls[:6]))
        self.assertTrue(all("second" in prompt for prompt in calls[6:]))

    def test_invalid_remote_configuration_fails_early(self):
        for settings in (
            {"LLM_BASE_URL": "https://openrouter.ai/api/v1"},
            {"LLM_BASE_URL": "http://openrouter.ai/api/v1", "LLM_API_KEY": "key"},
            {"LLM_BASE_URL": "https://openrouter.ai/invalid", "LLM_API_KEY": "key"},
            {"LLM_BASE_URL": "https://openrouter.ai/api/v1", "LLM_API_KEY": "key", "REMOTE_CONCURRENCY": "0"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                configure(**settings)


if __name__ == "__main__":
    unittest.main()
