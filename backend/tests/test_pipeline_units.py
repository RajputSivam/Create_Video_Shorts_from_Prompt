import asyncio
import io
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image
from pydantic import ValidationError

from backend import database, providers
from backend.config import MODE_CONFIGS
from backend.image_generation import ImageProviderChain, _failure_message, clean_image_prompt, generate_image
from backend.image_providers import configured_providers
from backend.main import JobRequest
from backend.providers import _gemini_api_key, _groq_api_key, generate_text
from backend.renderer import _mux_narration, probe_video
from backend.scene_splitter import split_scenes
from backend.script_generation import generate_script
from backend.subtitles import make_caption_frames
from backend.voiceover import generate_voice


class ModeConfigTests(unittest.TestCase):
    def test_export_shapes_and_duration_ranges_are_mode_aware(self):
        self.assertEqual((MODE_CONFIGS["video"].width, MODE_CONFIGS["video"].height), (1280, 720))
        self.assertEqual((MODE_CONFIGS["shorts"].width, MODE_CONFIGS["shorts"].height), (1080, 1920))
        self.assertEqual((MODE_CONFIGS["video"].min_seconds, MODE_CONFIGS["video"].max_seconds), (300, 900))
        self.assertEqual((MODE_CONFIGS["shorts"].min_seconds, MODE_CONFIGS["shorts"].max_seconds), (15, 60))

    def test_request_rejects_duration_outside_selected_mode(self):
        with self.assertRaises(ValidationError):
            JobRequest(mode="shorts", prompt="A story about the moon", duration=61, language="English", style="cinematic", voice="female")
        request = JobRequest(mode="shorts", prompt="A story about the moon", duration=45, language="English", style="cinematic", voice="female")
        self.assertEqual(request.duration, 45)

    def test_three_minute_video_is_valid_only_in_fast_preview(self):
        request = JobRequest(mode="video", fast_preview=True, prompt="A short preview story", duration=180, language="English", style="cinematic", voice="female")
        self.assertTrue(request.fast_preview)
        with self.assertRaises(ValidationError):
            JobRequest(mode="video", fast_preview=False, prompt="A short preview story", duration=180, language="English", style="cinematic", voice="female")


class QueueRecoveryTests(unittest.TestCase):
    def test_running_job_is_requeued_after_restart(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(database, "DB_PATH", Path(folder) / "jobs.sqlite3"):
            database.initialize()
            database.create_job({"id": "stale-job", "mode": "video", "prompt": "Recover this queued job", "duration": 180, "language": "English", "style": "cinematic", "voice": "female", "fast_preview": True})
            self.assertEqual(database.claim_next_queued_job(), "stale-job")
            self.assertEqual(database.get_job("stale-job")["status"], "running")
            self.assertEqual(database.recover_interrupted_jobs(), ["stale-job"])
            recovered = database.get_job("stale-job")
            self.assertEqual(recovered["status"], "queued")
            self.assertTrue(recovered["fast_preview"])
            database.update_job("stale-job", output_file=str(Path(folder) / "preview.mp4"))
            self.assertEqual(database.get_job("stale-job")["video_url"], "/media/stale-job/preview.mp4")

    def test_expired_running_job_is_failed_without_blocking_queue(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(database, "DB_PATH", Path(folder) / "jobs.sqlite3"):
            database.initialize()
            database.create_job({"id": "expired-job", "mode": "video", "prompt": "A stale worker test", "duration": 300, "language": "English", "style": "cinematic", "voice": "female"})
            self.assertEqual(database.claim_next_queued_job(), "expired-job")
            expired = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
            database.update_job("expired-job", heartbeat_at=expired)
            self.assertEqual(database.recover_interrupted_jobs(), [])
            recovered = database.get_job("expired-job")
            self.assertEqual(recovered["status"], "failed")
            self.assertEqual(recovered["stage"], "Render timed out")

    def test_fast_preview_is_claimed_before_full_quality_queue(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(database, "DB_PATH", Path(folder) / "jobs.sqlite3"):
            database.initialize()
            common = {"mode": "video", "prompt": "Queue priority test", "duration": 300, "language": "English", "style": "cinematic", "voice": "female"}
            database.create_job({**common, "id": "full-job", "fast_preview": False})
            database.create_job({**common, "id": "preview-job", "fast_preview": True})
            self.assertEqual(database.claim_next_queued_job(), "preview-job")


class ProviderConfigTests(unittest.TestCase):
    def test_google_alias_is_trimmed_and_unquoted(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": '  "example-key"  '}, clear=True):
            self.assertEqual(_gemini_api_key(), "example-key")

    def test_empty_keys_are_missing_and_error_names_providers(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "  ''  ", "GROQ_API_KEY": "   "}, clear=True):
            self.assertIsNone(_gemini_api_key())
            self.assertIsNone(_groq_api_key())
            with self.assertRaisesRegex(RuntimeError, "Gemini needs GEMINI_API_KEY or GOOGLE_API_KEY.*Groq needs GROQ_API_KEY"):
                asyncio.run(generate_text("test prompt"))

    def test_image_providers_are_enabled_only_with_required_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_providers(), [])
        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": "pollinations", "CLOUDFLARE_API_TOKEN": "token", "GEMINI_API_KEY": "gemini", "PEXELS_API_KEY": "pexels"}, clear=True):
            self.assertEqual([provider.name for provider in configured_providers()], ["Pollinations", "Gemini", "Pexels"])
        with patch.dict(os.environ, {"CLOUDFLARE_API_TOKEN": "token", "CLOUDFLARE_ACCOUNT_ID": "account"}, clear=True):
            self.assertEqual([provider.name for provider in configured_providers()], ["Cloudflare Workers AI"])


class ProviderFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_groq_is_tried_after_gemini_returns_http_404(self):
        request = httpx.Request("POST", "https://example.test/generate")
        response = httpx.Response(404, request=request, json={"error": {"message": "model not found"}})
        failure = httpx.HTTPStatusError("not found", request=request, response=response)
        gemini = AsyncMock(side_effect=failure)
        groq = AsyncMock(return_value="Groq fallback worked")

        with (
            patch.object(providers, "_gemini", gemini),
            patch.object(providers, "_groq", groq),
            patch.object(providers, "_gemini_api_key", return_value="test-gemini-key"),
            patch.object(providers, "_groq_api_key", return_value="test-groq-key"),
        ):
            self.assertEqual(await providers.generate_text("test prompt"), "Groq fallback worked")

        gemini.assert_awaited_once_with("test prompt")
        groq.assert_awaited_once_with("test prompt")


class ImageGenerationTests(unittest.TestCase):
    def _jpeg(self, width=320, height=180):
        image = Image.effect_noise((width, height), 80).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG")
        return output.getvalue()

    def test_image_prompt_is_short_and_excludes_full_video_description(self):
        job = {"mode": "video", "style": "cinematic", "prompt": "Private full video description " * 100}
        scene = {"image_prompt": "A lone lighthouse on a rocky coast " * 100}
        prompt = clean_image_prompt(job, scene)
        self.assertLessEqual(len(prompt), 300)
        self.assertNotIn("Private full video description", prompt)
        self.assertNotIn("  ", prompt)

    def test_provider_failure_message_redacts_keys_and_urls(self):
        class Provider:
            name = "Pollinations"
            key = "secret-token"

        request = httpx.Request("GET", "https://image.example.test/private/path")
        response = httpx.Response(402, request=request, json={"error": {"message": "Key secret-token rejected; see https://example.test/help"}})
        error = httpx.HTTPStatusError("payment required", request=request, response=response)
        message = _failure_message(Provider(), error)
        self.assertIn("HTTP 402", message)
        self.assertNotIn("secret-token", message)
        self.assertNotIn("https://", message)

    def test_quota_failure_message_is_concise(self):
        class Provider:
            name = "Gemini"

        request = httpx.Request("POST", "https://example.test/generate")
        response = httpx.Response(429, request=request, json={"error": {"message": "large quota body with retry URL https://example.test/details"}})
        error = httpx.HTTPStatusError("rate limited", request=request, response=response)
        message = _failure_message(Provider(), error)
        self.assertEqual(message, "Gemini: HTTP 429 (rate limit or provider quota exceeded)")

    def test_successful_image_is_cached_across_jobs(self):
        class StubProvider:
            name = "Stub image provider"
            attribution = None

            def __init__(self):
                self.calls = 0

            def generate_image(self, prompt, width, height):
                self.calls += 1
                return self._image

        provider = StubProvider()
        provider._image = self._jpeg()
        job = {"id": "job-one", "mode": "shorts", "style": "storybook", "prompt": "A long-form job description"}
        scene = {"image_prompt": "A tiny boat crossing a quiet sea"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "jobs"
            with (
                patch("backend.image_generation.database.DATA_DIR", root),
                patch("backend.image_generation.configured_providers", return_value=[provider]),
            ):
                first = generate_image(job, scene, 0, root / "job-one" / "scene.jpg")
                second = generate_image({**job, "id": "job-two"}, scene, 0, root / "job-two" / "scene.jpg")

        self.assertEqual(provider.calls, 1)
        self.assertEqual(first.provider, "Stub image provider")
        self.assertEqual(second.provider, "Stub image provider")
        self.assertFalse(second.placeholder)

    def test_all_provider_failures_return_a_text_free_gradient(self):
        class FailedProvider:
            name = "Unavailable provider"
            attribution = None

            def generate_image(self, prompt, width, height):
                raise RuntimeError("service unavailable")

        job = {"id": "job-placeholder", "mode": "shorts", "style": "cinematic", "prompt": "Story description"}
        scene = {"title": "A mountain dawn", "image_prompt": "An alpine lake at sunrise"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "jobs"
            with (
                patch("backend.image_generation.database.DATA_DIR", root),
                patch("backend.image_generation.configured_providers", return_value=[FailedProvider()]),
            ):
                result = generate_image(job, scene, 2, root / "job-placeholder" / "scene.jpg")
                with Image.open(result.path) as image:
                    self.assertEqual(image.size, (768, 1280))
                    self.assertEqual(image.getpixel((0, 500)), image.getpixel((767, 500)))

        self.assertTrue(result.placeholder)
        self.assertEqual(result.provider, "placeholder")
        self.assertIn("Unavailable provider", result.failures[0])

    def test_narration_mux_maps_video_and_audio_and_encodes_aac(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "joined.mp4"
            output = Path(folder) / "final.mp4"
            video.write_bytes(b"input")

            def run(command, **kwargs):
                Path(command[-1]).write_bytes(b"muxed")
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            with patch("backend.renderer._ffmpeg", return_value="ffmpeg"), patch("backend.renderer.subprocess.run", side_effect=run) as subprocess_run:
                _mux_narration(video, output)

            command = subprocess_run.call_args.args[0]
            self.assertEqual(command.count(str(video)), 2)
            self.assertEqual(command[command.index("-map") + 1], "0:v")
            self.assertEqual(command[command.index("-map") + 3], "1:a")
            self.assertIn("-c:a", command)
            self.assertEqual(command[command.index("-c:a") + 1], "aac")
            self.assertIn("-shortest", command)
            self.assertTrue(output.exists())

    def test_ffprobe_summary_is_parsed_and_logged(self):
        ffprobe_result = SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"codec_type":"video"},{"codec_type":"audio"}],"format":{"duration":"1.0"}}',
            stderr="",
        )
        with patch("backend.renderer._ffprobe", return_value="ffprobe"), patch("backend.renderer.subprocess.run", return_value=ffprobe_result) as run:
            result = probe_video(Path("test.mp4"))
        self.assertEqual(len(result["streams"]), 2)
        self.assertIn("-show_streams", run.call_args.args[0])

    def test_image_quota_error_opens_job_scoped_circuit_breaker(self):
        class LimitedProvider:
            name = "Limited image provider"
            attribution = None

            def __init__(self):
                self.calls = 0

            def generate_image(self, prompt, width, height):
                self.calls += 1
                request = httpx.Request("POST", "https://example.test/image")
                response = httpx.Response(429, request=request, json={"error": {"message": "quota"}})
                raise httpx.HTTPStatusError("limited", request=request, response=response)

        class WorkingProvider:
            name = "Working image provider"
            attribution = None

            def __init__(self):
                self.calls = 0

            def generate_image(self, prompt, width, height):
                self.calls += 1
                return b"image bytes"

        limited = LimitedProvider()
        working = WorkingProvider()
        chain = ImageProviderChain([limited, working])
        self.assertEqual(chain.generate("prompt", 640, 360)[1], "Working image provider")
        _, _, _, failures = chain.generate("prompt", 640, 360)
        self.assertEqual(limited.calls, 1)
        self.assertEqual(working.calls, 2)
        self.assertTrue(any("skipped for this job" in item for item in failures))


class VoiceCacheTests(unittest.TestCase):
    def test_tts_cache_is_shared_by_narration_and_voice(self):
        job = {"id": "voice-cache-test", "mode": "video", "language": "English", "voice": "female"}
        scene = {"narration_text": "The cached narrator speaks."}
        duration = 4.25
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "jobs"
            cache = data_dir.parent / "voice_cache"
            import hashlib
            digest = hashlib.sha256(f"{scene['narration_text']}|en-US-JennyNeural|+0%".encode("utf-8")).hexdigest()
            cache.mkdir(parents=True)
            cached_bytes = b"cached mp3 bytes" * 160
            (cache / f"{digest}.mp3").write_bytes(cached_bytes)
            destination = data_dir / job["id"] / "audio" / "scene-001.mp3"
            with (
                patch("backend.voiceover.database.DATA_DIR", data_dir),
                patch("backend.voiceover.MP3", return_value=SimpleNamespace(info=SimpleNamespace(length=duration))),
            ):
                self.assertEqual(generate_voice(job, scene, 0, destination), duration)
                self.assertEqual(destination.read_bytes(), cached_bytes)


class GenerationUnitTests(unittest.TestCase):
    def test_45_second_short_requests_about_140_words_and_caches_script(self):
        job = {"mode": "shorts", "duration": 45, "language": "English", "prompt": "A moonlit train", "style": "cinematic", "voice": "female"}
        with tempfile.TemporaryDirectory() as folder:
            cache_path = Path(folder) / "script.txt"
            provider = AsyncMock(return_value="Look closer. The night train is already here.")
            with patch("backend.script_generation.generate_text", provider):
                script = generate_script(job, cache_path)
                self.assertEqual(script, cache_path.read_text(encoding="utf-8"))
                self.assertIn("140 words", provider.await_args.args[0])
                self.assertIn("hook", provider.await_args.args[0])
                self.assertEqual(list(Path(folder).glob("*.part")), [])

    def test_scene_splitter_keeps_one_character_description_and_caches(self):
        job = {"mode": "shorts", "duration": 45, "language": "English", "prompt": "A moonlit train", "style": "cinematic", "voice": "female"}
        result = {
            "character_description": "A young conductor with a red scarf and silver spectacles",
            "scenes": [{"narration_text": "The train arrives.", "image_prompt": "A train crossing a bridge", "duration": 5}],
        }
        with tempfile.TemporaryDirectory() as folder:
            cache_path = Path(folder) / "scenes.json"
            provider = patch("backend.scene_splitter.generate_json", return_value=result)
            with provider as generate_json:
                scenes = split_scenes(job, "The train arrives.", cache_path)
                self.assertIn("about 9 sequential visual scenes", generate_json.call_args.args[0])
                self.assertEqual(scenes[0]["character_description"], result["character_description"])
                self.assertEqual(split_scenes(job, "ignored after cache", cache_path), scenes)
                generate_json.assert_called_once()

    def test_video_scene_split_targets_about_one_scene_per_18_seconds(self):
        job = {"mode": "video", "duration": 180, "language": "English", "prompt": "A short preview story", "style": "cinematic", "voice": "female"}
        result = {"character_description": "A hiker in a red coat", "scenes": [{"narration_text": "The trail opens.", "image_prompt": "A mountain trail", "duration": 18}]}
        with tempfile.TemporaryDirectory() as folder:
            cache_path = Path(folder) / "scenes.json"
            with patch("backend.scene_splitter.generate_json", return_value=result) as generate_json:
                split_scenes(job, "The trail opens.", cache_path)
        self.assertIn("about 10 sequential visual scenes", generate_json.call_args.args[0])
        self.assertIn("15-20 seconds", generate_json.call_args.args[0])

    def test_video_scene_split_merges_excess_scenes_without_losing_narration(self):
        job = {"mode": "video", "duration": 180, "language": "English", "prompt": "A short preview story", "style": "cinematic", "voice": "female"}
        source_scenes = [
            {"narration_text": f"Sentence {index}.", "image_prompt": f"Visual {index}", "duration": 3.6}
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = {"character_description": "A hiker", "scenes": source_scenes}
            with patch("backend.scene_splitter.generate_json", return_value=result):
                scenes = split_scenes(job, "all narration", Path(folder) / "scenes.json")
        self.assertEqual(len(scenes), 10)
        self.assertEqual(" ".join(scene["narration_text"] for scene in scenes), " ".join(f"Sentence {index}." for index in range(20)))

    def test_shorts_captions_return_word_timed_safe_overlays(self):
        frames = make_caption_frames("A tiny test story loops back", 360, 640, "word", 3.0)
        self.assertEqual(len(frames), 6)
        self.assertTrue(all(frame.width <= 360 and frame.height <= 640 for frame, _, _, _, _ in frames))
        self.assertAlmostEqual(sum(duration for _, _, duration, _, _ in frames), 3.0, places=2)


if __name__ == "__main__":
    unittest.main()