import tempfile
import json
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from workers.captions import _word_times, _wrap_caption
from workers.doctor import preflight_job
from workers.common import load_job, save_json
from workers.images import _bounded_prompt, find_near_duplicate_images
from workers.music import create_music
from workers.pipeline import process_job
from workers.script_llm import (
    generate_script,
    _ollama_generate,
    expected_scene_count,
    validate_script,
    verify_ollama_unloaded,
)
from workers.sfx import generate_sfx
from workers.tts import _voice_model
from workers.upscale import upscale_images


class LocalWorkerTests(unittest.TestCase):
    def _hindi_script(self, job):
        narration = [
            "कॉफी की शुरुआत इथियोपिया के पहाड़ों में हुई जहाँ किसान लाल जामुन सावधानी से चुनते थे।",
            "व्यापारी इन बीजों को दूर देशों तक ले गए और नए स्वाद से लोगों को परिचित कराया।",
            "यमन के सूफी साधक रात की प्रार्थनाओं में जागने के लिए कॉफी पीते थे।",
            "इस्तांबुल के उस्मानी कॉफीघरों में लोग मिलते बातचीत करते और खबरें साझा करते थे।",
            "बाबा बुंदन सात बीज लेकर भारत लौटे और उन्हें चिकमगलूर की पहाड़ियों में बोया।",
            "धीरे धीरे खेती फैली और कॉफी दुनिया भर की सुबहों तथा मिलन का हिस्सा बनी।",
        ]
        prompts = [
            "Ethiopian farmer picking ripe red coffee cherries on a green mountain hillside at sunrise, woven basket, distant valley, historically inspired illustration, broad cinematic framing, warm golden light",
            "Arab traders transporting sacks of coffee beans along an old desert caravan route, camels crossing ochre dunes, distant stone market, historically inspired illustration, panoramic composition, warm sunlight",
            "Yemeni Sufi monks drinking coffee during a quiet night prayer, lantern lit courtyard, simple cups, starry sky, historically inspired illustration, intimate cinematic framing, deep blue shadows",
            "Ottoman men talking over coffee in a crowded Istanbul coffeehouse, ornate arches, brass trays, patterned carpets, historically inspired illustration, warm lantern glow, wide interior composition",
            "Baba Budan carrying seven coffee seeds through misty Chikmagalur hills, lush slopes, modest travel clothes, historically inspired illustration, sweeping landscape framing, early morning light, on foot",
            "Coffee farmers tending flowering shrubs on Indian hills, fresh beans drying in courtyard, village homes in distance, historically inspired illustration, balanced cinematic framing, clear warm daylight",
        ]
        return {"scenes": [
            {"narration": text, "image_prompt": prompt}
            for text, prompt in zip(narration, prompts)
        ]}

    def test_job_state_save_retries_permission_error_with_increasing_delays(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "job.json"
            original_replace = Path.replace
            attempts = 0

            def replace(path, destination):
                nonlocal attempts
                if path.name.startswith("job.json.") and path.name.endswith(".part"):
                    attempts += 1
                    if attempts < 3:
                        raise PermissionError("temporarily locked")
                return original_replace(path, destination)

            with patch("workers.common.time.sleep") as sleep, patch.object(Path, "replace", replace):
                save_json(target, {"status": "running"})

            self.assertEqual(load_job(target), {"status": "running"})
            self.assertEqual(attempts, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.2, 0.4])

    def test_job_state_save_falls_back_to_direct_write_after_ten_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "job.json"
            with patch("workers.common.time.sleep") as sleep, patch.object(Path, "replace", side_effect=PermissionError("locked")):
                save_json(target, {"progress": 4})

            self.assertEqual(load_job(target), {"progress": 4})
            self.assertEqual(sleep.call_count, 9)

    def test_job_state_save_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "job.json"
            original_write_text = Path.write_text

            def write_text(path, *args, **kwargs):
                if path == target:
                    raise PermissionError("job state still locked")
                return original_write_text(path, *args, **kwargs)

            with patch("workers.common.time.sleep"), patch.object(Path, "replace", side_effect=PermissionError("locked")), patch.object(Path, "write_text", write_text):
                save_json(target, {"progress": 4})

    def test_job_state_reader_retries_permission_error(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "job.json"
            target.write_text('{"status": "running"}', encoding="utf-8")
            original_read_text = Path.read_text
            attempts = 0

            def read_text(path, *args, **kwargs):
                nonlocal attempts
                if path == target:
                    attempts += 1
                    if attempts < 3:
                        raise PermissionError("temporarily locked")
                return original_read_text(path, *args, **kwargs)

            with patch("workers.common.time.sleep") as sleep, patch.object(Path, "read_text", read_text):
                self.assertEqual(load_job(target), {"status": "running"})

            self.assertEqual(attempts, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.2, 0.4])

    def test_requested_six_scene_hindi_script_passes_strict_schema(self):
        job = {"id": "unit-test", "mode": "shorts", "duration": 30, "scene_count": 6, "language": "Hindi", "prompt": "Coffee history", "style": "storybook"}
        count = expected_scene_count(job)
        script = validate_script(self._hindi_script(job), job, count)
        self.assertEqual(count, 6)
        self.assertEqual(len(script["scenes"]), count)
        self.assertTrue(all({"narration", "image_prompt", "mood", "motion", "transition", "sfx"} <= scene.keys() for scene in script["scenes"]))
        self.assertIn("title", script)
        self.assertGreaterEqual(script["word_count"], 59)
        self.assertLessEqual(script["word_count"], 97)
        self.assertTrue(all(25 <= len(scene["image_prompt"].split()) <= 40 for scene in script["scenes"]))

    def test_english_narration_is_rejected_for_hindi_job(self):
        job = {"mode": "shorts", "duration": 30, "scene_count": 6, "language": "Hindi", "prompt": "Coffee history"}
        script = self._hindi_script(job)
        script["scenes"][0]["narration"] = "Coffee spread around the world through traders and old coffee houses."
        with self.assertRaisesRegex(ValueError, "not Hindi"):
            validate_script(script, job, 6)

    def test_placeholder_script_is_rejected(self):
        job = {"mode": "shorts", "duration": 30, "scene_count": 6, "language": "English", "prompt": "Coffee history"}
        script = self._hindi_script(job)
        script["scenes"][0]["narration"] = "Here is the surprising story behind coffee and its journey through history."
        with self.assertRaisesRegex(ValueError, "placeholder"):
            validate_script(script, job, 6)

    def test_devanagari_in_image_prompt_is_rejected(self):
        job = {"mode": "shorts", "duration": 30, "scene_count": 6, "language": "Hindi", "prompt": "Coffee history"}
        script = self._hindi_script(job)
        script["scenes"][0]["image_prompt"] += " कॉफी"
        with self.assertRaisesRegex(ValueError, "Devanagari"):
            validate_script(script, job, 6)

    def test_ollama_failure_retries_then_fails_without_placeholder_script(self):
        job = {"id": "offline-test", "mode": "shorts", "duration": 30, "scene_count": 6, "language": "Hindi", "prompt": "Coffee history"}
        with tempfile.TemporaryDirectory() as folder, \
                patch("workers.script_llm.job_directory", return_value=Path(folder)), \
                patch("workers.script_llm._ollama_generate", side_effect=RuntimeError("Ollama down")) as generate:
            with self.assertRaisesRegex(RuntimeError, "Script generation failed: Ollama down"):
                generate_script(job)
            self.assertEqual(generate.call_count, 3)
            self.assertFalse((Path(folder) / "script.json").exists())

    def test_pipeline_marks_ollama_failure_failed_without_creating_video(self):
        job = {
            "id": "ollama-job-test", "mode": "shorts", "duration": 30, "scene_count": 6,
            "language": "Hindi", "voice": "female", "prompt": "Coffee history",
        }
        passed = {"status": "passed", "detail": "test prerequisite"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_folder = root / "jobs" / job["id"]
            job_path = job_folder / "job.json"
            save_json(job_path, job)
            with (
                patch("workers.pipeline.job_directory", return_value=job_folder),
                patch("workers.qa.job_directory", return_value=job_folder),
                patch("workers.pipeline.preflight_job", return_value={
                    "status": "passed",
                    "checks": {name: passed for name in ("config", "ollama", "voice", "media_tools", "models", "resources")},
                }),
                patch("workers.pipeline.generate_script", side_effect=RuntimeError("Script generation failed: Ollama is down")),
                patch("workers.pipeline.emit"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Ollama is down"):
                    process_job(job)
            state = load_job(job_path)
            self.assertEqual(state["status"], "failed")
            self.assertIn("Ollama is down", state["error"])
            self.assertEqual(state["qa_report"]["status"], "failed")
            self.assertFalse((job_folder / "script.json").exists())
            self.assertFalse((root / "outputs" / job["id"] / "video.mp4").exists())

    def test_sdxl_prompt_uses_short_style_suffix_and_stays_under_clip_limit(self):
        class Tokenizer:
            def __call__(self, prompt, **_kwargs):
                return {"input_ids": prompt.split()}

        pipeline = type("Pipeline", (), {"tokenizer": Tokenizer(), "tokenizer_2": Tokenizer()})()
        source = "A small bird discovers a seed in a quiet garden at sunrise and meets another bird. " * 4
        prompt, tokens = _bounded_prompt(pipeline, source, "storybook", 1)
        self.assertLessEqual(tokens, 77)
        self.assertLessEqual(tokens, 70)
        self.assertIn("storybook illustration, warm soft colors", prompt)

    def test_ollama_script_request_unloads_model_after_completion(self):
        script = {"title": "Test", "scenes": []}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"response": json.dumps(script)}
        ).encode()
        job = {"prompt": "A small story", "mode": "shorts", "duration": 30, "language": "Hindi"}
        with patch("workers.script_llm.urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(_ollama_generate(job, 8), script)
        request = urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data)["keep_alive"], 0)

    def test_pipeline_can_verify_an_empty_ollama_resident_model_list(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"models":[]}'
        with patch.dict("os.environ", {"OLLAMA_URL": "http://127.0.0.1:11434"}), patch(
            "workers.script_llm.urllib.request.urlopen", return_value=response
        ) as urlopen:
            verify_ollama_unloaded(required=True)
        self.assertEqual(urlopen.call_args.args[0].full_url, "http://127.0.0.1:11434/api/ps")

    def test_identical_images_are_detected_by_perceptual_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            one, two = Path(folder) / "one.jpg", Path(folder) / "two.jpg"
            Image.effect_noise((64, 96), 80).convert("RGB").save(one)
            two.write_bytes(one.read_bytes())
            self.assertEqual(find_near_duplicate_images([one, two]), [(1, 2)])

    def test_caption_chunk_wraps_to_two_safe_lines(self):
        lines = _wrap_caption("Coffee history moves across old trade routes", 7, 25)
        self.assertEqual(len(lines), 2)
        self.assertLessEqual(max(len(line) for line in lines), 25)

    def test_missing_voice_file_fails_job_preflight(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("workers.doctor._job_voice_path", return_value=Path(folder) / "missing.onnx"), \
                patch("workers.doctor.shutil.which", return_value="piper.exe"), \
                patch("workers.doctor.urllib.request.urlopen", side_effect=OSError("offline")), \
                patch("workers.doctor._cached_snapshot", return_value=Path(folder)), \
                patch("workers.doctor.importlib.util.find_spec", return_value=object()), \
                patch("workers.doctor.subprocess.run", return_value=SimpleNamespace(returncode=0)), \
                patch("workers.doctor.shutil.disk_usage", return_value=SimpleNamespace(free=8 * 1024**3)), \
                patch("workers.doctor._available_memory_bytes", return_value=8 * 1024**3):
            report = preflight_job({"mode": "shorts", "duration": 30, "scene_count": 6, "language": "English", "voice": "female"})
        self.assertEqual(report["checks"]["voice"]["status"], "failed")

    def test_local_music_and_sfx_are_valid_wav(self):
        job = {"id": "audio-test", "mode": "shorts", "music_mood": "mystery", "seed": 42}
        script = {"scenes": [{"mood": "mystery"}]}
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch("workers.music.job_directory", return_value=Path(folder)),
                patch("workers.sfx.job_directory", return_value=Path(folder)),
            ):
                music = create_music(job, script, 2)
                sounds = generate_sfx(job)
            with wave.open(str(music), "rb") as stream:
                self.assertEqual(stream.getnchannels(), 2)
                self.assertGreater(stream.getnframes(), stream.getframerate())
            self.assertEqual(set(sounds), {"whoosh", "ambience"})

    def test_caption_words_use_whisper_timestamps_without_fallback(self):
        words = [
            SimpleNamespace(word="कॉफी", start=0.0, end=0.5),
            SimpleNamespace(word="इतिहास", start=0.5, end=1.0),
        ]
        model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: ([SimpleNamespace(words=words)], None))
        timed = _word_times("voice.wav", "कॉफी इतिहास", 1.0, model)
        self.assertEqual(timed, [("कॉफी", 0.0, 0.5), ("इतिहास", 0.5, 1.0)])
        with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
            _word_times("voice.wav", "कॉफी इतिहास", 1.0)

    def test_upscale_uses_requested_short_canvas_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "small.jpg"
            Image.new("RGB", (32, 48), "navy").save(original)
            job = {"id": "upscale-test", "mode": "shorts"}
            with patch("workers.upscale.job_directory", return_value=Path(folder)):
                results = upscale_images(job, [{"scene": 1, "path": str(original), "placeholder": True}])
            with Image.open(results[0]["render_path"]) as image:
                self.assertEqual(image.size, (1080, 1920))

    def test_hindi_narrator_gender_uses_matching_local_voice(self):
        with patch.dict("os.environ", {"PIPER_MODEL_HI_FEMALE": "", "PIPER_MODEL_HI_MALE": ""}):
            female = _voice_model({"voice": "female", "language": "Hindi"})
            male = _voice_model({"voice": "male", "language": "Hindi"})
        self.assertEqual(female.name, "hi_IN-priyamvada-medium.onnx")
        self.assertEqual(male.name, "hi_IN-rohan-medium.onnx")


if __name__ == "__main__":
    unittest.main()
