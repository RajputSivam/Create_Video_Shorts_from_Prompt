from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from workers.render import _final_render_command, _probe_video


class RenderTests(unittest.TestCase):
    def test_final_graph_transitions_and_burns_captions_once_with_audio_maps(self):
        job = {"mode": "shorts", "language": "English"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = _final_render_command(
                job,
                [root / "one.mp4", root / "two.mp4"],
                [4.0, 5.0],
                root / "music.wav",
                root / "captions.ass",
                {"transition": "dissolve"},
                root / "final.mp4",
            )

        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("xfade=transition=dissolve", graph)
        self.assertIn("acrossfade=d=0.250", graph)
        self.assertEqual(graph.count("subtitles=filename="), 1)
        self.assertIn("-map", command)
        self.assertIn("[captioned]", command)
        self.assertIn("[a]", command)
        self.assertEqual(command.count("-i"), 3)
        self.assertEqual(command[command.index("-ar") + 1], "48000")

    def test_probe_rejects_render_with_no_real_images(self):
        streams = [
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ]
        response = CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps({"streams": streams, "format": {"duration": "4.0"}}),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "video.mp4"
            output.write_bytes(b"x" * 5000)
            with patch("workers.render.subprocess.run", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "no real generated images"):
                    _probe_video(output, 4.0, 0)


if __name__ == "__main__":
    unittest.main()
