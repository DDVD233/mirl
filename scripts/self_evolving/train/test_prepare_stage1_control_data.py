import json
import tempfile
import unittest
from pathlib import Path

from prepare_stage1_control_data import prepare


class PreparedControlDataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source, self.output = self.root / "source", self.root / "output"
        self.source.mkdir()
        self.output.mkdir()
        (self.source / "ecg_images").mkdir()
        self.image = self.source / "ecg_images/example.png"
        self.image.write_bytes(b"test fixture; only existence is checked")
        self.originals = {}
        for name, n in (("train", 5719), ("test", 2452)):
            rows = [
                {
                    "extra_info": {"hadm_id": i},
                    "prompt": "<image>",
                    "images": [
                        {
                            "image": "/scratch/self_evolving_datasets/mimiciv_rare/ecg_images/example.png",
                            "max_pixels": 65536,
                        }
                    ],
                }
                for i in range(n)
            ]
            path = self.source / (name + ".jsonl")
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.originals[name] = path.read_bytes()

    def test_paths_resolve_without_changing_originals(self):
        prepare(self.source, self.output, self.source)
        for name in ("train", "test"):
            self.assertEqual((self.source / (name + ".jsonl")).read_bytes(), self.originals[name])
            first = json.loads((self.output / (name + ".jsonl")).read_text().splitlines()[0])
            self.assertEqual(first["images"][0]["image"], str(self.image))
            self.assertEqual(first["images"][0]["max_pixels"], 65536)

    def test_missing_images_fail_before_training(self):
        self.image.unlink()
        with self.assertRaisesRegex(ValueError, "missing images"):
            prepare(self.source, self.output, self.source)

    def test_changed_prepared_input_requires_new_directory(self):
        (self.output / "train.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Prepared input changed"):
            prepare(self.source, self.output, self.source)


if __name__ == "__main__":
    unittest.main()
