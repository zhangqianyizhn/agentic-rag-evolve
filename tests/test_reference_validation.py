import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentic_rag_evolve.reference_validation import validate_store


class ReferenceValidationTest(unittest.TestCase):
    def test_validates_relocatable_vector_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "report_emb.npy", np.asarray([[1.0, 0.0]], dtype=np.float16))
            (root / "report_idmap.json").write_text(
                json.dumps([{"node_id": "0", "paragraph_index": 0}]), encoding="utf-8"
            )
            (root / "report_corpus.json").write_text(
                json.dumps({
                    "nodes": [{"id": "0", "paragraphs": ["text"]}],
                    "vector_store": {
                        "matrix_path": "/old/machine/report_emb.npy",
                        "id_map_path": "/old/machine/report_idmap.json",
                        "model_name": "embedding-v1",
                    },
                }),
                encoding="utf-8",
            )

            result = validate_store(root)

            self.assertEqual(result.document_count, 1)
            self.assertEqual(result.vector_dimension, 2)
            self.assertEqual(result.vector_count, 1)


if __name__ == "__main__":
    unittest.main()
