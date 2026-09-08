import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag_evolve.diagnostics import analyze_evidence_ladder, store_fingerprint


class EvidenceLadderTest(unittest.TestCase):
    def _store(self, root: Path) -> Path:
        store = root / "store"
        store.mkdir()
        (store / "report_corpus.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "n1",
                            "paragraphs": ["Revenue was $12 million in fiscal 2022."],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return store

    def test_reports_source_references_at_every_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            evidence = "Revenue was $12 million in fiscal 2022."
            trajectory = {
                "task_id": "q1",
                "answer": evidence,
                "turns": [
                    {
                        "round": 1,
                        "tools": [
                            {
                                "call_id": "search-1",
                                "name": "bm25_search",
                                "ok": True,
                                "result": {
                                    "results": [
                                        {
                                            "text": evidence,
                                            "ref": {"doc_id": "1", "node_id": "n1"},
                                        }
                                    ]
                                },
                            },
                            {
                                "call_id": "read-1",
                                "name": "read_section",
                                "ok": True,
                                "result": {
                                    "ref": {"doc_id": "1", "node_id": "n1"},
                                    "paragraphs": [
                                        {"paragraph_index": 0, "text": evidence}
                                    ],
                                },
                            },
                        ],
                    }
                ],
            }
            result = analyze_evidence_ladder(
                trajectory=trajectory,
                evaluation={"gold_evidence": [evidence]},
                store_path=store,
                payload_root=root,
                expected_store_fingerprint=store_fingerprint(store),
            )

        self.assertEqual(result["primary_signal"], "fully_covered")
        self.assertEqual(
            {layer: summary["recall"] for layer, summary in result["layers"].items()},
            {"corpus": 1.0, "candidate": 1.0, "read": 1.0, "answer": 1.0},
        )
        layers = result["evidence"][0]["layers"]
        self.assertEqual(layers["corpus"]["references"][0]["corpus_file"], "report_corpus.json")
        self.assertEqual(layers["candidate"]["references"][0]["tool_call_id"], "search-1")
        self.assertEqual(layers["read"]["references"][0]["tool_call_id"], "read-1")

    def test_identifies_candidate_as_earliest_missing_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            result = analyze_evidence_ladder(
                trajectory={
                    "task_id": "q1",
                    "answer": "No supporting number was found.",
                    "turns": [],
                },
                evaluation={
                    "gold_evidence": ["Revenue was $12 million in fiscal 2022."]
                },
                store_path=store,
                payload_root=root,
            )

        self.assertEqual(result["primary_signal"], "candidate")
        self.assertEqual(result["earliest_missing_counts"]["candidate"], 1)
        self.assertEqual(result["layers"]["corpus"]["recall"], 1.0)
        self.assertEqual(result["layers"]["candidate"]["recall"], 0.0)

    def test_does_not_combine_unrelated_passages_into_false_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            result = analyze_evidence_ladder(
                trajectory={
                    "task_id": "q1",
                    "answer": "",
                    "turns": [
                        {
                            "round": 1,
                            "tools": [
                                {
                                    "call_id": "search-1",
                                    "name": "bm25_search",
                                    "ok": True,
                                    "result": {
                                        "results": [
                                            {"text": "Revenue was twelve"},
                                            {"text": "million in fiscal 2022"},
                                        ]
                                    },
                                }
                            ],
                        }
                    ],
                },
                evaluation={
                    "gold_evidence": ["Revenue was twelve million in fiscal 2022"]
                },
                store_path=store,
                payload_root=root,
            )

        self.assertFalse(result["evidence"][0]["layers"]["candidate"]["matched"])

    def test_matches_gold_across_adjacent_text_and_html_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "store"
            store.mkdir()
            header = "CompanySubsidiariesSummary of Segment Data"
            table = (
                "<table><tr><td>(Dollars in millions)Years ended</td>"
                "<td>2022</td></tr><tr><td>Revenue</td><td>$12</td></tr></table>"
            )
            (store / "report_corpus.json").write_text(
                json.dumps(
                    {"nodes": [{"id": "n1", "paragraphs": [header, table]}]}
                ),
                encoding="utf-8",
            )
            gold = (
                "Company Subsidiaries\nSummary of Segment Data\n"
                "Dollars in millions\nYears ended\n2022\nRevenue\n$12"
            )
            result = analyze_evidence_ladder(
                trajectory={
                    "task_id": "q1",
                    "answer": "Revenue was $12 million.",
                    "turns": [
                        {
                            "round": 1,
                            "tools": [
                                {
                                    "call_id": "search-1",
                                    "name": "bm25_search",
                                    "ok": True,
                                    "result": {
                                        "results": [
                                            {
                                                "text": header,
                                                "neighbors": [{"text": table}],
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    ],
                },
                evaluation={"gold_evidence": [gold]},
                store_path=store,
                payload_root=root,
            )

        evidence = result["evidence"][0]
        self.assertTrue(evidence["layers"]["corpus"]["matched"])
        self.assertTrue(evidence["layers"]["candidate"]["matched"])
        self.assertEqual(evidence["earliest_missing_layer"], "read")

    def test_no_gold_evidence_is_not_reported_as_fully_covered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            result = analyze_evidence_ladder(
                trajectory={"task_id": "q1", "answer": "", "turns": []},
                evaluation={"gold_evidence": []},
                store_path=store,
                payload_root=root,
            )

        self.assertEqual(result["primary_signal"], "no_gold_evidence")

    def test_rejects_store_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            with self.assertRaisesRegex(ValueError, "store fingerprint"):
                analyze_evidence_ladder(
                    trajectory={"task_id": "q1", "answer": "", "turns": []},
                    evaluation={"gold_evidence": []},
                    store_path=store,
                    payload_root=root,
                    expected_store_fingerprint="wrong",
                )


if __name__ == "__main__":
    unittest.main()
