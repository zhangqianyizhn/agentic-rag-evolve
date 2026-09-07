from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_section(
    doc_index: Any,
    doc_id: Optional[str],
    node_id: str,
    start_paragraph: int,
    end_paragraph: int,
    include_images: bool = True,
) -> Dict[str, Any]:
    doc_id_str = str(doc_id or "")
    node_id_str = str(node_id) if node_id is not None else ""

    node = (doc_index.nodes_by_doc.get(doc_id_str) or {}).get(node_id_str)
    if not node:
        return {
            "ok": False,
            "error": f"node_id '{node_id_str}' not found in doc '{doc_id_str}'",
            "ref": {"doc_id": doc_id_str, "node_id": node_id_str, "paragraph_indexes": []},
            "paragraphs": [],
        }

    paragraphs = node.get("paragraphs", [])
    n = len(paragraphs)

    start_paragraph = max(0, min(start_paragraph, n))
    if end_paragraph == -1:
        end_paragraph = n
    end_paragraph = max(start_paragraph, min(end_paragraph, n))

    slice_paragraphs = paragraphs[start_paragraph:end_paragraph]

    out_items: List[Dict[str, Any]] = []
    for rel_idx, paragraph in enumerate(slice_paragraphs):
        paragraph_index = start_paragraph + rel_idx

        if isinstance(paragraph, str):
            if paragraph.strip():
                out_items.append({"paragraph_index": paragraph_index, "type": "text", "text": paragraph})
            continue

        if isinstance(paragraph, dict):
            paragraph_type = paragraph.get("type", "text")
            content = paragraph.get("content", "")

            if paragraph_type == "image" and include_images:
                if content:
                    out_items.append(
                        {
                            "paragraph_index": paragraph_index,
                            "type": "text",
                            "text": f"[image_ocr]{content}[/image_ocr]",
                        }
                    )
                image_path = paragraph.get("image_path")
                if image_path and Path(image_path).exists():
                    try:
                        with open(image_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        mime_type, _ = mimetypes.guess_type(image_path)
                        if mime_type:
                            out_items.append(
                                {
                                    "paragraph_index": paragraph_index,
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                                }
                            )
                    except Exception:
                        pass
                elif content:
                    out_items.append(
                        {
                            "paragraph_index": paragraph_index,
                            "type": "text",
                            "text": f"[image_ocr]{content}[/image_ocr]",
                        }
                    )
            elif content:
                out_items.append({"paragraph_index": paragraph_index, "type": "text", "text": content})
        else:
            out_items.append({"paragraph_index": paragraph_index, "type": "text", "text": str(paragraph)})

    paragraph_indexes = sorted({item["paragraph_index"] for item in out_items})
    return {
        "ok": True,
        "ref": {"doc_id": doc_id_str, "node_id": node_id_str, "paragraph_indexes": paragraph_indexes},
        "paragraphs": out_items,
    }
