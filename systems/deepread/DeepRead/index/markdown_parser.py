from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_heading(line: str) -> Optional[Dict[str, Any]]:
    """Return {'level': int, 'title': str} if line is a markdown heading (#,##,...), else None."""
    m = re.match(r"^\s*(#{1,6})\s+(.*)$", line)
    if m:
        return {"level": len(m.group(1)), "title": m.group(2).strip()}
    return None


def _extract_html_table(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Extract a full <table>...</table> block starting at start_idx. Returns (block, next_index)."""
    buf = []
    i = start_idx
    first_line = lines[i]
    buf.append(first_line)

    if "</table>" in first_line.lower():
        return "\n".join(buf), i + 1

    i += 1
    while i < len(lines):
        buf.append(lines[i])
        if "</table>" in lines[i].lower():
            i += 1
            break
        i += 1
    return "\n".join(buf), i


def _extract_md_table(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Extract a markdown pipe-style table. Consecutive lines starting with '|' or optional align lines."""
    buf = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\|.*\|\s*$", line) or re.match(
            r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", line
        ):
            buf.append(line)
            i += 1
        else:
            break
    return "\n".join(buf), i


def _extract_image(line: str, md_dir: str) -> Optional[Dict[str, Any]]:
    """Detect <img ...> or ![]() in a line and return a paragraph dict with absolute image_path."""
    # HTML image tag
    m = re.search(r"<img\s+[^>]*src=[\"']([^\"']+)[\"'][^>]*", line, re.IGNORECASE)
    if m:
        src = m.group(1).strip()
        alt_m = re.search(r"alt=[\"']([^\"']+)[\"']", line, re.IGNORECASE)
        alt_text = alt_m.group(1).strip() if alt_m else ""
        if src.startswith(("http://", "https://", "data:")):
            image_path = src
        else:
            image_path = os.path.normpath(os.path.join(md_dir, src))
        return {"type": "image", "content": alt_text, "image_path": image_path}

    # Markdown image syntax ![alt](src)
    m2 = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", line)
    if m2:
        alt_text = m2.group(1).strip()
        src = m2.group(2).strip()
        if src.startswith(("http://", "https://", "data:")):
            image_path = src
        else:
            image_path = os.path.normpath(os.path.join(md_dir, src))
        return {"type": "image", "content": alt_text, "image_path": image_path}

    return None


def parse_markdown_to_corpus(md_path: str) -> Dict[str, Any]:
    """
    Parse a Markdown file into a corpus JSON compatible with MuReAct_TF DocIndex:
    - Recognizes headings (#, ##, ...). If the minimum heading level > 1, normalize so min level becomes 1.
    - Treats each table (HTML <table>...</table> or pipe-style markdown) as a single paragraph string.
    - Treats each image (<img ...> / ![]()) as a dict paragraph: {type: 'image', content: alt, image_path: absolute_path}.
    - Other text blocks are grouped by blank lines.
    - Maintains parent-child hierarchy with ids (flat sequential numbering: "1", "2", "3", ...).
    """
    md_dir = os.path.dirname(md_path)
    content = _read_file(md_path)
    lines = content.splitlines()

    # Pass 1: find minimum heading level present to normalize hierarchy
    heading_levels = []
    for ln in lines:
        info = _is_heading(ln)
        if info:
            heading_levels.append(info["level"])
    min_level = min(heading_levels) if heading_levels else 1
    level_offset = min_level - 1

    nodes: List[Dict[str, Any]] = []
    node_map: Dict[str, Dict[str, Any]] = {}
    stack: List[Dict[str, Any]] = []  # each: {'id': str, 'level': int}
    next_id = 0

    def _alloc_id() -> str:
        nonlocal next_id
        next_id += 1
        return str(next_id)

    front_matter_id: Optional[str] = None

    def _ensure_front_matter(paragraph: Any):
        nonlocal front_matter_id, nodes, node_map, stack
        if front_matter_id is None:
            front_matter_id = _alloc_id()
            fm = {"id": front_matter_id, "title": "前言", "paragraphs": [], "children": []}
            nodes.append(fm)
            node_map[front_matter_id] = fm
            stack = [{"id": front_matter_id, "level": 1}]
        node_map[front_matter_id]["paragraphs"].append(paragraph)

    def _new_node(level: int, title: str) -> Dict[str, Any]:
        nonlocal stack, nodes, node_map

        # Top-level node or empty stack
        if level == 1 or not stack:
            node_id = _alloc_id()
            node = {"id": node_id, "title": title.strip(), "paragraphs": [], "children": []}
            nodes.append(node)
            node_map[node_id] = node
            stack = [{"id": node_id, "level": 1}]
            return node

        # Pop until we find parent at level-1
        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if not stack:
            # No valid parent, treat as top-level
            node_id = _alloc_id()
            node = {"id": node_id, "title": title.strip(), "paragraphs": [], "children": []}
            nodes.append(node)
            node_map[node_id] = node
            stack = [{"id": node_id, "level": 1}]
            return node

        parent = stack[-1]
        parent_id = parent["id"]

        node_id = _alloc_id()
        node = {"id": node_id, "title": title.strip(), "paragraphs": [], "children": []}
        nodes.append(node)
        node_map[node_id] = node

        # Link to immediate parent
        node_map[parent_id]["children"].append(node_id)
        stack.append({"id": node_id, "level": level})
        return node

    def _append_paragraph(node: Optional[Dict[str, Any]], paragraph: Any):
        if node is None:
            _ensure_front_matter(paragraph)
            return
        node["paragraphs"].append(paragraph)

    current_node: Optional[Dict[str, Any]] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _is_heading(line)

        if heading:
            normalized = max(1, heading["level"] - level_offset)
            if stack:
                parent_level = stack[-1]["level"]
                if normalized > parent_level + 1:
                    level = parent_level + 1  # avoid multi-level jumps
                else:
                    level = normalized
            else:
                level = normalized

            current_node = _new_node(level, heading["title"])
            i += 1
            continue

        # HTML table block
        if "<table" in line.lower():
            block, nxt = _extract_html_table(lines, i)
            _append_paragraph(current_node, block)
            i = nxt
            continue

        # pipe-style markdown table block
        if re.match(r"^\s*\|.*\|\s*$", line):
            block, nxt = _extract_md_table(lines, i)
            _append_paragraph(current_node, block)
            i = nxt
            continue

        # image block
        img_para = _extract_image(line, md_dir)
        if img_para is not None:
            _append_paragraph(current_node, img_para)
            i += 1
            continue

        # collect text paragraph until blank line or block boundary
        if line.strip() == "":
            i += 1
            continue

        buf = [line]
        i += 1
        while i < len(lines):
            peek = lines[i]
            if peek.strip() == "":
                i += 1
                break
            if (
                _is_heading(peek)
                or "<table" in peek.lower()
                or re.match(r"^\s*\|.*\|\s*$", peek)
                or _extract_image(peek, md_dir) is not None
            ):
                break
            buf.append(peek)
            i += 1

        paragraph_text = "\n".join(buf)
        _append_paragraph(current_node, paragraph_text)

    return {"nodes": nodes}

