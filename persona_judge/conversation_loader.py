# persona_judge/conversation_loader.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def load_conversations(base_dir: str | Path, persona_id: str) -> List[Dict[str, Any]]:
    """
    Load raw conversations for a given persona_id.

    Expected layout:
      base_dir/
        persona_id/
          raw_conversations.json

    The json file is expected to contain a list of dicts like:
      {"role": "user" or "assistant", "content": "text..."}
    """
    base_path = Path(base_dir)
    persona_path = base_path / persona_id
    raw_path = persona_path / "raw_conversations.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"raw_conversations.json not found at {raw_path}")

    with raw_path.open("r", encoding="utf8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("raw_conversations.json must contain a list")

    conversations: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        # normalize whitespace a bit
        content_norm = " ".join(content.split())
        conversations.append({"role": role, "content": content_norm})

    return conversations
