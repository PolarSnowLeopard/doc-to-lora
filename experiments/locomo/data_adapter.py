"""
LoCoMo 数据适配器。

解析 locomo10.json，提取 session-level 对话文本和 QA 标注。
数据下载: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

LoCoMo 数据结构:
  - 10 个超长对话（19-35 sessions, 300+ 轮, 9K-26K tokens）
  - 每个对话包含 QA 标注（5 类: single-hop / multi-hop / temporal / commonsense / adversarial）
  - 评测指标: token-level F1 (with stemming)
"""
import json
import os
from dataclasses import dataclass, field


QA_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "commonsense",
    4: "single-hop",
    5: "adversarial",
}


@dataclass
class LoCoMoConversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[dict] = field(default_factory=list)
    qa: list[dict] = field(default_factory=list)


def _extract_sessions(conversation: dict) -> list[dict]:
    """从 conversation 字段中提取按时间排序的 session 列表。"""
    sessions = []
    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        session_idx = int(sk.split("_")[1])
        date_key = f"{sk}_date_time"
        turns = conversation[sk]
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if text:
                lines.append(f"{speaker}: {text}")
        sessions.append({
            "session_id": session_idx,
            "date": conversation.get(date_key, ""),
            "text": "\n".join(lines),
            "n_turns": len(turns),
        })
    return sessions


def load_conversations(data_path: str) -> list[LoCoMoConversation]:
    """加载 locomo10.json 并解析为 LoCoMoConversation 列表。"""
    with open(data_path, "r") as f:
        raw = json.load(f)

    conversations = []
    for sample in raw:
        conv = sample["conversation"]
        sessions = _extract_sessions(conv)
        conversations.append(LoCoMoConversation(
            sample_id=sample["sample_id"],
            speaker_a=conv.get("speaker_a", ""),
            speaker_b=conv.get("speaker_b", ""),
            sessions=sessions,
            qa=sample.get("qa", []),
        ))
    return conversations


def session_to_text(session: dict) -> str:
    """返回 session 的对话文本（供 D2L 编码）。"""
    return session["text"]
