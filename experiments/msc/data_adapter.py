"""
MSC (Multi-Session Chat) 数据适配器。

将 HuggingFace nayohan/multi_session_chat 数据集
转换为 CMP pipeline 可用的格式。

MSC 数据结构:
  - 每个 dialogue_id 有 4 个 session (session 0-3)
  - 每个 session 包含 10-14 轮两人对话
  - persona 随 session 推进不断积累

对应原始论文 (Xu et al., ACL 2022) 的 session 编号:
  HF session 0 = paper session 1
  HF session 1 = paper session 2, ...
"""
import os
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MSCDialogue:
    dialogue_id: int
    split: str
    sessions: list[dict] = field(default_factory=list)


def session_to_text(session: dict) -> str:
    """将单个 session 的对话转为纯文本。"""
    lines = []
    for utt, spk in zip(session["utterances"], session["speakers"]):
        lines.append(f"{spk}: {utt}")
    return "\n".join(lines)


def load_dialogues(
    split: str = "test",
    cache_dir: str | None = None,
    max_dialogues: int = 0,
) -> list[MSCDialogue]:
    """从 HuggingFace 加载 MSC 数据，按 dialogue_id 分组。

    Args:
        split: "train" / "validation" / "test"
        cache_dir: HuggingFace 缓存目录
        max_dialogues: 限制返回对话数量（0=不限）
    """
    from datasets import load_dataset

    ds = load_dataset(
        "nayohan/multi_session_chat", split=split, cache_dir=cache_dir,
    )

    grouped: dict[int, list] = defaultdict(list)
    for row in ds:
        grouped[row["dialoug_id"]].append(row)

    dialogues = []
    for did, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["session_id"])
        sessions = []
        for row in rows:
            sess = {
                "session_id": row["session_id"],
                "persona1": row["persona1"],
                "persona2": row["persona2"],
                "utterances": row["dialogue"],
                "speakers": row["speaker"],
            }
            sess["text"] = session_to_text(sess)
            sessions.append(sess)
        dialogues.append(MSCDialogue(
            dialogue_id=did, split=split, sessions=sessions,
        ))
        if max_dialogues > 0 and len(dialogues) >= max_dialogues:
            break

    return dialogues


def get_opening_text(session: dict) -> str:
    """提取 session 的第一条发言（session opening）。"""
    if session["utterances"]:
        return f"{session['speakers'][0]}: {session['utterances'][0]}"
    return ""
