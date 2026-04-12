"""
PERMA 数据适配器：将 PERMA benchmark 的多 session 对话数据
转换为 Doc-to-LoRA 可以 internalize 的纯文本格式。
"""
import json
import os
from dataclasses import dataclass
from typing import Optional


PERMA_DATA_ROOT = os.environ.get(
    "PERMA_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "data"),
)

COUNTRY_USERS = {
    "Canada": [334], "Mexico": [354], "Finland": [123],
    "United States": [1377], "Australia": [507],
    "United Kingdom": [914], "Switzerland": [112],
    "Israel": [419], "Russian Federation": [108], "Belgium": [109],
}
ALL_USER_IDS = [uid for ids in COUNTRY_USERS.values() for uid in ids]


@dataclass
class PermaTask:
    user_id: int
    task_id: str
    task_type: int            # 1=Zero-Memory, 2=In-Time, 3=Post-Intervention
    topic: list[str]
    sessions: list[dict]      # [{"text": str, "date": str}, ...]
    question: str
    options: list[str]
    gold_label: str
    preferences: list[str]


def _session_to_text(conversation: list[dict]) -> str:
    return "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in conversation
    )


def _flatten_sessions(context_list: list) -> list[dict]:
    """将 PERMA 的 context 格式转为 [{text, date}, ...]"""
    sessions = []
    for entry in context_list:
        conv = entry[0]  # list of {role, content}
        date = entry[1] if len(entry) > 1 else ""
        sessions.append({
            "text": _session_to_text(conv),
            "date": str(date),
        })
    return sessions


def load_tasks(
    user_ids: Optional[list[int]] = None,
    noise: bool = False,
    multi_domain: bool = False,
) -> list[PermaTask]:
    if user_ids is None:
        user_ids = ALL_USER_IDS

    version = "_multi" if multi_domain else ""
    suffix = "_n" if noise else "_c"
    tasks = []

    for uid in user_ids:
        task_path = os.path.join(
            PERMA_DATA_ROOT, "tasks", f"user{uid}",
            f"input_data{version}{suffix}.json",
        )
        if not os.path.exists(task_path):
            continue

        with open(task_path, "r") as f:
            data = json.load(f)

        meta_dir = os.path.join(
            PERMA_DATA_ROOT, "evaluation", f"user{uid}", "meta", "overall",
        )

        for ev in data.get("overall", []):
            task_id = ev.get("task_id", "")
            task_type = int(ev.get("type", 0))
            context = ev.get("context", [])
            sessions = _flatten_sessions(context)

            meta_path = os.path.join(meta_dir, f"{task_id}_{task_type}.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                meta = json.load(f)

            tasks.append(PermaTask(
                user_id=uid,
                task_id=task_id,
                task_type=task_type,
                topic=ev.get("topic", []),
                sessions=sessions,
                question=meta.get("question", ""),
                options=meta.get("options", []),
                gold_label=meta.get("gold_label", ""),
                preferences=ev.get("preferences", []),
            ))

    return tasks


def sessions_to_full_text(sessions: list[dict]) -> str:
    """将所有 session 拼接为一个完整文本（供 Oracle 模式使用）"""
    parts = []
    for i, s in enumerate(sessions):
        parts.append(f"[Session {i+1} | {s['date']}]\n{s['text']}")
    return "\n\n".join(parts)


def session_to_text(session: dict) -> str:
    """单个 session 转文本（供 Single-shot / 增量模式使用）"""
    return f"[{session['date']}]\n{session['text']}"
