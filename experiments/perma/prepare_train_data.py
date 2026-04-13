"""
将 PERMA 数据转换为 Doc-to-LoRA 训练格式 (parquet)。

输出字段:
  context:   对话 session 文本 (送入超网络 internalize)
  prompts:   [MCQ 问题+选项]
  responses: [gold label]

数据增强:
  对每个 task 生成多种 context 变体 —— 全部 session / 最后 N 个 session

用法:
  PERMA_DATA_ROOT=experiments/perma/data \
  python experiments/perma/prepare_train_data.py \
      --test_user 334 \
      --output_dir experiments/perma/train_data
"""
import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from experiments.perma.data_adapter import (
    ALL_USER_IDS,
    load_tasks,
    session_to_text,
    sessions_to_full_text,
)


def build_mcq_prompt(question: str, options: list[str]) -> str:
    n = len(options)
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    return (
        f"Question: {question}\n\n"
        f"Options:\n{opts}\n\n"
        f"Answer with the letter only (A-{chr(65 + n - 1)}):"
    )


def make_samples(tasks, augment=True):
    """从 task 列表生成训练样本"""
    samples = []
    for task in tasks:
        prompt = build_mcq_prompt(task.question, task.options)
        gold = task.gold_label

        # 变体 1: 全部 session
        if len(task.sessions) > 0:
            full_ctx = sessions_to_full_text(task.sessions)
            samples.append({
                "context": full_ctx,
                "prompts": [prompt],
                "responses": [gold],
            })

        if not augment:
            continue

        # 变体 2: 只用最后一个 session
        if len(task.sessions) > 1:
            last_ctx = session_to_text(task.sessions[-1])
            samples.append({
                "context": last_ctx,
                "prompts": [prompt],
                "responses": [gold],
            })

        # 变体 3: 最后 2 个 session
        if len(task.sessions) > 2:
            ctx = "\n\n".join(
                session_to_text(s) for s in task.sessions[-2:]
            )
            samples.append({
                "context": ctx,
                "prompts": [prompt],
                "responses": [gold],
            })

        # 变体 4: 最后一半 session
        if len(task.sessions) > 3:
            half = len(task.sessions) // 2
            ctx = "\n\n".join(
                session_to_text(s) for s in task.sessions[half:]
            )
            samples.append({
                "context": ctx,
                "prompts": [prompt],
                "responses": [gold],
            })

    return samples


def save_parquet(samples, path):
    table = pa.table({
        "context": [s["context"] for s in samples],
        "prompts": [s["prompts"] for s in samples],
        "responses": [s["responses"] for s in samples],
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)
    print(f"  Saved {len(samples)} samples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_user", type=int, default=334,
                        help="Held-out user for testing")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/perma/train_data")
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    augment = not args.no_augment
    train_users = [u for u in ALL_USER_IDS if u != args.test_user]
    test_users = [args.test_user]

    print(f"Train users ({len(train_users)}): {train_users}")
    print(f"Test user: {test_users}")

    train_tasks = load_tasks(user_ids=train_users, noise=False, multi_domain=False)
    test_tasks = load_tasks(user_ids=test_users, noise=False, multi_domain=False)

    print(f"Train tasks: {len(train_tasks)}")
    print(f"Test tasks: {len(test_tasks)}")

    train_samples = make_samples(train_tasks, augment=augment)
    test_samples = make_samples(test_tasks, augment=False)

    save_parquet(train_samples, os.path.join(args.output_dir, "train.parquet"))
    save_parquet(test_samples, os.path.join(args.output_dir, "test.parquet"))

    # 统计
    ctx_lens = [len(s["context"]) for s in train_samples]
    print(f"\nTrain context length stats (chars):")
    print(f"  min={min(ctx_lens)}, max={max(ctx_lens)}, "
          f"avg={sum(ctx_lens)//len(ctx_lens)}")


if __name__ == "__main__":
    main()
