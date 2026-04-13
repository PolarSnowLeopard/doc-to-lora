"""读取 diagnostic JSON 结果文件，输出汇总表格"""
import json, os, glob, sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/perma/results"
type_names = {1: "Zero-Memory", 2: "In-Time", 3: "Post-Intervention"}

files = sorted(glob.glob(os.path.join(results_dir, "diagnostic_*.json")))
if not files:
    print(f"No result files found in {results_dir}")
    sys.exit(1)

all_modes = {}
for f in files:
    mode = os.path.basename(f).replace("diagnostic_", "").replace(".json", "")
    with open(f) as fh:
        all_modes[mode] = json.load(fh)

def acc(results):
    if not results:
        return 0, 0, 0
    c = sum(r["correct"] for r in results)
    return c, len(results), c / len(results)

print("=" * 80)
print("PERMA Diagnostic Results Summary")
print("=" * 80)

# Overall table
types = sorted({r["task_type"] for res in all_modes.values() for r in res})
header = f"{'Mode':<20} {'Overall':>10}"
for t in types:
    header += f" {'Type '+str(t):>15}"
print(header)
print("-" * len(header))

for mode, results in all_modes.items():
    c, n, a = acc(results)
    row = f"{mode:<20} {a:>9.1%} ({c}/{n})"
    for t in types:
        group = [r for r in results if r["task_type"] == t]
        c2, n2, a2 = acc(group)
        row += f" {a2:>9.1%} ({c2}/{n2})"
    print(row)

c_rand, n_rand = 0, 0
for results in all_modes.values():
    n_rand = len(results)
    break
print(f"{'random (1/8)':<20} {'12.5%':>10}")
print()

# Per-user breakdown
user_ids = sorted({r.get("user_id", 0) for res in all_modes.values() for r in res})
if len(user_ids) > 1:
    print("=" * 80)
    print("Per-User Breakdown")
    print("=" * 80)
    for uid in user_ids:
        print(f"\n--- User {uid} ---")
        header = f"{'Mode':<20} {'Overall':>10}"
        for t in types:
            header += f" {'Type '+str(t):>15}"
        print(header)
        print("-" * len(header))
        for mode, results in all_modes.items():
            user_res = [r for r in results if r.get("user_id") == uid]
            if not user_res:
                continue
            c, n, a = acc(user_res)
            row = f"{mode:<20} {a:>9.1%} ({c}/{n})"
            for t in types:
                group = [r for r in user_res if r["task_type"] == t]
                c2, n2, a2 = acc(group)
                row += f" {a2:>9.1%} ({c2}/{n2})"
            print(row)
