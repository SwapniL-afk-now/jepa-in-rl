import json

FILES = {
    ("DAPO", "pass@8"): {
        "math": "dapo_pass8_math_5seed.json",
        "code": ["dapo_pass8_livecodebench_5seed.json", "dapo_he_mbpp_seedsA.json", "dapo_he_mbpp_seedsB.json"],
    },
    ("DAPO", "avg@8"): {
        "math": "dapo_avg8_math_5seed.json",
        "code": ["dapo_avg8_code_5seed.json"],
    },
    ("JEPO", "pass@8"): {
        "math": "jepo_pass8_math_5seed.json",
        "code": ["jepo_pass8_livecodebench_5seed.json", "jepo_he_mbpp_seedsA.json", "jepo_he_mbpp_seedsB.json"],
    },
    ("JEPO", "avg@8"): {
        "math": "jepo_avg8_math_5seed.json",
        "code": ["jepo_avg8_code_5seed.json"],
    },
}

SEEDS = [1234, 2025, 7777, 4242, 31337]
MATH_BENCHES = ["amc23", "aime24", "aime25", "aime26"]
CODE_BENCHES = ["humanevalplus", "mbppplus", "livecodebench"]
BENCH_LABELS = {
    "amc23": "AMC23", "aime24": "AIME24", "aime25": "AIME25", "aime26": "AIME26",
    "humanevalplus": "HumanEval+", "mbppplus": "MBPP+", "livecodebench": "LiveCodeBench",
}

def load_data(method, ckpt):
    spec = FILES[(method, ckpt)]
    math_d = json.load(open(spec["math"]))
    per_seed = {}
    for f in spec["code"]:
        d = json.load(open(f))
        for seed, benches in d["per_seed"].items():
            per_seed.setdefault(seed, {}).update(benches)
    for seed, benches in math_d["per_seed"].items():
        per_seed.setdefault(seed, {}).update(benches)
    return per_seed

lines = []
lines.append("# DAPO vs JEPO — full per-seed results (all checkpoints)")
lines.append("")
lines.append("Qwen2.5-3B-Instruct. 5 fixed seeds (1234, 2025, 7777, 4242, 31337), n=8 rollout budget per seed.")
lines.append("")

for method in ["DAPO", "JEPO"]:
    for ckpt in ["pass@8", "avg@8"]:
        data = load_data(method, ckpt)
        lines.append(f"## {method} best-{ckpt}")
        lines.append("")
        header = "| Benchmark | " + " | ".join(str(s) for s in SEEDS) + " |"
        sep = "|---|" + "---|" * len(SEEDS)
        lines.append(header)
        lines.append(sep)
        for bench_list in [MATH_BENCHES, CODE_BENCHES]:
            for bench in bench_list:
                label = BENCH_LABELS[bench]
                for metric_key, metric_label in [("pass@1", "pass@1"), ("pass@8", "pass@8")]:
                    vals = [data[str(s)][bench][metric_key] * 100 for s in SEEDS]
                    row = f"| {label} {metric_label} | " + " | ".join(f"{v:.2f}" for v in vals) + " |"
                    lines.append(row)
        lines.append("")

with open("results_full_5seed_all_checkpoints.md", "w") as f:
    f.write("\n".join(lines))
print("done")
