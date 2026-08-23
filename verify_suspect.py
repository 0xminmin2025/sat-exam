#!/usr/bin/env python3
"""对可疑批次做第二模型盲答复核。

单模型分歧可能是模型自身能力问题；两个独立模型一致反对答案键，
才能判定是数据错。
"""
import json, os, sys, random, hashlib, urllib.request, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_sat as A

MODEL_B = "Pro/MiniMaxAI/MiniMax-M2.5"
N = 20   # 每批次抽样数

SUSPECT = [
    ("rw",   "Bluebook", "Standard English Conventions"),
    ("math", "Bluebook", "Problem-Solving and Data Analysis"),
    ("math", "Bluebook", "Geometry and Trigonometry"),
    ("math", "Bluebook", "Algebra"),
]

_lock = threading.Lock()


def load_rows():
    rows = []
    for part, fn in A.FILES.items():
        for q in json.load(open(os.path.join(A.DOCS, fn))):
            cp = A.cache_path(q, part, A.MODEL_A)
            if not os.path.exists(cp):
                continue
            try:
                b = json.load(open(cp)).get("answer")
            except Exception:
                continue
            if b:
                rows.append((part, q, b))
    return rows


def main():
    rows = load_rows()
    print(f"已有模型A盲答: {len(rows)}\n")
    print(f"{'批次':<52}{'一致反对':>10}{'样本':>6}{'判定':>16}")
    print("-" * 86)

    for part, src, dom in SUSPECT:
        cand = []
        for p, q, b in rows:
            if p != part:
                continue
            if A.batch_key(q) != (src, dom):
                continue
            if A.same_answer(b, q.get("answer")):
                continue          # 只验有分歧的
            cand.append((q, b))
        if not cand:
            print(f"{part}/{src}/{dom:<28} 无可验证样本")
            continue

        random.seed(13)
        sample = random.sample(cand, min(N, len(cand)))

        agree = 0
        done = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(A.blind_ask, q, part, MODEL_B): (q, b)
                    for q, b in sample}
            for fut in as_completed(futs):
                q, b_a = futs[fut]
                try:
                    b_b = fut.result()
                except Exception:
                    b_b = None
                if not b_b:
                    continue
                done += 1
                key = q.get("answer")
                # 两模型答案一致，且都不等于答案键
                if A.same_answer(b_b, b_a) and not A.same_answer(b_b, key):
                    agree += 1

        pct = agree / done * 100 if done else 0
        verdict = ("答案键确认错误" if pct >= 80 else
                   "部分错误" if pct >= 50 else "模型侧问题")
        label = f"{part}/{src}/{dom}"
        print(f"{label:<52}{pct:>9.0f}%{done:>6}{verdict:>16}")


if __name__ == "__main__":
    main()
