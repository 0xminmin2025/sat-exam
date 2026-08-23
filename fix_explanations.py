#!/usr/bin/env python3
"""检测并清除"串位解析"——解析讲的是另一道题。

背景：SAT 题库的解析是分批生成后拼接的，存在错位，导致题干问 A、
解析讲 B。这比答案键错更有害（学生会照着错的思路学）。

判据（分数学/读写两套，因为两者的文本特征完全不同）：

  数学 —— 用【数字指纹】
    数学题的身份标识是具体数值(220/900/11.60)，不是词汇。
    实测词汇重叠对数学不可靠：串位解析也能靠 equation/function/graph
    这些通用词刷到 12-17% 重叠，会漏判。
    题干数字与解析数字零交集 = 几乎必然串位。

  读写 —— 用【词汇重叠】
    RW 题干多是套话("Which choice completes the text...")，
    真正内容在 passage 和 options 里，所以取三者最大重叠。
    注意阈值要低：RW 正常解析的重叠率本就比数学低。

  通用 —— 【解析自述答案 vs 答案键冲突】
    纯文本比对，最硬的证据，不依赖任何阈值。
    但答案键为多值('11 or -7')时要跳过，否则误报。

用法:
  python3 fix_explanations.py            # 只报告，不改
  python3 fix_explanations.py --apply    # 删除串位解析
"""
import json, os, re, argparse
from collections import Counter

HOME = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HOME, "docs")
FILES = {"math": "q_math.json", "rw": "q_rw.json"}


def toks(s):
    return set(re.findall(r"[a-zA-Z]{4,}", (s or "").lower()))


def nums(s):
    """数字指纹。去掉 0-3 这类高频噪音，只留有辨识度的数值。"""
    out = set()
    for x in re.findall(r"\d+\.?\d*", (s or "").replace(",", "")):
        try:
            if len(x) >= 2 or float(x) > 3:
                out.add(x.rstrip("."))
        except ValueError:
            pass
    return out


def refs_of(q, fn):
    """题目侧的参考集合：题干 / passage / 选项。"""
    r = [fn(q.get("stem")), fn(q.get("passage"))]
    if q.get("options"):
        u = set()
        for v in q["options"].values():
            u |= fn(v)
        r.append(u)
    return r


def same_answer(a, b):
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return False
    if re.fullmatch(r"[A-Da-d]", a) or re.fullmatch(r"[A-Da-d]", b):
        return a.upper()[:1] == b.upper()[:1]
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.lower() == b.lower()


def judge(part, q):
    """返回 (是否串位, 原因)。保守优先——拿不准就判正常，宁漏不误删。"""
    e = (q.get("explanation") or "").strip()
    if not e:
        return False, ""

    # 信号①：解析自述答案与答案键冲突（最硬，跨科目通用）
    key = str(q.get("answer") or "").strip()
    if key and not re.search(r"\bor\b|,", key):
        m = re.search(r"correct answer is\s*\**\s*\(?([A-D]\b|[\d.]+)", e, re.I)
        if m:
            said = m.group(1).strip().rstrip(".")
            if not same_answer(said, key):
                return True, "解析自述答案与答案键冲突"

    if part == "math":
        # 信号②：数字指纹 + 词汇 双信号
        #
        # 单用数字阈值切不干净：18-26% 区间里既有真串位(表格题配 y=12x+12)，
        # 也有正常解析(平行线求w，解析也在讲平行线，只是没复述所有数字)。
        # 单用词汇也不行：串位解析能靠 equation/function/graph 刷到 12-17%。
        #
        # 所以要求两个信号同时低——数字对不上"且"用词也对不上，才判串位。
        # 这样牺牲一点召回，但避免误删正常解析（错删比漏删更难挽回，
        # 因为原文虽存了备份，用户看到的就是没解析）。
        ref_n = set()
        for s in refs_of(q, nums):
            ref_n |= s
        en = nums(e)
        ov_n = len(ref_n & en) / len(ref_n) if (len(ref_n) >= 4 and en) else None

        best_w = 0.0
        has_w = False
        for r in refs_of(q, toks):
            if len(r) >= 5:
                has_w = True
                best_w = max(best_w, len(r & toks(e)) / len(r))

        if ov_n is not None:
            if ov_n == 0:
                return True, "数字指纹零交集"
            # 数字偏低且词汇也偏低 = 双证据
            if ov_n <= 0.25 and has_w and best_w < 0.30:
                return True, "数字+词汇双低"
        else:
            # 数字不足以判断，退回纯词汇，阈值压到极低只抓最明显的
            if has_w and best_w < 0.03:
                return True, "词汇几乎无重叠"
    else:
        # 读写：三者取最大重叠，阈值 5%
        best = 0.0
        ok = False
        for r in refs_of(q, toks):
            if len(r) >= 5:
                ok = True
                best = max(best, len(r & toks(e)) / len(r))
        if ok and best < 0.05:
            return True, "词汇几乎无重叠"

    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    grand = Counter()
    for part, fn in FILES.items():
        path = os.path.join(DOCS, fn)
        qs = json.load(open(path))
        hits = []
        for q in qs:
            bad, why = judge(part, q)
            if bad:
                hits.append((q, why))
        reasons = Counter(w for _, w in hits)
        n_expl = sum(1 for q in qs if (q.get("explanation") or "").strip())
        print(f"\n[{part}] 共 {len(qs)} 题，有解析 {n_expl} 道")
        print(f"   判定串位 {len(hits)} 道 ({len(hits)/max(n_expl,1)*100:.1f}%)")
        for r, c in reasons.most_common():
            print(f"      {r}: {c}")
        grand[part] = len(hits)

        if a.apply:
            for q, why in hits:
                # 保留原文备查，不是直接抹掉——将来重生成时可对照
                q["explanation_bad"] = q.get("explanation")
                q["explanation_bad_reason"] = why
                q["explanation"] = ""
            tmp = path + ".tmp"
            json.dump(qs, open(tmp, "w"), ensure_ascii=False,
                      separators=(",", ":"))
            os.replace(tmp, path)
            print(f"   已清除 {len(hits)} 道的解析（原文存入 explanation_bad）")

    print(f"\n合计 {sum(grand.values())} 道")
    if not a.apply:
        print("（这是预演，加 --apply 才会真正修改）")


if __name__ == "__main__":
    main()
