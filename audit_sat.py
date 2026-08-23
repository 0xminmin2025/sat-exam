#!/usr/bin/env python3
"""SAT 答案键体检：盲答交叉验证 + 批次分歧率统计。

与 AP 版(audit_answers.py)的差异：
  - SAT 无 year 字段，批次维度用 (sources, domain)
  - 含 407 道填空题(答案为数值)，需数值等价比较而非字母比较
  - 原始数据无 answer_disputed，必须先跑盲答生成

用法:
  python3 audit_sat.py --pilot 150      # 分层抽样探路，估算整体分歧率
  python3 audit_sat.py --full           # 全量盲答
  python3 audit_sat.py --report         # 只出报告(读已有 blind_answer)
  python3 audit_sat.py --verify         # 对可疑批次做第二模型复核
  python3 audit_sat.py --mark           # 标记可疑批次
"""
import json, os, argparse, random, re, threading, urllib.request, hashlib, time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

HOME = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HOME, "docs")
CACHE = os.path.join(HOME, ".blind_cache")
API = "https://api.siliconflow.cn/v1/chat/completions"

FILES = {"math": "q_math.json", "rw": "q_rw.json"}
TESTNAME = {"math": "SAT Math", "rw": "SAT Reading and Writing"}

# 与生成解析所用模型独立，保证是真正的第二意见
MODEL_A = "Pro/moonshotai/Kimi-K2.5"
MODEL_B = "Pro/MiniMaxAI/MiniMax-M2.5"

SUSPECT_RATE = 30.0
MIN_N = 10

_lock = threading.Lock()
_stats = Counter()


def _load_key():
    k = os.environ.get("SILICONFLOW_KEY", "").strip()
    if k:
        return k
    p = os.path.join(HOME, ".env.local")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("SILICONFLOW_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


KEY = _load_key()

PROMPT_MC = """Answer this {subj} multiple-choice question.
Reason briefly, then respond with ONLY JSON: {{"answer":"<single letter A-D>"}}
The value MUST be exactly one of the letters shown below. Never output a number.

{passage}{stem}

{opts}"""

PROMPT_FR = """Answer this {subj} student-produced-response question.
This question has NO answer choices. You must compute the value yourself.
Reason briefly, then respond with ONLY JSON: {{"answer":"<the numeric value>"}}
The value MUST be a number (or a simple fraction like 3/4). Never output a letter.

{passage}{stem}"""


def norm_num(s):
    """数值答案归一化：'15' '15.0' ' 15 ' 视为相等；分数 3/4 转小数。"""
    s = str(s).strip().replace(" ", "").replace(",", "")
    s = s.strip("$%")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            d = float(m.group(2))
            return round(float(m.group(1)) / d, 6) if d else None
        except Exception:
            return None
    try:
        return round(float(s), 6)
    except Exception:
        return None


def same_answer(a, b):
    """兼容字母题和数值题的答案比较。"""
    if a is None or b is None:
        return False
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return False
    if re.fullmatch(r"[A-Da-d]", a) or re.fullmatch(r"[A-Da-d]", b):
        return a.upper()[:1] == b.upper()[:1]
    na, nb = norm_num(a), norm_num(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-6
    return a.lower() == b.lower()


def cache_path(q, part, model):
    """缓存键用题干内容哈希，不能用 question_id。
    实测 question_id 在本数据集里严重重复(math 1720题仅1105个唯一id，
    单个id最多被33道不同的题共用)，用它做键会导致不同题共用缓存、结果张冠李戴。"""
    d = os.path.join(CACHE, model.replace("/", "_"))
    os.makedirs(d, exist_ok=True)
    raw = json.dumps({"s": q.get("stem"), "o": q.get("options"),
                      "p": (q.get("passage") or "")[:400]},
                     ensure_ascii=False, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()[:16]
    return os.path.join(d, f"{part}_{h}.json")


def blind_ask(q, part, model, session_timeout=120):
    """盲答：只给题干和选项，不给答案键和已有解析。"""
    qid = q.get("question_id")
    cp = cache_path(q, part, model)
    if os.path.exists(cp):
        try:
            with _lock:
                _stats["cached"] += 1
            return json.load(open(cp)).get("answer")
        except Exception:
            pass

    # 选项判空要严：{} 和 {"A":"", "B":None} 都算填空题
    opts_d = {k: v for k, v in (q.get("options") or {}).items()
              if (v or "").strip()}
    passage = (q.get("passage") or "").strip()
    passage = (passage + "\n\n") if passage else ""
    if len(opts_d) >= 2:
        opts = "\n".join(f"({k}) {v}" for k, v in sorted(opts_d.items()))
        prompt = PROMPT_MC.format(subj=TESTNAME[part], passage=passage,
                                  stem=q.get("stem", ""), opts=opts)
        want_letter = True
    else:
        prompt = PROMPT_FR.format(subj=TESTNAME[part], passage=passage,
                                  stem=q.get("stem", ""))
        want_letter = False

    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 800}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = None
    for attempt in range(3):          # 网络抖动重试，实测单次失败率约 20%
        try:
            t = json.load(urllib.request.urlopen(req, timeout=session_timeout))
            t = t["choices"][0]["message"]["content"]
            break
        except Exception:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    if t is None:
        with _lock:
            _stats["fail"] += 1
        return None
    t = t.replace("```json", "").replace("```", "").strip()
    try:
        ans = json.loads(t[t.find("{"):t.rfind("}") + 1],
                         strict=False).get("answer", "")
    except Exception:
        with _lock:
            _stats["parse_fail"] += 1
        return None
    ans = str(ans).strip()
    if not ans:
        with _lock:
            _stats["parse_fail"] += 1
        return None
    # 格式校验：选择题必须回字母，填空题必须回数值。
    # 不符就判为无效，不写缓存——否则脏数据会被永久固化成"分歧"。
    is_letter = bool(re.fullmatch(r"[A-Da-d]", ans))
    if want_letter and not is_letter:
        with _lock:
            _stats["fmt_bad"] += 1
        return None
    if not want_letter and is_letter:
        with _lock:
            _stats["fmt_bad"] += 1
        return None
    json.dump({"answer": ans}, open(cp, "w"), ensure_ascii=False)
    with _lock:
        _stats["ok"] += 1
    return ans


def batch_key(q):
    """批次维度：来源 + 领域。SAT 无年份，来源是最接近'一份卷子'的粒度。"""
    src = q.get("sources") or []
    if isinstance(src, str):
        src = [src]
    return ("+".join(sorted(src)) or "?", q.get("domain") or "?")


def run_blind(part, qs, model, workers=8, tag=""):
    todo = [q for q in qs if str(q.get("answer") or "").strip()]
    print(f"[{part}{tag}] 盲答 {len(todo)} 题 (模型 {model.split('/')[-1]})",
          flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(blind_ask, q, part, model): q for q in todo}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                a = fut.result()
            except Exception:
                a = None
            if a:
                q["_blind"] = a
            done += 1
            if done % 100 == 0:
                print(f"   {done}/{len(todo)}  ok={_stats['ok']} "
                      f"cache={_stats['cached']} fail={_stats['fail']}",
                      flush=True)
    return todo


def report(rows, title="批次分歧率"):
    stat = defaultdict(lambda: [0, 0])
    for part, q in rows:
        b = q.get("_blind")
        if not b:
            continue
        k = (part,) + batch_key(q)
        stat[k][1] += 1
        if not same_answer(b, q.get("answer")):
            stat[k][0] += 1

    print(f"\n{title}")
    print(f"{'部分':<6}{'来源':<16}{'领域':<34}{'分歧':>5}{'样本':>6}{'分歧率':>8}")
    print("-" * 78)
    suspects = []
    tot_d = tot_n = 0
    for k, (d, n) in sorted(stat.items(), key=lambda x: -x[1][0] / max(x[1][1], 1)):
        tot_d += d
        tot_n += n
        if n < MIN_N:
            continue
        rate = d / n * 100
        flag = "  ⚠️" if rate >= SUSPECT_RATE else ""
        if flag:
            suspects.append((k, d, n, rate))
        print(f"{k[0]:<6}{k[1]:<16}{k[2]:<34}{d:>5}{n:>6}{rate:>7.0f}%{flag}")
    print("-" * 78)
    if tot_n:
        print(f"整体分歧率 {tot_d}/{tot_n} = {tot_d/tot_n*100:.1f}%")
    return suspects


def _toks(s):
    return set(re.findall(r"[a-zA-Z]{4,}", (s or "").lower()))


def check_mismatch(rows):
    """检测解析与题干串位。

    两个独立信号：
      A. 词汇重叠率 —— 解析里的实词与题干/passage 几乎不重叠，说明在讲另一道题
      B. 解析自述答案与答案键冲突 —— 纯文本比对，不依赖模型判断

    信号B 要注意误报：答案键 '11 or -7' 而解析写 '11' 其实一致，
    所以只在答案键本身是单值时才比。
    """
    print("\n" + "=" * 78)
    print("解析质量体检（与答案键无关的独立问题）")
    print("=" * 78)

    bad_overlap, bad_claim = [], []
    for part, q in rows:
        e = (q.get("explanation") or "").strip()
        if not e:
            continue
        st = _toks(q.get("stem")) | _toks(q.get("passage"))
        et = _toks(e)
        if st and et and len(st) >= 8:
            ov = len(st & et) / len(st)
            if ov < 0.06:
                bad_overlap.append((part, q, ov))

        key = str(q.get("answer") or "").strip()
        # 答案键含多值(如 '11 or -7' / '15, -5')时跳过，避免误报
        if key and not re.search(r"\bor\b|,", key):
            m = re.search(r"correct answer is\s*\**\s*\(?([A-D]\b|[\d.]+/?[\d.]*)",
                          e, re.I)
            if m:
                said = m.group(1).strip().rstrip(".")
                if not same_answer(said, key):
                    bad_claim.append((part, q, said, key))

    print(f"\n① 解析与题干几乎无词汇重叠（疑似串位）: {len(bad_overlap)} 道")
    for part, q, ov in bad_overlap[:5]:
        print(f"   [{part}] {(q.get('stem') or '')[:60]}")
        print(f"        解析却在讲: {(q.get('explanation') or '')[:70]}")

    print(f"\n② 解析自述答案与答案键冲突: {len(bad_claim)} 道")
    for part, q, said, key in bad_claim[:5]:
        print(f"   [{part}] 答案键={key!r} 解析说={said!r} | "
              f"{(q.get('stem') or '')[:55]}")

    return bad_overlap, bad_claim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if not KEY:
        print("缺少 SILICONFLOW_KEY")
        return

    rows = []
    for part, fn in FILES.items():
        qs = json.load(open(os.path.join(DOCS, fn)))
        if a.pilot:
            # 分层抽样：按 (来源,领域) 均匀取，保证每个批次都有样本
            buckets = defaultdict(list)
            for q in qs:
                if str(q.get("answer") or "").strip():
                    buckets[batch_key(q)].append(q)
            random.seed(7)
            per = max(3, a.pilot // max(len(buckets), 1))
            sel = []
            for b, lst in buckets.items():
                sel += random.sample(lst, min(per, len(lst)))
            qs = sel
        run_blind(part, qs, MODEL_A, workers=a.workers,
                  tag="-pilot" if a.pilot else "")
        rows += [(part, q) for q in qs]

    suspects = report(rows, "SAT 答案键体检" + ("（抽样）" if a.pilot else ""))
    print(f"\n统计: ok={_stats['ok']} cache={_stats['cached']} "
          f"fail={_stats['fail']} parse_fail={_stats['parse_fail']} "
          f"fmt_bad={_stats['fmt_bad']}")
    if suspects:
        print(f"\n⚠️ {len(suspects)} 个批次分歧率 ≥{SUSPECT_RATE:.0f}%")

    # 解析串位是独立于答案键的问题，全量时一并报
    if not a.pilot:
        all_rows = []
        for part, fn in FILES.items():
            for q in json.load(open(os.path.join(DOCS, fn))):
                all_rows.append((part, q))
        check_mismatch(all_rows)


if __name__ == "__main__":
    main()
