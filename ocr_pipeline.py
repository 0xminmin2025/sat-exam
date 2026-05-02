#!/usr/bin/env python3
"""
SAT PDF OCR Pipeline
PDF → 图片 → 百炼 qwen-vl-max → 结构化JSON

支持并发处理，自动跳过已处理的页面。
"""

import json, os, sys, time, base64, urllib.request, urllib.error
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# === Config ===
DASHSCOPE_KEY = "sk-b77970c0fda84e6e8a343bf31ff6183d"
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-vl-max"
DPI = 200
MAX_WORKERS = 6  # Parallel OCR requests
OUTPUT_DIR = Path(os.path.expanduser("~/sat-exam/ocr_output"))
PROGRESS_FILE = OUTPUT_DIR / "_progress.json"

# === OCR Prompt ===
OCR_PROMPT = """You are an OCR expert extracting SAT test questions from images.

Extract ALL questions from this page. Each question should be a separate JSON object.

For EACH question found, output:
{
  "question_id": "the ID shown (e.g. 91e7ea5e), or null if not shown",
  "test": "Math" or "Reading and Writing",
  "domain": "the domain if shown, or infer from content",
  "skill": "the skill if shown, or infer from content",
  "difficulty": "Easy" or "Medium" or "Hard" (from the squares ■□□ / ■■□ / ■■■), or null,
  "passage": "the reading passage text if any, null for math",
  "stem": "the question text, including any function definitions like h(x) = 2(x-4)^2 - 32",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."} or null if not multiple choice,
  "answer": "the correct answer letter, or the numeric answer",
  "explanation": "the explanation text if shown on this page, null otherwise"
}

Rules:
- For math formulas, use LaTeX: $h(x) = 2(x-4)^2 - 32$
- If the page has MULTIPLE questions, return an array of objects
- If the page is a cover page, table of contents, or has no questions, return: []
- If the page only has answer explanations (no new question), still extract with the question_id
- For Chinese text mixed with English, preserve both languages
- Return ONLY valid JSON (array of objects), no markdown fences"""

def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}

def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2))

def pdf_to_images(pdf_path):
    """Convert PDF pages to base64-encoded PNG images."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=DPI)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        images.append({
            "page_num": page_num + 1,
            "base64": img_b64,
            "width": pix.width,
            "height": pix.height
        })
    doc.close()
    return images

def ocr_page(img_b64, retries=3):
    """Send image to Dashscope for OCR, return parsed JSON."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_KEY}"
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": OCR_PROMPT}
            ]}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                DASHSCOPE_URL,
                data=json.dumps(data).encode(),
                headers=headers
            )
            resp = urllib.request.urlopen(req, timeout=90)
            result = json.loads(resp.read())
            content = result['choices'][0]['message']['content']
            tokens = result.get('usage', {})

            # Parse JSON from response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            questions = json.loads(content)
            if isinstance(questions, dict):
                questions = [questions]

            return questions, tokens

        except json.JSONDecodeError as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None, {"error": f"JSON parse error: {str(e)}", "raw": content[:500]}

        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
            elif attempt < retries - 1:
                time.sleep(2)
            else:
                return None, {"error": f"HTTP {e.code}: {body}"}

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None, {"error": str(e)}

    return None, {"error": "Max retries exceeded"}

def process_pdf(pdf_path, progress):
    """Process a single PDF file."""
    rel_path = str(pdf_path)
    if rel_path in progress and progress[rel_path].get("status") == "done":
        return progress[rel_path].get("questions", []), 0, 0

    print(f"\n📄 Processing: {os.path.basename(pdf_path)}")
    images = pdf_to_images(pdf_path)
    print(f"   {len(images)} pages")

    all_questions = []
    total_tokens = 0
    pages_processed = 0

    for img in images:
        page_key = f"{rel_path}:p{img['page_num']}"
        if page_key in progress and progress[page_key].get("status") == "done":
            cached = progress[page_key].get("questions", [])
            all_questions.extend(cached)
            continue

        sys.stdout.write(f"   p{img['page_num']}/{len(images)}...")
        sys.stdout.flush()
        questions, tokens = ocr_page(img["base64"])
        pages_processed += 1

        if questions is not None:
            # Tag each question with source info
            for q in questions:
                q["_source_file"] = os.path.basename(pdf_path)
                q["_source_page"] = img["page_num"]

            all_questions.extend(questions)
            t = tokens.get("total_tokens", 0)
            total_tokens += t

            progress[page_key] = {
                "status": "done",
                "questions": questions,
                "tokens": t
            }
            sys.stdout.write(f" {len(questions)}q\n")
            sys.stdout.flush()
        else:
            progress[page_key] = {
                "status": "error",
                "error": tokens
            }
            print(f"   ❌ Page {img['page_num']} error: {tokens}")

        # Rate limiting
        time.sleep(0.5)

    progress[rel_path] = {
        "status": "done",
        "total_pages": len(images),
        "questions_found": len(all_questions),
        "total_tokens": total_tokens
    }

    q_count = len(all_questions)
    print(f"   ✅ {q_count} questions extracted, {total_tokens} tokens")
    return all_questions, pages_processed, total_tokens

def process_directory(dir_path, output_name=None):
    """Process all PDFs in a directory."""
    dir_path = Path(dir_path)
    if not output_name:
        output_name = dir_path.name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    # Find all PDFs
    pdfs = sorted(dir_path.rglob("*.pdf"))
    print(f"\n{'='*60}")
    print(f"📚 Processing: {dir_path.name}")
    print(f"   {len(pdfs)} PDF files found")
    print(f"{'='*60}")

    all_questions = []
    total_pages = 0
    total_tokens = 0

    for i, pdf in enumerate(pdfs):
        print(f"\n[{i+1}/{len(pdfs)}]", end="")
        questions, pages, tokens = process_pdf(str(pdf), progress)
        all_questions.extend(questions)
        total_pages += pages
        total_tokens += tokens

        # Save progress every PDF (not every 5)
        save_progress(progress)

    save_progress(progress)

    # Filter out empty results and save
    valid_questions = [q for q in all_questions if q and (q.get("stem") or q.get("question_id"))]

    output_file = OUTPUT_DIR / f"{output_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(valid_questions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"🎉 完成: {output_name}")
    print(f"   PDF数: {len(pdfs)}")
    print(f"   处理页数: {total_pages}")
    print(f"   提取题目: {len(valid_questions)}")
    print(f"   总tokens: {total_tokens}")
    cost = total_tokens * 0.005 / 1000  # rough average
    print(f"   估算成本: ¥{cost:.2f}")
    print(f"   输出: {output_file}")
    print(f"{'='*60}")

    return valid_questions

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.path.expanduser("~/sat-exam/raw/SAT-文法分类题库")

    process_directory(target)
