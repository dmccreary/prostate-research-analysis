"""
qwen_classifier.py
Classifies prostate cancer abstracts using Qwen (via Ollama API) and
produces a full classification report.

Usage:
    python qwen_classifier.py                          # run on all val_set rows
    python qwen_classifier.py --n 20                   # first N rows (stratified)
    python qwen_classifier.py --input data/test_set.csv
    python qwen_classifier.py --input data/test_set.csv --output data/qwen_test_results.csv
"""

import argparse
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

load_dotenv("Pubmed_API.env")

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434/api/generate"
MODEL             = "qwen3.6:27b"
DEFAULT_DATA_PATH = Path("data/val_set.csv")
THINK             = True   # set True to enable Qwen chain-of-thought (slower)
CHECKPOINT_EVERY  = 25     # flush partial results to the output CSV every N articles
EUTILS_BASE       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# ── Publication-type pre-filter ──────────────────────────────────────────────
EXCLUDE_TYPES   = {t.lower() for t in [
    "Case Reports", "Systematic Review", "Meta-Analysis", "Observational Study",
    "Retracted Publication", "Comment", "Practice Guideline",
    "Historical Article", "Consensus Statement", "Letter", "Editorial",
    "Conference Proceedings", "News", "Guideline", "Portrait", "Lecture",
    "Biography", "Twin Study", "Duplicate Publication", "Technical Report",
    "Clinical Conference", "Introductory Journal Article", "Video-Audio Media",
    "Corrected and Republished Article", "Patient Education Handout", "Interview",
    "Autobiography", "Retraction Notice",
]}


def matched_excluded_types(types_str: str) -> list:
    """Return the article's publication types that are on the exclusion list."""
    return [t.strip() for t in str(types_str).split(";")
            if t.strip().lower() in EXCLUDE_TYPES]


# Advanced/castration-resistant disease acronyms in the title/abstract mark a
# study as out-of-scope (not a localized risk-group cohort). Word-boundaried so
# "mCRPC" etc. match cleanly and nothing inside other words does.
# KEEP IN SYNC WITH search_articles.py.
EXCLUDE_KEYWORDS_RE = re.compile(r"\b(?:m?CRPC|m?HRPC|m?CSPC|m?HSPC)\b", re.IGNORECASE)


def matched_excluded_keywords(text: str) -> list:
    """Return the excluded advanced-disease acronyms found in the text."""
    return sorted({m.group(0).upper() for m in EXCLUDE_KEYWORDS_RE.finditer(str(text))})


# Title-ONLY keywords: a study whose TITLE contains these is *about* an
# out-of-scope topic (e.g. salvage therapy after primary-treatment failure).
# Checked against the title only — these words in the abstract are fine, because
# in-scope primary-treatment studies report them as outcomes. (0 positives carry
# "salvage" in the title; it would otherwise drop 6 positives via the abstract.)
# KEEP IN SYNC WITH search_articles.py.
EXCLUDE_TITLE_KEYWORDS_RE = re.compile(r"\bsalvage", re.IGNORECASE)


def matched_excluded_title_keywords(title: str) -> list:
    """Return the title-only excluded keywords found in the title."""
    return sorted({m.group(0).lower() for m in EXCLUDE_TITLE_KEYWORDS_RE.finditer(str(title))})


def fetch_publication_types(pmids: list, api_key) -> dict:
    """Fetch {pmid: 'type; type; ...'} from PubMed efetch (batches of 200)."""
    out, delay = {}, (0.11 if api_key else 0.34)
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        params = {"db": "pubmed", "id": ",".join(batch),
                  "rettype": "xml", "retmode": "xml"}
        if api_key:
            params["api_key"] = api_key
        try:
            resp = requests.get(EUTILS_BASE + "efetch.fcgi", params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for node in (root.findall(".//PubmedArticle")
                         + root.findall(".//PubmedBookArticle")):
                pmid = node.findtext(".//PMID", "")
                types = [pt.text.strip() for pt in node.findall(".//PublicationType")
                         if pt.text and pt.text.strip()]
                out[pmid] = "; ".join(types)
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"  [WARN] efetch publication types failed at batch {i}: {exc}")
        time.sleep(delay)
    return out

SYSTEM_PROMPT = """\
You are a biomedical literature screening assistant.
Your job is to decide whether a PubMed abstract is RELEVANT to a systematic \
review of prostate cancer treatment outcomes.

An abstract is POSITIVE (relevant) if it reports:
- Discuss prostate cancer treatment
- A minimum sample size of 50 patients
- A minimum median follow up time for any time of therapy or risk-group of exactly at least 5 years \
or 60 months. If the median follow up time is not stated, assume that this criteria is met.
- Treated with a specific modality (e.g. radical prostatectomy, radiation \
therapy, brachytherapy, ADT)
- With quantitative survival or biochemical outcomes (e.g. BRFS, OS, MFS, CSS, PSA)

Otherwise, it should be classified as NEGATIVE.

Respond with ONLY a single JSON object, containing the label and your justification:
{"label": "Positive", "reason": "<why it is relevant>"}
or
{"label": "Negative", "reason": "<why it is not relevant>"}
"""

# ── Helpers ──────────────────────────────────────────────────────────────────
def classify_abstract(abstract: str) -> tuple[str, str]:
    """Send one abstract to Ollama/Qwen and return (label, reason).

    label  is 'Positive' / 'Negative' / 'Unknown' / 'Error';
    reason is the model's brief justification ('' if none/parse failed).
    """
    payload = {
        "model": MODEL,
        "prompt": f"Abstract:\n{abstract}\n\nClassify this abstract.",
        "system": SYSTEM_PROMPT,
        "stream": False,
        "think": THINK,
        "options": {
            "temperature": 0.0,
            # Thinking generates a chain-of-thought BEFORE the JSON answer, and
            # thinking tokens count against this budget. Too few tokens and the
            # answer is truncated mid-`reason` (unterminated JSON) or `response`
            # comes back empty. Give thinking generous headroom.
            "num_predict": 8192 if THINK else 256,
        },
    }

    raw = ""
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        raw  = (body.get("response", "") or "").strip()

        # Newer Ollama returns reasoning in a separate `thinking` field, but some
        # versions embed it as <think>...</think> inside `response`. If `response`
        # is empty, fall back to whatever is in `thinking`.
        if not raw:
            raw = (body.get("thinking", "") or "").strip()

        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = clean.strip("`").strip()
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()

        # Find every flat {...} object and use the LAST one that parses to a dict
        # with a "label" key. When thinking leaks into the output it contains
        # several draft objects; the model's final answer is the last one. (A
        # greedy first-to-last match would span multiple objects and make
        # json.loads raise "Extra data".)
        data = None
        for obj in reversed(re.findall(r"\{[^{}]*\}", clean, flags=re.DOTALL)):
            try:
                parsed = json.loads(obj)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "label" in parsed:
                data = parsed
                break
        if data is None:                      # last resort: whole cleaned string
            data = json.loads(clean)

        label  = str(data.get("label", "")).strip().capitalize()
        reason = str(data.get("reason", data.get("Reason", ""))).strip()
        label  = label if label in ("Positive", "Negative") else "Unknown"
        return label, reason
    except Exception as e:
        # Salvage: if the JSON is truncated/invalid (e.g. a long reason got cut
        # off mid-string), the label still appears early and intact. Recover it
        # by regex rather than dropping the whole row to "Error".
        m = re.search(r'"label"\s*:\s*"(positive|negative)"', raw, re.IGNORECASE)
        if m:
            label = m.group(1).capitalize()
            rmatch = re.search(r'"[Rr]eason"\s*:\s*"(.*)', raw, re.DOTALL)
            reason = rmatch.group(1).strip() if rmatch else ""
            print(f"  [WARN] Salvaged label={label} from malformed JSON ({e})")
            return label, reason + " [truncated]" if reason else reason
        print(f"  [ERROR] {e} | raw={raw!r}")
        return "Error", ""


def build_report(df: pd.DataFrame, input_name: str = "") -> str:
    """Build a detailed classification report string."""
    valid = df[df["qwen_label"].isin(["Positive", "Negative"])].copy()
    skipped = len(df) - len(valid)

    true = valid["label"].str.strip().str.capitalize()
    pred = valid["qwen_label"]

    # Confusion matrix
    cm = confusion_matrix(true, pred, labels=["Positive", "Negative"])
    tn, fp, fn, tp = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]

    total    = len(valid)
    accuracy = accuracy_score(true, pred)

    # sklearn classification report (per-class precision/recall/F1)
    sk_report = classification_report(
        true, pred, labels=["Positive", "Negative"],
        target_names=["Positive", "Negative"],
        digits=4,
    )

    # Balanced accuracy = 0.5 * (Sensitivity + Specificity)
    # This equals the AUC-ROC when only hard binary labels are available
    # (no continuous probability output from Qwen).
    try:
        binary_true = (true == "Positive").astype(int)
        binary_pred = (pred == "Positive").astype(int)
        auc = roc_auc_score(binary_true, binary_pred)
        auc_line = f"  Balanced Accuracy (= hard-label AUC): {auc:.4f}"
    except Exception:
        auc_line = "  Balanced Accuracy: N/A (only one class present)"

    # Confusion matrix table
    cm_str = (
        "\nConfusion Matrix (rows=true, cols=predicted):\n"
        f"                Pred Positive  Pred Negative\n"
        f"  True Positive      {tp:>6}         {fn:>6}\n"
        f"  True Negative      {fp:>6}         {tn:>6}\n"
    )

    lines = [
        "=" * 60,
        f"  Qwen Classification Report -- {input_name}",
        f"  Model   : {MODEL}",
        f"  Samples : {total} evaluated  ({skipped} skipped - no abstract)",
        "=" * 60,
        "",
        "Per-class metrics (sklearn):",
        sk_report,
        "-" * 60,
        "Summary:",
        f"  Accuracy        : {accuracy:.4f}",
        auc_line,
        f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        cm_str,
    ]

    if skipped > 0:
        no_abs = df[~df["qwen_label"].isin(["Positive", "Negative"])][
            ["pmid", "label", "qwen_label"]
        ]
        lines += [
            f"Skipped rows ({skipped}):",
            no_abs.to_string(index=False),
            "",
        ]

    # Misclassified rows
    wrong = valid[true != pred][["pmid", "title", "label", "qwen_label"]]
    if not wrong.empty:
        lines += [
            "-" * 60,
            f"Misclassified ({len(wrong)}):",
            wrong.to_string(index=False),
            "",
        ]
    else:
        lines += ["-" * 60, "No misclassifications!", ""]

    lines.append("=" * 60)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Classify abstracts with Qwen via Ollama."
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_DATA_PATH),
        help="Input CSV with an 'abstract' and 'label' column (default: data/val_set.csv).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path. Defaults to data/qwen_<stem>_results.csv.",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Stratified sample size (half Positive, half Negative). "
             "Omit to run all rows.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing output CSV and rerun from scratch "
             "(the existing results file is overwritten).",
    )
    args = parser.parse_args()

    data_path = Path(args.input)
    stem      = data_path.stem                                   # e.g. "test_set"
    out_csv   = Path(args.output) if args.output else Path(f"data/qwen_{stem}_results.csv")
    rep_path  = Path(f"data/qwen_{stem}_classification_report.txt")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if args.n is not None:
        half = args.n // 2
        pos  = df[df["label"].str.strip() == "Positive"].head(half)
        neg  = df[df["label"].str.strip() == "Negative"].head(args.n - half)
        df   = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Publication-type pre-filter: ensure an `article_types` column ───────
    # Use it if already present (e.g. from screen_article_type.py); otherwise
    # fetch the types from PubMed so the filter can be applied before Qwen.
    def _norm_pmid(p):
        p = str(p).strip()
        return p[:-2] if p.endswith(".0") else p

    if "article_types" not in df.columns:
        if "pmid" in df.columns:
            api_key = os.environ.get("NCBI_API_KEY")
            pmids = [_norm_pmid(p) for p in df["pmid"] if pd.notna(p) and str(p).strip()]
            print(f"[filter] Fetching publication types for {len(pmids)} PMIDs from PubMed "
                  f"({'API key' if api_key else 'no key'}) ...")
            tmap = fetch_publication_types(pmids, api_key)
            df["article_types"] = df["pmid"].apply(lambda p: tmap.get(_norm_pmid(p), ""))
        else:
            print("[filter] No 'article_types' or 'pmid' column — publication-type filter disabled.")
            df["article_types"] = ""

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Optionally start fresh: drop any existing results so resume is bypassed
    if args.no_resume and out_csv.exists():
        out_csv.unlink()
        print(f"[no-resume] removed existing {out_csv} — rerunning from scratch.")

    # ── Resume: collect already-done PMIDs from an existing output CSV ───────
    done_pmids: set[str] = set()
    existing_fieldnames = None
    if out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv, encoding="utf-8-sig")
        prev.columns = prev.columns.str.strip()
        existing_fieldnames = list(prev.columns)
        if "pmid" in prev.columns:
            done_pmids = set(prev["pmid"].astype(str))
            print(f"[resume] {len(done_pmids)} rows already in {out_csv} — skipping them.")

    todo = (df[~df["pmid"].astype(str).isin(done_pmids)]
            if done_pmids and "pmid" in df.columns else df)

    # Output column order: reuse the existing header when resuming, else the
    # input columns plus the three appended Qwen fields.
    fieldnames = existing_fieldnames or (
        list(df.columns) + [c for c in ("qwen_label", "qwen_reason", "qwen_seconds")
                            if c not in df.columns])

    def append_rows(rows: list) -> None:
        """True append. Writes the header (+BOM) only when creating the file;
        subsequent appends add rows without re-writing the file."""
        if not rows:
            return
        new_file = (not out_csv.exists()) or out_csv.stat().st_size == 0
        with out_csv.open("a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    print(f"Running Qwen ({MODEL}) on {len(todo)} abstracts from {data_path} "
          f"({len(done_pmids)} already done) ...\n")

    buffer: list = []
    for idx, (_, row) in enumerate(todo.iterrows(), start=1):
        abstract   = str(row.get("abstract", "")).strip()
        pmid_disp  = str(row.get("pmid",     ""))
        title_disp = str(row.get("title",    ""))[:55]
        truth      = str(row.get("label",    "")).strip()

        excluded_types = matched_excluded_types(row.get("article_types", ""))
        excluded_kw    = matched_excluded_keywords(f"{row.get('title','')} {abstract}")
        excluded_title = matched_excluded_title_keywords(row.get("title", ""))
        if excluded_types:
            # Filtered out by publication type — classify Negative, skip Qwen
            qwen_label, qwen_reason, elapsed = "Negative", \
                f"[filtered] excluded publication type: {', '.join(excluded_types)}", 0.0
        elif excluded_kw:
            # Filtered out by advanced-disease keyword — classify Negative, skip Qwen
            qwen_label, qwen_reason, elapsed = "Negative", \
                f"[filtered] excluded keyword: {', '.join(excluded_kw)}", 0.0
        elif excluded_title:
            # Filtered out by title keyword (study is ABOUT it) — classify Negative
            qwen_label, qwen_reason, elapsed = "Negative", \
                f"[filtered] excluded title keyword: {', '.join(excluded_title)}", 0.0
        elif not abstract or abstract.lower() == "nan":
            qwen_label, qwen_reason, elapsed = "NoAbstract", "", 0.0
        else:
            t0 = time.perf_counter()
            qwen_label, qwen_reason = classify_abstract(abstract)
            elapsed = time.perf_counter() - t0

        tag = "OK" if qwen_label == truth else "XX"
        print(f"[{idx:>4}/{len(todo)}] {tag}  {elapsed:6.1f}s  "
              f"truth={truth:<9} qwen={qwen_label:<9} | PMID {pmid_disp:<9} | {title_disp}")

        buffer.append({**row.to_dict(), "qwen_label": qwen_label,
                       "qwen_reason": qwen_reason, "qwen_seconds": round(elapsed, 2)})

        # Checkpoint: append the buffered rows to disk every N articles
        if idx % CHECKPOINT_EVERY == 0:
            append_rows(buffer)
            print(f"  [checkpoint] appended {len(buffer)} rows "
                  f"({idx}/{len(todo)}) -> {out_csv}")
            buffer.clear()

        time.sleep(0.1)

    append_rows(buffer)          # flush the final partial batch (< CHECKPOINT_EVERY)
    buffer.clear()
    print(f"\nResults saved  -> {out_csv}")

    # Reload the full set (resumed + new) for the timing summary and report
    out_df = pd.read_csv(out_csv, encoding="utf-8-sig")
    out_df.columns = out_df.columns.str.strip()

    # Per-datapoint generation-time summary (only rows actually sent to Qwen)
    timed = (out_df.loc[out_df["qwen_seconds"] > 0, "qwen_seconds"]
             if "qwen_seconds" in out_df.columns else pd.Series(dtype=float))
    if len(timed):
        print(f"\nQwen generation time per datapoint (n={len(timed)}):  "
              f"mean={timed.mean():.1f}s  median={timed.median():.1f}s  "
              f"min={timed.min():.1f}s  max={timed.max():.1f}s  total={timed.sum():.1f}s")

    # Build and print report
    report = build_report(out_df, input_name=data_path.name)
    print("\n" + report)

    # Save report to file
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved   -> {rep_path}")


if __name__ == "__main__":
    main()
