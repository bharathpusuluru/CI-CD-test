#!/usr/bin/env python3
"""
Adaptive CI/CD — pre-build risk predictor (runs INSIDE GitHub Actions).

On every push it:
  1. Extracts pre-build features from the pushed commit using `git`.
  2. Loads the model trained in Colab (model.joblib).
  3. Predicts failure risk, maps it to a pipeline LANE (LOW/MED/HIGH).
  4. Writes lane/risk/score to $GITHUB_OUTPUT so downstream jobs can adapt.
  5. Prints an explanation for the developer.

Feature parity note: many TravisTorrent features (sloc, team_size, project
history, etc.) aren't available for a single live commit. We compute the
git-derivable ones and leave the rest as NaN — the model's built-in imputer
fills them with the training medians. So live predictions lean on the
change-shape features (churn, files, docs-only), which is honest and fine.
"""
import os, subprocess, sys, json, random, datetime
import numpy as np
import pandas as pd
import joblib

# ---- policy thresholds (from your Colab Pareto analysis) ----
LOW_MAX  = 0.10   # p < 0.10  -> LOW  (fast lane)
HIGH_MIN = 0.20   # p >= 0.20 -> HIGH (extended + manual gate); between -> MED

# ---- exploration: occasionally force a full run on a LOW commit so we still
#      observe its true outcome (fixes the censored-feedback problem, RQ3) ----
EXPLORE_RATE = 0.15

DOC_EXT  = {".md", ".rst", ".txt", ".adoc"}
SRC_EXT  = {".py", ".java", ".js", ".ts", ".rb", ".go", ".c", ".cpp", ".cc",
            ".h", ".hpp", ".cs", ".php", ".scala", ".kt", ".rs", ".swift"}


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def get_diff():
    """Return (files, added, deleted) for the pushed commit vs its parent."""
    parent = sh(["git", "rev-parse", "HEAD~1"])
    base = parent if parent else sh(
        ["git", "hash-object", "-t", "tree", "/dev/null"])  # empty tree (first commit)
    numstat = sh(["git", "diff", "--numstat", base, "HEAD"])
    files = []
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            add, dele, path = parts
            files.append((path,
                          int(add) if add.isdigit() else 0,
                          int(dele) if dele.isdigit() else 0))
    return files


def classify(path):
    ext = os.path.splitext(path)[1].lower()
    low = path.lower()
    if "test" in low or "spec" in low:
        return "test"
    if ext in DOC_EXT or low.startswith("docs/"):
        return "doc"
    if ext in SRC_EXT:
        return "src"
    return "other"


def build_features(feature_names):
    files = get_diff()
    src_churn = test_churn = 0
    n_add = n_del = n_mod = 0
    src_files = doc_files = other_files = test_add = test_del = 0
    for path, add, dele in files:
        kind = classify(path)
        churn = add + dele
        if kind == "test":
            test_churn += churn
            if dele == 0 and add > 0: test_add += 1
            elif add == 0 and dele > 0: test_del += 1
        elif kind == "src":
            src_churn += churn; src_files += 1
        elif kind == "doc":
            doc_files += 1
        else:
            other_files += 1
        if dele == 0 and add > 0: n_add += 1
        elif add == 0 and dele > 0: n_del += 1
        else: n_mod += 1

    total_churn = src_churn + test_churn
    total_files = len(files)
    is_docs_only = int(src_files == 0 and doc_files > 0 and total_files > 0)

    # git-derivable features; everything else -> NaN (imputed by the model)
    known = {
        "git_diff_src_churn": src_churn,
        "git_diff_test_churn": test_churn,
        "total_churn": total_churn,
        "gh_diff_files_added": n_add,
        "gh_diff_files_deleted": n_del,
        "gh_diff_files_modified": n_mod,
        "total_files_changed": total_files,
        "gh_diff_tests_added": test_add,
        "gh_diff_tests_deleted": test_del,
        "gh_diff_src_files": src_files,
        "gh_diff_doc_files": doc_files,
        "gh_diff_other_files": other_files,
        "gh_num_commits_in_push": 1,
        "is_docs_only": is_docs_only,
        "gh_is_pr": 0,
    }
    row = {f: known.get(f, np.nan) for f in feature_names}
    return pd.DataFrame([row])[feature_names], known, is_docs_only


def explain(known, p, lane):
    reasons = []
    if known["is_docs_only"]:
        reasons.append("documentation-only change (low risk)")
    if known["total_churn"] > 300:
        reasons.append(f"large code churn ({known['total_churn']} lines)")
    if known["total_files_changed"] > 15:
        reasons.append(f"many files touched ({known['total_files_changed']})")
    if known["gh_diff_src_files"] > 0 and known["gh_diff_tests_added"] == 0 \
            and known["git_diff_src_churn"] > 100:
        reasons.append("substantial source change with no new tests")
    if not reasons:
        reasons.append("change shape resembles historically low-risk commits")
    return reasons


def risk_meter(p, width=20):
    """ASCII risk bar, e.g. [█████████░░░░░░░░░░░] 45%"""
    fill = int(round(p * width))
    return "`[" + "█" * fill + "░" * (width - fill) + f"]` **{p*100:.0f}%**"


def build_summary(p, risk, lane, reasons, known):
    emoji = {"LOW": "🟢", "MED": "🟡", "HIGH": "🔴"}[lane]
    lane_action = {
        "LOW":  "**Fast lane** — lint + unit tests only (skips expensive stages)",
        "MED":  "**Standard pipeline** — full build + unit + integration tests",
        "HIGH": "**Extended validation** — full tests + security scan + manual approval gate",
    }[lane]
    reason_md = "\n".join(f"- {r}" for r in reasons)
    return f"""## {emoji} Pre-Build Risk Prediction

| | |
|---|---|
| **Risk level** | {emoji} **{risk}** |
| **Risk score** | {risk_meter(p)} |
| **Pipeline lane** | `{lane}` |
| **Files changed** | {known['total_files_changed']} |
| **Code churn** | {known['total_churn']} lines |

### 🔀 Selected pipeline
{lane_action}

### 💡 Why this prediction
{reason_md}

<sub>Predicted by the Adaptive CI/CD risk model before any build stage ran.</sub>
"""


def main():
    bundle = joblib.load("model.joblib")
    model, feats = bundle["model"], bundle["features"]

    X, known, is_docs = build_features(feats)
    p = float(model.predict_proba(X)[:, 1][0])

    predicted_lane = "LOW" if p < LOW_MAX else ("HIGH" if p >= HIGH_MIN else "MED")

    # exploration: sometimes upgrade a LOW commit to a full run to keep ground truth flowing
    explored = int(predicted_lane == "LOW" and random.random() < EXPLORE_RATE)
    lane = "MED" if explored else predicted_lane

    risk = {"LOW": "Low", "MED": "Medium", "HIGH": "High"}[lane]
    reasons = explain(known, p, lane)
    if explored:
        reasons.append("selected for exploration (full run to verify a low-risk prediction)")

    # ---- human-readable output in the Actions log ----
    print("=" * 52)
    print(f"  PRE-BUILD RISK PREDICTION")
    print(f"  Risk score : {p:.3f}")
    print(f"  Risk level : {risk}")
    print(f"  Pipeline   : {lane} lane")
    print(f"  Why        : " + "; ".join(reasons))
    print("=" * 52)

    summary_md = build_summary(p, risk, lane, reasons, known)

    # ---- rich visual report card on the Actions run summary page ----
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(summary_md)

    # ---- save summary so a later job can post it as a PR comment ----
    with open("risk_summary.md", "w") as f:
        f.write(summary_md)

    # ---- record the prediction so the log-outcome job can pair it with the
    #      real build result (closes the learning loop) ----
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "commit": os.environ.get("GITHUB_SHA", sh(["git", "rev-parse", "HEAD"])),
        "score": round(p, 4),
        "predicted_lane": predicted_lane,
        "lane_run": lane,
        "explored": explored,
        **{k: (None if pd.isna(v) else v) for k, v in known.items()},
    }
    with open("prediction.json", "w") as f:
        json.dump(record, f)

    # ---- machine-readable output for downstream jobs ----
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"lane={lane}\n")
            f.write(f"risk={risk}\n")
            f.write(f"score={p:.3f}\n")
            f.write(f"explored={explored}\n")


if __name__ == "__main__":
    main()
