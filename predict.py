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
import os, subprocess, sys
import numpy as np
import pandas as pd
import joblib

# ---- policy thresholds (from your Colab Pareto analysis) ----
LOW_MAX  = 0.10   # p < 0.10  -> LOW  (fast lane)
HIGH_MIN = 0.20   # p >= 0.20 -> HIGH (extended + manual gate); between -> MED

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


def main():
    bundle = joblib.load("model.joblib")
    model, feats = bundle["model"], bundle["features"]

    X, known, is_docs = build_features(feats)
    p = float(model.predict_proba(X)[:, 1][0])

    lane = "LOW" if p < LOW_MAX else ("HIGH" if p >= HIGH_MIN else "MED")
    risk = {"LOW": "Low", "MED": "Medium", "HIGH": "High"}[lane]
    reasons = explain(known, p, lane)

    # ---- human-readable output in the Actions log ----
    print("=" * 52)
    print(f"  PRE-BUILD RISK PREDICTION")
    print(f"  Risk score : {p:.3f}")
    print(f"  Risk level : {risk}")
    print(f"  Pipeline   : {lane} lane")
    print(f"  Why        : " + "; ".join(reasons))
    print("=" * 52)

    # ---- machine-readable output for downstream jobs ----
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"lane={lane}\n")
            f.write(f"risk={risk}\n")
            f.write(f"score={p:.3f}\n")


if __name__ == "__main__":
    main()
