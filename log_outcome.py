#!/usr/bin/env python3
"""
Close the learning loop: pair the pre-build PREDICTION with the ACTUAL build
outcome and append one row to logs/outcomes.csv.

Runs in the `log-outcome` job AFTER the lanes finish. The lane result is passed
in via the OUTCOME env var ('passed' / 'failed'), computed from the GitHub job
results in the workflow.

This CSV is the retraining dataset. Crucially, rows where explored=1 (or a full
lane ran) carry a TRUE outcome; a fast-tracked LOW commit that was NOT explored
has outcome='unobserved' — that missing label is exactly the censored-feedback
signal, and the EXPLORE_RATE in predict.py is what keeps enough true labels
flowing to retrain safely.
"""
import os, csv, json, datetime

LOG = "logs/outcomes.csv"


def main():
    with open("prediction.json") as f:
        rec = json.load(f)

    outcome = os.environ.get("OUTCOME", "unobserved").strip() or "unobserved"
    # A fast-tracked LOW commit that wasn't explored never ran the full pipeline,
    # so its true pass/fail is unknown -> record as censored.
    if rec.get("lane_run") == "LOW" and not rec.get("explored"):
        outcome = "unobserved"

    row = {
        "logged_at": datetime.datetime.utcnow().isoformat(),
        "commit": rec.get("commit"),
        "score": rec.get("score"),
        "predicted_lane": rec.get("predicted_lane"),
        "lane_run": rec.get("lane_run"),
        "explored": rec.get("explored"),
        "outcome": outcome,                       # passed / failed / unobserved
        "total_files_changed": rec.get("total_files_changed"),
        "total_churn": rec.get("total_churn"),
        "git_diff_src_churn": rec.get("git_diff_src_churn"),
        "is_docs_only": rec.get("is_docs_only"),
    }

    os.makedirs("logs", exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"Logged: lane={row['lane_run']} outcome={row['outcome']} "
          f"explored={row['explored']} -> {LOG}")


if __name__ == "__main__":
    main()
