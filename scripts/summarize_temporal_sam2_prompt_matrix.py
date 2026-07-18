#!/usr/bin/env python3
"""Summarize frozen automatic-prompt SAM 2 outputs without reading GT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.summarize_temporal_sam2_outputs import (
    summarize_frozen_sam2_outputs,
    write_sam2_output_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-matrix-summary", required=True, type=Path)
    parser.add_argument("--sam2-output-root", required=True, type=Path)
    args = parser.parse_args()
    result = summarize_frozen_sam2_outputs(args.prompt_matrix_summary, args.sam2_output_root)
    write_sam2_output_summary(result, args.sam2_output_root)
    print(json.dumps({
        "sample_count": result["sample_count"],
        "cuda_oom_count": result["cuda_oom_count"],
        "selected_highest_score_count": result["selected_highest_score_count"],
        "ground_truth_used": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
