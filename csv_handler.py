"""
CSV HANDLER — saves all results to one CSV file
"""

import csv
import os
from config import CSV_OUTPUT_PATH

HEADERS = [
    "folder_name",
    "file_name",
    "blur",
    "contrast",
    "brightness",
    "resolution",
    "dpi",
    "noise",
    "skew_angle_deg",
    "shadow_coverage_pct",
    "orientation",
    "text_clarity",
    "perspective_distortion_pct",
    "overall_quality_score",
    "overall_quality",
]


class CSVHandler:

    def __init__(self):
        self.path        = CSV_OUTPUT_PATH
        self.file_exists = os.path.exists(self.path)

    def setup(self):
        if not self.file_exists:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HEADERS)
                writer.writeheader()
            print(f"  ✅ Created: {self.path}")
        else:
            print(f"  ✅ Appending to: {self.path}")

    def write(self, result):
        row = {h: result.get(h, "") for h in HEADERS}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writerow(row)

    def close(self):
        print(f"\n  ✅ All results saved to: {self.path}")