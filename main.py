"""
MAIN — Document Quality Engine

HOW TO RUN:
  Step 1 — Edit config.py: add all your folder paths
  Step 2 — Delete old quality_scores.csv if it exists
  Step 3 — Run: python main.py
"""

from document_quality_calculator import (
    scan_folders,
    analyze_document,
    print_scores,
)
from csv_handler import CSVHandler
from config import FOLDER_PATHS


def main():
    print("\n" + "="*60)
    print("  DOCUMENT QUALITY ENGINE")
    print("="*60)

    print(f"\n  Scanning {len(FOLDER_PATHS)} folder(s)...")
    all_files = scan_folders(FOLDER_PATHS)

    if not all_files:
        print("\n  ❌ No documents found.")
        print("     Check FOLDER_PATHS in config.py")
        return

    print(f"  Found {len(all_files)} document(s)\n")

    csv = CSVHandler()
    csv.setup()

    success = 0
    errors  = 0

    for i, file_info in enumerate(all_files, 1):
        print(f"\n  [{i}/{len(all_files)}] {file_info['file_name']}")
        try:
            result = analyze_document(file_info)
            print_scores(result)
            csv.write(result)
            success += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors += 1

    csv.close()

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Total   : {len(all_files)}")
    print(f"  Success : {success}")
    print(f"  Errors  : {errors}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
