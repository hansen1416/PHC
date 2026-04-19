#!/usr/bin/env python3
"""
Copy all .pt files from a local directory to a Google Drive folder using rclone.

Default behavior:
- Scans /home/hlz/repos/humos/output recursively
- Uploads only *.pt files
- Copies them to gdrive:humos_output
- Skips unchanged files

Examples:
    python upload_pt_to_gdrive.py
    python upload_pt_to_gdrive.py --remote mydrive --dest-folder humos_output
    python upload_pt_to_gdrive.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy all .pt files from a local folder to Google Drive via rclone."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/hlz/repos/humos/output"),
        help="Local source directory (default: /home/hlz/repos/humos/output)",
    )
    parser.add_argument(
        "--remote",
        default="gdrive",
        help="Rclone remote name (default: gdrive)",
    )
    parser.add_argument(
        "--dest-folder",
        default="humos_output",
        help="Destination folder inside the remote (default: humos_output)",
    )
    parser.add_argument(
        "--transfers",
        type=int,
        default=4,
        help="Number of concurrent file transfers (default: 4)",
    )
    parser.add_argument(
        "--checkers",
        type=int,
        default=8,
        help="Number of checkers for file comparison (default: 8)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to save the rclone log",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually uploading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use verbose rclone logging",
    )
    return parser.parse_args()


def ensure_rclone_exists() -> None:
    if shutil.which("rclone") is None:
        print("Error: rclone is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)


def ensure_source_exists(src: Path) -> None:
    if not src.exists():
        print(f"Error: source directory does not exist: {src}", file=sys.stderr)
        sys.exit(1)
    if not src.is_dir():
        print(f"Error: source path is not a directory: {src}", file=sys.stderr)
        sys.exit(1)


def count_pt_files(src: Path) -> int:
    return sum(1 for _ in src.rglob("*.pt"))


def build_rclone_command(args: argparse.Namespace) -> list[str]:
    destination = f"{args.remote}:{args.dest_folder}"

    cmd = [
        "rclone",
        "copy",
        str(args.src),
        destination,
        "--include",
        "*.pt",
        "--create-empty-src-dirs",
        "--transfers",
        str(args.transfers),
        "--checkers",
        str(args.checkers),
        "--progress",
        "--stats",
        "10s",
    ]

    if args.dry_run:
        cmd.append("--dry-run")

    if args.verbose:
        cmd.extend(["-vv"])
    else:
        cmd.extend(["--log-level", "INFO"])

    if args.log_file is not None:
        cmd.extend(["--log-file", str(args.log_file)])

    return cmd


def main() -> None:
    args = parse_args()
    ensure_rclone_exists()
    ensure_source_exists(args.src)

    pt_count = count_pt_files(args.src)
    if pt_count == 0:
        print(f"No .pt files found under: {args.src}")
        return

    destination = f"{args.remote}:{args.dest_folder}"
    print(f"Found {pt_count} .pt files")
    print(f"Source      : {args.src}")
    print(f"Destination : {destination}")
    print(f"Mode        : {'DRY RUN' if args.dry_run else 'COPY'}")

    cmd = build_rclone_command(args)
    print("Running:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\nUpload failed with exit code {exc.returncode}.", file=sys.stderr)
        sys.exit(exc.returncode)

    print("\nDone.")


if __name__ == "__main__":
    main()
