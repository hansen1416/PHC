#!/bin/bash
# =============================================================================
# hhi / humos2phc data prep script
# Copy ~half of humos_output from Google Drive → local (preserves folder structure)
# Keeps exact record of copied files in copied_files.txt
# =============================================================================

set -euo pipefail

REMOTE="gdrive:humos_output"
LOCAL_DEST="${HOME}/humos_output_half"          # <-- change if you want another path
RECORD_FILE="copied_files.txt"
ALL_FILES_LIST="all_humos_files.txt"

echo "=== hhi HUMOS copy (half dataset) ==="
echo "Remote : $REMOTE"
echo "Local  : $LOCAL_DEST"
echo "Record : $RECORD_FILE"

# 1. Create local destination
mkdir -p "$LOCAL_DEST"

# 2. List ALL files (relative paths, handles subfolders)
echo "Listing all files from Google Drive..."
rclone lsf --files-only --recursive "$REMOTE" > "$ALL_FILES_LIST"
TOTAL=$(wc -l < "$ALL_FILES_LIST")
