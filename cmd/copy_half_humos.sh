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
HALF=$((TOTAL / 2))

echo "Total files found : $TOTAL"
echo "Will copy ~half   : $HALF files"

# 3. Randomly select half (shuf is fast and fair)
echo "Selecting random half..."
shuf "$ALL_FILES_LIST" | head -n "$HALF" > "$RECORD_FILE"

# 4. Copy only the selected files (rclone --files-from is perfect for this)
echo "Starting rclone copy (this may take a while)..."
rclone copy \
  "$REMOTE" \
  "$LOCAL_DEST" \
  --files-from="$RECORD_FILE" \
  --progress \
  --transfers=32 \
  --checkers=64 \
  --drive-chunk-size=256M \
  --fast-list

echo "=== DONE ==="
echo "Copied files: $(wc -l < "$RECORD_FILE") / $TOTAL"
echo "Record saved to: $RECORD_FILE"
echo "Local data at : $LOCAL_DEST"
echo "You can now point humos2phc_data_parallel.py at $LOCAL_DEST"

----

rclone copy   gdrive:humos_output   ~/datasets/humos_output_part1   --files-from=all_humos_part1.txt   --progress   --transfers=32   --checkers=64   --drive-chunk-size=256M   --fast-list