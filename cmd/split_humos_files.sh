#!/bin/bash
# hhi project helper
# Split cmd/all_humos_files.txt (22,459 HUMOS motions) into 4 parallel parts
# Ready for: rclone copy --files-from=... or humos2phc_data_parallel.py

INPUT_FILE="cmd/all_humos_files.txt"
OUTPUT_PREFIX="cmd/all_humos_part"

# Check file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: $INPUT_FILE not found!"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$INPUT_FILE")
echo "📊 Total HUMOS files to split: $TOTAL_LINES"

NUM_PARTS=4
LINES_PER_PART=$(( (TOTAL_LINES + NUM_PARTS - 1) / NUM_PARTS ))
echo "➗ Splitting into $NUM_PARTS parts (~$LINES_PER_PART lines each)"

# Split using GNU split (first 3 parts get ceiling size, last gets remainder)
split -l "$LINES_PER_PART" -d --additional-suffix=.txt "$INPUT_FILE" "${OUTPUT_PREFIX}_"

# Rename to clean part1.txt … part4.txt
for i in {0..3}; do
    old_file="${OUTPUT_PREFIX}_$(printf "%02d" $i).txt"
    new_file="${OUTPUT_PREFIX}$((i+1)).txt"
    if [ -f "$old_file" ]; then
        mv "$old_file" "$new_file"
        echo "✅ Created: $new_file  ($(wc -l < "$new_file") lines)"
    fi
done

echo ""
echo "🎉 Split completed! You now have:"
ls -1 "${OUTPUT_PREFIX}"*.txt
echo ""
echo "Next steps (example parallel rclone):"
echo "   rclone copy gdrive:humos_output ~/humos_output_half --files-from=cmd/all_humos_part1.txt --progress --transfers=32 ..."
echo "   (repeat for part2, part3, part4)"
echo ""
echo "Or feed directly into humos2phc_data_parallel.py with --files-from=..."