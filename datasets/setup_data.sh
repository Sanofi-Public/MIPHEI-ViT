#!/usr/bin/env bash
set -e

# ============================================
# CONFIG
# ============================================
DOWNLOAD_DIR="download"
PREPROCESS_DIR="preprocessing"

DATASETS=()
GLOBAL_DATA_DIR=""
AVAILABLE_DATASETS=()

# Detect available datasets automatically (based on *_download.sh)
for f in "$DOWNLOAD_DIR"/*_download.sh; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    name="${base%_download.sh}"
    AVAILABLE_DATASETS+=("$name")
done


# ============================================
# HELP
# ============================================
usage() {
    echo "Usage:"
    echo "  ./data_setup.sh --data_dir <path> --dataset <name>"
    echo "  ./data_setup.sh --data_dir <path> --all"
    echo
    echo "Arguments:"
    echo "  --data_dir <path>     Root folder where datasets will be stored"
    echo "  --dataset <name>      Select a specific dataset (can repeat)"
    echo "  --all                 Process all available datasets"
    echo
    echo "Available datasets:"
    for d in "${AVAILABLE_DATASETS[@]}"; do
        echo "  - $d"
    done
    exit 1
}


# ============================================
# PARSE ARGS
# ============================================
if [ "$#" -eq 0 ]; then
    usage
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dataset)
            shift
            DATASETS+=("$1")
            ;;
        --all)
            DATASETS=("${AVAILABLE_DATASETS[@]}")
            ;;
        --data_dir)
            shift
            GLOBAL_DATA_DIR="$1"
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "❌ Unknown argument: $1"
            usage
            ;;
    esac
    shift
done

# Required argument
if [ -z "$GLOBAL_DATA_DIR" ]; then
    echo "❌ Missing required argument: --data_dir <path>"
    usage
fi

# If user didn't select dataset(s)
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "❌ No dataset selected."
    usage
fi

# Normalize path
GLOBAL_DATA_DIR=$(realpath "$GLOBAL_DATA_DIR")
mkdir -p "$GLOBAL_DATA_DIR"


# ============================================
# FUNCTIONS
# ============================================

run_dataset() {
    local dataset="$1"
    local download_script="$DOWNLOAD_DIR/${dataset}_download.sh"
    local preprocess_script="$PREPROCESS_DIR/${dataset}_preprocess.py"

    echo
    echo "============================================"
    echo " 🚀 Processing dataset: $dataset"
    echo "============================================"

    # Check expected scripts exist
    if [ ! -f "$download_script" ]; then
        echo "❌ Missing download script: $download_script"
        exit 1
    fi

    if [ ! -f "$preprocess_script" ]; then
        echo "❌ Missing preprocessing script: $preprocess_script"
        exit 1
    fi

    # Dataset-specific folder: <data_dir>/<dataset>
    DATASET_DIR="$GLOBAL_DATA_DIR/$dataset"
    mkdir -p "$DATASET_DIR"

    echo "📁 Target dataset folder: $DATASET_DIR"

    # ---------- Run download ----------
    echo "⬇️  Running download script..."
    bash "$download_script" "$DATASET_DIR"
    echo "   ✓ Download complete"

    # ---------- Run preprocessing ----------
    echo "🧪 Running preprocessing script..."
    python "$preprocess_script" --data_dir "$DATASET_DIR"
    echo "   ✓ Preprocessing complete"
}


# ============================================
# MAIN LOOP
# ============================================
for dataset in "${DATASETS[@]}"; do
    run_dataset "$dataset"
done

echo
echo "✨ All requested datasets processed successfully!"
