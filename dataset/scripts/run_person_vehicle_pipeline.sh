#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/dataset/person_vehicle"
ANNOTATION_DIR="${REPO_ROOT}/dataset/annotations"
ANNOTATION_PATH="${1:-${ANNOTATION_DIR}/instances_train2017.json}"
TRAIN_SIZE="${2:-6000}"
VAL_SIZE="${3:-1000}"
TEST_SIZE="${4:-1000}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Error: neither python nor python3 is available in PATH."
  exit 1
fi

mkdir -p "${ANNOTATION_DIR}"

if [ ! -f "${ANNOTATION_PATH}" ]; then
  echo "COCO annotation file not found at: ${ANNOTATION_PATH}"
  echo "Downloading COCO annotations..."
  curl -L "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
    -o "${ANNOTATION_DIR}/annotations_trainval2017.zip"
  unzip -o "${ANNOTATION_DIR}/annotations_trainval2017.zip" -d "${ANNOTATION_DIR}"

  if [ -f "${ANNOTATION_DIR}/annotations/instances_train2017.json" ]; then
    ANNOTATION_PATH="${ANNOTATION_DIR}/annotations/instances_train2017.json"
  elif [ -f "${ANNOTATION_DIR}/instances_train2017.json" ]; then
    ANNOTATION_PATH="${ANNOTATION_DIR}/instances_train2017.json"
  else
    echo "Error: COCO annotation JSON was not found after download."
    exit 1
  fi
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_dataset.py" \
  --annotation "${ANNOTATION_PATH}" \
  --output-dir "${DATASET_ROOT}" \
  --train "${TRAIN_SIZE}" \
  --val "${VAL_SIZE}" \
  --test "${TEST_SIZE}"

printf "\nDone. Final dataset root: %s\n" "${DATASET_ROOT}"
