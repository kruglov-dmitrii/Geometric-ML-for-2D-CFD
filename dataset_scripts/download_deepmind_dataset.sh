#!/usr/bin/env bash
set -e

DATASET_NAME="cylinder_flow"
OUTPUT_DIR="./deepmind_datasets/$DATASET_NAME"
BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/$DATASET_NAME"

mkdir -p "$OUTPUT_DIR"

for file in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
  wget -O "$OUTPUT_DIR/$file" "$BASE_URL/$file"
done