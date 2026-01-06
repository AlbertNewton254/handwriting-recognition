#!/bin/env bash
set -e

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/landlord/handwriting-recognition"
ZIP_FILE="handwritten_recognition.zip"
DATA_DIR="data"

echo "Downloading Dataset..."
curl -L -o "$ZIP_FILE" "$DATASET_URL"

echo "Extracting dataset..."
unzip "$ZIP_FILE" > /dev/null 2>&1
rm "$ZIP_FILE"

# Cleanup previous directory
echo "Cleaning up previous data directory..."
if [ -d "$DATA_DIR" ]; then
    rm -rf "$DATA_DIR"
fi

echo "Creating data directory..."
mkdir "$DATA_DIR"

echo "Organizing files..."

# Folders
if [ -d train_v2/train ]; then
    mv train_v2/train "$DATA_DIR/train"
fi

if [ -d test_v2/test ]; then
    mv test_v2/test "$DATA_DIR/test"
fi

if [ -d validation_v2/validation ]; then
    mv validation_v2/validation "$DATA_DIR/validation"
fi

# CSV files
if [ -f written_name_train_v2.csv ]; then
    mv written_name_train_v2.csv "$DATA_DIR/train.csv"
fi

if [ -f written_name_test_v2.csv ]; then
    mv written_name_test_v2.csv "$DATA_DIR/test.csv"
fi

if [ -f written_name_validation_v2.csv ]; then
    mv written_name_validation_v2.csv "$DATA_DIR/validation.csv"
fi

# Cleanup
rm -rf train_v2 test_v2 validation_v2

echo "Done."
echo "Final structure:"

ls "$DATA_DIR"