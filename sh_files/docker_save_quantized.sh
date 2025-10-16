#!/bin/bash
# Docker Script - Save Quantized Model
# This script runs inside Docker to create a pre-quantized model cache

echo "========================================"
echo "Docker: Save Quantized Model"
echo "========================================"
echo ""
echo "This will:"
echo "1. Load CodeLlama-7B-Instruct"
echo "2. Apply 4-bit quantization"
echo "3. Save to ./quantized_model_cache/"
echo ""
echo "Time required: ~5-7 minutes (one-time)"
echo "Disk space: ~3-4 GB"
echo ""

docker-compose run --rm lai-nux-tool-shell python save_quantized_model.py

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo ""
echo "The quantized model is now cached and will be used automatically."
echo "Next Docker run will be 4x faster!"

