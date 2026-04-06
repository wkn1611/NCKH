#!/usr/bin/env bash
# =============================================================================
# DRIVER DROWSINESS DETECTION - Project Scaffolding Script
# Run from: /Users/thanhphong/Documents/NCKH/
# Usage: bash setup_project.sh
# =============================================================================

set -e  # Exit immediately on any error

ROOT="Drowsiness_Project"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚗  Scaffolding Drowsiness Detection Project..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# --- Create top-level directories ---
mkdir -p "$ROOT/src/perception"
mkdir -p "$ROOT/src/extraction"
mkdir -p "$ROOT/src/intelligence"
mkdir -p "$ROOT/src/utils"
mkdir -p "$ROOT/models"
mkdir -p "$ROOT/scripts"

echo "✅  Directories created."

# --- src/ root files ---
touch "$ROOT/src/main.py"       # Entry point: Master Loop
touch "$ROOT/src/context.py"    # Shared Master Context Class

# --- perception/ module ---
# Handles camera initialization and MediaPipe face mesh setup
touch "$ROOT/src/perception/__init__.py"
touch "$ROOT/src/perception/camera.py"       # Camera stream manager (OpenCV)
touch "$ROOT/src/perception/face_mesh.py"    # MediaPipe FaceMesh wrapper

# --- extraction/ module ---
# Handles all geometric feature extraction from landmarks
touch "$ROOT/src/extraction/__init__.py"
touch "$ROOT/src/extraction/ear.py"          # Eye Aspect Ratio (EAR)
touch "$ROOT/src/extraction/mar.py"          # Mouth Aspect Ratio (MAR)
touch "$ROOT/src/extraction/head_pose.py"    # Pitch / Yaw estimation
touch "$ROOT/src/extraction/roi.py"          # ROI crop from frame

# --- intelligence/ module ---
# Handles TFLite model loading and inference
touch "$ROOT/src/intelligence/__init__.py"
touch "$ROOT/src/intelligence/model_runner.py"  # TFLite interpreter wrapper
touch "$ROOT/src/intelligence/classifier.py"    # Eye/Mouth CNN classifier logic

# --- utils/ module ---
# Handles logging, GPIO alerts, and on-screen visualization
touch "$ROOT/src/utils/__init__.py"
touch "$ROOT/src/utils/logger.py"            # Structured logger
touch "$ROOT/src/utils/gpio_alert.py"        # GPIO buzzer / LED triggers
touch "$ROOT/src/utils/visualizer.py"        # OpenCV drawing / overlay

# --- Top-level files ---
touch "$ROOT/.cursorrules"         # Context master (copy CONTEXT_MASTER.md here)
touch "$ROOT/requirements.txt"     # Fixed version dependencies (see requirements.txt)

echo "✅  All .py files and top-level files created."

# --- Print final tree ---
echo ""
echo "📁  Final structure:"
find "$ROOT" | sort | sed 's|[^/]*/|  |g; s|  \([^ ]\)|├── \1|'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Scaffold complete! Navigate to: $ROOT/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
