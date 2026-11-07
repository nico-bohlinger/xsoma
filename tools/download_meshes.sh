#!/bin/bash
# Download the G1 meshes that assets/g1_29dof.xml references (19 MB) from Holosoma's repository at the commit XSoma's task is
# pinned to, and check them against tools/mesh_sha256.txt. Only rendering needs them (tools/render_gif.py); training and
# evaluation use the collision geometry, which is all primitives.
#   bash tools/download_meshes.sh
set -euo pipefail
COMMIT=79ba7142398e6e565e559e75620a889581d5966f
URL=https://raw.githubusercontent.com/amazon-far/holosoma/$COMMIT/src/holosoma/holosoma/data/robots/g1/meshes
cd "$(dirname "$0")/.."
mkdir -p assets/meshes
while read -r sha name; do
  [ -f "assets/meshes/$name" ] || curl -fsSL "$URL/$name" -o "assets/meshes/$name"
done < tools/mesh_sha256.txt
(cd assets/meshes && shasum -a 256 -c ../../tools/mesh_sha256.txt --quiet) && echo "G1 meshes in assets/meshes ($(ls assets/meshes | wc -l | tr -d ' ') files, checked)"
