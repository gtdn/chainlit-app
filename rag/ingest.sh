#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: ./ingest.sh <chemin_du_dossier>"
    exit 1
fi

FOLDER_PATH="$1"

if [ ! -d "$FOLDER_PATH" ]; then
    echo "Erreur : dossier introuvable -> $FOLDER_PATH"
    exit 1
fi

echo "Extraction : $FOLDER_PATH"
python3 ingest.py "$FOLDER_PATH"