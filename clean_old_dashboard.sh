#!/usr/bin/env bash
# ================================================================
# clean_old_dashboard.sh
# ================================================================
# Deletes ONLY the old dashboard files, nothing else. Does not
# touch project data, models, or anything unrelated to the dashboard.
#
# RUN from inside /Volumes/Apple/images/ :
#   chmod +x clean_old_dashboard.sh
#   ./clean_old_dashboard.sh
# ================================================================

echo "=================================================="
echo "Files/folders that WILL BE DELETED:"
echo "=================================================="
found_any=0
for item in app.py static templates doctor_photo.jpg dashboard_config.json __pycache__; do
    if [ -e "$item" ]; then
        echo "  - $item"
        found_any=1
    fi
done

if [ "$found_any" -eq 0 ]; then
    echo "  (none of these exist here -- nothing to delete)"
    exit 0
fi

echo ""
read -p "Type 'yes' to delete the items above: " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cancelled -- nothing was deleted."
    exit 0
fi

echo ""
for item in app.py static templates doctor_photo.jpg dashboard_config.json __pycache__; do
    if [ -e "$item" ]; then
        rm -rf "$item"
        echo "  deleted: $item"
    fi
done

echo ""
echo "Done. Old dashboard files removed. Now extract the new zip into this folder."
