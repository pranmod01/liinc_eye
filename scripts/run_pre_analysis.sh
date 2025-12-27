#!/bin/bash
# Script to run all PRE-decision analysis notebooks in order

set -e  # Exit on error

echo "=================================="
echo "Running PRE-Decision Analysis"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo -e "${BLUE}Working directory: $BASE_DIR${NC}"
echo ""

# Function to run a notebook
run_notebook() {
    local notebook_path=$1
    local description=$2

    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}Running: $description${NC}"
    echo -e "${BLUE}File: $notebook_path${NC}"
    echo -e "${BLUE}===================================================${NC}"

    if [ ! -f "$notebook_path" ]; then
        echo -e "${RED}Error: Notebook not found: $notebook_path${NC}"
        exit 1
    fi

    jupyter nbconvert --to notebook --execute --inplace "$notebook_path" --ExecutePreprocessor.timeout=600

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: $description${NC}"
    else
        echo -e "${RED}✗ Failed: $description${NC}"
        exit 1
    fi
    echo ""
}

# 1. Main Late Fusion Model
run_notebook \
    "notebooks/fusion_models_PRE/late_fusion_model_PRE.ipynb" \
    "Main Late Fusion Model (PRE)"

# 2. Multi-seed validation
run_notebook \
    "notebooks/fusion_models_PRE/multi_seed_late_fusion_PRE.ipynb" \
    "Multi-Seed Late Fusion (PRE)"

# 3. Visit-stratified analysis
run_notebook \
    "notebooks/fusion_models_PRE/late_fusion_by_visit_PRE.ipynb" \
    "Late Fusion by Visit (PRE)"

# 4. Ambiguity group analysis
run_notebook \
    "notebooks/fusion_models_PRE/ambiguity_group_late_fusion_PRE.ipynb" \
    "Ambiguity Group Late Fusion (PRE)"

# 5. Ambiguity group (balanced)
run_notebook \
    "notebooks/fusion_models_PRE/ambiguity_group_late_fusion_balanced_PRE.ipynb" \
    "Ambiguity Group Late Fusion - Balanced (PRE)"

# 6. Reaction time group analysis
run_notebook \
    "notebooks/fusion_models_PRE/reaction_time_group_late_fusion_PRE.ipynb" \
    "Reaction Time Group Late Fusion (PRE)"

# 7. Reaction time group (balanced)
run_notebook \
    "notebooks/fusion_models_PRE/reaction_time_group_late_fusion_balanced_PRE.ipynb" \
    "Reaction Time Group Late Fusion - Balanced (PRE)"

# 8. Outcome classification
run_notebook \
    "notebooks/classification_PRE/outcome_classification_PRE.ipynb" \
    "Outcome Classification (PRE)"

# 9. Ambiguity classification
run_notebook \
    "notebooks/classification_PRE/ambiguity_classification_PRE.ipynb" \
    "Ambiguity Classification (PRE)"

# 10. Subject repeat analysis
run_notebook \
    "notebooks/analysis_PRE/subject_repeat_analysis_PRE.ipynb" \
    "Subject Repeat Analysis (PRE)"

echo ""
echo -e "${GREEN}=================================="
echo -e "All PRE-Decision Analysis Complete!"
echo -e "==================================${NC}"
echo ""
echo "Results saved to:"
echo "  - data/results/model_outputs_PRE/"
echo "  - data/results/analysis_outputs_PRE/"
