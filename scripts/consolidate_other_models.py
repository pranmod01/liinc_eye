#!/usr/bin/env python3
"""
Consolidate other_models_POST notebooks to accept TIMEFRAME parameter.

This script reads the three POST notebooks and creates parameterized versions
that work with both PRE and POST data.
"""

import json
import re
from pathlib import Path

def add_timeframe_parameter(cell_source):
    """Replace hardcoded _POST with TIMEFRAME-based paths."""
    # Replace file paths
    cell_source = re.sub(
        r"features_POST/extracted_features_POST\.pkl",
        r"features_{TIMEFRAME}/extracted_features_{TIMEFRAME}.pkl",
        cell_source
    )
    cell_source = re.sub(
        r"analysis_outputs_POST/",
        r"analysis_outputs_{TIMEFRAME}/",
        cell_source
    )
    cell_source = re.sub(
        r"model_outputs_POST/",
        r"model_outputs_{TIMEFRAME}/",
        cell_source
    )
    cell_source = re.sub(
        r"_comparison_POST\.csv",
        r"_comparison_{TIMEFRAME}.csv",
        cell_source
    )
    cell_source = re.sub(
        r"_results_POST\.pkl",
        r"_results_{TIMEFRAME}.pkl",
        cell_source
    )

    return cell_source

def consolidate_notebook(input_path, output_path, title):
    """Consolidate a single notebook with TIMEFRAME parameter."""
    with open(input_path, 'r') as f:
        nb = json.load(f)

    # Create new first cell with TIMEFRAME configuration
    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================================\n",
            "# CONFIGURATION: Set timeframe for analysis\n",
            "# ============================================================================\n",
            "TIMEFRAME = 'POST'  # Change to 'PRE' for pre-decision analysis\n",
            "# ============================================================================\n",
            "\n",
            f"print(f\"\\n{{'='*70}}\")\n",
            f"print(f\"{title.upper()}: {{TIMEFRAME}}-DECISION PERIOD\")\n",
            f"print(f\"{{'='*70}}\\n\")"
        ]
    }

    # Update title in first markdown cell
    if nb['cells'][0]['cell_type'] == 'markdown':
        old_title = ''.join(nb['cells'][0]['source'])
        new_title = f"{title}\n\n**Parameterized**: Set `TIMEFRAME` to run PRE or POST analysis."
        nb['cells'][0]['source'] = [new_title]

    # Insert config cell after title
    nb['cells'].insert(1, config_cell)

    # Process all code cells to replace hardcoded paths
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if 'source' in cell:
                original_source = ''.join(cell['source'])
                updated_source = add_timeframe_parameter(original_source)

                # Split back into lines for proper formatting
                if updated_source != original_source:
                    cell['source'] = updated_source.split('\n')
                    # Add newlines back except for last line
                    cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                     for i, line in enumerate(cell['source'])]

    # Write consolidated notebook
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"✓ Created {output_path.name}")

def main():
    """Main consolidation process."""
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "notebooks" / "other_models_POST"
    output_dir = base_dir / "notebooks" / "other_models"
    output_dir.mkdir(exist_ok=True)

    notebooks = [
        {
            "input": "ensemble_baseline_POST.ipynb",
            "output": "ensemble_baseline.ipynb",
            "title": "Ensemble Methods Baseline"
        },
        {
            "input": "gradient_boosting_baseline_POST.ipynb",
            "output": "gradient_boosting_baseline.ipynb",
            "title": "Gradient Boosting Baseline"
        },
        {
            "input": "feature_engineering_baseline_POST.ipynb",
            "output": "feature_engineering_baseline.ipynb",
            "title": "Feature Engineering Baseline"
        }
    ]

    print("\nConsolidating other_models notebooks...")
    print("=" * 70)

    for nb_info in notebooks:
        input_path = input_dir / nb_info["input"]
        output_path = output_dir / nb_info["output"]

        if not input_path.exists():
            print(f"⚠️  Skipping {nb_info['input']} (not found)")
            continue

        consolidate_notebook(input_path, output_path, nb_info["title"])

    print("=" * 70)
    print("✓ Consolidation complete!")
    print(f"\nConsolidated notebooks saved to: {output_dir}")
    print("\nOld notebooks in notebooks/other_models_POST/ can now be archived.")

if __name__ == "__main__":
    main()