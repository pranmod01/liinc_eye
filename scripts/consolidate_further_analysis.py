#!/usr/bin/env python3
"""
Consolidate further_analysis_PRE notebooks to accept TIMEFRAME parameter.

This script reads the four PRE analysis notebooks and creates parameterized versions
that work with both PRE and POST data.
"""

import json
import re
from pathlib import Path

def add_timeframe_parameter(cell_source):
    """Replace hardcoded _PRE with TIMEFRAME-based paths."""
    # Replace file paths
    cell_source = re.sub(
        r"features_PRE/extracted_features_PRE\.pkl",
        r"features_{TIMEFRAME}/extracted_features_{TIMEFRAME}.pkl",
        cell_source
    )
    cell_source = re.sub(
        r"analysis_outputs_PRE/",
        r"analysis_outputs_{TIMEFRAME}/",
        cell_source
    )
    cell_source = re.sub(
        r"fusion_model_results_PRE/",
        r"fusion_model_results_{TIMEFRAME}/",
        cell_source
    )
    cell_source = re.sub(
        r"_summary_PRE\.csv",
        r"_summary_{TIMEFRAME}.csv",
        cell_source
    )
    cell_source = re.sub(
        r"_results_PRE\.pkl",
        r"_results_{TIMEFRAME}.pkl",
        cell_source
    )
    cell_source = re.sub(
        r"_PRE\.png",
        r"_{TIMEFRAME}.png",
        cell_source
    )
    cell_source = re.sub(
        r"_PRE\.pdf",
        r"_{TIMEFRAME}.pdf",
        cell_source
    )
    cell_source = re.sub(
        r"_PRE\.csv",
        r"_{TIMEFRAME}.csv",
        cell_source
    )

    # Replace feature column references
    cell_source = re.sub(
        r"col\.endswith\('_pre'\)",
        r"col.endswith(f'_{TIMEFRAME.lower()}')",
        cell_source
    )
    cell_source = re.sub(
        r"'_pre'",
        r"f'_{TIMEFRAME.lower()}'",
        cell_source
    )
    cell_source = re.sub(
        r'"_pre"',
        r'f"_{TIMEFRAME.lower()}"',
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
            "TIMEFRAME = 'PRE'  # Change to 'POST' for post-decision analysis\n",
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
        # Remove existing PRE/POST mention from title
        old_title = re.sub(r'\s*\(PRE-Decision\)', '', old_title)
        old_title = re.sub(r'\s*\(POST-Decision\)', '', old_title)
        new_title = f"{old_title.strip()}\n\n**Parameterized**: Set `TIMEFRAME` to run PRE or POST analysis."
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

        # Also update markdown cells that mention PRE/POST
        elif cell['cell_type'] == 'markdown':
            if 'source' in cell:
                original_source = ''.join(cell['source'])
                # Update headers mentioning PRE/POST
                updated_source = re.sub(r'\(PRE-Decision\)', r'({TIMEFRAME}-Decision)', original_source)
                updated_source = re.sub(r'\(POST-Decision\)', r'({TIMEFRAME}-Decision)', updated_source)
                updated_source = re.sub(r'PRE-decision', r'{TIMEFRAME}-decision', updated_source)
                updated_source = re.sub(r'POST-decision', r'{TIMEFRAME}-decision', updated_source)

                if updated_source != original_source:
                    cell['source'] = [updated_source]

    # Write consolidated notebook
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"✓ Created {output_path.name}")

def main():
    """Main consolidation process."""
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "notebooks" / "further_analysis_PRE"
    output_dir = base_dir / "notebooks" / "further_analysis"
    output_dir.mkdir(exist_ok=True)

    notebooks = [
        {
            "input": "feature_importance_shap_PRE.ipynb",
            "output": "feature_importance_shap.ipynb",
            "title": "Feature Importance Analysis (SHAP)"
        },
        {
            "input": "publication_figures_PRE.ipynb",
            "output": "publication_figures.ipynb",
            "title": "Publication Figures"
        },
        {
            "input": "statistical_testing_PRE.ipynb",
            "output": "statistical_testing.ipynb",
            "title": "Statistical Significance Testing"
        },
        {
            "input": "temporal_dynamics_PRE.ipynb",
            "output": "temporal_dynamics.ipynb",
            "title": "Temporal Dynamics Analysis"
        }
    ]

    print("\nConsolidating further_analysis notebooks...")
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
    print("\nOld notebooks in notebooks/further_analysis_PRE/ can now be archived.")

if __name__ == "__main__":
    main()