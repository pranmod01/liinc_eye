#!/usr/bin/env python3
"""Update POST notebooks to use renamed _POST files."""

import json
from pathlib import Path

def update_notebook_paths(notebook_path):
    """Update file paths in a notebook to use _POST suffix."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    updated = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Update file paths
            new_source = source
            new_source = new_source.replace(
                "'../data/results/extracted_features.pkl'",
                "'../data/results/features_POST/extracted_features_POST.pkl'"
            )
            new_source = new_source.replace(
                '"../data/results/extracted_features.pkl"',
                '"../data/results/features_POST/extracted_features_POST.pkl"'
            )
            new_source = new_source.replace(
                "'../data/results/eeg_features.pkl'",
                "'../data/results/features_POST/eeg_features_POST.pkl'"
            )
            new_source = new_source.replace(
                "'../data/results/'",
                "'../data/results/model_outputs_POST/'"
            )
            new_source = new_source.replace(
                '"../data/results/"',
                '"../data/results/model_outputs_POST/"'
            )

            if new_source != source:
                cell['source'] = new_source.split('\n')
                if cell['source'] and not cell['source'][-1].endswith('\n'):
                    cell['source'][-1] += '\n'
                updated = True

    if updated:
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        return True
    return False

def main():
    base_dir = Path(__file__).parent.parent / 'notebooks'

    folders = [
        'fusion_models_POST',
        'classification_POST',
        'analysis_POST',
        'other_models_POST'
    ]

    updated_count = 0
    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            continue

        for notebook_file in folder_path.rglob('*.ipynb'):
            if update_notebook_paths(notebook_file):
                print(f"✓ Updated: {notebook_file.name}")
                updated_count += 1

    print(f"\n✓ Updated {updated_count} notebooks")

if __name__ == '__main__':
    main()
