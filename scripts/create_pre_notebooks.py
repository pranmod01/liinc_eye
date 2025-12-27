#!/usr/bin/env python3
"""
Script to create PRE-decision versions of all POST-decision notebooks.
Replaces POST terminology with PRE and updates file paths.
"""

import json
import re
from pathlib import Path

def convert_post_to_pre_notebook(post_path, pre_path):
    """Convert a POST-decision notebook to PRE-decision version."""

    with open(post_path, 'r') as f:
        notebook = json.load(f)

    # Update all cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            # Update markdown content
            content = ''.join(cell['source'])
            content = content.replace('POST-decision', 'PRE-decision')
            content = content.replace('Post-decision', 'Pre-decision')
            content = content.replace('post-decision', 'pre-decision')
            content = content.replace('POST-SUBMIT', 'PRE-SUBMIT')
            content = content.replace('POST)', 'PRE)')
            content = content.replace('(POST', '(PRE')
            content = content.replace('Physiology (POST)', 'Physiology (PRE)')
            content = content.replace('after submit', 'before submit')
            content = content.replace('After submit', 'Before submit')
            content = content.replace('0 to 2 seconds after', '-2 to 0 seconds before')
            content = content.replace('response to outcome', 'anticipatory/deliberation')
            cell['source'] = content.split('\n')
            if cell['source'] and not cell['source'][-1].endswith('\n'):
                cell['source'][-1] += '\n'

        elif cell['cell_type'] == 'code':
            # Update code content
            content = ''.join(cell['source'])

            # Update file paths
            content = content.replace('../data/results/extracted_features.pkl',
                                     '../data/results/features_PRE/extracted_features_PRE.pkl')
            content = content.replace("'../data/results'",
                                     "'../data/results/model_outputs_PRE'")
            content = content.replace('../../data/results/',
                                     '../../data/results/')

            # Update variable names and labels
            content = content.replace('_post', '_pre')
            content = content.replace('POST)', 'PRE)')
            content = content.replace('(POST', '(PRE')
            content = content.replace('Physiology (POST)', 'Physiology (PRE)')
            content = re.sub(r'# POST-SUBMIT DATA ONLY.*',
                           '# PRE-SUBMIT DATA ONLY (-2 to 0 seconds before submit)', content)
            content = content.replace('post_submit_mask = (time_clean > 0) & (time_clean <= 2.0)',
                                     'pre_submit_mask = (time_clean >= -2.0) & (time_clean < 0)')
            content = content.replace('pupil_post = pupil_avg_clean[post_submit_mask]',
                                     'pupil_pre = pupil_avg_clean[pre_submit_mask]')
            content = content.replace('pupil_L_post = pupil_L_clean[post_submit_mask]',
                                     'pupil_L_pre = pupil_L_clean[pre_submit_mask]')
            content = content.replace('pupil_R_post = pupil_R_clean[post_submit_mask]',
                                     'pupil_R_pre = pupil_R_clean[pre_submit_mask]')
            content = content.replace('time_post = time_clean[post_submit_mask]',
                                     'time_pre = time_clean[pre_submit_mask]')

            # Update output file names
            content = re.sub(r"'([^']+)_results\.pkl'",
                           r"'\1_results_PRE.pkl'", content)
            content = re.sub(r"'([^']+)_comparison\.csv'",
                           r"'\1_comparison_PRE.csv'", content)

            cell['source'] = content.split('\n')
            if cell['source'] and not cell['source'][-1].endswith('\n'):
                cell['source'][-1] += '\n'

    # Save to new file
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pre_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"✓ Created: {pre_path.name}")

def main():
    """Main conversion function."""

    base_dir = Path(__file__).parent.parent / 'notebooks'

    # Define notebook conversions
    conversions = [
        # Main fusion models
        ('fusion_models_POST/late_fusion_model.ipynb',
         'fusion_models_PRE/late_fusion_model_PRE.ipynb'),
        ('fusion_models_POST/multi_seed_late_fusion.ipynb',
         'fusion_models_PRE/multi_seed_late_fusion_PRE.ipynb'),
        ('fusion_models_POST/late_fusion_by_visit.ipynb',
         'fusion_models_PRE/late_fusion_by_visit_PRE.ipynb'),

        # Ambiguity group models
        ('fusion_models_POST/ambiguity_group_late_fusion.ipynb',
         'fusion_models_PRE/ambiguity_group_late_fusion_PRE.ipynb'),
        ('fusion_models_POST/ambiguity_group_late_fusion_balanced.ipynb',
         'fusion_models_PRE/ambiguity_group_late_fusion_balanced_PRE.ipynb'),

        # Reaction time group models
        ('fusion_models_POST/reaction_time_group_late_fusion.ipynb',
         'fusion_models_PRE/reaction_time_group_late_fusion_PRE.ipynb'),
        ('fusion_models_POST/reaction_time_group_late_fusion_balanced.ipynb',
         'fusion_models_PRE/reaction_time_group_late_fusion_balanced_PRE.ipynb'),

        # Classification
        ('classification_POST/outcome_classification.ipynb',
         'classification_PRE/outcome_classification_PRE.ipynb'),
        ('classification_POST/ambiguity_classification.ipynb',
         'classification_PRE/ambiguity_classification_PRE.ipynb'),

        # Analysis
        ('analysis_POST/subject_repeat_analysis.ipynb',
         'analysis_PRE/subject_repeat_analysis_PRE.ipynb'),
        # NOTE: session_mapping.ipynb is shared (metadata only, no physiology features)
    ]

    print("Creating PRE-decision notebooks...")
    print("=" * 60)

    for post_rel, pre_rel in conversions:
        post_path = base_dir / post_rel
        pre_path = base_dir / pre_rel

        if post_path.exists():
            convert_post_to_pre_notebook(post_path, pre_path)
        else:
            print(f"⚠ Warning: {post_path.name} not found")

    print("=" * 60)
    print(f"✓ Conversion complete! Created {len([c for c in conversions])} notebooks")

if __name__ == '__main__':
    main()
