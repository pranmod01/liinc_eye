"""Create notebook 14_glmm_feature_ambiguity_interaction.ipynb"""
import json

NB_PATH = 'notebooks/further_analysis/14_glmm_feature_ambiguity_interaction.ipynb'

def code_cell(src):
    return {'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': src}

def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

cells = []

cells.append(md_cell(
    '# GLMM: Feature × Ambiguity Interaction\n'
    '\n'
    '**Question**: Does ambiguity *moderate* how individual gaze/physio features predict investment decisions?\n'
    '\n'
    'Notebook 11b showed that 11 gaze features are significantly driven by ambiguity (Direction A),\n'
    'but they were tested as main effects in Direction B. Here we add `feature × ambiguity`\n'
    'interaction terms to the ConditionalLogit model:\n'
    '\n'
    '```\n'
    'outcome ~ feature_z + ambiguity_z + feature_z:ambiguity_z + controls + (subject FE)\n'
    '```\n'
    '\n'
    'A significant interaction means the feature\'s predictive relationship with choice\n'
    'is **stronger or weaker depending on ambiguity** — i.e., ambiguity moderates the\n'
    'neural-to-behavioural coupling, not just its magnitude.\n'
    '\n'
    '**Scope**: We test this for all 11 ambiguity-significant gaze features, plus\n'
    'a secondary scan of EEG and pupil features as a null comparison.'
))

cells.append(code_cell("""\
import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
import pickle
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
np.random.seed(42)
OUT = Path('../../data/results/main/statistical_analyses/interaction')
OUT.mkdir(parents=True, exist_ok=True)
"""))

cells.append(md_cell('## 1. Load Data'))

cells.append(code_cell("""\
with open('../../data/features/extracted_features_PRE.pkl', 'rb') as f:
    feature_data = pickle.load(f)
merged_df   = feature_data['merged_df'].copy()
physio_cols = feature_data['physio_cols']
gaze_cols   = feature_data['gaze_cols']

with open('../../data/features/eeg_features_regional_bands.pkl', 'rb') as f:
    eeg_data = pickle.load(f)
eeg_df   = eeg_data['eeg_features_df'].copy()
eeg_cols = eeg_data['feature_columns']

behavior_keep = ['subject_id', 'trial_id', 'outcome',
                 'ambiguity', 'ev_difference', 'decision_time',
                 'prev_outcome', 'running_invest_rate']

print(f"Trials: {len(merged_df)}, Subjects: {merged_df['subject_id'].nunique()}")
"""))

cells.append(md_cell('## 2. Helpers'))

cells.append(code_cell("""\
def zscore_cols(df, cols):
    df = df.copy()
    for c in cols:
        mu, sd = df[c].mean(), df[c].std()
        df[f'{c}_z'] = (df[c] - mu) / sd if sd > 0 else 0.0
    return df


def fit_clogit(outcome, predictors_df, groups):
    df_fit = predictors_df.copy()
    df_fit['_outcome'] = outcome.values
    df_fit['_group']   = groups.values
    var_mask = df_fit.groupby('_group')['_outcome'].transform('std') > 0
    df_fit = df_fit[var_mask].reset_index(drop=True)
    if len(df_fit) == 0 or df_fit['_group'].nunique() < 5:
        return None
    try:
        model  = ConditionalLogit(df_fit['_outcome'],
                                  df_fit.drop(columns=['_outcome', '_group']),
                                  groups=df_fit['_group'])
        return model.fit(disp=False)
    except Exception:
        return None


CTRL_COLS = ['ev_difference_z', 'prev_outcome', 'running_invest_rate_z', 'decision_time_z']

print("Helpers defined.")
"""))

cells.append(md_cell(
    '## 3. Interaction Scan Function\n'
    '\n'
    'For each feature we fit two ConditionalLogit models and compare:\n'
    '- **Main-effects model**: `outcome ~ feature_z + ambiguity_z + controls`\n'
    '- **Interaction model**: adds `feature_z × ambiguity_z`\n'
    '\n'
    'We extract the interaction coefficient, its SE, p-value, and an LRT\n'
    '(likelihood-ratio test) comparing the two models.'
))

cells.append(code_cell("""\
def interaction_scan(feature_cols, df, label=''):
    rows = []
    for col in feature_cols:
        zcol = f'{col}_z'
        if zcol not in df.columns:
            continue
        interaction_col = f'{zcol}_x_amb'
        df[interaction_col] = df[zcol] * df['ambiguity_z']

        pred_main = df[[zcol, 'ambiguity_z'] + CTRL_COLS].copy()
        pred_int  = df[[zcol, 'ambiguity_z', interaction_col] + CTRL_COLS].copy()

        fit_main = fit_clogit(df['outcome'], pred_main, df['subject_id'])
        fit_int  = fit_clogit(df['outcome'], pred_int,  df['subject_id'])

        if fit_main is None or fit_int is None:
            continue

        beta_main = fit_main.params[zcol]
        beta_int  = fit_int.params[interaction_col]
        se_int    = fit_int.bse[interaction_col]
        p_int     = fit_int.pvalues[interaction_col]

        # LRT: -2 * (llf_main - llf_int) ~ chi2(1)
        from scipy.stats import chi2
        lrt_stat = -2 * (fit_main.llf - fit_int.llf)
        lrt_p    = chi2.sf(lrt_stat, df=1)

        rows.append({
            'feature':     col,
            'beta_main':   beta_main,          # main effect (log-odds)
            'beta_int':    beta_int,            # interaction log-odds
            'se_int':      se_int,
            'p_int':       p_int,
            'lrt_stat':    lrt_stat,
            'lrt_p':       lrt_p,
            'sig_int':     p_int < 0.05,
        })

    out = pd.DataFrame(rows).sort_values('p_int')
    return out

print("interaction_scan() defined.")
"""))

cells.append(md_cell(
    '## 4. Gaze × Ambiguity\n'
    '\n'
    'Focus: the 11 gaze features that are significantly driven by ambiguity (from notebook 11b).'
))

cells.append(code_cell("""\
gaze_model_cols = [c for c in gaze_cols if c not in
                   ['gaze_valid_pct', 'fixation_ratio', 'saccade_ratio', 'saccade_count']]

df_gaze = merged_df[behavior_keep + gaze_model_cols].dropna(subset=gaze_model_cols).copy().reset_index(drop=True)
df_gaze = zscore_cols(df_gaze, ['ambiguity', 'ev_difference', 'decision_time',
                                  'running_invest_rate'] + gaze_model_cols)

print(f"Gaze dataset: {len(df_gaze)} trials, {df_gaze['subject_id'].nunique()} subjects")

gaze_int = interaction_scan(gaze_model_cols, df_gaze, label='Gaze')

print(f"\\nGaze × Ambiguity interaction scan ({len(gaze_int)} features):")
print(f"  Significant interactions (p<.05): {gaze_int['sig_int'].sum()} / {len(gaze_int)}")
print()
print(gaze_int[['feature', 'beta_main', 'beta_int', 'se_int', 'p_int', 'lrt_p', 'sig_int']]
      .to_string(index=False))
"""))

cells.append(md_cell('## 5. Pupil × Ambiguity'))

cells.append(code_cell("""\
pupil_cols = [c for c in physio_cols if c.endswith('_pre')]

df_pupil = merged_df[behavior_keep + pupil_cols].dropna(subset=pupil_cols).copy().reset_index(drop=True)
df_pupil = zscore_cols(df_pupil, ['ambiguity', 'ev_difference', 'decision_time',
                                   'running_invest_rate'] + pupil_cols)

print(f"Pupil dataset: {len(df_pupil)} trials, {df_pupil['subject_id'].nunique()} subjects")

pupil_int = interaction_scan(pupil_cols, df_pupil, label='Pupil')

print(f"\\nPupil × Ambiguity interaction scan ({len(pupil_int)} features):")
print(f"  Significant interactions: {pupil_int['sig_int'].sum()} / {len(pupil_int)}")
print()
print(pupil_int[['feature', 'beta_main', 'beta_int', 'se_int', 'p_int', 'lrt_p', 'sig_int']]
      .to_string(index=False))
"""))

cells.append(md_cell('## 6. EEG × Ambiguity (Null Comparison)'))

cells.append(code_cell("""\
df_eeg = merged_df[behavior_keep].merge(eeg_df[['trial_id'] + eeg_cols], on='trial_id', how='inner')
df_eeg = df_eeg.dropna(subset=eeg_cols).reset_index(drop=True)
df_eeg = zscore_cols(df_eeg, ['ambiguity', 'ev_difference', 'decision_time',
                                'running_invest_rate'] + eeg_cols)

print(f"EEG dataset: {len(df_eeg)} trials, {df_eeg['subject_id'].nunique()} subjects")

eeg_int = interaction_scan(eeg_cols, df_eeg, label='EEG')

print(f"\\nEEG × Ambiguity interaction scan ({len(eeg_int)} features):")
print(f"  Significant interactions: {eeg_int['sig_int'].sum()} / {len(eeg_int)}")
print()
print(eeg_int[['feature', 'beta_main', 'beta_int', 'se_int', 'p_int', 'lrt_p', 'sig_int']]
      .to_string(index=False))
"""))

cells.append(md_cell(
    '## 7. Visualization\n'
    '\n'
    'Forest plots of interaction coefficients (log-odds of `feature × ambiguity`).\n'
    'A positive interaction means the feature predicts investing *more strongly* at high ambiguity;\n'
    'negative means the feature\'s relationship with choice *attenuates* under ambiguity.'
))

cells.append(code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
panel_data = [
    (gaze_int,  'Gaze × Ambiguity',  '#C73E1D'),
    (pupil_int, 'Pupil × Ambiguity', '#2196F3'),
    (eeg_int,   'EEG × Ambiguity\\n(null comparison)', '#A23B72'),
]

for ax, (df_i, title, color) in zip(axes, panel_data):
    df_plot = df_i.sort_values('beta_int').reset_index(drop=True)
    short = [f.replace('gaze_', '').replace('screen_', 'scr_')
               .replace('_pre', '').replace('eeg_', '') for f in df_plot['feature']]
    bar_colors = [color if s else '#BDBDBD' for s in df_plot['sig_int']]
    ax.barh(short, df_plot['beta_int'], color=bar_colors, alpha=0.8, zorder=3)
    ax.errorbar(df_plot['beta_int'], short,
                xerr=1.96 * df_plot['se_int'], fmt='none',
                color='#333', capsize=3, linewidth=1.2, zorder=4)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    # Annotate significant rows
    for idx, row in df_plot.iterrows():
        if row['sig_int']:
            ax.text(row['beta_int'] + 1.96*row['se_int'] + 0.002, idx,
                    f'* p={row["p_int"]:.3f}', va='center', fontsize=8)
    n_sig = df_plot['sig_int'].sum()
    ax.set_title(f'{title}\\n({n_sig}/{len(df_plot)} sig.)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Interaction log-odds\\n(feature_z × ambiguity_z)', fontsize=10)
    ax.grid(axis='x', alpha=0.4)

fig.suptitle('GLMM Feature × Ambiguity Interaction\\n'
             'Positive = feature predicts investing more strongly at high ambiguity',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'glmm_feature_ambiguity_interaction.png', dpi=300, bbox_inches='tight')
plt.show()
"""))

cells.append(md_cell('## 8. Save & Summary'))

cells.append(code_cell("""\
gaze_int['modality']  = 'Gaze'
pupil_int['modality'] = 'Pupil'
eeg_int['modality']   = 'EEG'

all_int = pd.concat([gaze_int, pupil_int, eeg_int], ignore_index=True)
all_int.to_csv(OUT / 'glmm_feature_ambiguity_interactions.csv', index=False)

print("=" * 65)
print("SUMMARY: Feature × Ambiguity Interactions")
print("=" * 65)
for mod_df, label in [(gaze_int, 'Gaze'), (pupil_int, 'Pupil'), (eeg_int, 'EEG')]:
    sig = mod_df[mod_df['sig_int']]
    print(f"\\n{label} ({len(mod_df)} features, {len(sig)} sig. interactions):")
    if len(sig):
        print(sig[['feature', 'beta_int', 'p_int']].to_string(index=False))
    else:
        print("  None significant")

print("\\nInterpretation:")
print("  Positive β_int: feature predicts INVEST more strongly at HIGH ambiguity")
print("  Negative β_int: feature predicts invest LESS strongly at high ambiguity")
print("  (i.e., the feature-choice coupling attenuates under uncertainty)")
"""))

nb = {
    'nbformat': 4,
    'nbformat_minor': 4,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Created {NB_PATH} with {len(cells)} cells")
