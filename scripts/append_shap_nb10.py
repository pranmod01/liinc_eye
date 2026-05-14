"""Append feature-level SHAP cells to notebook 10_modality_shapley_by_ambiguity.ipynb"""
import json

NB_PATH = 'notebooks/further_analysis/10_modality_shapley_by_ambiguity.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

shap_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': (
        '## 9. Feature-Level SHAP Values by Ambiguity\n'
        '\n'
        '**Question**: Within each ambiguity stratum, *which individual features* are most '
        'predictive of investment decisions?\n'
        '\n'
        'The modality-level Shapley analysis reveals that Gaze dominates at low ambiguity '
        'and Behavior at medium/high — but not which specific features drive that. Here we '
        'train one RF per stratum on **all features combined** and compute SHAP values to '
        'rank individual features within each ambiguity context.'
    )
}

shap_code = {
    'cell_type': 'code',
    'metadata': {},
    'outputs': [],
    'execution_count': None,
    'source': """\
try:
    import shap
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
    import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

all_feature_cols = behavior_cols + physio_cols + gaze_cols + eeg_cols
modality_labels = (
    ['Behavior'] * len(behavior_cols) +
    ['Physio']   * len(physio_cols) +
    ['Gaze']     * len(gaze_cols) +
    ['EEG']      * len(eeg_cols)
)

shap_results = {}   # amb_level -> DataFrame of mean |SHAP| per feature

for amb_level in ['Low', 'Medium', 'High']:
    print(f"\\nComputing SHAP values — {amb_level} ambiguity...")
    amb_df = merged_df[merged_df['ambiguity_group'] == amb_level].copy()

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(amb_df[all_feature_cols])
    y = amb_df['outcome'].values

    clf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    # Binary RF: shap_values returns list of 2 arrays (class 0, class 1); use class 1
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': all_feature_cols,
        'modality': modality_labels,
        'mean_abs_shap': mean_abs_shap,
        'ambiguity': amb_level
    }).sort_values('mean_abs_shap', ascending=False)
    shap_df['rank'] = range(1, len(shap_df) + 1)
    shap_results[amb_level] = shap_df
    print(f"  Top 5: {shap_df['feature'].head(5).tolist()}")

print("\\nDone.")
"""
}

rank_code = {
    'cell_type': 'code',
    'metadata': {},
    'outputs': [],
    'execution_count': None,
    'source': """\
# Rank-shift table: how does each feature's rank change across ambiguity levels?
rank_wide = pd.DataFrame({
    'feature':  shap_results['Low']['feature'].values,
    'modality': shap_results['Low']['modality'].values,
})
for amb in ['Low', 'Medium', 'High']:
    rank_wide = rank_wide.merge(
        shap_results[amb][['feature', 'mean_abs_shap', 'rank']].rename(
            columns={'mean_abs_shap': f'shap_{amb}', 'rank': f'rank_{amb}'}),
        on='feature'
    )

rank_wide['max_rank_shift'] = (
    rank_wide[['rank_Low', 'rank_Medium', 'rank_High']].max(axis=1) -
    rank_wide[['rank_Low', 'rank_Medium', 'rank_High']].min(axis=1)
)
rank_wide = rank_wide.sort_values('rank_Low')

print("Feature ranks across ambiguity levels (sorted by Low-ambiguity rank):")
print(rank_wide[['feature', 'modality', 'rank_Low', 'rank_Medium', 'rank_High', 'max_rank_shift']]
      .to_string(index=False))
"""
}

viz_code = {
    'cell_type': 'code',
    'metadata': {},
    'outputs': [],
    'execution_count': None,
    'source': """\
mod_colors = {'Behavior': '#2E86AB', 'Physio': '#F18F01', 'Gaze': '#C73E1D', 'EEG': '#A23B72'}
top_n = 15

fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)

for ax, amb in zip(axes, ['Low', 'Medium', 'High']):
    top = shap_results[amb].head(top_n).iloc[::-1]  # reverse for horizontal bar
    bar_colors = [mod_colors[m] for m in top['modality']]
    ax.barh(top['feature'], top['mean_abs_shap'], color=bar_colors, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Mean |SHAP| value', fontweight='bold')
    ax.set_title(f'{amb} Ambiguity\\n(top {top_n} features)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=m) for m, c in mod_colors.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Feature-Level SHAP Values by Ambiguity Stratum\\n'
             '(RF trained on all features within stratum; descriptive, not LOSO)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUTPUT_DIR / 'shap_by_ambiguity_feature_level.png', dpi=300, bbox_inches='tight')
plt.show()
"""
}

heatmap_code = {
    'cell_type': 'code',
    'metadata': {},
    'outputs': [],
    'execution_count': None,
    'source': """\
# Heatmap: mean |SHAP| per feature × ambiguity, normalised within each stratum
shap_wide = rank_wide[['feature', 'modality', 'shap_Low', 'shap_Medium', 'shap_High']].copy()
top20 = shap_wide.sort_values('shap_Low', ascending=False).head(20).set_index('feature')

heat_data = top20[['shap_Low', 'shap_Medium', 'shap_High']]
heat_norm = heat_data.div(heat_data.max())
mod_order = top20['modality']

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(heat_norm.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Low', 'Medium', 'High'], fontsize=12)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
ax.set_xlabel('Ambiguity Level', fontweight='bold')
ax.set_title('Relative Feature Importance across Ambiguity\\n'
             '(top-20 by Low-ambiguity SHAP, normalised per column)',
             fontweight='bold', fontsize=11)
for i, mod in enumerate(mod_order):
    ax.add_patch(plt.Rectangle((-0.45, i - 0.5), 0.35, 1,
                               color=mod_colors[mod], clip_on=False))
plt.colorbar(im, ax=ax, label='Normalised |SHAP|')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=m) for m, c in mod_colors.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0.05, 0.04, 1, 1])
plt.savefig(OUTPUT_DIR / 'shap_heatmap_ambiguity.png', dpi=300, bbox_inches='tight')
plt.show()
"""
}

save_code = {
    'cell_type': 'code',
    'metadata': {},
    'outputs': [],
    'execution_count': None,
    'source': """\
for amb in ['Low', 'Medium', 'High']:
    shap_results[amb].to_csv(OUTPUT_DIR / f'shap_feature_level_{amb.lower()}_ambiguity.csv', index=False)
rank_wide.to_csv(OUTPUT_DIR / 'shap_feature_rank_by_ambiguity.csv', index=False)

print("Feature-level SHAP results saved.")
print("\\nTop-3 features per ambiguity stratum:")
for amb in ['Low', 'Medium', 'High']:
    top3 = shap_results[amb].head(3)
    print(f"  {amb}: {list(zip(top3['feature'], top3['modality']))}")
"""
}

new_cells = [shap_md, shap_code, rank_code, viz_code, heatmap_code, save_code]
nb['cells'].extend(new_cells)

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Notebook 10 updated — appended {len(new_cells)} cells")
