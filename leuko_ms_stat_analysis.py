import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact, shapiro
import warnings
warnings.filterwarnings('ignore')

SIGNIFICANCE_LEVEL = 0.05

# Load data
df = pd.read_csv('Combined.csv')

# Create diagnosis_group

df['diagnosis_group'] = np.where(
    df['basic_diag_leuko'] == 2, 'Leukodystrophy',
    np.where(df['basic_diag_diff'] == 4, 'Multiple Sclerosis', None)
)
df = df[df['diagnosis_group'].isin(['Leukodystrophy', 'Multiple Sclerosis'])].reset_index(drop=True)

# List of features to analyze
feature_cols = [col for col in df.columns if (
    (col.startswith('basic') or col.startswith('exam') or col.startswith('mh')) and
    col not in ['basic_diag_leuko', 'basic_diag_diff', 'diagnosis_group']
)]

results = []

for col in feature_cols:
    if df[col].dropna().empty:
        continue
    # Determine variable type
    unique_vals = df[col].dropna().unique()
    n_unique = len(unique_vals)
    group1 = df[df['diagnosis_group']=='Leukodystrophy'][col].dropna()
    group2 = df[df['diagnosis_group']=='Multiple Sclerosis'][col].dropna()
    # Skip if one group is empty
    if group1.empty or group2.empty:
        continue
    # Numeric (continuous) variable
    if np.issubdtype(df[col].dtype, np.number) and n_unique > 10:
        # Normality test
        norm1 = shapiro(group1)[1] > 0.05 if len(group1) >= 3 else False
        norm2 = shapiro(group2)[1] > 0.05 if len(group2) >= 3 else False
        if norm1 and norm2:
            stat, p = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            test = 't-test'
        else:
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            test = 'Mann-Whitney U'
    # Categorical variable (binary or multi)
    elif n_unique <= 10:
        contingency = pd.crosstab(df['diagnosis_group'], df[col])
        if contingency.shape == (2,2):
            try:
                stat, p = fisher_exact(contingency)
                test = 'Fisher Exact'
            except:
                stat, p, _, _ = chi2_contingency(contingency)
                test = 'Chi-square'
        else:
            stat, p, _, _ = chi2_contingency(contingency)
            test = 'Chi-square'
    else:
        continue
    if p < SIGNIFICANCE_LEVEL:
        results.append({
            'variable': col,
            'test': test,
            'p_value': p,
            'group1_mean': group1.mean() if np.issubdtype(df[col].dtype, np.number) else group1.value_counts(normalize=True).to_dict(),
            'group2_mean': group2.mean() if np.issubdtype(df[col].dtype, np.number) else group2.value_counts(normalize=True).to_dict(),
            'n_unique': n_unique
        })

# Save significant results
summary = []
summary.append('Significant variables (p < 0.05) distinguishing Leukodystrophy vs MS\n')
for r in results:
    summary.append(f"{r['variable']} | Test: {r['test']} | p={r['p_value']:.4g}")
    summary.append(f"  Leukodystrophy: {r['group1_mean']}")
    summary.append(f"  Multiple Sclerosis: {r['group2_mean']}\n")

with open('stat_significant_results.txt', 'w') as f:
    f.write('\n'.join(summary))

print('Statistical analysis complete. See stat_significant_results.txt for significant results.') 