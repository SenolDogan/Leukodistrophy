import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact, shapiro
from fpdf import FPDF
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
SIGNIFICANCE_LEVEL = 0.05

# 1. Load data
df = pd.read_csv('Combined.csv')
df['diagnosis_group'] = np.where(
    df['basic_diag_leuko'] == 2, 'Leukodystrophy',
    np.where(df['basic_diag_diff'] == 4, 'Multiple Sclerosis', None)
)
df = df[df['diagnosis_group'].isin(['Leukodystrophy', 'Multiple Sclerosis'])].reset_index(drop=True)

# 2. ML Pipeline (all basic, exam, mh features)
feature_cols = [col for col in df.columns if (
    (col.startswith('basic') or col.startswith('exam') or col.startswith('mh')) and
    col not in ['basic_diag_leuko', 'basic_diag_diff', 'diagnosis_group']
)]
X = df[feature_cols].copy()
y = df['diagnosis_group'].map({'Leukodystrophy': 0, 'Multiple Sclerosis': 1})
# Fill missing values
for col in X.select_dtypes(include=['float64', 'int64']).columns:
    X[col].fillna(X[col].median(), inplace=True)
for col in X.select_dtypes(include=['object']).columns:
    X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else '', inplace=True)
# Encode categorical
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))
# Drop columns with any remaining NaN values
X = X.dropna(axis=1)
feature_cols = list(X.columns)
# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
# SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
# Random Forest
rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf.fit(X_train_bal, y_train_bal)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
rf_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba)
}
# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'Random Forest (AUC={rf_metrics["roc_auc"]})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('rf_roc_curve.png')
plt.close()
# Feature importance
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(15)
plt.figure(figsize=(7,5))
sns.barplot(y='feature', x='importance', data=feat_imp)
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# 3. Statistical Analysis
stat_results = []
for col in feature_cols:
    if df[col].dropna().empty:
        continue
    unique_vals = df[col].dropna().unique()
    n_unique = len(unique_vals)
    group1 = df[df['diagnosis_group']=='Leukodystrophy'][col].dropna()
    group2 = df[df['diagnosis_group']=='Multiple Sclerosis'][col].dropna()
    if group1.empty or group2.empty:
        continue
    if np.issubdtype(df[col].dtype, np.number) and n_unique > 10:
        norm1 = shapiro(group1)[1] > 0.05 if len(group1) >= 3 else False
        norm2 = shapiro(group2)[1] > 0.05 if len(group2) >= 3 else False
        if norm1 and norm2:
            stat, p = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            test = 't-test'
        else:
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            test = 'Mann-Whitney U'
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
        stat_results.append({
            'variable': col,
            'test': test,
            'p_value': p,
            'group1_mean': group1.mean() if np.issubdtype(df[col].dtype, np.number) else group1.value_counts(normalize=True).to_dict(),
            'group2_mean': group2.mean() if np.issubdtype(df[col].dtype, np.number) else group2.value_counts(normalize=True).to_dict(),
            'n_unique': n_unique
        })

# 4. PDF Report
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Leukodystrophy vs Multiple Sclerosis: ML & Statistical Report', ln=1, align='C')
        self.ln(4)
    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, ln=1)
        self.ln(2)
    def section_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(1)
    def add_image(self, path, w=120):
        if os.path.exists(path):
            self.image(path, w=w)
            self.ln(4)

pdf = PDF()
pdf.add_page()
pdf.section_title('1. Introduction')
pdf.section_body('This report summarizes the results of machine learning classification and classical statistical analysis to distinguish Leukodystrophy and Multiple Sclerosis patients using the Combined.csv dataset. All features starting with basic, exam, or mh were included.')

pdf.section_title('2. Machine Learning Results')
pdf.section_body(f"Random Forest Classifier Test Set Metrics:\nAccuracy: {rf_metrics['accuracy']:.3f}\nPrecision: {rf_metrics['precision']:.3f}\nRecall: {rf_metrics['recall']:.3f}\nF1 Score: {rf_metrics['f1']:.3f}\nROC AUC: {rf_metrics['roc_auc']:.3f}")
pdf.section_body('ROC Curve:')
pdf.add_image('rf_roc_curve.png', w=100)
pdf.section_body('Top 15 Feature Importances:')
pdf.add_image('rf_feature_importance.png', w=120)

pdf.section_title('3. Statistically Significant Variables')
if stat_results:
    for r in stat_results:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 7, f"{r['variable']} (p={r['p_value']:.4g}, {r['test']})", ln=1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f"  Leukodystrophy: {r['group1_mean']}", ln=1)
        pdf.cell(0, 6, f"  Multiple Sclerosis: {r['group2_mean']}", ln=1)
        pdf.ln(1)
else:
    pdf.section_body('No statistically significant variables found (p < 0.05).')

pdf.output('LeukoMS_Report.pdf')
print('PDF report generated: LeukoMS_Report.pdf') 