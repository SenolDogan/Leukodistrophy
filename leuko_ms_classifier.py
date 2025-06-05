import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_prepare_data():
    """Load and prepare the data for analysis"""
    print("\n1. Loading and Preparing Data...")
    
    # Load datasets
    combined_df = pd.read_csv('Combined.csv')
    metadata_df = pd.read_csv('metadata.csv')
    
    # Create diagnosis_group column with dtype=object (corrected codes)
    combined_df['diagnosis_group'] = np.where(
        combined_df['basic_diag_leuko'] == 2, 'Leukodystrophy',
        np.where(combined_df['basic_diag_diff'] == 4, 'Multiple Sclerosis', None)
    ).astype(object)
    
    # Filter to only Leukodystrophy and MS
    combined_df = combined_df[combined_df['diagnosis_group'].isin(['Leukodystrophy', 'Multiple Sclerosis'])]
    combined_df = combined_df.reset_index(drop=True)
    
    print("\nDataset shapes:")
    print(f"Combined data: {combined_df.shape}")
    print(f"Metadata: {metadata_df.shape}")
    
    return combined_df, metadata_df

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    print("\n2. Exploratory Data Analysis...")
    
    # Data info
    print("\nData Info:")
    print(df.info())
    
    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Save numerical statistics
    df.describe().to_csv('numerical_statistics.csv')
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    plt.title('Distribution of Target Classes')
    df['diagnosis_group'].value_counts().plot(kind='bar')
    plt.tight_layout()
    plt.savefig('target_distribution.png')
    plt.close()
    
    return df

def preprocess_data(df):
    """Preprocess the data for modeling using all basic, exam, and mh features."""
    print("\n3. Preprocessing Data...")
    
    # Create binary target: Leukodystrophy vs MS
    df['target'] = df['diagnosis_group'].map({
        'Leukodystrophy': 0,
        'Multiple Sclerosis': 1
    })
    
    # Select all columns starting with 'basic', 'exam', or 'mh' (except diagnosis columns)
    feature_cols = [col for col in df.columns if (
        (col.startswith('basic') or col.startswith('exam') or col.startswith('mh')) and
        col not in ['basic_diag_leuko', 'basic_diag_diff', 'diagnosis_group', 'target']
    )]
    X = df[feature_cols].copy()
    y = df['target']
    # Fill missing values
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col].fillna(X[col].median(), inplace=True)
    for col in X.select_dtypes(include=['object']).columns:
        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else '', inplace=True)
    # Encode categorical variables
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
    # Drop columns with any remaining NaN values
    X = X.dropna(axis=1)
    # Update feature_cols to match X
    feature_cols = list(X.columns)
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, y, feature_cols

def train_and_evaluate_models(X, y, feature_cols):
    """Train and evaluate multiple models, return feature importances."""
    print("\n4. Training and Evaluating Models...")
    from collections import defaultdict
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE)
    }
    # Model parameters for grid search
    param_grid = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    }
    best_models = {}
    model_results = {}
    feature_importances = defaultdict(list)
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid[name], cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train_balanced, y_train_balanced)
        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        # Training set metrics
        y_train_pred = best_model.predict(X_train_balanced)
        y_train_proba = best_model.predict_proba(X_train_balanced)[:, 1]
        train_results = {
            'accuracy': accuracy_score(y_train_balanced, y_train_pred),
            'precision': precision_score(y_train_balanced, y_train_pred),
            'recall': recall_score(y_train_balanced, y_train_pred),
            'f1': f1_score(y_train_balanced, y_train_pred),
            'roc_auc': roc_auc_score(y_train_balanced, y_train_proba)
        }
        # Test set metrics
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        test_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        # Cross-validation ROC AUC
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        model_results[name] = {
            'train': train_results,
            'test': test_results,
            'cv_roc_auc_mean': cv_mean,
            'cv_roc_auc_std': cv_std
        }
        # Feature importances (only for Random Forest)
        if name == 'Random Forest':
            importances = best_model.feature_importances_
            for i, col in enumerate(feature_cols):
                feature_importances[col].append(importances[i])
    return best_models, model_results, feature_importances

def save_results(model_results):
    """Save model results to a file"""
    print("\n5. Saving Results...")
    
    with open('model_results.txt', 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=======================\n\n")
        for model_name, results in model_results.items():
            f.write(f"{model_name}:\n")
            f.write("-" * len(model_name) + "\n")
            f.write("Training Set:\n")
            for metric, value in results['train'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("Test Set:\n")
            for metric, value in results['test'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write(f"Cross-Validation ROC AUC (5-fold): {results['cv_roc_auc_mean']:.4f} ± {results['cv_roc_auc_std']:.4f}\n")
            f.write("\n")

def save_feature_stats(df, feature_cols):
    """Save patient counts and percentages for each feature by group."""
    results = []
    for group in ['Leukodystrophy', 'Multiple Sclerosis']:
        group_df = df[df['diagnosis_group'] == group]
        N = len(group_df)
        results.append(f"\n--- {group} (N={N}) ---\n")
        for col in feature_cols:
            if col in group_df.columns:
                val_counts = group_df[col].value_counts(dropna=True)
                if not val_counts.empty:
                    most_common = val_counts.idxmax()
                    count = val_counts.max()
                    percent = 100 * count / N if N > 0 else 0
                    results.append(f"  {col}: {most_common} ({count}, {percent:.1f}%)")
    with open('feature_patient_stats.txt', 'w') as f:
        f.write('\n'.join(results))

def most_common_findings(df):
    """Find and save the most common clinical findings and HPO codes for each group, in English, with N and %."""
    clinical_cols = [
        'exam_fati','exam_gait','exam_progress','exam_muscpath___1','exam_muscpath___2','exam_muscpath___3','exam_muscpath___4','exam_muscpath___5','exam_muscpath___6','exam_muscpath___7',
        'exam_speech','exam_seiz','exam_spast','exam_pysign','exam_vertigo','exam_bd','exam_bowel___1','exam_bowel___2','exam_bowel___3','exam_bowel___4','exam_bowel___5','exam_bowel___6',
        'exam_prs','exam_numb','exam_pain___1','exam_pain___2','exam_pain___3','exam_pain___4','exam_pain___5','exam_atax','exam_trem','exam_loss','exam_color','exam_emd','exam_dipl',
        'exam_np___0','exam_np___1','exam_np___2','exam_np___3','exam_np___4'
    ]
    hpo_cols = ['exam_hpo1','exam_hpo2','exam_hpo3','exam_hpo4','exam_hpo5','exam_hpo6','mh_hpo_pi1','mh_hpo_pi2','mh_hpo_pi3','mh_hpo_pi4','mh_hpo_pi5','mh_hpo_pi6']
    
    results = []
    for group in ['Leukodystrophy', 'Multiple Sclerosis']:
        group_df = df[df['diagnosis_group'] == group]
        N = len(group_df)
        results.append(f"\n--- {group} (N={N}) ---\n")
        # Clinical findings (binary or categorical)
        results.append("Most common clinical findings (N, %):")
        for col in clinical_cols:
            if col in group_df.columns:
                val_counts = group_df[col].value_counts(dropna=True)
                if not val_counts.empty:
                    most_common = val_counts.idxmax()
                    count = val_counts.max()
                    percent = 100 * count / N if N > 0 else 0
                    results.append(f"  {col}: {most_common} ({count}, {percent:.1f}%)")
        # HPO codes
        hpo_values = []
        for col in hpo_cols:
            if col in group_df.columns:
                hpo_values += group_df[col].dropna().astype(str).tolist()
        from collections import Counter
        hpo_counter = Counter([h for h in hpo_values if h and h != '0' and h != 'nan'])
        results.append("Most common HPO codes (N, %):")
        for hpo, count in hpo_counter.most_common(10):
            percent = 100 * count / N if N > 0 else 0
            results.append(f"  HPO:{hpo} ({count}, {percent:.1f}%)")
    with open('most_common_findings.txt', 'w') as f:
        f.write('\n'.join(results))

def main():
    """Main function to run the entire pipeline"""
    print("Starting Leukodystrophy vs MS Classification Pipeline...")
    
    # Load and prepare data
    combined_df, metadata_df = load_and_prepare_data()
    
    # Perform EDA
    df = perform_eda(combined_df)
    
    # Preprocess data
    X, y, feature_cols = preprocess_data(df)
    
    # Train and evaluate models
    best_models, model_results, feature_importances = train_and_evaluate_models(X, y, feature_cols)
    
    # Save results
    save_results(model_results)
    
    # Save feature stats
    save_feature_stats(df, feature_cols)
    
    # Most common findings
    most_common_findings(df)
    
    print("\nPipeline completed successfully!")
    print("Check the following files for results:")
    print("- model_results.txt")
    print("- numerical_statistics.csv")
    print("- target_distribution.png")
    print("- roc_curve_random_forest.png")
    print("- roc_curve_xgboost.png")
    print("- feature_importance.png")
    print("- most_common_findings.txt")
    print("- feature_patient_stats.txt")

if __name__ == "__main__":
    main() 
