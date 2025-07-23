import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

def preprocess_data(df):
    print("\n====== DATA PREPROCESSING ======\n")

    # Drop duplicates
    df = df.drop_duplicates()
    print("Removed duplicate rows.")

    # Handle missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values detected. Filling with median (numerical) or mode (categorical).")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print("No missing values detected.")

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"Encoding categorical columns: {list(cat_cols)}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        print("No categorical columns to encode.")

    # Feature scaling
    if 'target' in df.columns:
        features = df.drop(columns=['target'])
        target = df['target']
    else:
        features = df
        target = None

    num_cols = features.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(features[num_cols]), columns=num_cols)

    if target is not None:
        df_scaled['target'] = target.values

    print("Feature scaling complete.\n")
    return df_scaled

def descriptive_analysis(df):
    print("\n====== DESCRIPTIVE ANALYSIS ======\n")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print(df.describe(include='all').T)
    print("\nValue counts for each column:\n")
    for col in df.columns:
        print(f"{col}:")
        print(df[col].value_counts())
        print()

def eda_charts(df, output_dir="eda_charts"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEDA charts will be saved in: {output_dir}/\n")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_hist.png")
        plt.close()

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_box.png")
        plt.close()

    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

    print("Charts saved. Review them for visual insights.\n")

def predictive_analysis(df):
    print("\n====== PREDICTIVE ANALYSIS ======\n")
    target_col = "target"
    if target_col not in df.columns:
        print("Target column not found. Skipping predictive analysis.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"Random Forest Classifier - Accuracy: {acc:.2f}\n")
    print(report)
    print("Confusion Matrix:\n", cm)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Top features influencing heart disease prediction:\n")
    print(feature_importances.sort_values(ascending=False).head(7))
    return feature_importances

def prescriptive_analysis(feature_importances):
    print("\n====== PRESCRIPTIVE ANALYSIS ======\n")
    top_features = feature_importances.sort_values(ascending=False).head(3).index.tolist()
    print(f"To reduce heart disease risk, focus on managing these key factors: {', '.join(top_features)}.\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python heart_disease_analysis.py heart.csv")
        return

    fname = sys.argv[1]
    if not os.path.exists(fname):
        print(f"File not found: {fname}")
        return

    print(f"Loading data from {fname} ...\n")
    df_raw = pd.read_csv(fname)

    descriptive_analysis(df_raw)   # Raw data description
    eda_charts(df_raw)             # EDA on raw data

    df = preprocess_data(df_raw)   # Cleaned and scaled data for modeling

    feature_importances = predictive_analysis(df)
    if feature_importances is not None:
        prescriptive_analysis(feature_importances)

if __name__ == "__main__":
    main()
