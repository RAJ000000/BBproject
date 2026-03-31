import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif,
    RFE,
    SequentialFeatureSelector,
    RFECV
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ---------------------------------------------------
# 1. Load and combine datasets
# ---------------------------------------------------
def load_model1_dataset(real_csv_path, vae_csv_path, random_state=42):
    """
    Load real CMU and VAE synthetic datasets.
    Real CMU label = 0
    VAE synthetic label = 1
    """

    real_df = pd.read_csv(real_csv_path)
    vae_df = pd.read_csv(vae_csv_path)

    print("Real CMU shape:", real_df.shape)
    print("VAE shape     :", vae_df.shape)

    # Keep only common columns
    common_cols = [col for col in real_df.columns if col in vae_df.columns]

    real_df = real_df[common_cols].copy()
    vae_df = vae_df[common_cols].copy()

    print("\nNumber of common columns:", len(common_cols))
    print("First few common columns:", common_cols[:10])

    # Add binary labels
    real_df["label"] = 0
    vae_df["label"] = 1

    # Combine
    combined_df = pd.concat([real_df, vae_df], ignore_index=True)

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\nCombined shape:", combined_df.shape)
    print("\nLabel counts:")
    print(combined_df["label"].value_counts())

    return combined_df


# ---------------------------------------------------
# 2. Preprocess data
# ---------------------------------------------------
def preprocess_model1_data(df, drop_cols=None):
    """
    Remove non-feature columns and return X, y.
    """

    if drop_cols is None:
        drop_cols = ["subject", "sessionIndex", "rep"]

    df = df.copy()

    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    print("\nDropping non-feature columns:", actual_drop_cols)

    X = df.drop(columns=actual_drop_cols + ["label"], errors="ignore")
    y = df["label"]

    # Convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill missing values with column mean
    X = X.fillna(X.mean())

    print("\nFeature matrix shape:", X.shape)
    print("Target shape        :", y.shape)

    print("\nFirst 5 rows of processed features:")
    print(X.head())

    return X, y


# ---------------------------------------------------
# 3. Split and scale
# ---------------------------------------------------
def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    80/20 split and standard scaling
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTrain shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------
# 4. Filter Method 1: Mutual Information
# ---------------------------------------------------
def filter_mutual_info(X_train, y_train, X_test, k=10):
    scores = mutual_info_classif(X_train, y_train, random_state=42)

    score_df = pd.DataFrame({
        "feature": X_train.columns,
        "score": scores
    }).sort_values(by="score", ascending=False)

    selected_features = score_df.head(k)["feature"].tolist()

    print("\nTop features by Mutual Information:")
    print(score_df.head(k))

    return X_train[selected_features], X_test[selected_features], selected_features, score_df


# ---------------------------------------------------
# 5. Filter Method 2: ANOVA F-test
# ---------------------------------------------------
def filter_anova(X_train, y_train, X_test, k=10):
    scores, pvalues = f_classif(X_train, y_train)

    score_df = pd.DataFrame({
        "feature": X_train.columns,
        "score": scores,
        "pvalue": pvalues
    }).sort_values(by="score", ascending=False)

    selected_features = score_df.head(k)["feature"].tolist()

    print("\nTop features by ANOVA F-test:")
    print(score_df.head(k))

    return X_train[selected_features], X_test[selected_features], selected_features, score_df


# ---------------------------------------------------
# 6. Filter Method 3: Correlation-based filtering
# ---------------------------------------------------
def filter_correlation(X_train, y_train, X_test, k=10):
    """
    Select top-k features based on absolute correlation with target.
    """

    train_df = X_train.copy()
    train_df["label"] = y_train.values

    corr_series = train_df.corr(numeric_only=True)["label"].drop("label")
    corr_abs = corr_series.abs().sort_values(ascending=False)

    selected_features = corr_abs.head(k).index.tolist()

    score_df = pd.DataFrame({
        "feature": corr_abs.index,
        "abs_correlation_with_label": corr_abs.values
    })

    print("\nTop features by Correlation Filtering:")
    print(score_df.head(k))

    return X_train[selected_features], X_test[selected_features], selected_features, score_df


# ---------------------------------------------------
# 7. Wrapper Method 1: RFE with Random Forest
# ---------------------------------------------------
def wrapper_rfe_rf(X_train, y_train, X_test, k=10, random_state=42):
    estimator = RandomForestClassifier(random_state=random_state, n_estimators=100)

    selector = RFE(
        estimator=estimator,
        n_features_to_select=k
    )

    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.support_].tolist()

    print("\nTop features by RFE with Random Forest:")
    print(selected_features)

    return X_train[selected_features], X_test[selected_features], selected_features, selector


# ---------------------------------------------------
# 8. Wrapper Method 2: SFS with Random Forest
# ---------------------------------------------------
def wrapper_sfs_rf(X_train, y_train, X_test, k=10, random_state=42):
    estimator = RandomForestClassifier(random_state=random_state, n_estimators=100)

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=k,
        direction="forward",
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.get_support()].tolist()

    print("\nTop features by SFS with Random Forest:")
    print(selected_features)

    return X_train[selected_features], X_test[selected_features], selected_features, selector


# ---------------------------------------------------
# 9. Wrapper Method 3: RFECV with Random Forest
# ---------------------------------------------------
def wrapper_rfecv_rf(X_train, y_train, X_test, min_features_to_select=5, random_state=42):
    estimator = RandomForestClassifier(random_state=random_state, n_estimators=100)

    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )

    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.support_].tolist()

    print("\nTop features by RFECV with Random Forest:")
    print(selected_features)
    print("Number of selected features:", len(selected_features))

    return X_train[selected_features], X_test[selected_features], selected_features, selector


# ---------------------------------------------------
# 10. Classifiers
# ---------------------------------------------------
def get_classifiers(random_state=42):
    models = {
        "SVM": SVC(probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state)
    }
    return models


# ---------------------------------------------------
# 11. Evaluate classifiers with 80/20 split + 10-fold CV
# ---------------------------------------------------
def evaluate_classifiers(X_train, X_test, y_train, y_test, cv=10):
    models = get_classifiers()
    results = {}

    for name, model in models.items():
        print("\n" + "=" * 70)
        print("Model:", name)

        # Fit on training set
        model.fit(X_train, y_train)

        # Test set prediction
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("80/20 Test Accuracy:")
        print(acc)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(report)

        # ROC-AUC
        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
            print("ROC-AUC:", roc_auc)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_score)
            print("ROC-AUC:", roc_auc)

        # 10-fold cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

        print("\n10-Fold CV Accuracy Scores:")
        print(cv_scores)

        print("Mean CV Accuracy:", cv_scores.mean())
        print("Std CV Accuracy :", cv_scores.std())

        results[name] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "roc_auc": roc_auc,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }

    return results


# ---------------------------------------------------
# 12. Best model by test accuracy
# ---------------------------------------------------
def get_best_model(results):
    best_name = max(results, key=lambda x: results[x]["accuracy"])
    best_model = results[best_name]["model"]

    print("\nBest model:", best_name)
    print("Best test accuracy:", results[best_name]["accuracy"])
    print("Best CV mean      :", results[best_name]["cv_mean"])

    return best_name, best_model


# ---------------------------------------------------
# 13. ROC data for one chosen model
# ---------------------------------------------------
def get_roc_data(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print("This model does not support ROC scoring.")
        return None, None, None

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc_value = roc_auc_score(y_test, y_score)

    print("\nROC-AUC:", auc_value)

    return fpr, tpr, auc_value


# ---------------------------------------------------
# 14. Show summary table of results
# ---------------------------------------------------
def summarize_results(results):
    rows = []

    for model_name, info in results.items():
        rows.append({
            "Model": model_name,
            "Test Accuracy": info["accuracy"],
            "ROC-AUC": info["roc_auc"],
            "CV Mean Accuracy": info["cv_mean"],
            "CV Std": info["cv_std"]
        })

    summary_df = pd.DataFrame(rows).sort_values(by="Test Accuracy", ascending=False)

    print("\nSummary of all classifiers:")
    print(summary_df)

    return summary_df