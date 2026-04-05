import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif,
    VarianceThreshold,
    RFE
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# 1. Load and combine data
def load_data(real_file, synthetic_file):
    real_df = pd.read_csv(real_file)
    synthetic_df = pd.read_csv(synthetic_file)

    common_cols = [col for col in real_df.columns if col in synthetic_df.columns]

    real_df = real_df[common_cols].copy()
    synthetic_df = synthetic_df[common_cols].copy()

    real_df["label"] = 0
    synthetic_df["label"] = 1

    df = pd.concat([real_df, synthetic_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Combined shape:", df.shape)
    print("\nClass counts:")
    print(df["label"].value_counts())

    return df


# 2. Preprocess data
def preprocess_data(df):
    df = df.copy()

    drop_cols = ["subject", "sessionIndex", "rep"]
    drop_cols = [col for col in drop_cols if col in df.columns]

    print("\nDropping columns:", drop_cols)

    X = df.drop(columns=drop_cols + ["label"], errors="ignore")
    y = df["label"]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    print("\nFeature shape:", X.shape)
    print("Label shape:", y.shape)

    return X, y


# 3. Split data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# 4. Scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# 5. Filter 1: Mutual Information
def select_mutual_info(X_train, y_train, X_test, k=10):
    scores = mutual_info_classif(X_train, y_train, random_state=42)

    score_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Score": scores
    }).sort_values(by="Score", ascending=False)

    selected = score_df.head(k)["Feature"].tolist()

    print("\nTop Mutual Information Features:")
    print(score_df.head(k))

    return X_train[selected], X_test[selected], selected, score_df


# 6. Filter 2: ANOVA
def select_anova(X_train, y_train, X_test, k=10):
    scores, pvalues = f_classif(X_train, y_train)

    score_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Score": scores,
        "P_Value": pvalues
    }).sort_values(by="Score", ascending=False)

    selected = score_df.head(k)["Feature"].tolist()

    print("\nTop ANOVA Features:")
    print(score_df.head(k))

    return X_train[selected], X_test[selected], selected, score_df


# 7. Filter 3: Correlation
def select_correlation(X_train, y_train, X_test, k=10):
    temp_df = X_train.copy()
    temp_df["label"] = y_train.values

    corr = temp_df.corr(numeric_only=True)["label"].drop("label")
    corr = corr.abs().sort_values(ascending=False)

    score_df = pd.DataFrame({
        "Feature": corr.index,
        "Score": corr.values
    })

    selected = score_df.head(k)["Feature"].tolist()

    print("\nTop Correlation Features:")
    print(score_df.head(k))

    return X_train[selected], X_test[selected], selected, score_df


# 8. Filter 4: Variance Threshold
def select_variance(X_train, X_test, threshold=0.0, k=10):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)

    var_values = X_train.var().sort_values(ascending=False)

    score_df = pd.DataFrame({
        "Feature": var_values.index,
        "Score": var_values.values
    })

    selected = score_df.head(k)["Feature"].tolist()

    print("\nTop Variance Features:")
    print(score_df.head(k))

    return X_train[selected], X_test[selected], selected, score_df


# 9. Filter 5: Mean Difference
def select_mean_difference(X_train, y_train, X_test, k=10):
    real_mean = X_train[y_train == 0].mean()
    synthetic_mean = X_train[y_train == 1].mean()

    diff = (real_mean - synthetic_mean).abs().sort_values(ascending=False)

    score_df = pd.DataFrame({
        "Feature": diff.index,
        "Score": diff.values
    })

    selected = score_df.head(k)["Feature"].tolist()

    print("\nTop Mean Difference Features:")
    print(score_df.head(k))

    return X_train[selected], X_test[selected], selected, score_df


# 10. Wrapper 1: RFE Logistic Regression
def select_rfe_lr(X_train, y_train, X_test, k=10):
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    selector = RFE(model, n_features_to_select=k, step=2)

    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()

    print("\nRFE Logistic Regression Features:")
    print(selected)

    return X_train[selected], X_test[selected], selected


# 11. Wrapper 2: RFE Linear SVM
def select_rfe_svm(X_train, y_train, X_test, k=10):
    model = LinearSVC(max_iter=5000)
    selector = RFE(model, n_features_to_select=k, step=2)

    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()

    print("\nRFE Linear SVM Features:")
    print(selected)

    return X_train[selected], X_test[selected], selected


# 12. Wrapper 3: RFE Random Forest
def select_rfe_rf(X_train, y_train, X_test, k=10):
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(model, n_features_to_select=k, step=2)

    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()

    print("\nRFE Random Forest Features:")
    print(selected)

    return X_train[selected], X_test[selected], selected


# 13. Wrapper 4: Model-based LR (top absolute coefficients)
def select_sfm_lr(X_train, y_train, X_test, k=10):
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=1000
    )
    model.fit(X_train, y_train)

    importance = np.abs(model.coef_[0])

    score_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Score": importance
    }).sort_values(by="Score", ascending=False)

    selected = score_df.head(k)["Feature"].tolist()

    print("\nSelectFromModel Logistic Regression Features:")
    print(score_df.head(k))

    return X_train[selected], X_test[selected], selected, score_df


# 14. Wrapper 5: Model-based RF (top feature importances)
def select_sfm_rf(X_train, y_train, X_test, k=10):
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    importance = model.feature_importances_

    score_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Score": importance
    }).sort_values(by="Score", ascending=False)

    selected = score_df.head(k)["Feature"].tolist()

    print("\nSelectFromModel Random Forest Features:")
    print(score_df.head(k))

    return X_train[selected], X_test[selected], selected, score_df


# 15. Models
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Linear SVM": LinearSVC(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    return models


# 16. Evaluate models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = []

    for name, model in models.items():
        print("\n" + "=" * 60)
        print("Model:", name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("Accuracy:", acc)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_score)

        print("ROC-AUC:", roc_auc)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "ROC_AUC": roc_auc,
            "Confusion_Matrix": cm,
            "Model_Object": model
        })

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

    print("\nResults Table:")
    print(results_df[["Model", "Accuracy", "ROC_AUC"]])

    return results_df


# 17. Validate best model
def validate_best_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=3)

    print("\n3-Fold CV Scores:", scores)
    print("CV Mean:", scores.mean())

    return scores


# 18. Plot confusion matrix
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


# 19. Plot ROC
def plot_roc(model, X_test, y_test, title="ROC Curve"):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print("ROC not available for this model")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_value = roc_auc_score(y_test, y_score)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()


# 20. Plot feature distribution
def plot_feature_distribution(df, feature_name):
    real = df[df["label"] == 0][feature_name]
    synthetic = df[df["label"] == 1][feature_name]

    print("\nFeature:", feature_name)
    print("Real Mean:", real.mean())
    print("Synthetic Mean:", synthetic.mean())
    print("Real Std:", real.std())
    print("Synthetic Std:", synthetic.std())

    plt.figure(figsize=(6, 4))
    plt.hist(real, bins=30, alpha=0.5, label="Real")
    plt.hist(synthetic, bins=30, alpha=0.5, label="Synthetic")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature_name}")
    plt.legend()
    plt.show()


# 21. Feature selection summary
def feature_selection_summary(*feature_lists):
    all_features = []

    for feature_list in feature_lists:
        all_features.extend(feature_list)

    counts = Counter(all_features)

    summary_df = pd.DataFrame({
        "Feature": list(counts.keys()),
        "Times_Selected": list(counts.values())
    }).sort_values(by="Times_Selected", ascending=False)

    print("\nFeature Selection Summary:")
    print(summary_df)

    return summary_df