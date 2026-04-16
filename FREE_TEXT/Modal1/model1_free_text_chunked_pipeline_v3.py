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


# ---------------------------------------------------
# 1. Clean raw free-text dataset
# ---------------------------------------------------
def clean_free_text_df(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    junk_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if junk_cols:
        df = df.drop(columns=junk_cols)

    expected_cols = [
        "participant",
        "session",
        "key1",
        "key2",
        "DU.key1.key1",
        "DD.key1.key2",
        "DU.key1.key2",
        "UD.key1.key2",
        "UU.key1.key2"
    ]

    df = df[[c for c in expected_cols if c in df.columns]].copy()

    timing_cols = [
        "DU.key1.key1",
        "DD.key1.key2",
        "DU.key1.key2",
        "UD.key1.key2",
        "UU.key1.key2"
    ]

    for col in timing_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["participant", "session"])

    for col in timing_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


# ---------------------------------------------------
# 2. Split sessions FIRST
# ---------------------------------------------------
def split_sessions(df, test_size=0.2, random_state=42):
    session_df = df[["participant", "session"]].drop_duplicates().reset_index(drop=True)

    train_sessions, test_sessions = train_test_split(
        session_df,
        test_size=test_size,
        random_state=random_state
    )

    train_df = df.merge(train_sessions, on=["participant", "session"], how="inner")
    test_df = df.merge(test_sessions, on=["participant", "session"], how="inner")

    return train_df, test_df


# ---------------------------------------------------
# 3. Chunk one dataframe by participant/session
# ---------------------------------------------------
def make_chunks(df, chunk_size=50):
    chunks = []

    grouped = df.groupby(["participant", "session"], sort=False)

    for (participant, session), group in grouped:
        group = group.reset_index(drop=True)

        n = len(group)
        num_chunks = n // chunk_size

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = start + chunk_size
            chunk = group.iloc[start:end].copy()

            if len(chunk) == chunk_size:
                chunk["chunk_id"] = chunk_id
                chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=list(df.columns) + ["chunk_id"])

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------
# 4. Aggregate chunk-level features
#    v3: no min/max, no ratio features
# ---------------------------------------------------
def aggregate_chunk_features(chunked_df):
    df = chunked_df.copy()

    hold_col = "DU.key1.key1"
    dd_col = "DD.key1.key2"
    du_flight_col = "DU.key1.key2"
    ud_col = "UD.key1.key2"
    uu_col = "UU.key1.key2"

    group_cols = ["participant", "session", "chunk_id"]
    grouped = df.groupby(group_cols)

    agg_df = grouped.agg(
        DU_hold_mean=(hold_col, "mean"),
        DU_hold_std=(hold_col, "std"),

        DD_mean=(dd_col, "mean"),
        DD_std=(dd_col, "std"),

        DU_flight_mean=(du_flight_col, "mean"),
        DU_flight_std=(du_flight_col, "std"),

        UD_mean=(ud_col, "mean"),
        UD_std=(ud_col, "std"),

        UU_mean=(uu_col, "mean"),
        UU_std=(uu_col, "std"),

        total_events=(hold_col, "count")
    ).reset_index()

    total_time_df = grouped[dd_col].sum().reset_index(name="total_time")
    agg_df = agg_df.merge(total_time_df, on=group_cols, how="left")

    agg_df["avg_time_per_event"] = agg_df["total_time"] / (agg_df["total_events"] + 1e-5)

    base_mean_cols = [
        "DU_hold_mean",
        "DD_mean",
        "DU_flight_mean",
        "UD_mean",
        "UU_mean"
    ]

    agg_df["overall_mean"] = agg_df[base_mean_cols].mean(axis=1)
    agg_df["overall_std"] = agg_df[base_mean_cols].std(axis=1)

    agg_df = agg_df.fillna(0)

    return agg_df


# ---------------------------------------------------
# 5. Full train/test preparation
# ---------------------------------------------------
def prepare_train_test_data(real_file, synthetic_file, chunk_size=50, test_size=0.2, random_state=42):
    real_df = pd.read_csv(real_file, low_memory=False)
    synthetic_df = pd.read_csv(synthetic_file, low_memory=False)

    real_df = clean_free_text_df(real_df)
    synthetic_df = clean_free_text_df(synthetic_df)

    real_train_raw, real_test_raw = split_sessions(real_df, test_size=test_size, random_state=random_state)
    syn_train_raw, syn_test_raw = split_sessions(synthetic_df, test_size=test_size, random_state=random_state)

    real_train_chunks = make_chunks(real_train_raw, chunk_size=chunk_size)
    real_test_chunks = make_chunks(real_test_raw, chunk_size=chunk_size)
    syn_train_chunks = make_chunks(syn_train_raw, chunk_size=chunk_size)
    syn_test_chunks = make_chunks(syn_test_raw, chunk_size=chunk_size)

    real_train = aggregate_chunk_features(real_train_chunks)
    real_test = aggregate_chunk_features(real_test_chunks)
    syn_train = aggregate_chunk_features(syn_train_chunks)
    syn_test = aggregate_chunk_features(syn_test_chunks)

    real_train["label"] = 0
    real_test["label"] = 0
    syn_train["label"] = 1
    syn_test["label"] = 1

    train_df = pd.concat([real_train, syn_train], ignore_index=True)
    test_df = pd.concat([real_test, syn_test], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("Real raw shape:", real_df.shape)
    print("Synthetic raw shape:", synthetic_df.shape)

    print("\nReal train chunks shape:", real_train.shape)
    print("Real test chunks shape:", real_test.shape)
    print("Synthetic train chunks shape:", syn_train.shape)
    print("Synthetic test chunks shape:", syn_test.shape)

    print("\nFinal train shape:", train_df.shape)
    print("Final test shape:", test_df.shape)

    print("\nTrain class counts:")
    print(train_df["label"].value_counts())

    print("\nTest class counts:")
    print(test_df["label"].value_counts())

    return train_df, test_df


# ---------------------------------------------------
# 6. Preprocess train/test
# ---------------------------------------------------
def preprocess_train_test(train_df, test_df):
    drop_cols = ["participant", "session", "chunk_id"]
    drop_cols = [c for c in drop_cols if c in train_df.columns]

    print("\nDropping columns:", drop_cols)

    X_train = train_df.drop(columns=drop_cols + ["label"], errors="ignore").copy()
    y_train = train_df["label"].copy()

    X_test = test_df.drop(columns=drop_cols + ["label"], errors="ignore").copy()
    y_test = test_df["label"].copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    # clip using train quantiles only
    for col in X_train.columns:
        low = X_train[col].quantile(0.01)
        high = X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(low, high)
        X_test[col] = X_test[col].clip(low, high)

    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# 7. Scale
# ---------------------------------------------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------
# 8. Filter methods
# ---------------------------------------------------
def select_mutual_info(X_train, y_train, X_test, k=10):
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    score_df = pd.DataFrame({"Feature": X_train.columns, "Score": scores}).sort_values(by="Score", ascending=False)
    selected = score_df.head(k)["Feature"].tolist()
    print("\nTop Mutual Information Features:")
    print(score_df.head(k))
    return X_train[selected], X_test[selected], selected, score_df


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


def select_correlation(X_train, y_train, X_test, k=10):
    temp_df = X_train.copy()
    temp_df["label"] = y_train.values
    corr = temp_df.corr(numeric_only=True)["label"].drop("label")
    corr = corr.abs().sort_values(ascending=False)
    score_df = pd.DataFrame({"Feature": corr.index, "Score": corr.values})
    selected = score_df.head(k)["Feature"].tolist()
    print("\nTop Correlation Features:")
    print(score_df.head(k))
    return X_train[selected], X_test[selected], selected, score_df


def select_variance(X_train, X_test, threshold=0.0, k=10):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)
    var_values = X_train.var().sort_values(ascending=False)
    score_df = pd.DataFrame({"Feature": var_values.index, "Score": var_values.values})
    selected = score_df.head(k)["Feature"].tolist()
    print("\nTop Variance Features:")
    print(score_df.head(k))
    return X_train[selected], X_test[selected], selected, score_df


def select_mean_difference(X_train, y_train, X_test, k=10):
    real_mean = X_train[y_train == 0].mean()
    synthetic_mean = X_train[y_train == 1].mean()
    diff = (real_mean - synthetic_mean).abs().sort_values(ascending=False)
    score_df = pd.DataFrame({"Feature": diff.index, "Score": diff.values})
    selected = score_df.head(k)["Feature"].tolist()
    print("\nTop Mean Difference Features:")
    print(score_df.head(k))
    return X_train[selected], X_test[selected], selected, score_df


# ---------------------------------------------------
# 9. Wrapper/model-based methods
# ---------------------------------------------------
def select_rfe_lr(X_train, y_train, X_test, k=10):
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    selector = RFE(model, n_features_to_select=k, step=2)
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    print("\nRFE Logistic Regression Features:")
    print(selected)
    return X_train[selected], X_test[selected], selected


def select_rfe_svm(X_train, y_train, X_test, k=10):
    model = LinearSVC(max_iter=5000)
    selector = RFE(model, n_features_to_select=k, step=2)
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    print("\nRFE Linear SVM Features:")
    print(selected)
    return X_train[selected], X_test[selected], selected


def select_rfe_rf(X_train, y_train, X_test, k=10):
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    selector = RFE(model, n_features_to_select=k, step=2)
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    print("\nRFE Random Forest Features:")
    print(selected)
    return X_train[selected], X_test[selected], selected


def select_sfm_lr(X_train, y_train, X_test, k=10):
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    model.fit(X_train, y_train)
    importance = np.abs(model.coef_[0])
    score_df = pd.DataFrame({"Feature": X_train.columns, "Score": importance}).sort_values(by="Score", ascending=False)
    selected = score_df.head(k)["Feature"].tolist()
    print("\nSelectFromModel Logistic Regression Features:")
    print(score_df.head(k))
    return X_train[selected], X_test[selected], selected, score_df


def select_sfm_rf(X_train, y_train, X_test, k=10):
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    score_df = pd.DataFrame({"Feature": X_train.columns, "Score": importance}).sort_values(by="Score", ascending=False)
    selected = score_df.head(k)["Feature"].tolist()
    print("\nSelectFromModel Random Forest Features:")
    print(score_df.head(k))
    return X_train[selected], X_test[selected], selected, score_df


# ---------------------------------------------------
# 10. Models
#    v3: limited RF depth
# ---------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Linear SVM": LinearSVC(max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }


# ---------------------------------------------------
# 11. Evaluate
# ---------------------------------------------------
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


# ---------------------------------------------------
# 12. Validate best model
# ---------------------------------------------------
def validate_best_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=3)
    print("\n3-Fold CV Scores:", scores)
    print("CV Mean:", scores.mean())
    return scores


# ---------------------------------------------------
# 13. Plot confusion matrix
# ---------------------------------------------------
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


# ---------------------------------------------------
# 14. Plot ROC
# ---------------------------------------------------
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


# ---------------------------------------------------
# 15. Plot feature distribution
# ---------------------------------------------------
def plot_feature_distribution_from_train_test(train_df, test_df, feature_name):
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    real = full_df[full_df["label"] == 0][feature_name]
    synthetic = full_df[full_df["label"] == 1][feature_name]

    print("\nFeature:", feature_name)
    print("Real Mean:", real.mean())
    print("Synthetic Mean:", synthetic.mean())
    print("Real Std:", real.std())
    print("Synthetic Std:", synthetic.std())

    plt.figure(figsize=(6, 4))
    plt.hist(real, bins=20, alpha=0.5, label="Real")
    plt.hist(synthetic, bins=20, alpha=0.5, label="Synthetic")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature_name}")
    plt.legend()
    plt.show()


# ---------------------------------------------------
# 16. Feature summary
# ---------------------------------------------------
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