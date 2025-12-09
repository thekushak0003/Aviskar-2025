# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------
# 0. BASIC STREAMLIT CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Student Dropout Prediction System",
    layout="wide",
)

sns.set(style="whitegrid")

DATA_PATH = "students.csv"  # assumes app.py is in same folder as students.csv


# -------------------------------------------------------------------
# 1. DATA LOADING + PREPROCESSING / FEATURE ENGINEERING
# -------------------------------------------------------------------
@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    - Converts yes/no to 1/0
    - Maps gender
    - Creates binary target Is_Dropout
    - Adds engineered features (success ratios, risk score, etc.)
    - Handles missing values
    """
    df = df_raw.copy()

    # ---- yes/no columns -> 1/0 ------------------------------------
    _yes_no = {"yes", "no"}
    yesno_cols = [
        col
        for col in df.columns
        if df[col].dropna().astype(str).str.strip().str.lower().isin(_yes_no).all()
    ]
    if yesno_cols:
        df[yesno_cols] = df[yesno_cols].apply(
            lambda s: s.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
            .astype("Int64")
        )

    # ---- gender mapping (if present in text form) -----------------
    if "Gender (1=Male, 0=Female)" in df.columns:
        df["Gender (1=Male, 0=Female)"] = (
            df["Gender (1=Male, 0=Female)"]
            .map({"Male": 1, "Female": 0})
            .fillna(df["Gender (1=Male, 0=Female)"])
        )

    # ---- binary target: Is_Dropout -------------------------------
    if "Student Status" in df.columns:
        df["Is_Dropout"] = df["Student Status"].map(
            {"Dropout": 1, "Enrolled": 0, "Graduate": 0}
        )

    # ----------------------------------------------------------------
    # Feature engineering (only if required columns exist)
    # ----------------------------------------------------------------
    # Academic load + performance
    if {
        "Enrolled Units (1st Sem)",
        "Enrolled Units (2nd Sem)",
    }.issubset(df.columns):
        df["Total Enrolled Units"] = (
            df["Enrolled Units (1st Sem)"] + df["Enrolled Units (2nd Sem)"]
        )

    if {
        "Approved Units (1st Sem)",
        "Approved Units (2nd Sem)",
    }.issubset(df.columns):
        df["Total Approved Units"] = (
            df["Approved Units (1st Sem)"] + df["Approved Units (2nd Sem)"]
        )

    if {"Total Enrolled Units", "Total Approved Units"}.issubset(df.columns):
        df["Overall Success Ratio"] = df["Total Approved Units"] / df[
            "Total Enrolled Units"
        ].replace(0, np.nan)
        df["Overall Success Ratio"] = df["Overall Success Ratio"].fillna(0)

    # Financial risk score
    risk_cols = []
    if "Is Debtor" in df.columns:
        risk_cols.append("Is Debtor")
    if "Tuition Fees Up-to-Date" in df.columns:
        df["Not UpToDate Fees"] = 1 - df["Tuition Fees Up-to-Date"]
        risk_cols.append("Not UpToDate Fees")
    if "Scholarship Holder" in df.columns:
        df["No Scholarship"] = 1 - df["Scholarship Holder"]
        risk_cols.append("No Scholarship")
    if risk_cols:
        df["Financial Risk Score"] = df[risk_cols].sum(axis=1)

    # Support needs count
    support_cols = [
        c
        for c in [
            "Special Educational Needs",
            "Displaced Student",
            "International Student",
        ]
        if c in df.columns
    ]
    if support_cols:
        df["Support Needs Count"] = df[support_cols].sum(axis=1)

    # ----------------------------------------------------------------
    # Missing value handling (simple, robust)
    # ----------------------------------------------------------------
    num_cols = df.select_dtypes(include=["int64", "float64", "Int64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# -------------------------------------------------------------------
# 2. MODEL TRAINING + METRICS
# -------------------------------------------------------------------
@st.cache_resource
def train_model(df_pre: pd.DataFrame):
    """
    Returns:
        model: trained Pipeline (preprocessing + RandomForest)
        feature_cols: list of feature columns used
        metrics: dict with train/test/CV metrics
        df_model: dataframe actually used for modeling (with Is_Dropout)
    """
    df_model = df_pre.copy()

    # Drop rows where target unknown
    df_model = df_model.dropna(subset=["Is_Dropout"])

    y = df_model["Is_Dropout"].astype(int)

    # Drop target + original Student Status from features
    X = df_model.drop(columns=["Is_Dropout", "Student Status"], errors="ignore")

    feature_cols = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Separate numeric / categorical for preprocessing
    num_cols = X.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # Regularized RandomForest to avoid overfitting
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])

    # Fit on train
    pipe.fit(X_train, y_train)

    # Train / test accuracy
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # CV accuracy on full data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipe,
        X,
        y,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
    )

    # Classification report (on test set)
    report_dict = classification_report(
        y_test,
        y_pred_test,
        target_names=["Not Dropout", "Dropout"],
        output_dict=True,
    )

    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "classification_report": report_dict,
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
    }

    return pipe, feature_cols, metrics, df_model


# -------------------------------------------------------------------
# 3. LOAD DATA + TRAIN MODEL (ONCE, VIA CACHING)
# -------------------------------------------------------------------
df_raw = load_raw_data(DATA_PATH)
df_pre = preprocess_and_engineer(df_raw)
model, feature_cols, metrics, df_model = train_model(df_pre)

# -------------------------------------------------------------------
# 4. STREAMLIT PAGES
# -------------------------------------------------------------------
st.title("🎓 Student Dropout Prediction System")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Model Performance", "Predict Dropout", "About"],
)


# -------------------------------------------------------------------
# PAGE: OVERVIEW
# -------------------------------------------------------------------
if page == "Overview":
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Shape")
        st.write(f"Rows: {df_raw.shape[0]}  |  Columns: {df_raw.shape[1]}")

    with col2:
        st.subheader("Modeling Data Shape (after FE)")
        st.write(f"Rows: {df_model.shape[0]}  |  Columns: {df_model.shape[1]}")

    st.subheader("Sample of Dataset")
    st.dataframe(df_raw.head())

    # Target distribution
    if "Is_Dropout" in df_model.columns:
        st.subheader("Target Distribution (Dropout vs Not Dropout)")
        tgt_counts = df_model["Is_Dropout"].value_counts()
        tgt_perc = df_model["Is_Dropout"].value_counts(normalize=True) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.write("Counts:")
            st.write(tgt_counts.rename(index={0: "Not Dropout", 1: "Dropout"}))
        with col2:
            st.write("Percentage:")
            st.write(
                tgt_perc.rename(index={0: "Not Dropout", 1: "Dropout"}).round(2)
            )

        fig, ax = plt.subplots()
        sns.barplot(
            x=tgt_counts.index.map({0: "Not Dropout", 1: "Dropout"}),
            y=tgt_counts.values,
            ax=ax,
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Target Class Distribution")
        st.pyplot(fig)


# -------------------------------------------------------------------
# PAGE: EDA
# -------------------------------------------------------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis")

    numeric_cols = (
        df_model.select_dtypes(include=["int64", "float64", "Int64"])
        .columns.drop("Is_Dropout")
        .tolist()
    )
    cat_cols = df_model.select_dtypes(include=["object", "category"]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(
        ["Numeric vs Target", "Categorical vs Target", "Correlation"]
    )

    with tab1:
        st.subheader("Numeric Feature Distribution by Dropout")

        if numeric_cols:
            num_col = st.selectbox("Select numeric column", numeric_cols)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            sns.histplot(
                data=df_model,
                x=num_col,
                hue="Is_Dropout",
                multiple="stack",
                ax=axes[0],
            )
            axes[0].set_title(f"{num_col} distribution by Is_Dropout")
            axes[0].set_xlabel(num_col)

            sns.boxplot(
                data=df_model,
                x="Is_Dropout",
                y=num_col,
                ax=axes[1],
            )
            axes[1].set_title(f"{num_col} vs Is_Dropout")
            axes[1].set_xlabel("Is_Dropout")
            axes[1].set_xticklabels(["Not Dropout", "Dropout"])

            st.pyplot(fig)
        else:
            st.info("No numeric columns detected.")

    with tab2:
        st.subheader("Categorical Feature vs Dropout")

        if cat_cols:
            cat_col = st.selectbox("Select categorical column", cat_cols)

            ctab = pd.crosstab(
                df_model[cat_col],
                df_model["Is_Dropout"],
                normalize="index",
            )
            ctab.columns = ["Not Dropout", "Dropout"]

            st.write("Row-wise proportion:")
            st.dataframe(ctab.round(3))

            fig, ax = plt.subplots(figsize=(8, 4))
            ctab.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Proportion")
            ax.set_title(f"{cat_col} vs Is_Dropout")
            st.pyplot(fig)
        else:
            st.info("No categorical columns detected.")

    with tab3:
        st.subheader("Correlation Heatmap (Numeric Features)")
        corr_cols = numeric_cols + ["Is_Dropout"]
        if corr_cols:
            corr = df_model[corr_cols].corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr,
                annot=False,
                cmap="coolwarm",
                center=0,
                ax=ax,
            )
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.info("No numeric columns to correlate.")


# -------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# -------------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Accuracy")
    st.write(f"Train Accuracy: **{metrics['train_accuracy']:.4f}**")
    st.write(f"Test Accuracy : **{metrics['test_accuracy']:.4f}**")
    st.write(
        f"5-fold CV Accuracy: **{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}**"
    )

    st.info(
        "If Train >> Test or CV mean is much lower than Test, it suggests overfitting."
    )

    # Classification report
    st.subheader("Classification Report (on test split)")
    report_df = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report_df.style.format("{:.3f}"))

    # Confusion matrix
    st.subheader("Confusion Matrix (on test split)")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=["Not Dropout", "Dropout"],
    )
    disp.plot(ax=ax)
    st.pyplot(fig)


# -------------------------------------------------------------------
# PAGE: PREDICT DROPOUT
# -------------------------------------------------------------------
elif page == "Predict Dropout":
    st.header("Predict Dropout Risk for a Single Student")

    # Build defaults from dataset medians/modes for feature_cols
    df_feats = df_model[feature_cols].copy()

    defaults = {}
    for col in feature_cols:
        if df_feats[col].dtype in ["int64", "float64", "Int64"]:
            defaults[col] = float(df_feats[col].median())
        else:
            defaults[col] = df_feats[col].mode()[0]

    # Start with defaults and let the user override key ones
    user_input = defaults.copy()

    with st.form("prediction_form"):
        st.subheader("Basic Information")

        if "Age at Enrollment" in feature_cols:
            user_input["Age at Enrollment"] = st.number_input(
                "Age at Enrollment",
                min_value=15,
                max_value=70,
                value=int(defaults["Age at Enrollment"]),
            )

        if "Gender (1=Male, 0=Female)" in feature_cols:
            gender = st.selectbox("Gender", ["Male", "Female"])
            user_input["Gender (1=Male, 0=Female)"] = 1 if gender == "Male" else 0

        if "Scholarship Holder" in feature_cols:
            schol = st.selectbox("Scholarship Holder", ["No", "Yes"])
            user_input["Scholarship Holder"] = 1 if schol == "Yes" else 0

        if "Is Debtor" in feature_cols:
            debtor = st.selectbox("Is Debtor", ["No", "Yes"])
            user_input["Is Debtor"] = 1 if debtor == "Yes" else 0

        if "Tuition Fees Up-to-Date" in feature_cols:
            fees = st.selectbox("Tuition Fees Up-to-Date", ["No", "Yes"])
            user_input["Tuition Fees Up-to-Date"] = 1 if fees == "Yes" else 0

        # Academic units
        if "Enrolled Units (1st Sem)" in feature_cols:
            user_input["Enrolled Units (1st Sem)"] = st.number_input(
                "Enrolled Units (1st Sem)",
                min_value=0,
                max_value=100,
                value=int(defaults["Enrolled Units (1st Sem)"]),
            )

        if "Approved Units (1st Sem)" in feature_cols:
            user_input["Approved Units (1st Sem)"] = st.number_input(
                "Approved Units (1st Sem)",
                min_value=0,
                max_value=100,
                value=int(defaults["Approved Units (1st Sem)"]),
            )

        if "Enrolled Units (2nd Sem)" in feature_cols:
            user_input["Enrolled Units (2nd Sem)"] = st.number_input(
                "Enrolled Units (2nd Sem)",
                min_value=0,
                max_value=100,
                value=int(defaults["Enrolled Units (2nd Sem)"]),
            )

        if "Approved Units (2nd Sem)" in feature_cols:
            user_input["Approved Units (2nd Sem)"] = st.number_input(
                "Approved Units (2nd Sem)",
                min_value=0,
                max_value=100,
                value=int(defaults["Approved Units (2nd Sem)"]),
            )

        # Example for course / nationality if they exist
        if "Course Name" in feature_cols:
            user_input["Course Name"] = st.selectbox(
                "Course Name", sorted(df_model["Course Name"].unique())
            )

        if "Nationality" in feature_cols:
            user_input["Nationality"] = st.selectbox(
                "Nationality", sorted(df_model["Nationality"].unique())
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])

        # Important: reindex to match training features
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        proba = model.predict_proba(input_df)[0][1]  # probability of Dropout (1)
        pred_class = model.predict(input_df)[0]

        st.subheader("Prediction Result")
        st.write(f"**Predicted Class:** {'Dropout' if pred_class == 1 else 'Not Dropout'}")
        st.write(f"**Dropout Probability:** {proba:.3f}")

        if proba >= 0.7:
            st.error("⚠ High risk of dropout. Early intervention strongly recommended.")
        elif proba >= 0.4:
            st.warning("⚠ Moderate risk of dropout. Monitor and support the student.")
        else:
            st.success("✅ Low risk of dropout.")


# -------------------------------------------------------------------
# PAGE: ABOUT
# -------------------------------------------------------------------
elif page == "About":
    st.header("About this Project")
    st.markdown(
        """
This Streamlit application is built for your **Data Vision / Student Dropout Prediction**
problem.

**What it does end-to-end:**

1. Loads `students.csv`
2. Performs:
   - Cleaning (`yes/no` → 1/0, gender mapping)
   - Feature engineering (success ratios, risk scores, support needs)
   - Missing value imputation
3. Trains a **RandomForest + preprocessing Pipeline** with:
   - Train / Test split
   - 5-fold Stratified Cross-Validation
4. Shows:
   - EDA dashboards
   - Train / Test / CV accuracy → helps you detect **overfitting**
   - Confusion matrix & classification report
5. Provides an interactive **“Predict Dropout”** form to estimate risk for a single student.

You can modify:
- The feature engineering inside `preprocess_and_engineer`
- The model and hyperparameters inside `train_model`
- The UI elements in each page section.

This is designed to be a clean, competition / project ready base that matches
your current folder structure (`students.csv` + analysis notebooks + images).
"""
    )
