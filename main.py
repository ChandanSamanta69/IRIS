"""
Streamlit app: k-Nearest Neighbors (k-NN) Iris Classifier
Deploy with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42

st.set_page_config(
    page_title="k-NN Iris Classifier",
    page_icon="🌸",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached data / model helpers — Streamlit reruns the whole script on
# every interaction, so caching avoids redoing work unnecessarily.
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return iris, df


@st.cache_data
def get_split(_X, _y):
    return train_test_split(
        _X, _y, test_size=0.2, random_state=RANDOM_STATE, stratify=_y
    )


@st.cache_resource
def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


@st.cache_data
def cv_curve(X_train_scaled, y_train, k_max=25):
    ks = list(range(1, k_max + 1))
    scores = []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="accuracy")
        scores.append(cv.mean())
    return ks, scores


@st.cache_resource
def train_model(X_train_scaled, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    return knn


iris, df = load_data()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = get_split(X, y)
scaler = fit_scaler(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

ks, cv_scores = cv_curve(X_train_scaled, y_train)
best_k = ks[int(np.argmax(cv_scores))]

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")
k = st.sidebar.slider(
    "Number of neighbors (k)",
    min_value=1, max_value=25, value=best_k, step=1,
    help="How many nearest training flowers to vote on.",
)
st.sidebar.caption(f"💡 Best k found via 5-fold CV: **{best_k}**")

st.sidebar.header("🌼 Flower Measurements (cm)")
sepal_length = st.sidebar.slider("Sepal length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

model = train_model(X_train_scaled, y_train, k)
y_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🌸 k-Nearest Neighbors Iris Classifier")
st.markdown(
    "Predicts a flower's **species** by finding the `k` most similar flowers "
    "in the training set and taking a majority vote among their species."
)

col1, col2, col3 = st.columns(3)
col1.metric("Current k", k)
col2.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
col3.metric("Best CV k", best_k)

st.divider()

# ----------------------------------------------------------------------
# Live prediction
# ----------------------------------------------------------------------
st.subheader("🔮 Try a Prediction")

input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=feature_names,
)
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

pred_col, prob_col = st.columns([1, 2])

with pred_col:
    st.success(f"Predicted species: **{target_names[prediction].capitalize()}**")
    st.dataframe(
        input_df.rename(columns=lambda c: c.replace(" (cm)", "")).T.rename(columns={0: "Value (cm)"}),
        use_container_width=True,
    )

with prob_col:
    prob_df = pd.DataFrame({"Species": target_names, "Probability": probabilities})
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(data=prob_df, x="Species", y="Probability", hue="Species", palette="viridis", legend=False, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Vote share among {k} nearest neighbors")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ----------------------------------------------------------------------
# Model performance
# ----------------------------------------------------------------------
st.subheader("📊 Model Performance")

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("**k vs. Cross-Validation Accuracy**")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(ks, [s * 100 for s in cv_scores], marker="o", color="#2b6cb0")
    ax1.axvline(k, color="#e53e3e", linestyle="--", label=f"Selected k = {k}")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Mean CV Accuracy (%)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

with perf_col2:
    st.markdown("**Confusion Matrix (Test Set)**")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)
    plt.close(fig2)

with st.expander("📄 Full classification report"):
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.divider()

# ----------------------------------------------------------------------
# Dataset explorer
# ----------------------------------------------------------------------
st.subheader("🔍 Explore the Dataset")

tab1, tab2 = st.tabs(["Scatter plot", "Raw data"])

with tab1:
    feat_x = st.selectbox("X-axis feature", feature_names, index=2)
    feat_y = st.selectbox("Y-axis feature", feature_names, index=3)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=feat_x, y=feat_y, hue="species", palette="viridis", s=60, ax=ax3)
    ax3.scatter(
        input_df[feat_x], input_df[feat_y],
        color="red", marker="X", s=200, label="Your input", zorder=5,
    )
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

with tab2:
    st.dataframe(df, use_container_width=True)

st.caption("Built with scikit-learn + Streamlit · Dataset: Fisher's Iris (150 samples, 3 species)")