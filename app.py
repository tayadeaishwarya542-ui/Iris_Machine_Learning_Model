import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

st.title("🌸 Iris Species Predictor")
st.write("A simple ML app trained on the Iris dataset. Adjust the sliders to predict the species.")

# ---- Load & train (cached so it only runs once) ----
@st.cache_resource
def load_and_train():
    df = pd.read_csv("IRIS.csv")
    X = df.drop(columns="species")
    y = df["species"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    acc_lr = accuracy_score(y_test, pred_lr)
    acc_rf = accuracy_score(y_test, pred_rf)

    best_model, best_name, best_acc = (lr, "Logistic Regression", acc_lr) if acc_lr >= acc_rf \
        else (rf, "Random Forest", acc_rf)

    return df, le, lr, rf, acc_lr, acc_rf, best_model, best_name, best_acc, y_test, pred_lr, pred_rf


df, le, lr, rf, acc_lr, acc_rf, best_model, best_name, best_acc, y_test, pred_lr, pred_rf = load_and_train()

# ---- Sidebar: model comparison ----
st.sidebar.header("Model Comparison")
st.sidebar.metric("Logistic Regression Accuracy", f"{acc_lr:.2%}")
st.sidebar.metric("Random Forest Accuracy", f"{acc_rf:.2%}")
st.sidebar.success(f"Best model in use: **{best_name}** ({best_acc:.2%})")

# ---- Input sliders ----
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", float(df.sepal_length.min()), float(df.sepal_length.max()), float(df.sepal_length.mean()))
    petal_length = st.slider("Petal Length (cm)", float(df.petal_length.min()), float(df.petal_length.max()), float(df.petal_length.mean()))
with col2:
    sepal_width = st.slider("Sepal Width (cm)", float(df.sepal_width.min()), float(df.sepal_width.max()), float(df.sepal_width.mean()))
    petal_width = st.slider("Petal Width (cm)", float(df.petal_width.min()), float(df.petal_width.max()), float(df.petal_width.mean()))

input_df = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width],
})

# ---- Prediction ----
if st.button("Predict Species"):
    pred = best_model.predict(input_df)
    species = le.inverse_transform(pred)[0]
    proba = best_model.predict_proba(input_df)[0]

    st.success(f"Predicted Species: **{species}**")

    proba_df = pd.DataFrame({"Species": le.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Species"))

st.divider()

# ---- Model Insights (EDA graphs + confusion matrices) ----
with st.expander("📊 See Model Insights & Data Exploration"):

    st.subheader("Feature Pairplot")
    fig1 = sns.pairplot(df, hue="species")
    st.pyplot(fig1)

    st.subheader("Feature Distribution by Species")
    fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for ax, feat in zip(axes.flat, features):
        sns.boxplot(data=df, x="species", y=feat, ax=ax)
        ax.set_title(f"{feat} by species")
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.subheader("Confusion Matrices (on test data)")
    fig4, axes4 = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(confusion_matrix(y_test, pred_lr), annot=True, fmt="d",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes4[0], cmap="Blues")
    axes4[0].set_title("Logistic Regression")

    sns.heatmap(confusion_matrix(y_test, pred_rf), annot=True, fmt="d",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes4[1], cmap="Greens")
    axes4[1].set_title("Random Forest")
    plt.tight_layout()
    st.pyplot(fig4)

st.caption("Dataset: Iris (150 samples, 3 species) | Models: Logistic Regression & Random Forest")
