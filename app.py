import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    acc_rf = accuracy_score(y_test, rf.predict(X_test))

    best_model, best_name, best_acc = (lr, "Logistic Regression", acc_lr) if acc_lr >= acc_rf \
        else (rf, "Random Forest", acc_rf)

    return df, le, lr, rf, acc_lr, acc_rf, best_model, best_name, best_acc


df, le, lr, rf, acc_lr, acc_rf, best_model, best_name, best_acc = load_and_train()

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
st.caption("Dataset: Iris (150 samples, 3 species) | Models: Logistic Regression & Random Forest")