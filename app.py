import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wine Quality Regression", layout="wide")

st.title("🍷 Wine Quality Prediction – Linear Regression (OLS)")
st.write("Aplikasi ini menampilkan proses analisis regresi linear berganda untuk memprediksi kualitas wine berdasarkan karakteristik kimia.")

# =====================
# Load Data
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    df = df.drop_duplicates()
    return df

df = load_data()

# =====================
# Sidebar
# =====================
st.sidebar.header("Navigasi")
menu = st.sidebar.radio("Pilih Menu", [
    "Dataset",
    "EDA",
    "Preprocessing",
    "Regression Model",
    "Prediction"
])

# =====================
# Dataset
# =====================
if menu == "Dataset":
    st.subheader("Dataset Wine Quality")
    st.write("Jumlah data:", df.shape[0])
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

# =====================
# EDA
# =====================
elif menu == "EDA":
    st.subheader("Distribusi Quality")
    fig, ax = plt.subplots()
    sns.countplot(x="quality", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Hubungan Alcohol vs Quality")
    fig, ax = plt.subplots()
    sns.scatterplot(x="alcohol", y="quality", data=df, ax=ax)
    st.pyplot(fig)

# =====================
# Preprocessing
# =====================
elif menu == "Preprocessing":
    st.subheader("Preprocessing Data")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    st.write("Fitur setelah StandardScaler:")
    st.dataframe(X_scaled.head())

# =====================
# Regression Model
# =====================
elif menu == "Regression Model":
    st.subheader("Model Regresi Linear Berganda (OLS)")

    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()

    st.write("**R-squared:**", round(model.rsquared, 3))
    st.write("**Adjusted R-squared:**", round(model.rsquared_adj, 3))

    st.text(model.summary())

    st.subheader("Residual Plot")
    y_hat = model.predict(X_const)
    residual = y - y_hat

    fig, ax = plt.subplots()
    ax.scatter(y_hat, residual, alpha=0.5)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    st.pyplot(fig)

# =====================
# Prediction
# =====================
elif menu == "Prediction":
    st.subheader("Prediksi Kualitas Wine")

    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    st.write("Masukkan nilai fitur wine:")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([input_data])
    input_df_const = sm.add_constant(input_df)

    if st.button("Prediksi Quality"):
        prediction = model.predict(input_df_const)
        st.success(f"Prediksi Kualitas Wine: {round(prediction.iloc[0], 2)}")
