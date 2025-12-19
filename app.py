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
    # Use sep=';' if your CSV uses semicolons
    df = pd.read_csv("winequality-red.csv", sep=';') 
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.replace('"', '') # Strip whitespace and quotes
    return df

df_orig = load_data()  # simpan dataset asli

# Debug: tampilkan kolom untuk memastikan 'quality' ada
st.write("Kolom asli CSV:", df_orig.columns.tolist())

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
    st.write("Jumlah data:", df_orig.shape[0])
    st.dataframe(df_orig.head())

    st.subheader("Statistik Deskriptif")
    st.dataframe(df_orig.describe())

# =====================
# EDA
# =====================
elif menu == "EDA":
    st.subheader("Distribusi Quality")
    fig, ax = plt.subplots()
    sns.countplot(x="quality", data=df_orig, ax=ax)
    st.pyplot(fig)

    st.subheader("Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_orig.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Hubungan Alcohol vs Quality")
    fig, ax = plt.subplots()
    sns.scatterplot(x="alcohol", y="quality", data=df_orig, ax=ax)
    st.pyplot(fig)

# =====================
# Preprocessing
# =====================
elif menu == "Preprocessing":
    st.subheader("Preprocessing Data")

    X = df_orig.drop("quality", axis=1)
    y = df_orig["quality"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    st.write("Fitur setelah StandardScaler:")
    st.dataframe(X_scaled_df.head())

# =====================
# Regression Model
# =====================
elif menu == "Regression Model":
    st.subheader("🤖 Model Regresi Linear Berganda (OLS)")

    # Menyiapkan data
    X = df_orig.drop("quality", axis=1)
    y = df_orig["quality"]
    X_const = sm.add_constant(X)

    # Fit Model
    model = sm.OLS(y, X_const).fit()

    # 1. Menampilkan Metric Utama dalam Kolom
    col1, col2, col3 = st.columns(3)
    col1.metric("R-squared", f"{model.rsquared:.3f}")
    col2.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    col3.metric("F-statistic", f"{model.fvalue:.2f}")

    # 2. Menampilkan Tabel Koefisien (Tabel Rapi)
    st.write("### Tabel Koefisien Regresi")
    # Mengambil hasil summary koefisien ke dalam DataFrame
    coef_df = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-values": model.tvalues,
        "P-values": model.pvalues.round(4)
    })
    st.table(coef_df) # Menggunakan table agar statis dan rapi

    # 3. Analisis Residual (Visualisasi tetap dipertahankan)
    st.write("### Analisis Residual")
    y_hat = model.predict(X_const)
    residual = y - y_hat

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(y_hat, residual, alpha=0.5, color='royalblue')
    ax.axhline(0, linestyle="--", color='red')
    ax.set_xlabel("Predicted Quality")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

    # Opsi: Jika masih butuh melihat detail teknis asli
    with st.expander("Lihat Full Model Summary (Raw Text)"):
        st.text(model.summary())

# =====================
# Prediction
# =====================
elif menu == "Prediction":
    st.subheader("Prediksi Kualitas Wine")

    X = df_orig.drop("quality", axis=1)
    y = df_orig["quality"]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    st.write("Masukkan nilai fitur wine:")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, float(df_orig[col].min()), float(df_orig[col].max()), float(df_orig[col].mean()))

    input_df = pd.DataFrame([input_data])
    input_df_const = sm.add_constant(input_df)

    if st.button("Prediksi Quality"):
        prediction = model.predict(input_df_const)
        st.success(f"Prediksi Kualitas Wine: {round(prediction.iloc[0], 2)}")
