import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wine Quality Regression", layout="wide")

st.title("🍷 Wine Quality Prediction – Linear Regression (OLS)")
st.write("Aplikasi ini menampilkan proses analisis regresi linear berganda untuk memprediksi kualitas wine.")

# =====================
# Load Data
# =====================
@st.cache_data
def load_data():
    # Menggunakan sep=';' karena dataset wine quality UCI biasanya menggunakan semicolon
    df = pd.read_csv("winequality-red.csv", sep=';') 
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.replace('"', '')
    return df

df_orig = load_data()

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
    st.write(f"Jumlah data setelah hapus duplikat: **{df_orig.shape[0]} baris**")
    st.dataframe(df_orig.head(), use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df_orig.describe(), use_container_width=True)

# =====================
# EDA
# =====================
elif menu == "EDA":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Quality")
        fig, ax = plt.subplots()
        sns.countplot(x="quality", data=df_orig, ax=ax, palette="viridis")
        st.pyplot(fig)

    with col2:
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots()
        sns.heatmap(df_orig.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("Hubungan Alcohol vs Quality")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(x="alcohol", y="quality", data=df_orig, alpha=0.5, ax=ax)
    st.pyplot(fig)

# =====================
# Preprocessing
# =====================
elif menu == "Preprocessing":
    st.subheader("Preprocessing Data (Scaling)")
    X = df_orig.drop("quality", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    st.write("Fitur setelah StandardScaler (Z-Score):")
    st.dataframe(X_scaled_df.head(), use_container_width=True)

# =====================
# Regression Model
# =====================
elif menu == "Regression Model":
    st.subheader("🤖 Model Regresi Linear Berganda (OLS)")

    X = df_orig.drop("quality", axis=1)
    y = df_orig["quality"]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # 1. Metric Utama
    c1, c2, c3 = st.columns(3)
    c1.metric("R-squared", f"{model.rsquared:.3f}")
    c2.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    c3.metric("F-statistic", f"{model.fvalue:.2f}")

    # 2. Tabel Koefisien yang Rapi
    st.write("### 📋 Tabel Koefisien Model")
    coef_df = pd.DataFrame({
        "Feature": model.params.index,
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        "t-Stat": model.tvalues.values,
        "P-Value": model.pvalues.values
    }).set_index("Feature")
    
    # Menampilkan tabel dengan highlight p-value signifikan (< 0.05)
    st.table(coef_df.style.format("{:.4f}"))

    # 3. Residual Plot
    st.write("### 📉 Residual Plot")
    y_hat = model.predict(X_const)
    residual = y - y_hat
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(y_hat, residual, alpha=0.4, color='teal')
    ax.axhline(0, linestyle="--", color='red')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    st.pyplot(fig)

# =====================
# Prediction
# =====================
elif menu == "Prediction":
    st.subheader("🔮 Prediksi Kualitas Wine")

    X = df_orig.drop("quality", axis=1)
    y = df_orig["quality"]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    st.write("Masukkan parameter kimia untuk prediksi:")
    
    # Layout kolom untuk input agar rapi
    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            input_data[col] = st.number_input(
                col, 
                float(df_orig[col].min()), 
                float(df_orig[col].max()), 
                float(df_orig[col].mean())
            )

    if st.button("Hitung Prediksi", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        input_df_const = sm.add_constant(input_df, has_constant='add')
        
        # Pastikan kolom const ada di awal jika sm.add_constant tidak menambahkannya otomatis
        if 'const' not in input_df_const.columns:
            input_df_const.insert(0, 'const', 1.0)
            
        # Urutkan kolom sesuai model
        input_df_const = input_df_const[X_const.columns]
        
        prediction = model.predict(input_df_const)
        st.success(f"### Estimasi Skor Kualitas: {prediction.iloc[0]:.2f}")
