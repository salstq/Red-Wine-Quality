import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman
st.set_page_config(page_title="Wine Quality Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# =====================
# Load Data
# =====================
@st.cache_data
def load_data():
    # Menggunakan sep=';' karena dataset wine quality UCI biasanya memakai semicolon
    try:
        df = pd.read_csv("winequality-red.csv", sep=';')
        if df.shape[1] <= 1: # Jika gagal deteksi kolom, coba comma
            df = pd.read_csv("winequality-red.csv", sep=',')
    except:
        # Fallback dummy data jika file tidak ditemukan saat testing
        return pd.DataFrame()
        
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.replace('"', '')
    return df

df_orig = load_data()

# =====================
# Sidebar Layout
# =====================
st.sidebar.image("https://www.svgrepo.com/show/275331/wine-glass-wine.svg", width=100)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", [
    "📊 Dashboard Dataset",
    "🔍 Exploratory Data Analysis",
    "⚙️ Preprocessing",
    "🤖 Regression Model",
    "🍷 Predict Quality"
])

st.sidebar.divider()
st.sidebar.info("Aplikasi ini menggunakan Ordinary Least Squares (OLS) untuk analisis regresi.")

# =====================
# Main Logic
# =====================

if df_orig.empty:
    st.error("File 'winequality-red.csv' tidak ditemukan. Pastikan file berada di folder yang sama.")
else:
    # 1. DATASET
    if menu == "📊 Dashboard Dataset":
        st.title("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Baris", df_orig.shape[0])
        col2.metric("Total Fitur", df_orig.shape[1] - 1)
        col3.metric("Target Variable", "Quality")

        st.subheader("Cuplikan Data")
        st.dataframe(df_orig.head(10), use_container_width=True)

        with st.expander("Lihat Statistik Deskriptif"):
            st.table(df_orig.describe().T)

    # 2. EDA
    elif menu == "🔍 Exploratory Data Analysis":
        st.title("🔍 Exploratory Data Analysis")
        
        tab1, tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])
        
        with tab1:
            st.subheader("Distribusi Target (Quality)")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x="quality", data=df_orig, palette="viridis", ax=ax)
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Korelasi Antar Fitur")
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(df_orig.corr()))
            sns.heatmap(df_orig.corr(), annot=True, fmt=".2f", cmap="RdBu", mask=mask, ax=ax)
            st.pyplot(fig)
            
            st.subheader("Hubungan Fitur vs Quality")
            feature = st.selectbox("Pilih fitur untuk dibandingkan:", df_orig.columns[:-1])
            fig, ax = plt.subplots()
            sns.boxplot(x="quality", y=feature, data=df_orig, palette="magma", ax=ax)
            st.pyplot(fig)

    # 3. PREPROCESSING
    elif menu == "⚙️ Preprocessing":
        st.title("⚙️ Data Preprocessing")
        st.write("Langkah: Pemisahan Fitur, Target, dan Scaling menggunakan `StandardScaler`.")
        
        X = df_orig.drop("quality", axis=1)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sebelum Scaling (Original)**")
            st.dataframe(X.head())
        with col2:
            st.write("**Sesudah Scaling (Z-Score)**")
            st.dataframe(X_scaled.head())

    # 4. REGRESSION MODEL
    elif menu == "🤖 Regression Model":
        st.title("🤖 Linear Regression (OLS)")
        
        X = df_orig.drop("quality", axis=1)
        y = df_orig["quality"]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        # Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("R-Squared", f"{model.rsquared:.3f}")
        c2.metric("Adj. R-Squared", f"{model.rsquared_adj:.3f}")
        c3.metric("F-Statistic", f"{model.fvalue:.2f}")

        st.subheader("Model Summary")
        st.text(model.summary().as_text())

        st.subheader("Analisis Residual")
        y_hat = model.predict(X_const)
        residual = y - y_hat
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_hat, residual, alpha=0.5, color='purple')
        ax.axhline(0, linestyle="--", color='red')
        ax.set_xlabel("Predicted Quality")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

    # 5. PREDICTION
    elif menu == "🍷 Predict Quality":
        st.title("🍷 Wine Quality Predictor")
        st.write("Gunakan slider di bawah untuk mengatur nilai karakteristik kimia wine.")

        X = df_orig.drop("quality", axis=1)
        y = df_orig["quality"]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        # Buat form input yang rapi dalam kolom
        with st.form("prediction_form"):
            cols = st.columns(3)
            input_data = {}
            
            for i, col in enumerate(X.columns):
                with cols[i % 3]:
                    input_data[col] = st.slider(
                        label=col,
                        min_value=float(df_orig[col].min()),
                        max_value=float(df_orig[col].max()),
                        value=float(df_orig[col].mean())
                    )
            
            submitted = st.form_submit_button("Hitung Prediksi Kualitas")

        if submitted:
            input_df = pd.DataFrame([input_data])
            # Tambahkan konstanta secara manual karena ini data baru satu baris
            input_df.insert(0, 'const', 1.0)
            
            prediction = model.predict(input_df)
            
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.write("### Hasil Prediksi:")
                score = round(prediction.iloc[0], 2)
                st.title(f"⭐ {score}")
            with res_col2:
                if score >= 6.5:
                    st.success("Kualitas: **EXCELLENT** (High Quality)")
                elif score >= 5.5:
                    st.info("Kualitas: **AVERAGE** (Good Enough)")
                else:
                    st.warning("Kualitas: **POOR** (Low Quality)")
