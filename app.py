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
            
            st.subheader("Hubungan Fitur vs
