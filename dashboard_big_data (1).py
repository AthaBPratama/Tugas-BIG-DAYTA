import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import os

st.title("Dashboard Big Data Laptop")

st.set_page_config(
    page_title="Dashboard Big Data",
    page_icon="💻",
    layout="wide"
)

# Membaca dataset asli
df = pd.read_csv("dataset_final.csv")

st.success("✅ Dataset berhasil dimuat.")

# Tampilkan 5 data teratas
st.subheader("Data Laptop")
st.dataframe(df.head())



st.write(f"**Total Laptop:** {len(df)}")
st.write(f"**Rata-rata Harga:** Rp {df['Price'].mean():,.0f}")
st.write(f"**Harga Tertinggi:** Rp {df['Price'].max():,.0f}")
st.write(f"**Harga Terendah:** Rp {df['Price'].min():,.0f}")

# Menampilkan informasi tipe data kolom
st.subheader("5 Data Pertama")
st.dataframe(df.head())

# Grafik 1: Rata-rata Harga Berdasarkan Brand
df_brand = df.groupby('Brand')['Price'].mean().reset_index().sort_values(by='Price', ascending=False)
fig1 = px.bar(df_brand, x='Brand', y='Price',
             title='Rata-rata Harga Laptop Berdasarkan Brand',
             labels={'Price': 'Rata-rata Harga (Rp)'},
             color='Brand', template='plotly_dark')
st.plotly_chart(fig1, use_container_width=True)

# Grafik 2: Distribusi RAM di Pasar
fig2 = px.pie(df, names='RAM', title='Persentase Distribusi Kapasitas RAM Laptop',
             template='plotly_dark', hole=0.3)
st.plotly_chart(fig2, use_container_width=True)

# Bersihkan data dari nilai kosong jika ada
df_clean = df.dropna(subset=['RAM', 'SSD', 'Price'])

# Menentukan variabel independen (X) dan dependen (y)
X = df_clean[['RAM', 'SSD']]
y = df_clean['Price']

# Inisialisasi dan latih model
model = LinearRegression()
model.fit(X, y)

# Mengecek skor akurasi (R-squared)
score = model.score(X, y)
st.write(f"**Nilai R² Model:** {score:.2f}")

# Visualisasi Hasil Prediksi vs Harga Asli
df_clean = df_clean.copy()
df_clean['Prediksi_Harga'] = model.predict(X)
fig3 = px.scatter(df_clean, x='Price', y='Prediksi_Harga', color='Brand',
                  title="Perbandingan Harga Asli vs Hasil Prediksi Model",
                 labels={'Price': 'Harga Aktual (Rp)', 'Prediksi_Harga': 'Harga Prediksi (Rp)'},
                 template='plotly_dark')
st.plotly_chart(fig3, use_container_width=True)

#@title 🎛️ Simulator Prediksi Harga Laptop Baru (Geser Parameter)
#@markdown Pilih spesifikasi di bawah ini untuk mengestimasi harga jual pasarannya:

Input_RAM_GB = st.slider(
    "RAM (GB)",
    min_value=4,
    max_value=64,
    step=4,
    value=16
)

Input_SSD_GB = st.slider(
    "SSD (GB)",
    min_value=128,
    max_value=2048,
    step=128,
    value=512
)

# Melakukan prediksi berdasarkan input form
input_data = pd.DataFrame({
    "RAM":[Input_RAM_GB],
    "SSD":[Input_SSD_GB]
})

input_data = pd.DataFrame({
    "RAM": [Input_RAM_GB],
    "SSD": [Input_SSD_GB]
})

harga_prediksi = model.predict(input_data)[0]



st.subheader("📊 Hasil Estimasi")

st.write(f"**RAM:** {Input_RAM_GB} GB")
st.write(f"**SSD:** {Input_SSD_GB} GB")

st.success(
    f"💰 Perkiraan Harga Jual: Rp {max(0, harga_prediksi):,.0f}"
)
