import streamlit as st 
import numpy as np
from joblib import load

gizi_level = ['normal', 'Severly stunted', 'stunted', 'tinggi']

st.title("BalitaPenerus: Membangun Balita Indonesia yang Sehat dan Produktif")
st.header("Cek asupan gizi balita anda")
st.text("Balita sehat, Indonesia kuat")

model = load("model/stunting-knn.joblib")

with st.form("stunting"):
    jenis_kelamin = st.radio("Jenis kelamin: ", ["Laki-laki", "Perempuan"])
    if (jenis_kelamin == "Laki-laki"):
        jenis_kelamin = 0
    else:
        jenis_kelamin = 1
    
    tinggi = st.number_input("Tinggi badan (cm): ")
    umur = st.number_input("Umur (bulan):")

    submitted =  st.form_submit_button("Cek status gizi")

    if submitted:
        result = model.predict(np.array([umur, jenis_kelamin, tinggi]).reshape(1,-1))[0]
        st.markdown(f"""# Hasil prediksi: {gizi_level[result]}""")

        if result == 0:
            st.success("Gizi normal")
        elif result == 1:
            st.warning("Direkomendasikan konsultasi ke Dokter ahli ")
        elif result == 2:
            st.warning("Direkomendasikan konsultasi ke Dokter ahli ")
        elif result == 3:
            st.warning("Gizi baik")
        


md_about = '''
# About this project

Proyek ini dibuat dengan tujuan edukasi sekaligus awareness tentang bahayanya stunting pada balita.

## Dataset
[Stunting Toddler (Balita) Detection (121K rows)](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows)

By: RENDI PUTRA PRADANA

## Model Detail
**Algoritma**: K-Nearest Neigbors

**n-neighbors**: 17

**weights**: distance

**n-data trained**: 96.799

**Accuracy**: 99.89%

### Notebook
**[Stunting prediction](https://www.kaggle.com/code/developbyarya/stunting-prediction/)**

## Further explanation
'''
st.markdown(md_about)
        
        

