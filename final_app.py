# === FUZZY SYSTEM SETUP ===
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt  # optional for visualizing
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go  # Gauge Chart

# 1. Definisikan variabel fuzzy
fcaox = ctrl.Antecedent(np.arange(0.0, 2.6, 0.01), 'fcaox')
status = ctrl.Consequent(np.arange(0, 101, 1), 'status')

# 2. Fungsi keanggotaan FCaOX
fcaox['terlalu_rendah'] = fuzz.trapmf(fcaox.universe, [0.0, 0.0, 0.2, 0.5])
fcaox['normal'] = fuzz.trimf(fcaox.universe, [0.3, 1.25, 2.1])
fcaox['terlalu_tinggi'] = fuzz.trapmf(fcaox.universe, [2.0, 2.2, 2.5, 2.5])

# 3. Fungsi keanggotaan status
status['rendah'] = fuzz.trimf(status.universe, [0, 0, 40])
status['normal'] = fuzz.trimf(status.universe, [30, 50, 70])
status['tinggi'] = fuzz.trimf(status.universe, [60, 100, 100])

# 4. Aturan fuzzy
rule1 = ctrl.Rule(fcaox['terlalu_rendah'], status['rendah'])
rule2 = ctrl.Rule(fcaox['normal'], status['normal'])
rule3 = ctrl.Rule(fcaox['terlalu_tinggi'], status['tinggi'])

# 5. Sistem fuzzy
status_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
status_sim = ctrl.ControlSystemSimulation(status_ctrl)

# Fungsi klasifikasi fuzzy
def fuzzy_klasifikasi(fcaox_val):
    if fcaox_val < 0.0 or fcaox_val > 2.5:
        return "❌ Nilai FCaOX di luar jangkauan [0.0 - 2.5]"
    status_sim.input['fcaox'] = fcaox_val
    status_sim.compute()
    output = status_sim.output['status']
    
    # Penambahan keterangan teknis berdasarkan nilai FCaOX
    if output <= 40:
        return "🔴 FCaOX Terlalu Rendah\n📌 Indikasi: Overheating pada kiln"
    elif output <= 65:
        return "🟡 FCaOX Normal\n✅ Kondisi optimal"
    else:
        return "🟢 FCaOX Terlalu Tinggi\n⚠ LSF terlalu rendah dan CaO terlalu tinggi"

# === STREAMLIT DASHBOARD ===
# Load model SVR dan scaler
svr_model = joblib.load("final_model_svr.pkl")
scaler = joblib.load("final_scaler.pkl")

st.set_page_config(page_title="Estimasi FCaOX dengan SVR", layout="wide")

# Judul
st.title("📊 Estimasi FCaOX Menggunakan SVR")

# Sidebar untuk input
st.sidebar.header("🔧 Input Parameter Proses")
al2o3 = st.sidebar.number_input("Al2O3", value=0.0)
cao = st.sidebar.number_input("CaO", value=0.0)
lsf = st.sidebar.number_input("LSF", value=0.0)
c3s = st.sidebar.number_input("C3S", value=0.0)
c2s = st.sidebar.number_input("C2S", value=0.0)

# Susun data
input_data = np.array([[al2o3, cao, lsf, c3s, c2s]])

if any([al2o3, cao, lsf, c3s, c2s]):
    # Scaling
    scaled_data = scaler.transform(input_data)

    # Prediksi FCaOX
    predicted_fcaox = svr_model.predict(scaled_data)[0]

    # Tampilkan hasil
    st.subheader("📈 Hasil Estimasi FCaOX")
    st.metric(label="Estimasi FCaOX", value=f"{predicted_fcaox:.4f}")

    # === Gauge Chart ===
    st.subheader("📟 Gauge Chart FCaOX")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_fcaox,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediksi FCaOX"},
        gauge={
            'axis': {'range': [0, 2.5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0.0, 0.5], 'color': "red"},
                {'range': [0.5, 2.0], 'color': "yellow"},
                {'range': [2.0, 2.5], 'color': "green"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': predicted_fcaox
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Klasifikasi fuzzy
    klasifikasi_fuzzy = fuzzy_klasifikasi(predicted_fcaox)
    st.write("🧠 *Klasifikasi Berdasarkan Sistem Fuzzy:*")
    st.info(klasifikasi_fuzzy)

    # Tampilkan data input
    st.subheader("🔎 Data Input")
    df_input = pd.DataFrame(input_data, columns=['Al2O3', 'CaO', 'LSF', 'C3S', 'C2S'])
    st.dataframe(df_input, use_container_width=True)

    # Evaluasi target
    st.subheader("📋 Evaluasi Terhadap Target Quality Plan")
    def check_target(param, value, min_val, max_val):
        if min_val <= value <= max_val:
            return "✅ Dalam Target"
        else:
            return "❌ Di Luar Target"

    domain_eval = {
        "LSF (94.5 - 99.5)": check_target("LSF", lsf, 94.5, 99.5),
        "C3S (59 - 65)": check_target("C3S", c3s, 59.0, 65.0),
    }
    df_domain = pd.DataFrame(list(domain_eval.items()), columns=["Parameter", "Status"])
    st.dataframe(df_domain, use_container_width=True)

else:
    st.warning("🛠 Masukkan nilai parameter terlebih dahulu untuk melihat hasil prediksi.")