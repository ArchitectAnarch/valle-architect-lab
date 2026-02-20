import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DEL CENTRO DE MANDO ---
st.set_page_config(page_title="VALLE ARCHITECT | Lab Quant", layout="wide", initial_sidebar_state="expanded")

st.title("⚙️ VALLE ARCHITECT - Motor Cuantitativo Multi-Estrategia")
st.markdown("Simulador de grado institucional con proyección de equidad (Equity Curve) y recolección de dividendos.")

# --- PANEL DE CONTROL LATERAL ---
st.sidebar.header("📡 Radares y Mercado")
ticker = st.sidebar.text_input("Símbolo (Ej. HNT-USD, BTC-USD)", value="HNT-USD")

# Selector de temporalidad inteligente
intervalos = {
    "1 Minuto (Solo últimos 7 días)": "1m",
    "5 Minutos (Últimos 60 días)": "5m",
    "15 Minutos (Últimos 60 días)": "15m",
    "30 Minutos (Últimos 60 días)": "30m",
    "1 Hora (Últimos 730 días)": "1h",
    "1 Día (Historial Completo)": "1d",
    "1 Semana (Historial Completo)": "1wk"
}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=2)
intervalo = intervalos[intervalo_sel]

dias_historia = st.sidebar.slider("Días de historial a descargar", min_value=1, max_value=730, value=30)
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=13364.0, step=1000.0)

st.sidebar.header("🧠 Selección de Estrategia")
estrategia_activa = st.sidebar.radio("Motor de Ejecución:", ["TRINITY V357 (Dividend Yield)", "JUGGERNAUT V356 (Lineal)"])

st.sidebar.header("🎯 Calibración de Parámetros")
tp_pct = st.sidebar.slider("🎯 Take Profit (%)", 0.5, 10.0, 3.0 if "TRINITY" in estrategia_activa else 2.2, 0.1)
sl_pct = st.sidebar.slider("🛑 Stop Loss (%)", 0.5, 10.0, 1.5 if "TRINITY" in estrategia_activa else 1.4, 0.1)
whale_factor = st.sidebar.slider("🐋 Factor Ballena (Volumen x)", 1.0, 5.0, 2.5, 0.1)
radar_sens = st.sidebar.slider("📡 Sensibilidad Radar (%)", 0.1, 5.0, 1.5, 0.1)

reinvest_pct = 0.0
if "TRINITY" in estrategia_activa:
    reinvest_pct = st.sidebar.slider("💵 Reinversión de Ganancias (%)", 0.0, 100.0, 50.0, 5.0)

# --- EXTRACCIÓN Y CÁLCULO DE DATOS ---
@st.cache_data(ttl=60)
def cargar_datos(sym, iv, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    try:
        df = yf.download(sym, start=start_date, end=end_date, interval=iv, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        return pd.DataFrame()

with st.spinner('Sincronizando con los servidores del mercado...'):
    df = cargar_datos(ticker, intervalo, dias_historia)

if not df.empty:
    # Construcción de Indicadores (El Cerebro)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Vol_MA'] = ta.sma(df['Volume'], length=20)
    
    # Bandas de Bollinger para simular DEFCON y Radares
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={bb.columns[0]: 'BBL', bb.columns[1]: 'BBM', bb.columns[2]: 'BBU'}, inplace=True)
    else:
        df['BBU'] = df['Close']
        df['BBL'] = df['Close']
