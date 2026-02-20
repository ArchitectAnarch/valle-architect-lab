import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas_ta as ta
import pandas as pd
from datetime import datetime, timedelta

# Configuración de página a pantalla completa
st.set_page_config(page_title="VALLE ARCHITECT | Lab", layout="wide", initial_sidebar_state="expanded")

st.title("⚙️ VALLE ARCHITECT - Laboratorio Cuantitativo")
st.markdown("Dashboard táctico para pruebas de algoritmos y superposición de capas.")

# --- PANEL DE CONTROL (BARRA LATERAL) ---
st.sidebar.header("🎯 Parámetros de Radar")

# 1. Selección de Activo y Tiempo
ticker = st.sidebar.text_input("Símbolo (Ej. HNT-USD, BTC-USD)", value="HNT-USD")
intervalo = st.sidebar.selectbox("Temporalidad", ["5m", "15m", "30m", "60m", "1d"], index=2)

# 2. Rango de Fechas
dias_historia = st.sidebar.slider("Días de historial a descargar", min_value=1, max_value=60, value=7)

# 3. Superposición de Capas (Scripts)
st.sidebar.header("🛡️ Capas de Algoritmo")
show_aegis = st.sidebar.checkbox("Activar Escudo AEGIS (EMA 200)", value=True)
show_bb = st.sidebar.checkbox("Activar Bandas de Bollinger", value=False)

# --- EXTRACCIÓN DE DATOS ---
@st.cache_data(ttl=300) # Guarda en caché por 5 mins para no sobrecargar
def cargar_datos(sym, iv, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(sym, start=start_date, end=end_date, interval=iv, progress=False)
    return df

with st.spinner('Escaneando mercado en tiempo real...'):
    df = cargar_datos(ticker, intervalo, dias_historia)

if df.empty:
    st.error("No se encontraron datos. Verifique el símbolo o el rango de tiempo.")
else:
    # Aplanar el índice si yfinance devuelve multi-índice
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # --- PROCESAMIENTO MATEMÁTICO (CAPAS) ---
    if show_aegis:
        df['EMA_200'] = ta.ema(df['Close'], length=200)
    if show_bb:
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)

    # --- MOTOR GRÁFICO (PLOTLY) ---
    fig = go.Figure()

    # Capa 1: Velas Japonesas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Mercado", increasing_line_color='#00FF00', decreasing_line_color='#FF0000'
    ))

    # Capa 2: Escudo AEGIS
    if show_aegis and 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_200'], mode='lines', name='AEGIS EMA 200',
            line=dict(color='orange', width=2)
        ))

    # Capa 3: Bandas de Bollinger
    if show_bb and 'BBL_20_2.0' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], mode='lines', line=dict(color='cyan', width=1, dash='dot'), name='BB Top'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], mode='lines', line=dict(color='cyan', width=1, dash='dot'), name='BB Bot', fill='tonexty', fillcolor='rgba(0,255,255,0.05)'))

    # Formato del gráfico
    fig.update_layout(
        template='plotly_dark',
        height=700,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar datos crudos (Auditoría)
    with st.expander("Ver Datos Cuantitativos Crudos"):
        st.dataframe(df.tail(10))
