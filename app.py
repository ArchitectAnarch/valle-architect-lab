import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(page_title="VALLE ARCHITECT | Lab Quant", layout="wide", initial_sidebar_state="expanded")

st.title("⚙️ VALLE ARCHITECT - Motor de Backtesting Quant")
st.markdown("Simulador algorítmico en tiempo real con ejecución de métricas institucionales.")

# --- PANEL DE CONTROL (BARRA LATERAL) ---
st.sidebar.header("💰 Capital y Activo")
ticker = st.sidebar.text_input("Símbolo (Ej. HNT-USD)", value="HNT-USD")
intervalo = st.sidebar.selectbox("Temporalidad", ["5m", "15m", "30m", "60m", "1d"], index=1)
dias_historia = st.sidebar.slider("Días de historial", min_value=1, max_value=60, value=30)
capital_inicial = st.sidebar.number_input("Capital Inicial ($)", value=13364.0, step=1000.0)

st.sidebar.header("🎯 Parámetros JUGGERNAUT")
tp_pct = st.sidebar.slider("🎯 Take Profit (%)", 0.5, 10.0, 2.2, 0.1)
sl_pct = st.sidebar.slider("🛑 Stop Loss (%)", 0.5, 10.0, 1.4, 0.1)
whale_factor = st.sidebar.slider("🐋 Factor Ballena (Volumen x)", 1.0, 5.0, 1.9, 0.1)
radar_sens = st.sidebar.slider("📡 Sensibilidad Radar EMA (%)", 0.1, 5.0, 2.4, 0.1)

# --- EXTRACCIÓN DE DATOS ---
@st.cache_data(ttl=300)
def cargar_datos(sym, iv, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(sym, start=start_date, end=end_date, interval=iv, progress=False)
    return df

with st.spinner('Descargando datos y calculando matrices...'):
    df = cargar_datos(ticker, intervalo, dias_historia)

if not df.empty:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # --- MOTORES MATEMÁTICOS (Python Juggernaut Proxy) ---
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Vol_MA'] = ta.sma(df['Volume'], length=20)
    
    # Lógica de disparo (Ejemplo adaptado de su Juggernaut)
    # 1. El precio debe estar cerca de la EMA 200 (Radar Sensibilidad)
    distancia_ema = (abs(df['Close'] - df['EMA_200']) / df['Close']) * 100
    radar_activo = distancia_ema <= radar_sens
    # 2. Debe haber un pico de volumen (Ballena)
    ballena_activa = df['Volume'] > (df['Vol_MA'] * whale_factor)
    # 3. Tendencia alcista (Cierre verde)
    vela_verde = df['Close'] > df['Open']
    
    df['Signal_Buy'] = radar_activo & ballena_activa & vela_verde

    # --- MOTOR DE BACKTESTING STRICTO ---
    trades = []
    en_posicion = False
    precio_entrada = 0.0
    capital_actual = capital_inicial
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # Evaluar salidas si estamos en posición
        if en_posicion:
            tp_price = precio_entrada * (1 + (tp_pct / 100))
            sl_price = precio_entrada * (1 - (sl_pct / 100))
            
            # Asumimos que si el 'High' toca el TP, cerramos con ganancia. Si 'Low' toca SL, cerramos con pérdida.
            if row['High'] >= tp_price:
                ganancia = capital_actual * (tp_pct / 100)
                capital_actual += ganancia
                trades.append({'Tipo': 'WIN', 'Retorno': (tp_pct/100), 'Ganancia_$': ganancia, 'Fecha': df.index[i]})
                en_posicion = False
            elif row['Low'] <= sl_price:
                perdida = capital_actual * (sl_pct / 100)
                capital_actual -= perdida
                trades.append({'Tipo': 'LOSS', 'Retorno': -(sl_pct/100), 'Ganancia_$': -perdida, 'Fecha': df.index[i]})
                en_posicion = False
                
        # Evaluar entradas si NO estamos en posición
        if not en_posicion and row['Signal_Buy']:
            precio_entrada = row['Close']
            en_posicion = True

    # --- CÁLCULO DE MÉTRICAS INSTITUCIONALES ---
    total_trades = len(trades)
    if total_trades > 0:
        df_trades = pd.DataFrame(trades)
        wins = len(df_trades[df_trades['Tipo'] == 'WIN'])
        losses = len(df_trades[df_trades['Tipo'] == 'LOSS'])
        win_rate = (wins / total_trades) * 100
        
        gross_profit = df_trades[df_trades['Tipo'] == 'WIN']['Ganancia_$'].sum()
        gross_loss = abs(df_trades[df_trades['Tipo'] == 'LOSS']['Ganancia_$'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0
        ratio_wl = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Sharpe Ratio (Simplificado Anualizado aprox)
        returns = df_trades['Retorno']
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(total_trades) if returns.std() != 0 else 0
    else:
        win_rate = profit_factor = ratio_wl = sharpe_ratio = gross_profit = gross_loss = 0

    # --- RENDERIZADO DEL DASHBOARD ---
    st.subheader("📊 Métricas de Rendimiento JUGGERNAUT")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capital Final", f"${capital_actual:,.2f}", f"{(capital_actual - capital_inicial):,.2f} USD")
    col2.metric("Total Operaciones", f"{total_trades}")
    col3.metric("Tasa de Acierto (Win Rate)", f"{win_rate:.1f}%")
    col4.metric("Factor de Beneficio (PF)", f"{profit_factor:.2f}x")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Sharpe Ratio (Estabilidad)", f"{sharpe_ratio:.2f}")
    col6.metric("Ratio Ganancia/Pérdida", f"{ratio_wl:.2f}")
    col7.metric("Motor de Riesgo", f"TP {tp_pct}% / SL {sl_pct}%")
    col8.metric("Protección Activa", "Lote Fijo 100%")

    st.markdown("---")

    # --- MOTOR GRÁFICO ---
    st.subheader("📈 Mapa de Ejecución Táctica")
    fig = go.Figure()

    # Velas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Mercado"
    ))
    
    # AEGIS EMA
    if 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='AEGIS EMA 200', line=dict(color='orange', width=2)))

    # Marcadores de Compra
    compras = df[df['Signal_Buy']]
    if not compras.empty:
        fig.add_trace(go.Scatter(
            x=compras.index, y=compras['Low'] * 0.99, mode='markers', name='Radar Disparo',
            marker=dict(symbol='triangle-up', color='magenta', size=12)
        ))

    fig.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Esperando conexión de datos...")
