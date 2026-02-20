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
st.markdown("Simulador Institucional con Paneles de Mando Específicos y Equity Curve.")

# --- PANEL DE CONTROL LATERAL ---
st.sidebar.header("📡 Radares y Mercado")
ticker = st.sidebar.text_input("Símbolo (Ej. HNT-USD, BTC-USD)", value="HNT-USD")

intervalos = {
    "5 Minutos (Máx 60 días)": "5m",
    "15 Minutos (Máx 60 días)": "15m",
    "30 Minutos (Máx 60 días)": "30m",
    "1 Hora (Máx 730 días)": "1h",
    "1 Día (Ilimitado)": "1d"
}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=1)
intervalo = intervalos[intervalo_sel]

dias_historia = st.sidebar.slider("Días de historial a descargar", min_value=1, max_value=60, value=30)
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=13364.0, step=1000.0)

# --- SELECCIÓN DE ESTRATEGIA Y PANELES MUTABLES ---
st.sidebar.header("🧠 Selección de Estrategia")
estrategia_activa = st.sidebar.radio("Motor de Ejecución:", ["TRINITY V357 (Dividend Yield)", "JUGGERNAUT V356 (Lineal + Escudos)"])

st.sidebar.header(f"🎯 Parámetros: {estrategia_activa.split(' ')[0]}")

# Parámetros Comunes
tp_pct = st.sidebar.slider("🎯 Take Profit (%)", 0.5, 10.0, 3.0 if "TRINITY" in estrategia_activa else 2.2, 0.1)
sl_pct = st.sidebar.slider("🛑 Stop Loss (%)", 0.5, 10.0, 1.5 if "TRINITY" in estrategia_activa else 1.4, 0.1)
whale_factor = st.sidebar.slider("🐋 Factor Ballena (Volumen x)", 1.0, 5.0, 2.5, 0.1)
radar_sens = st.sidebar.slider("📡 Sensibilidad Radar (%)", 0.1, 5.0, 1.5, 0.1)

# Parámetros Específicos
reinvest_pct = 0.0
use_macro_shield = False
use_knife_shield = False

if "TRINITY" in estrategia_activa:
    reinvest_pct = st.sidebar.slider("💵 Reinversión de Ganancias (%)", 0.0, 100.0, 50.0, 5.0)
    st.sidebar.info("Modo TRINITY: Interés compuesto dinámico con recolección de dividendos activa.")
else:
    st.sidebar.markdown("**🛡️ Escudos AEGIS (Anti-Crash)**")
    use_macro_shield = st.sidebar.checkbox("Bloquear compras bajo EMA 200", value=True)
    use_knife_shield = st.sidebar.checkbox("Bloquear Cuchillos Cayendo (> 1.5 ATR)", value=True)
    st.sidebar.info("Modo JUGGERNAUT: Crecimiento lineal con lote fijo y escudos de tendencia macro.")

# --- EXTRACCIÓN Y LIMPIEZA DE DATOS BLINDADA ---
@st.cache_data(ttl=60)
def cargar_datos(sym, iv, days):
    try:
        # Usamos Ticker().history para evitar el problema de MultiIndex de yf.download()
        ticker_obj = yf.Ticker(sym)
        df = ticker_obj.history(period=f"{days}d", interval=iv)
        # Limpiar zonas horarias para compatibilidad
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df
    except Exception as e:
        return pd.DataFrame()

with st.spinner('Construyendo gráficas y matrices tácticas...'):
    df = cargar_datos(ticker, intervalo, dias_historia)

if not df.empty and len(df) > 50:
    # --- CONSTRUCCIÓN DE INDICADORES (EL CEREBRO PYTHON) ---
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Vol_MA'] = ta.sma(df['Volume'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Bandas de Bollinger para Squeeze/Defcon
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={bb.columns[0]: 'BBL', bb.columns[1]: 'BBM', bb.columns[2]: 'BBU'}, inplace=True)
    else:
        df['BBU'] = df['Close']

    # --- LÓGICA DE ESCUDOS AEGIS (Solo Juggernaut) ---
    df['Macro_Safe'] = True
    df['Knife_Safe'] = True
    
    if use_macro_shield:
        df['Macro_Safe'] = df['Close'] > df['EMA_200']
    
    if use_knife_shield:
        # Caída fuerte previa > 1.5 ATR
        cuerpo_previo = df['Open'].shift(1) - df['Close'].shift(1)
        atr_previo = df['ATR'].shift(1)
        df['Knife_Safe'] = ~(cuerpo_previo > (atr_previo * 1.5))

    # --- LÓGICA DE DETECCIÓN DE SEÑALES ---
    distancia_ema = (abs(df['Close'] - df['EMA_200']) / df['Close']) * 100
    df['Radar_Activo'] = distancia_ema <= radar_sens
    df['Ballena_Activa'] = df['Volume'] > (df['Vol_MA'] * whale_factor)
    df['Vela_Verde'] = df['Close'] > df['Open']
    df['Defcon_Break'] = (df['Close'] >= df['BBU'] * 0.999) & df['Vela_Verde']

    # La Ballena ignora los escudos en V356 y V357
    cond_pink_whale = df['Ballena_Activa'] & df['Vela_Verde']
    
    # Señales normales respetan los escudos (si están activos)
    cond_normal = (df['Radar_Activo'] | df['Defcon_Break']) & df['Vela_Verde']
    cond_normal = cond_normal & df['Macro_Safe'] & df['Knife_Safe']

    df['Signal_Buy'] = cond_pink_whale | cond_normal

    # --- MOTOR DE BACKTESTING STRICTO ---
    trades = []
    equity_curve = []
    dividend_curve = []
    
    en_posicion = False
    precio_entrada = 0.0
    active_capital = capital_inicial
    safe_dividends = 0.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        fecha = df.index[i]
        
        if en_posicion:
            tp_price = precio_entrada * (1 + (tp_pct / 100))
            sl_price = precio_entrada * (1 - (sl_pct / 100))
            
            # Take Profit Hit
            if row['High'] >= tp_price:
                if "TRINITY" in estrategia_activa:
                    # Calcula ganancia en base al capital activo total invertido
                    ganancia_neta = active_capital * (tp_pct / 100)
                    reinvested = ganancia_neta * (reinvest_pct / 100.0)
                    extracted = ganancia_neta - reinvested
                    active_capital += reinvested
                    safe_dividends += extracted
                else:
                    # JUGGERNAUT: Lote fijo basado en Capital Inicial
                    ganancia_neta = capital_inicial * (tp_pct / 100)
                    safe_dividends += ganancia_neta 
                    
                trades.append({'Tipo': 'WIN', 'Ganancia_$': ganancia_neta, 'Fecha': fecha})
                en_posicion = False
                
            # Stop Loss Hit
            elif row['Low'] <= sl_price:
                if "TRINITY" in estrategia_activa:
                    perdida_neta = active_capital * (sl_pct / 100)
                    active_capital -= perdida_neta
                else:
                    perdida_neta = capital_inicial * (sl_pct / 100)
                    safe_dividends -= perdida_neta # Castiga las ganancias netas del Juggernaut
                    
                trades.append({'Tipo': 'LOSS', 'Ganancia_$': -perdida_neta, 'Fecha': fecha})
                en_posicion = False
                
        # Entradas
        if not en_posicion and row['Signal_Buy']:
            precio_entrada = row['Close']
            en_posicion = True

        equity_curve.append(active_capital)
        dividend_curve.append(safe_dividends)

    df['Active_Capital'] = equity_curve
    df['Safe_Dividends'] = dividend_curve
    df['Total_Portfolio'] = df['Active_Capital'] + df['Safe_Dividends'] if "TRINITY" in estrategia_activa else capital_inicial + df['Safe_Dividends']

    # --- CÁLCULO DE MÉTRICAS WALL STREET ---
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
    else:
        win_rate = profit_factor = ratio_wl = gross_profit = gross_loss = 0

    capital_final_total = df['Total_Portfolio'].iloc[-1]
    retorno_pct = ((capital_final_total - capital_inicial) / capital_inicial) * 100

    # --- PANEL FRONTAL DE RESULTADOS ---
    st.markdown(f"### 📊 Auditoría Táctica: {estrategia_activa.split(' ')[0]}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valor Total Portafolio", f"${capital_final_total:,.2f}", f"{retorno_pct:,.2f}% Retorno")
    col2.metric("Salario (Dividendos/Neto)", f"${df['Safe_Dividends'].iloc[-1]:,.2f}")
    col3.metric("Capital Invertido Base", f"${df['Active_Capital'].iloc[-1] if 'TRINITY' in estrategia_activa else capital_inicial:,.2f}")
    col4.metric("Total de Disparos", f"{total_trades} Operaciones")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Tasa de Acierto (Win Rate)", f"{win_rate:.1f}%")
    col6.metric("Factor de Beneficio (PF)", f"{profit_factor:.2f}x")
    col7.metric("Ratio Ganancia/Pérdida", f"{ratio_wl:.2f}")
    col8.metric("Motor Riesgo/Recompensa", f"TP {tp_pct}% | SL {sl_pct}%")

    st.markdown("---")

    # --- MOTOR GRÁFICO DUAL (VELAS + EQUITY CURVE) ---
    st.subheader("📈 Mapa de Ejecución Cuantitativa")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Velas Japonesas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Mercado", increasing_line_color='#00FF00', decreasing_line_color='#FF0000'
    ), row=1, col=1)
    
    # EMA 200 (Escudo)
    if 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='AEGIS EMA 200', line=dict(color='orange', width=2)), row=1, col=1)

    # Marcadores de Disparo
    compras = df[df['Signal_Buy']]
    if not compras.empty:
        fig.add_trace(go.Scatter(
            x=compras.index, y=compras['Low'] * 0.98, mode='markers', name='Fuego Táctico',
            marker=dict(symbol='triangle-up', color='magenta', size=12, line=dict(color='white', width=1))
        ), row=1, col=1)

    # Gráfico 2: Equity Curve (Curva de Rentabilidad Total)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Total_Portfolio'], mode='lines', name='Portafolio Total ($)',
        line=dict(color='#00FF00', width=3)
    ), row=2, col=1)
    
    # Línea de Dividendos
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Safe_Dividends'], mode='lines', name='Flujo de Caja Extraído ($)',
        line=dict(color='#00FFFF', width=2, dash='dot')
    ), row=2, col=1)

    fig.update_layout(template='plotly_dark', height=800, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Error de Radar: Servidor de datos no respondió o no hay historial para esta moneda. Intente con 'BTC-USD' o reduzca los días.")
