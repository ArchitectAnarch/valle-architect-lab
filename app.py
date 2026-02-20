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

st.title("⚙️ VALLE ARCHITECT - Motor Cuantitativo Multi-Arquitectura")
st.markdown("Plataforma de simulación algorítmica con métricas de evaluación institucional.")

# --- PANEL DE CONTROL LATERAL ---
st.sidebar.header("📡 Radares y Mercado")
ticker = st.sidebar.text_input("Símbolo (Ej. HNT-USD, BTC-USD)", value="HNT-USD")

intervalos = {
    "15 Minutos (Límite API: 60 días)": "15m",
    "30 Minutos (Límite API: 60 días)": "30m",
    "1 Hora (Límite API: 730 días)": "1h",
    "1 Día (Historial Ilimitado Años)": "1d",
    "1 Semana (Historial Ilimitado)": "1wk"
}
intervalo_sel = st.sidebar.selectbox("Resolución de Temporalidad", list(intervalos.keys()), index=0)
intervalo = intervalos[intervalo_sel]

st.sidebar.header("📅 Rango Histórico de Evaluación")
col_date1, col_date2 = st.sidebar.columns(2)
# Configuración por defecto: Últimos 30 días
default_start = datetime.today() - timedelta(days=30)
start_date = col_date1.date_input("Fecha de Inicio", value=default_start)
end_date = col_date2.date_input("Fecha de Fin", value=datetime.today())

capital_inicial = st.sidebar.number_input("Capital Inicial Base (USD)", value=13364.0, step=1000.0)

# --- SELECCIÓN DE ARQUITECTURA (UI DINÁMICA) ---
st.sidebar.header("🧠 Selección de Arquitectura")
estrategia_activa = st.sidebar.radio("Motor de Ejecución:", ["TRINITY V357 (Interés Compuesto)", "JUGGERNAUT V356 (Ejecución Lineal)"])

st.sidebar.header(f"🎯 Calibración: {estrategia_activa.split(' ')[0]}")

# Parámetros Universales
tp_pct = st.sidebar.slider("🎯 Límite de Toma de Beneficios - TP (%)", 0.5, 10.0, 3.0 if "TRINITY" in estrategia_activa else 2.2, 0.1)
sl_pct = st.sidebar.slider("🛑 Umbral de Detención de Pérdidas - SL (%)", 0.5, 10.0, 1.5 if "TRINITY" in estrategia_activa else 1.4, 0.1)
whale_factor = st.sidebar.slider("🐋 Multiplicador de Volumen Institucional (xVol)", 1.0, 5.0, 2.5, 0.1)
radar_sens = st.sidebar.slider("📡 Sensibilidad de Impacto Radar (%)", 0.1, 5.0, 1.5, 0.1)

# Parámetros Mutables (Específicos por Estrategia)
reinvest_pct = 0.0
use_macro_shield = False
use_atr_shield = False

if "TRINITY" in estrategia_activa:
    st.sidebar.markdown("**💵 Módulo de Flujo de Caja**")
    reinvest_pct = st.sidebar.slider("Porcentaje de Reinversión de Ganancias (%)", 0.0, 100.0, 50.0, 5.0)
    st.sidebar.info("Modo TRINITY: El capital extraído (no reinvertido) se salvaguarda como flujo de caja externo.")
else:
    st.sidebar.markdown("**🛡️ Módulo de Protección AEGIS**")
    use_macro_shield = st.sidebar.checkbox("Bloqueo Macroeconómico Bajista (Filtro EMA 200)", value=True)
    use_atr_shield = st.sidebar.checkbox("Bloqueo por Volatilidad Bajista Extrema (> 1.5 ATR)", value=True)
    st.sidebar.info("Modo JUGGERNAUT: Lote de compra estricto basado en el capital inicial. Ejecución lineal pura sin interés compuesto.")

# --- MOTOR DE EXTRACCIÓN DE DATOS ---
@st.cache_data(ttl=60)
def cargar_datos(sym, start, end, iv):
    try:
        # Añadimos 1 día a end_date para asegurar que incluya el día actual completo
        end_adjusted = end + timedelta(days=1)
        df = yf.download(sym, start=start, end=end_adjusted, interval=iv, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        return pd.DataFrame()

with st.spinner('Ensamblando matrices de datos...'):
    df = cargar_datos(ticker, start_date, end_date, intervalo)

if not df.empty and len(df) > 20:
    # --- PROCESAMIENTO MATEMÁTICO CUANTITATIVO ---
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Vol_MA'] = ta.sma(df['Volume'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Bandas de Bollinger
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={bb.columns[0]: 'BBL', bb.columns[1]: 'BBM', bb.columns[2]: 'BBU'}, inplace=True)
    else:
        df['BBU'] = df['Close']

    # Lógica AEGIS (Solo afecta a Juggernaut)
    df['Macro_Safe'] = True
    df['ATR_Safe'] = True
    
    if "JUGGERNAUT" in estrategia_activa:
        if use_macro_shield:
            df['Macro_Safe'] = df['Close'] > df['EMA_200']
        if use_atr_shield:
            cuerpo_previo = df['Open'].shift(1) - df['Close'].shift(1)
            atr_previo = df['ATR'].shift(1)
            df['ATR_Safe'] = ~(cuerpo_previo > (atr_previo * 1.5))

    # Lógica de Detección de Disparo
    distancia_ema = (abs(df['Close'] - df['EMA_200']) / df['Close']) * 100
    df['Radar_Activo'] = distancia_ema <= radar_sens
    df['Vol_Anormal'] = df['Volume'] > (df['Vol_MA'] * whale_factor)
    df['Vela_Verde'] = df['Close'] > df['Open']
    df['Ruptura_Bandas'] = (df['Close'] >= df['BBU'] * 0.999) & df['Vela_Verde']

    # La Ballena (Vol Anormal) ignora los escudos en ambas versiones
    cond_volumen = df['Vol_Anormal'] & df['Vela_Verde']
    
    # Radares normales respetan escudos
    cond_tecnica = (df['Radar_Activo'] | df['Ruptura_Bandas']) & df['Vela_Verde']
    cond_tecnica = cond_tecnica & df['Macro_Safe'] & df['ATR_Safe']

    df['Signal_Buy'] = cond_volumen | cond_tecnica

    # --- SIMULADOR DE CAJA FUERTE (BACKTEST ENGINE) ---
    trades = []
    equity_curve = []
    
    en_posicion = False
    precio_entrada = 0.0
    
    # Variables de Rastreo
    active_capital = capital_inicial
    safe_dividends = 0.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        fecha = df.index[i]
        
        if en_posicion:
            tp_price = precio_entrada * (1 + (tp_pct / 100))
            sl_price = precio_entrada * (1 - (sl_pct / 100))
            
            # Ejecución TP
            if row['High'] >= tp_price:
                if "TRINITY" in estrategia_activa:
                    ganancia_neta = active_capital * (tp_pct / 100)
                    reinvested = ganancia_neta * (reinvest_pct / 100.0)
                    extracted = ganancia_neta - reinvested
                    active_capital += reinvested
                    safe_dividends += extracted
                    valor_reportado = active_capital + safe_dividends
                else: # JUGGERNAUT Lineal
                    ganancia_neta = capital_inicial * (tp_pct / 100)
                    active_capital += ganancia_neta 
                    valor_reportado = active_capital
                    
                trades.append({'Tipo': 'WIN', 'Ganancia_$': ganancia_neta, 'Fecha': fecha})
                en_posicion = False
                
            # Ejecución SL
            elif row['Low'] <= sl_price:
                if "TRINITY" in estrategia_activa:
                    perdida_neta = active_capital * (sl_pct / 100)
                    active_capital -= perdida_neta
                    valor_reportado = active_capital + safe_dividends
                else: # JUGGERNAUT Lineal
                    perdida_neta = capital_inicial * (sl_pct / 100)
                    active_capital -= perdida_neta 
                    valor_reportado = active_capital
                    
                trades.append({'Tipo': 'LOSS', 'Ganancia_$': -perdida_neta, 'Fecha': fecha})
                en_posicion = False
                
        # Evaluación Entrada
        if not en_posicion and row['Signal_Buy']:
            precio_entrada = row['Close']
            en_posicion = True

        # Acumular valor total del portafolio en la historia
        valor_actual = (active_capital + safe_dividends) if "TRINITY" in estrategia_activa else active_capital
        equity_curve.append(valor_actual)

    df['Total_Portfolio'] = equity_curve
    df['Rentabilidad_Pct'] = ((df['Total_Portfolio'] - capital_inicial) / capital_inicial) * 100

    # --- MÉTRICAS DE EVALUACIÓN INSTITUCIONAL ---
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
    retorno_pct_final = df['Rentabilidad_Pct'].iloc[-1]

    # --- PRESENTACIÓN FRONTAL DE DATOS ---
    st.markdown(f"### 📊 Auditoría Táctica: {estrategia_activa.split(' ')[0]}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valor Total Portafolio", f"${capital_final_total:,.2f}", f"{retorno_pct_final:,.2f}% Rendimiento")
    
    if "TRINITY" in estrategia_activa:
        col2.metric("Flujo de Caja Extraído", f"${safe_dividends:,.2f}")
        col3.metric("Capital Reinvertido", f"${active_capital:,.2f}")
    else:
        col2.metric("Beneficio Neto Lineal", f"${(capital_final_total - capital_inicial):,.2f}")
        col3.metric("Lote de Ejecución Constante", f"${capital_inicial:,.2f}")
        
    col4.metric("Ejecuciones de Mercado", f"{total_trades} Operaciones")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Tasa de Precisión (Win Rate)", f"{win_rate:.1f}%")
    col6.metric("Factor de Beneficio (PF)", f"{profit_factor:.2f}x")
    col7.metric("Ratio de Riesgo/Recompensa", f"{ratio_wl:.2f}")
    col8.metric("Motor Interno (TP/SL)", f"{tp_pct}% / {sl_pct}%")

    st.markdown("---")

    # --- MOTOR GRÁFICO AVANZADO ---
    st.subheader("📈 Mapa Algorítmico y Curva de Crecimiento")
    
    # Creamos subgráficas con un eje Y secundario en la gráfica de abajo (para mostrar %)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.65, 0.35],
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

    # 1. Velas Japonesas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Precio del Activo"
    ), row=1, col=1)
    
    # 2. Línea Macroeconómica EMA
    if 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='Filtro Macroeconómico (EMA 200)', line=dict(color='orange', width=2)), row=1, col=1)

    # 3. Marcadores de Ejecución
    compras = df[df['Signal_Buy']]
    if not compras.empty:
        fig.add_trace(go.Scatter(
            x=compras.index, y=compras['Low'] * 0.98, mode='markers', name='Impacto Algorítmico (Compra)',
            marker=dict(symbol='triangle-up', color='cyan', size=12, line=dict(color='white', width=1))
        ), row=1, col=1)

    # 4. Curva de Capital (En Dólares - Eje Izquierdo)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Total_Portfolio'], mode='lines', name='Crecimiento de Portafolio ($)',
        line=dict(color='#00FF00', width=3),
        hovertemplate='Fecha: %{x}<br>Capital: $%{y:,.2f}<extra></extra>'
    ), row=2, col=1, secondary_y=False)

    # 5. Sombra de Rentabilidad (En Porcentaje - Eje Derecho invisible para alinear)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Rentabilidad_Pct'], mode='lines', name='Rentabilidad Neta (%)',
        line=dict(color='rgba(0,0,0,0)'), # Línea invisible, solo para alimentar el eje Y derecho
        hovertemplate='Rentabilidad: %{y:.2f}%<extra></extra>'
    ), row=2, col=1, secondary_y=True)

    # Configuraciones de diseño de ejes
    fig.update_layout(
        template='plotly_dark', 
        height=850, 
        xaxis_rangeslider_visible=False, 
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Nombrar los ejes para claridad profesional
    fig.update_yaxes(title_text="Precio del Activo ($)", row=1, col=1)
    fig.update_yaxes(title_text="Capital Total (USD)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Rentabilidad (%)", row=2, col=1, secondary_y=True, ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Infracción de Datos: No existe suficiente historial para el rango de fechas seleccionado. Intente cambiar la temporalidad a '1 Día' si está evaluando un lapso de varios años.")
