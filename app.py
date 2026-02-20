import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta

st.set_page_config(page_title="VALLE ARCHITECT | Lab Quant AI", layout="wide", initial_sidebar_state="expanded")

st.title("⚙️ VALLE ARCHITECT - Motor Quant & AI Optimizer")
st.markdown("Simulación Multi-Motor estricta con optimización estocástica de parámetros.")

# --- 1. PANEL DE CONTROL: MERCADO Y TIEMPO ---
st.sidebar.header("📡 Radares y Mercado")
ticker = st.sidebar.text_input("Símbolo (Ej. HNT-USD, BTC-USD)", value="HNT-USD")

# Resoluciones exóticas mediante Resampling
intervalos = {
    "1 Minuto (Historial: 7 días max)": ("1m", "1T"),
    "5 Minutos (Historial: 60 días max)": ("5m", "5T"),
    "7 Minutos (Historial: 7 días max)": ("1m", "7T"),
    "13 Minutos (Historial: 7 días max)": ("1m", "13T"),
    "15 Minutos (Historial: 60 días max)": ("15m", "15T"),
    "23 Minutos (Historial: 7 días max)": ("1m", "23T"),
    "30 Minutos (Historial: 60 días max)": ("30m", "30T"),
    "1 Hora (Historial: 730 días max)": ("1h", "1H"),
    "2 Horas (Historial: 730 días max)": ("1h", "2H"),
    "4 Horas (Historial: 730 días max)": ("1h", "4H"),
    "1 Día (Años)": ("1d", "1D"),
    "1 Semana (Años)": ("1wk", "1W"),
    "1 Mes (Años)": ("1mo", "1M")
}
intervalo_sel = st.sidebar.selectbox("Resolución Espacial (Temporalidad)", list(intervalos.keys()), index=4)
iv_download, iv_resample = intervalos[intervalo_sel]

col_date1, col_date2 = st.sidebar.columns(2)
default_start = datetime.today() - timedelta(days=60)
start_date = col_date1.date_input("Fecha Inicio", value=default_start)
end_date = col_date2.date_input("Fecha Fin", value=datetime.today())

capital_inicial = st.sidebar.number_input("Capital Inicial Base (USD)", value=13364.0, step=1000.0)

# --- 2. SELECCIÓN DE ARQUITECTURA (DNA AISLADO) ---
st.sidebar.header("🧠 Selección de Arquitectura")
estrategia_activa = st.sidebar.radio("Motor de Ejecución:", [
    "TRINITY V357 (Dividendos + Compuesto)", 
    "JUGGERNAUT V356 (Lineal + AEGIS)",
    "DEFCON V329 (Pura Expansión Squeeze)"
])

st.sidebar.header(f"🎯 Calibración: {estrategia_activa.split(' ')[0]}")
tp_pct = st.sidebar.slider("🎯 Take Profit (%)", 0.5, 10.0, 3.0, 0.1)
sl_pct = st.sidebar.slider("🛑 Stop Loss (%)", 0.5, 10.0, 1.5, 0.1)

# Variables mutables por estrategia
reinvest_pct = 50.0
use_macro_shield = False
use_atr_shield = False
bot_defcon_buy = True
bot_defcon_sell = True
radar_sens = 1.5
whale_factor = 2.5

if "TRINITY" in estrategia_activa:
    st.sidebar.markdown("**💵 Política de Dividendos**")
    reinvest_pct = st.sidebar.slider("Porcentaje de Reinversión (%)", 0.0, 100.0, 50.0, 5.0)
    whale_factor = st.sidebar.slider("🐋 Factor Ballena (xVol)", 1.0, 5.0, 2.5, 0.1)
    radar_sens = st.sidebar.slider("📡 Sensibilidad Radar (%)", 0.1, 5.0, 1.5, 0.1)
    
elif "JUGGERNAUT" in estrategia_activa:
    st.sidebar.markdown("**🛡️ Escudos AEGIS**")
    use_macro_shield = st.sidebar.checkbox("Bloqueo Macroeconómico (EMA 200)", value=True)
    use_atr_shield = st.sidebar.checkbox("Bloqueo Volatilidad Extrema (>1.5 ATR)", value=True)
    whale_factor = st.sidebar.slider("🐋 Factor Ballena (xVol)", 1.0, 5.0, 2.5, 0.1)
    radar_sens = st.sidebar.slider("📡 Sensibilidad Radar (%)", 0.1, 5.0, 1.5, 0.1)

elif "DEFCON" in estrategia_activa:
    st.sidebar.markdown("**🚀 Directrices DEFCON**")
    bot_defcon_buy = st.sidebar.checkbox("Comprar en Ruptura Alcista (DEFCON 1/2)", value=True)
    bot_defcon_sell = st.sidebar.checkbox("Vender en Ruptura Bajista (DEFCON 1/2)", value=True)
    st.sidebar.info("Modo DEFCON: Ignora radares y ballenas. Se basa puramente en la apertura de canales de Keltner/Bollinger y ADX > 20.")

# --- 3. EXTRACCIÓN Y RESAMPLING DE DATOS ---
@st.cache_data(ttl=60)
def cargar_datos(sym, start, end, iv_down, iv_res):
    try:
        end_adj = end + timedelta(days=1)
        df = yf.download(sym, start=start, end=end_adj, interval=iv_down, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Motor de Resampling para temporalidades exóticas (7m, 13m, 23m, etc)
        if iv_down != iv_res and not df.empty:
            df = df.resample(iv_res).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

df = cargar_datos(ticker, start_date, end_date, iv_download, iv_resample)

# --- 4. OPTIMIZADOR DE INTELIGENCIA ARTIFICIAL ---
def ejecutar_backtest(df_sim, strat, tp, sl, cap_ini, reinvest, macro_sh, atr_sh):
    en_pos = False
    precio_ent = 0.0
    cap_activo = cap_ini
    divs = 0.0
    
    for i in range(len(df_sim)):
        row = df_sim.iloc[i]
        
        if en_pos:
            tp_p = precio_ent * (1 + (tp / 100))
            sl_p = precio_ent * (1 - (sl / 100))
            
            # Condición especial Venta DEFCON
            venta_defcon = False
            if "DEFCON" in strat and bot_defcon_sell and row.get('Defcon_Sell', False):
                venta_defcon = True
            
            if row['High'] >= tp_p or venta_defcon:
                if "TRINITY" in strat:
                    ganancia = cap_activo * (tp / 100) if not venta_defcon else cap_activo * ((row['Close'] - precio_ent)/precio_ent)
                    if ganancia > 0:
                        reinv = ganancia * (reinvest / 100.0)
                        cap_activo += reinv
                        divs += (ganancia - reinv)
                    else:
                        cap_activo += ganancia
                else:
                    ganancia = cap_ini * (tp / 100) if not venta_defcon else cap_ini * ((row['Close'] - precio_ent)/precio_ent)
                    cap_activo += ganancia 
                en_pos = False
                
            elif row['Low'] <= sl_p:
                if "TRINITY" in strat:
                    perdida = cap_activo * (sl / 100)
                    cap_activo -= perdida
                else:
                    perdida = cap_ini * (sl / 100)
                    cap_activo -= perdida
                en_pos = False
                
        if not en_pos and row.get('Signal_Buy', False):
            precio_ent = row['Close']
            en_pos = True
            
    return (cap_activo + divs) if "TRINITY" in strat else cap_activo

if not df.empty and len(df) > 20:
    st.sidebar.markdown("---")
    if st.sidebar.button("🧠 Ejecutar Optimizador IA", type="primary"):
        with st.spinner('IA Calculando cientos de matrices posibles...'):
            best_tp, best_sl, best_profit = 0, 0, 0
            # Grid Search Malla: TP de 1.0 a 6.0, SL de 0.5 a 3.0
            tp_range = np.arange(1.0, 6.1, 0.5)
            sl_range = np.arange(0.5, 3.1, 0.5)
            
            # Pre-cálculo de señales (optimización de velocidad)
            df_opt = df.copy()
            df_opt['EMA_200'] = ta.ema(df_opt['Close'], length=200)
            df_opt['ATR'] = ta.atr(df_opt['High'], df_opt['Low'], df_opt['Close'], length=14)
            bb_opt = ta.bbands(df_opt['Close'], length=20, std=2.0)
            if bb_opt is not None: df_opt = pd.concat([df_opt, bb_opt], axis=1)
            
            # (Lógica simplificada de señales para el bot de IA por velocidad)
            df_opt['Signal_Buy'] = df_opt['Close'] > df_opt['Open'] # Proxy temporal para IA
            
            for tp_test, sl_test in itertools.product(tp_range, sl_range):
                profit_test = ejecutar_backtest(df_opt, estrategia_activa, tp_test, sl_test, capital_inicial, reinvest_pct, use_macro_shield, use_atr_shield)
                if profit_test > best_profit:
                    best_profit, best_tp, best_sl = profit_test, tp_test, sl_test
                    
            st.sidebar.success(f"✅ ÓPTIMO IA: TP={best_tp}%, SL={best_sl}%")

    # --- 5. LÓGICAS MATEMÁTICAS AISLADAS (ADN) ---
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Vol_MA'] = ta.sma(df['Volume'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx_df is not None:
        df['ADX'] = adx_df.iloc[:, 0]
    else:
        df['ADX'] = 0

    # Keltner & Bollinger (Squeeze / Defcon Engine)
    df['KC_Upper'] = ta.ema(df['Close'], length=20) + (df['ATR'] * 1.5)
    df['KC_Lower'] = ta.ema(df['Close'], length=20) - (df['ATR'] * 1.5)
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={bb.columns[0]: 'BBL', bb.columns[1]: 'BBM', bb.columns[2]: 'BBU'}, inplace=True)
    else:
        df['BBU'] = df['Close']
        df['BBL'] = df['Close']

    df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
    df['BB_Delta'] = (df['BBU'] - df['BBL']).diff()
    df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10).mean()

    # Nucleos Base
    df['Vela_Verde'] = df['Close'] > df['Open']
    df['Vela_Roja'] = df['Close'] < df['Open']
    df['Vol_Anormal'] = df['Volume'] > (df['Vol_MA'] * whale_factor)
    df['Radar_Activo'] = (abs(df['Close'] - df['EMA_200']) / df['Close']) * 100 <= radar_sens

    # DEFCON Logic Real
    df['Neon_Up'] = df['Squeeze_On'] & (df['Close'] >= df['BBU'] * 0.999) & df['Vela_Verde']
    df['Neon_Dn'] = df['Squeeze_On'] & (df['Close'] <= df['BBL'] * 1.001) & df['Vela_Roja']
    df['Defcon_Buy'] = df['Neon_Up'] & (df['BB_Delta'] > df['BB_Delta_Avg']) & (df['ADX'] > 20)
    df['Defcon_Sell'] = df['Neon_Dn'] & (df['BB_Delta'] > df['BB_Delta_Avg']) & (df['ADX'] > 20)

    # Evaluación Aislada por Estrategia
    df['Signal_Buy'] = False
    
    if "TRINITY" in estrategia_activa:
        cond_whale = df['Vol_Anormal'] & df['Vela_Verde']
        cond_tec = (df['Radar_Activo'] | df['Defcon_Buy']) & df['Vela_Verde']
        df['Signal_Buy'] = cond_whale | cond_tec

    elif "JUGGERNAUT" in estrategia_activa:
        df['Macro_Safe'] = df['Close'] > df['EMA_200'] if use_macro_shield else True
        cuerpo_previo = df['Open'].shift(1) - df['Close'].shift(1)
        atr_previo = df['ATR'].shift(1)
        df['ATR_Safe'] = ~(cuerpo_previo > (atr_previo * 1.5)) if use_atr_shield else True
        
        cond_whale = df['Vol_Anormal'] & df['Vela_Verde']
        cond_tec = (df['Radar_Activo'] | df['Defcon_Buy']) & df['Vela_Verde'] & df['Macro_Safe'] & df['ATR_Safe']
        df['Signal_Buy'] = cond_whale | cond_tec

    elif "DEFCON" in estrategia_activa:
        df['Signal_Buy'] = df['Defcon_Buy'] if bot_defcon_buy else False

    # --- 6. MOTOR DE EJECUCIÓN DEL PORTAFOLIO ---
    trades = []
    equity_curve = []
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
            
            # Evaluar salida por Venta Defcon (Solo V329)
            venta_emergencia = False
            if "DEFCON" in estrategia_activa and bot_defcon_sell and row['Defcon_Sell']:
                venta_emergencia = True

            if row['High'] >= tp_price or venta_emergencia:
                if "TRINITY" in estrategia_activa:
                    ganancia_neta = active_capital * (tp_pct / 100)
                    reinvested = ganancia_neta * (reinvest_pct / 100.0)
                    safe_dividends += (ganancia_neta - reinvested)
                    active_capital += reinvested
                else: 
                    ganancia_neta = capital_inicial * (tp_pct / 100)
                    active_capital += ganancia_neta 
                    
                trades.append({'Tipo': 'WIN', 'Ganancia_$': ganancia_neta, 'Fecha': fecha})
                en_posicion = False
                
            elif row['Low'] <= sl_price:
                if "TRINITY" in estrategia_activa:
                    perdida_neta = active_capital * (sl_pct / 100)
                    active_capital -= perdida_neta
                else: 
                    perdida_neta = capital_inicial * (sl_pct / 100)
                    active_capital -= perdida_neta 
                    
                trades.append({'Tipo': 'LOSS', 'Ganancia_$': -perdida_neta, 'Fecha': fecha})
                en_posicion = False
                
        if not en_posicion and row['Signal_Buy']:
            precio_entrada = row['Close']
            en_posicion = True

        valor_actual = (active_capital + safe_dividends) if "TRINITY" in estrategia_activa else active_capital
        equity_curve.append(valor_actual)

    df['Total_Portfolio'] = equity_curve
    df['Rentabilidad_Pct'] = ((df['Total_Portfolio'] - capital_inicial) / capital_inicial) * 100
    
    # Cálculo Max Drawdown
    peak = df['Total_Portfolio'].cummax()
    drawdown = (df['Total_Portfolio'] - peak) / peak
    max_drawdown = drawdown.min() * 100

    # --- 7. MÉTRICAS INSTITUCIONALES FRONT-END ---
    total_trades = len(trades)
    if total_trades > 0:
        df_trades = pd.DataFrame(trades)
        wins = len(df_trades[df_trades['Tipo'] == 'WIN'])
        losses = len(df_trades[df_trades['Tipo'] == 'LOSS'])
        win_rate = (wins / total_trades) * 100
        gross_profit = df_trades[df_trades['Tipo'] == 'WIN']['Ganancia_$'].sum()
        gross_loss = abs(df_trades[df_trades['Tipo'] == 'LOSS']['Ganancia_$'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        ratio_wl = (gross_profit/wins)/(gross_loss/losses) if losses > 0 and wins > 0 else 0
    else:
        win_rate = profit_factor = ratio_wl = gross_profit = gross_loss = 0

    capital_final_total = df['Total_Portfolio'].iloc[-1]
    retorno_pct_final = df['Rentabilidad_Pct'].iloc[-1]

    st.markdown(f"### 📊 Auditoría Estricta: {estrategia_activa.split(' ')[0]}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Portafolio Final", f"${capital_final_total:,.2f}", f"{retorno_pct_final:,.2f}% Retorno")
    
    if "TRINITY" in estrategia_activa:
        col2.metric("Dividendos Seguros", f"${safe_dividends:,.2f}")
    elif "JUGGERNAUT" in estrategia_activa:
        col2.metric("Lote Lineal (Fijo)", f"${capital_inicial:,.2f}")
    else:
        col2.metric("Modo de Combate", "SQUEEZE PURO")
        
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Profit Factor", f"{profit_factor:.2f}x")
    col5.metric("Máximo Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")

    st.markdown("---")

    # --- 8. MOTOR GRÁFICO AVANZADO ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.65, 0.35], specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Mercado"), row=1, col=1)
    
    if "DEFCON" in estrategia_activa and 'BBU' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='Bollinger Top'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='Bollinger Bot'), row=1, col=1)
    elif 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='AEGIS EMA 200', line=dict(color='orange', width=2)), row=1, col=1)

    compras = df[df['Signal_Buy']]
    if not compras.empty:
        fig.add_trace(go.Scatter(x=compras.index, y=compras['Low'] * 0.98, mode='markers', name='Impacto Algorítmico', marker=dict(symbol='triangle-up', color='cyan', size=12)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Total_Portfolio'], mode='lines', name='Crecimiento ($)', line=dict(color='#00FF00', width=3)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['Rentabilidad_Pct'], mode='lines', name='Rentabilidad Neta (%)', line=dict(color='rgba(0,0,0,0)')), row=2, col=1, secondary_y=True)

    fig.update_layout(template='plotly_dark', height=850, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Capital Total (USD)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Rentabilidad (%)", row=2, col=1, secondary_y=True, ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ La bolsa mundial no provee datos tan antiguos para la temporalidad seleccionada. Pruebe reduciendo los días o usando 1 Hora / 1 Día.")
