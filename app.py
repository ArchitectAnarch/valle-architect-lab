import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta

st.set_page_config(page_title="ROCKET PROTOCOL | Lab Quant AI", layout="wide", initial_sidebar_state="expanded")

st.title("⚙️ ROCKET PROTOCOL LAB - Centro de Inteligencia Quant")
st.markdown("Simulación Multi-Arquitectura con Detección de Impacto Exacto y Optimización IA.")

# --- 1. PANEL DE CONTROL: MERCADO Y TIEMPO ---
st.sidebar.markdown("### 🚀 ROCKET PROTOCOL LAB")
ticker = st.sidebar.text_input("Símbolo (Ej. HNT-USD, BTC-USD)", value="HNT-USD")

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
dias_defecto = 6 if iv_download == "1m" else 59 if iv_download in ["5m", "15m", "30m"] else 365
default_start = datetime.today() - timedelta(days=dias_defecto)

start_date = col_date1.date_input("Fecha Inicio", value=default_start)
end_date = col_date2.date_input("Fecha Fin", value=datetime.today())

capital_inicial = st.sidebar.number_input("Capital Inicial Base (USD)", value=13364.0, step=1000.0)

# Motor de Auto-Corrección Temporal
dias_pedidos = (end_date - start_date).days
hoy = datetime.today().date()
if iv_download == "1m" and dias_pedidos > 6:
    st.sidebar.warning("⚠️ Temporalidad basada en minutos ajustada al límite de la bolsa (7 días).")
    start_date = hoy - timedelta(days=6)
elif iv_download in ["5m", "15m", "30m"] and dias_pedidos > 59:
    st.sidebar.warning("⚠️ Temporalidad intradía ajustada al límite de la bolsa (60 días).")
    start_date = hoy - timedelta(days=59)
elif iv_download == "1h" and dias_pedidos > 729:
    st.sidebar.warning("⚠️ Temporalidad horaria ajustada al límite de la bolsa (730 días).")
    start_date = hoy - timedelta(days=729)

# --- 2. SELECCIÓN DE ARQUITECTURA (DNA AISLADO) ---
st.sidebar.header("🧠 Selección de Arquitectura")
estrategia_activa = st.sidebar.radio("Motor de Ejecución:", [
    "TRINITY V357 (Dividendos + Compuesto)", 
    "JUGGERNAUT V356 (Lineal + AEGIS)",
    "DEFCON V329 (Pura Expansión Squeeze)"
])

st.sidebar.header(f"🎯 Calibración: {estrategia_activa.split(' ')[0]}")
tp_pct = st.sidebar.slider("🎯 Take Profit (%)", 0.5, 15.0, 3.0, 0.1)
sl_pct = st.sidebar.slider("🛑 Stop Loss (%)", 0.5, 10.0, 1.5, 0.1)

reinvest_pct = 50.0
use_macro_shield = False
use_atr_shield = False
bot_defcon_buy = True
bot_defcon_sell = True
radar_sens = 1.5
whale_factor = 2.5

if "TRINITY" in estrategia_activa:
    reinvest_pct = st.sidebar.slider("💵 Reinversión de Capital (%)", 0.0, 100.0, 50.0, 5.0)
    whale_factor = st.sidebar.slider("🐋 Multiplicador Volumen (xVol)", 1.0, 5.0, 2.5, 0.1)
    radar_sens = st.sidebar.slider("📡 Sensibilidad Impacto (%)", 0.1, 5.0, 1.5, 0.1)
elif "JUGGERNAUT" in estrategia_activa:
    use_macro_shield = st.sidebar.checkbox("Bloqueo Macroeconómico (EMA 200)", value=True)
    use_atr_shield = st.sidebar.checkbox("Bloqueo Volatilidad Extrema (>1.5 ATR)", value=True)
    whale_factor = st.sidebar.slider("🐋 Multiplicador Volumen (xVol)", 1.0, 5.0, 2.5, 0.1)
    radar_sens = st.sidebar.slider("📡 Sensibilidad Impacto (%)", 0.1, 5.0, 1.5, 0.1)
elif "DEFCON" in estrategia_activa:
    bot_defcon_buy = st.sidebar.checkbox("Entrada: Ruptura Alcista (DEFCON 1/2)", value=True)
    bot_defcon_sell = st.sidebar.checkbox("Salida Dinámica: Ruptura Bajista", value=True)

# --- 3. EXTRACCIÓN Y RESAMPLING DE DATOS ---
@st.cache_data(ttl=60)
def cargar_datos(sym, start, end, iv_down, iv_res):
    try:
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = datetime.combine(end, datetime.min.time()) + timedelta(days=1)
        df = yf.download(sym, start=start_dt, end=end_dt, interval=iv_down, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if iv_down != iv_res and not df.empty:
            df = df.resample(iv_res).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

with st.spinner('Descargando matrices de Wall Street...'):
    df = cargar_datos(ticker, start_date, end_date, iv_download, iv_resample)

# --- 4. PRE-CÁLCULO MATEMÁTICO (INDICADORES) ---
if not df.empty and len(df) > 20:
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Vol_MA'] = ta.sma(df['Volume'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx_df.iloc[:, 0] if adx_df is not None else 0

    df['KC_Upper'] = ta.ema(df['Close'], length=20) + (df['ATR'] * 1.5)
    df['KC_Lower'] = ta.ema(df['Close'], length=20) - (df['ATR'] * 1.5)
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={bb.columns[0]: 'BBL', bb.columns[1]: 'BBM', bb.columns[2]: 'BBU'}, inplace=True)
    else:
        df['BBU'], df['BBL'] = df['Close'], df['Close']

    df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
    df['BB_Delta'] = (df['BBU'] - df['BBL']).diff()
    df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10).mean()

    # Condiciones Base
    df['Vela_Verde'] = df['Close'] > df['Open']
    df['Vela_Roja'] = df['Close'] < df['Open']
    df['Vol_Anormal'] = df['Volume'] > (df['Vol_MA'] * whale_factor)
    df['Radar_Activo'] = (abs(df['Close'] - df['EMA_200']) / df['Close']) * 100 <= radar_sens

    # DEFCON Logic
    df['Neon_Up'] = df['Squeeze_On'] & (df['Close'] >= df['BBU'] * 0.999) & df['Vela_Verde']
    df['Neon_Dn'] = df['Squeeze_On'] & (df['Close'] <= df['BBL'] * 1.001) & df['Vela_Roja']
    df['Defcon_Buy'] = df['Neon_Up'] & (df['BB_Delta'] > df['BB_Delta_Avg']) & (df['ADX'] > 20)
    df['Defcon_Sell'] = df['Neon_Dn'] & (df['BB_Delta'] > df['BB_Delta_Avg']) & (df['ADX'] > 20)
    
    # Lógica de Cierre Dinámico (Thermometer Wall proxy)
    df['Therm_Wall_Sell'] = (df['RSI'] > 70) & (df['Close'] > df['BBU']) & df['Vela_Roja']

    # --- 5. ASIGNACIÓN DE SEÑALES POR ADN ---
    df['Signal_Buy'] = False
    df['Signal_Sell'] = False
    
    if "TRINITY" in estrategia_activa:
        df['Signal_Buy'] = (df['Vol_Anormal'] & df['Vela_Verde']) | ((df['Radar_Activo'] | df['Defcon_Buy']) & df['Vela_Verde'])
        df['Signal_Sell'] = df['Defcon_Sell'] | df['Therm_Wall_Sell']
    elif "JUGGERNAUT" in estrategia_activa:
        df['Macro_Safe'] = df['Close'] > df['EMA_200'] if use_macro_shield else True
        cuerpo_previo = df['Open'].shift(1) - df['Close'].shift(1)
        atr_previo = df['ATR'].shift(1)
        df['ATR_Safe'] = ~(cuerpo_previo > (atr_previo * 1.5)) if use_atr_shield else True
        df['Signal_Buy'] = (df['Vol_Anormal'] & df['Vela_Verde']) | ((df['Radar_Activo'] | df['Defcon_Buy']) & df['Vela_Verde'] & df['Macro_Safe'] & df['ATR_Safe'])
        df['Signal_Sell'] = df['Defcon_Sell'] | df['Therm_Wall_Sell']
    elif "DEFCON" in estrategia_activa:
        df['Signal_Buy'] = df['Defcon_Buy'] if bot_defcon_buy else False
        df['Signal_Sell'] = df['Defcon_Sell'] if bot_defcon_sell else False

    # --- 6. MOTOR DE BACKTESTING DE PRECISIÓN (Guarda Puntos de Impacto) ---
    def ejecutar_simulacion(df_sim, strat, tp, sl, cap_ini, reinvest):
        registro_trades = []
        curva_capital = []
        en_pos = False
        precio_ent = 0.0
        fecha_ent = None
        cap_activo = cap_ini
        divs = 0.0
        
        for i in range(len(df_sim)):
            row = df_sim.iloc[i]
            fecha = df_sim.index[i]
            
            if en_pos:
                tp_price = precio_ent * (1 + (tp / 100))
                sl_price = precio_ent * (1 - (sl / 100))
                
                # 1. Chequeo de TP
                if row['High'] >= tp_price:
                    ganancia = cap_activo * (tp / 100) if "TRINITY" in strat else cap_ini * (tp / 100)
                    if "TRINITY" in strat:
                        reinv = ganancia * (reinvest / 100.0)
                        divs += (ganancia - reinv)
                        cap_activo += reinv
                    else: cap_activo += ganancia
                    
                    registro_trades.append({'Fecha': fecha, 'Tipo': 'TP', 'Precio': tp_price, 'Ganancia_$': ganancia})
                    en_pos = False
                    
                # 2. Chequeo de SL
                elif row['Low'] <= sl_price:
                    perdida = cap_activo * (sl / 100) if "TRINITY" in strat else cap_ini * (sl / 100)
                    cap_activo -= perdida
                    registro_trades.append({'Fecha': fecha, 'Tipo': 'SL', 'Precio': sl_price, 'Ganancia_$': -perdida})
                    en_pos = False
                    
                # 3. Chequeo de Cierre Dinámico (Venta Algorítmica Anticipada)
                elif row['Signal_Sell']:
                    retorno_pct = (row['Close'] - precio_ent) / precio_ent
                    ganancia = cap_activo * retorno_pct if "TRINITY" in strat else cap_ini * retorno_pct
                    
                    if "TRINITY" in strat and ganancia > 0:
                        reinv = ganancia * (reinvest / 100.0)
                        divs += (ganancia - reinv)
                        cap_activo += reinv
                    else: cap_activo += ganancia
                    
                    tipo_cierre = 'DYNAMIC_WIN' if ganancia > 0 else 'DYNAMIC_LOSS'
                    registro_trades.append({'Fecha': fecha, 'Tipo': tipo_cierre, 'Precio': row['Close'], 'Ganancia_$': ganancia})
                    en_pos = False

            # Evaluar Entrada si no hay posición
            if not en_pos and row['Signal_Buy']:
                precio_ent = row['Close']
                fecha_ent = fecha
                en_pos = True
                registro_trades.append({'Fecha': fecha, 'Tipo': 'ENTRY', 'Precio': precio_ent, 'Ganancia_$': 0})

            valor_actual = (cap_activo + divs) if "TRINITY" in strat else cap_activo
            curva_capital.append(valor_actual)
            
        return curva_capital, divs, cap_activo, registro_trades

    # Ejecución de la simulación actual
    equity_curve, safe_dividends, active_capital, trades_log = ejecutar_simulacion(df, estrategia_activa, tp_pct, sl_pct, capital_inicial, reinvest_pct)
    df['Total_Portfolio'] = equity_curve
    df['Rentabilidad_Pct'] = ((df['Total_Portfolio'] - capital_inicial) / capital_inicial) * 100

    # --- 7. OPTIMIZADOR IA CON PROPUESTAS ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Central de Inteligencia")
    if st.sidebar.button("Ejecutar Optimización IA", type="primary"):
        with st.spinner('IA escaneando matrices operativas...'):
            best_tp, best_sl, best_profit = tp_pct, sl_pct, df['Total_Portfolio'].iloc[-1]
            tp_range = np.arange(1.0, 8.1, 0.5)
            sl_range = np.arange(0.5, 4.1, 0.5)
            
            for tp_test, sl_test in itertools.product(tp_range, sl_range):
                curva_test, _, _, _ = ejecutar_simulacion(df, estrategia_activa, tp_test, sl_test, capital_inicial, reinvest_pct)
                if curva_test[-1] > best_profit:
                    best_profit, best_tp, best_sl = curva_test[-1], tp_test, sl_test
            
            volatilidad_media = df['ATR'].mean() / df['Close'].mean() * 100
            st.sidebar.success(f"**Detección IA Completada**")
            st.sidebar.write(f"⚙️ **ÓPTIMO ENCONTRADO:** TP={best_tp}% | SL={best_sl}%")
            st.sidebar.info(f"**Fundamento IA:** La volatilidad promedio (ATR) del activo es de {volatilidad_media:.2f}%. Se propone un Stop Loss del {best_sl}% para evitar cacería de liquidez, y un TP asimétrico para maximizar el ratio R:R adaptado al régimen de tendencia actual.")

    # --- 8. MÉTRICAS FRONT-END ---
    df_trades = pd.DataFrame(trades_log) if len(trades_log) > 0 else pd.DataFrame()
    
    total_trades = 0
    wins, losses, win_rate, profit_factor, ratio_wl = 0, 0, 0, 0, 0
    
    if not df_trades.empty:
        df_exits = df_trades[df_trades['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
        total_trades = len(df_exits)
        if total_trades > 0:
            wins = len(df_exits[df_exits['Tipo'].isin(['TP', 'DYNAMIC_WIN'])])
            losses = len(df_exits[df_exits['Tipo'].isin(['SL', 'DYNAMIC_LOSS'])])
            win_rate = (wins / total_trades) * 100
            gross_profit = df_exits[df_exits['Ganancia_$'] > 0]['Ganancia_$'].sum()
            gross_loss = abs(df_exits[df_exits['Ganancia_$'] < 0]['Ganancia_$'].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    peak = df['Total_Portfolio'].cummax()
    drawdown = ((df['Total_Portfolio'] - peak) / peak) * 100
    max_drawdown = drawdown.min()

    st.markdown(f"### 📊 Auditoría Estricta: {estrategia_activa.split(' ')[0]}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Portafolio Final", f"${df['Total_Portfolio'].iloc[-1]:,.2f}", f"{df['Rentabilidad_Pct'].iloc[-1]:,.2f}% Retorno")
    
    if "TRINITY" in estrategia_activa: col2.metric("Dividendos Seguros", f"${safe_dividends:,.2f}")
    elif "JUGGERNAUT" in estrategia_activa: col2.metric("Capital Actual", f"${active_capital:,.2f}")
    else: col2.metric("Modo de Combate", "SQUEEZE PURO")
        
    col3.metric("Win Rate Absoluto", f"{win_rate:.1f}%")
    col4.metric("Profit Factor", f"{profit_factor:.2f}x")
    col5.metric("Máximo Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")

    # --- 9. MOTOR GRÁFICO AVANZADO (Puntos de Impacto Reales) ---
    st.markdown("---")
    st.subheader("📈 Mapa de Impacto Algorítmico")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.65, 0.35], specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

    # Velas
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Mercado"), row=1, col=1)
    
    # Indicadores
    if "DEFCON" in estrategia_activa and 'BBU' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='Bollinger Top'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='Bollinger Bot'), row=1, col=1)
    elif 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='Filtro EMA 200', line=dict(color='orange', width=2)), row=1, col=1)

    # 🎯 MARCADORES DE IMPACTO EXACTOS (Compras y Ventas)
    if not df_trades.empty:
        # Extraer Entradas
        entradas = df_trades[df_trades['Tipo'] == 'ENTRY']
        fig.add_trace(go.Scatter(
            x=entradas['Fecha'], y=entradas['Precio'] * 0.98, mode='markers', name='Ejecución de Compra',
            marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(color='white', width=1))
        ), row=1, col=1)
        
        # Extraer Salidas (TP, SL, y Ventas Dinámicas)
        salidas = df_trades[df_trades['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
        colores_salida = ['#00FF00' if t in ['TP', 'DYNAMIC_WIN'] else '#FF0000' for t in salidas['Tipo']]
        
        fig.add_trace(go.Scatter(
            x=salidas['Fecha'], y=salidas['Precio'] * 1.02, mode='markers', name='Ejecución de Cierre',
            marker=dict(symbol='triangle-down', color=colores_salida, size=14, line=dict(color='white', width=1)),
            text=salidas['Tipo'], hovertemplate="Cierre: %{text} a $%{y}<extra></extra>"
        ), row=1, col=1)

    # Curvas de Capital
    fig.add_trace(go.Scatter(x=df.index, y=df['Total_Portfolio'], mode='lines', name='Crecimiento ($)', line=dict(color='#00FF00', width=3)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['Rentabilidad_Pct'], mode='lines', name='Rentabilidad Neta (%)', line=dict(color='rgba(0,0,0,0)')), row=2, col=1, secondary_y=True)

    fig.update_layout(template='plotly_dark', height=850, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Capital Total (USD)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Rentabilidad (%)", row=2, col=1, secondary_y=True, ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Operación abortada: No hay datos suficientes de la API para construir las matrices de impacto.")
