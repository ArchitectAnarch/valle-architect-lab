import streamlit as st
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="ROCKET PROTOCOL | Lab Quant AI", layout="wide", initial_sidebar_state="expanded")

# --- MEMORIA IA BLINDADA ---
if 'tp_pct' not in st.session_state: st.session_state.tp_pct = 3.0
if 'sl_pct' not in st.session_state: st.session_state.sl_pct = 1.5
if 'whale_factor' not in st.session_state: st.session_state.whale_factor = 2.5
if 'radar_sens' not in st.session_state: st.session_state.radar_sens = 1.5
if 'reinvest_pct' not in st.session_state: st.session_state.reinvest_pct = 50.0

st.title("⚙️ ROCKET PROTOCOL LAB - Gemelo Digital Quant")
st.markdown("Sincronización Matemática Estricta con PineScript. Cero Sesgo de Simulación.")

# --- 1. PANEL DE CONTROL ---
st.sidebar.markdown("### 🚀 ROCKET PROTOCOL LAB")

exchanges_soportados = {"Coinbase (Pro)": "coinbase", "Binance": "binance", "Kraken": "kraken", "KuCoin": "kucoin"}
exchange_sel = st.sidebar.selectbox("🏦 Proveedor de Liquidez", list(exchanges_soportados.keys()))
id_exchange = exchanges_soportados[exchange_sel]

ticker = st.sidebar.text_input("Símbolo Exacto (Ej. HNT/USD, BTC/USDT)", value="HNT/USD")

intervalos = {
    "1 Minuto": ("1m", "1T"), "5 Minutos": ("5m", "5T"), 
    "7 Minutos": ("1m", "7T"), "13 Minutos": ("1m", "13T"), 
    "15 Minutos": ("15m", "15T"), "23 Minutos": ("1m", "23T"), 
    "30 Minutos": ("30m", "30T"), "1 Hora": ("1h", "1H"), 
    "4 Horas": ("4h", "4H"), "1 Día": ("1d", "1D")
}
intervalo_sel = st.sidebar.selectbox("Temporalidad (Velas)", list(intervalos.keys()), index=4)
iv_download, iv_resample = intervalos[intervalo_sel]

hoy = datetime.today().date()
limite_dias = 30 if iv_download == "1m" else 730 if iv_download in ["5m", "15m", "30m"] else 1800
fecha_minima = hoy - timedelta(days=limite_dias)

start_date, end_date = st.sidebar.slider("📅 Time Frame (Rango Histórico)", min_value=fecha_minima, max_value=hoy, value=(hoy - timedelta(days=30), hoy), format="YYYY-MM-DD")

st.sidebar.markdown("---")
capital_inicial = st.sidebar.number_input("Capital Inicial Base (USD)", value=13364.0, step=1000.0)
comision_pct = st.sidebar.number_input("Comisión por Trade (%)", value=0.25, step=0.05) / 100.0

# --- 2. ARQUITECTURA Y CALIBRACIÓN ---
st.sidebar.header("🧠 Selección de Arquitectura")
estrategia_activa = st.sidebar.radio("Motor de Ejecución:", [
    "TRINITY V357 (Dividendos + Compuesto)", 
    "JUGGERNAUT V356 (Lineal + AEGIS)",
    "DEFCON V329 (Pura Expansión Squeeze)"
])

st.sidebar.header(f"🎯 Calibración: {estrategia_activa.split(' ')[0]}")
with st.sidebar.form("calibracion_form"):
    st.session_state.tp_pct = st.slider("🎯 Take Profit (%)", 0.5, 15.0, value=float(st.session_state.tp_pct), step=0.1)
    st.session_state.sl_pct = st.slider("🛑 Stop Loss (%)", 0.5, 10.0, value=float(st.session_state.sl_pct), step=0.1)

    use_macro_shield, use_atr_shield, bot_defcon_buy, bot_defcon_sell = False, False, True, True

    if "TRINITY" in estrategia_activa:
        st.session_state.reinvest_pct = st.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.reinvest_pct), step=5.0)
        st.session_state.whale_factor = st.slider("🐋 Factor Ballena (xVol)", 1.0, 5.0, value=float(st.session_state.whale_factor), step=0.1)
        st.session_state.radar_sens = st.slider("📡 Hitbox Radar (%)", 0.1, 5.0, value=float(st.session_state.radar_sens), step=0.1)
    elif "JUGGERNAUT" in estrategia_activa:
        use_macro_shield = st.checkbox("Bloqueo Macro (EMA 200)", value=True)
        use_atr_shield = st.checkbox("Bloqueo Volatilidad (>1.5 ATR)", value=True)
        st.session_state.whale_factor = st.slider("🐋 Factor Ballena (xVol)", 1.0, 5.0, value=float(st.session_state.whale_factor), step=0.1)
        st.session_state.radar_sens = st.slider("📡 Hitbox Radar (%)", 0.1, 5.0, value=float(st.session_state.radar_sens), step=0.1)
    elif "DEFCON" in estrategia_activa:
        bot_defcon_buy = st.checkbox("Comprar en DEFCON 1/2", value=True)
        bot_defcon_sell = st.checkbox("Cerrar en DEFCON Inverso", value=True)
        
    submit_calibracion = st.form_submit_button("⚡ Aplicar Algoritmo")

# --- REPOSITORIO DE SCRIPTS ---
st.sidebar.markdown("---")
with st.sidebar.expander("📂 Repositorio de Scripts (PineScript)"):
    st.markdown("**TRINITY V357**")
    st.code("// Copie desde su archivo original de TradingView", language="pine")
    st.markdown("**JUGGERNAUT V356**")
    st.code("// Copie desde su archivo original de TradingView", language="pine")
    st.markdown("**DEFCON V329**")
    st.code("// Copie desde su archivo original de TradingView", language="pine")

# --- 3. EXTRACCIÓN CCXT ---
@st.cache_data(ttl=300)
def cargar_datos_ccxt(exchange_id, sym, start, end, iv_down, iv_res):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        
        all_ohlcv = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            if len(all_ohlcv) > 200000: break
            
        if not all_ohlcv: return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        if iv_down != iv_res:
            df = df.resample(iv_res).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

with st.spinner(f'Descargando matriz de {exchange_sel}...'):
    df = cargar_datos_ccxt(id_exchange, ticker, start_date, end_date, iv_download, iv_resample)

# --- 4. GEMELO DIGITAL: CÁLCULOS MATEMÁTICOS ESTRICTOS ---
if not df.empty and len(df) > 30:
    df['EMA_200'] = ta.ema(df['Close'], length=200).fillna(df['Close'])
    df['Vol_MA'] = ta.sma(df['Volume'], length=20).fillna(df['Volume'])
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(0.001)
    df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50.0)
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx_df.iloc[:, 0].fillna(0.0) if adx_df is not None else 0.0

    df['KC_Upper'] = ta.ema(df['Close'], length=20) + (df['ATR'] * 1.5)
    df['KC_Lower'] = ta.ema(df['Close'], length=20) - (df['ATR'] * 1.5)
    
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={bb.columns[0]: 'BBL', bb.columns[1]: 'BBM', bb.columns[2]: 'BBU'}, inplace=True)
    else:
        df['BBU'], df['BBL'] = df['Close'], df['Close']

    df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
    df['BB_Delta'] = (df['BBU'] - df['BBL']).diff().fillna(0)
    df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10).mean().fillna(0)
    
    df['Vela_Verde'] = df['Close'] > df['Open']
    df['Vela_Roja'] = df['Close'] < df['Open']
    df['Cuerpo_Vela'] = abs(df['Close'] - df['Open'])

    # 🛠️ CORRECCIÓN 1: PIVOT MATRIX (TARGET LOCK REAL)
    df['Pivot_Low_30'] = df['Low'].rolling(window=30, center=False).min()
    df['Pivot_High_30'] = df['High'].rolling(window=30, center=False).max()
    
    # --- 5. SIMULACIÓN ESTRICTA ---
    def generar_senales(df_sim, strat, w_factor, r_sens, macro_sh, atr_sh):
        # 🛠️ CORRECCIÓN 2: ANATOMÍA DE LA BALLENA ROSA (Volumen + Cuerpo > 0.3 ATR)
        df_sim['Whale_Cond'] = df_sim['Cuerpo_Vela'] > (df_sim['ATR'] * 0.3)
        df_sim['Vol_Anormal'] = (df_sim['Volume'] > (df_sim['Vol_MA'] * w_factor)) & df_sim['Whale_Cond']
        
        # 🛠️ CORRECCIÓN 1 CONTINUACIÓN: DISTANCIA A PIVOTES (No a EMA)
        dist_pivot_low = (abs(df_sim['Close'] - df_sim['Pivot_Low_30']) / df_sim['Close']) * 100
        dist_pivot_high = (abs(df_sim['Close'] - df_sim['Pivot_High_30']) / df_sim['Close']) * 100
        df_sim['Radar_Activo'] = (dist_pivot_low <= r_sens) | (dist_pivot_high <= r_sens)
        
        df_sim['Neon_Up'] = df_sim['Squeeze_On'] & (df_sim['Close'] >= df_sim['BBU'] * 0.999) & df_sim['Vela_Verde']
        df_sim['Neon_Dn'] = df_sim['Squeeze_On'] & (df_sim['Close'] <= df_sim['BBL'] * 1.001) & df_sim['Vela_Roja']
        df_sim['Defcon_Buy'] = df_sim['Neon_Up'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
        df_sim['Defcon_Sell'] = df_sim['Neon_Dn'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
        df_sim['Therm_Wall_Sell'] = (df_sim['RSI'] > 70) & (df_sim['Close'] > df_sim['BBU']) & df_sim['Vela_Roja']

        if "TRINITY" in strat:
            df_sim['Signal_Buy'] = (df_sim['Vol_Anormal'] & df_sim['Vela_Verde']) | ((df_sim['Radar_Activo'] | df_sim['Defcon_Buy']) & df_sim['Vela_Verde'])
            df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell']
        elif "JUGGERNAUT" in strat:
            df_sim['Macro_Safe'] = df_sim['Close'] > df_sim['EMA_200'] if macro_sh else True
            df_sim['ATR_Safe'] = ~(df_sim['Cuerpo_Vela'].shift(1).fillna(0) > (df_sim['ATR'].shift(1).fillna(0.001) * 1.5)) if atr_sh else True
            df_sim['Signal_Buy'] = (df_sim['Vol_Anormal'] & df_sim['Vela_Verde']) | ((df_sim['Radar_Activo'] | df_sim['Defcon_Buy']) & df_sim['Vela_Verde'] & df_sim['Macro_Safe'] & df_sim['ATR_Safe'])
            df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell']
        elif "DEFCON" in strat:
            df_sim['Signal_Buy'] = df_sim['Defcon_Buy'] if bot_defcon_buy else False
            df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] if bot_defcon_sell else False
        return df_sim

    def ejecutar_simulacion(df_sim, strat, tp, sl, cap_ini, reinvest, com_pct):
        registro_trades = []
        curva_capital = [cap_ini] * len(df_sim)
        en_pos = False
        precio_ent = 0.0
        cap_activo = cap_ini
        divs = 0.0
        
        for i in range(len(df_sim)):
            row = df_sim.iloc[i]
            fecha = df_sim.index[i]
            
            if en_pos:
                tp_price = precio_ent * (1 + (tp / 100))
                sl_price = precio_ent * (1 - (sl / 100))
                
                # EJECUCIÓN (Verificamos High y Low reales del día)
                if row['High'] >= tp_price:
                    ganancia_bruta = cap_activo * (tp / 100) if "TRINITY" in strat else cap_ini * (tp / 100)
                    costo_salida = (cap_activo + ganancia_bruta) * com_pct if "TRINITY" in strat else (cap_ini + ganancia_bruta) * com_pct
                    ganancia_neta = ganancia_bruta - costo_salida
                    
                    if "TRINITY" in strat:
                        reinv = ganancia_neta * (reinvest / 100.0)
                        divs += (ganancia_neta - reinv)
                        cap_activo += reinv
                    else: cap_activo += ganancia_neta
                    
                    registro_trades.append({'Fecha': fecha, 'Tipo': 'TP', 'Precio': tp_price, 'Ganancia_$': ganancia_neta})
                    en_pos = False
                    
                elif row['Low'] <= sl_price:
                    perdida_bruta = cap_activo * (sl / 100) if "TRINITY" in strat else cap_ini * (sl / 100)
                    costo_salida = (cap_activo - perdida_bruta) * com_pct if "TRINITY" in strat else (cap_ini - perdida_bruta) * com_pct
                    perdida_neta = perdida_bruta + costo_salida
                    cap_activo -= perdida_neta
                    registro_trades.append({'Fecha': fecha, 'Tipo': 'SL', 'Precio': sl_price, 'Ganancia_$': -perdida_neta})
                    en_pos = False
                    
                elif row['Signal_Sell']:
                    retorno_pct = (row['Close'] - precio_ent) / precio_ent
                    ganancia_bruta = cap_activo * retorno_pct if "TRINITY" in strat else cap_ini * retorno_pct
                    costo_salida = (cap_activo + ganancia_bruta) * com_pct if "TRINITY" in strat else (cap_ini + ganancia_bruta) * com_pct
                    ganancia_neta = ganancia_bruta - costo_salida
                    if "TRINITY" in strat and ganancia_neta > 0:
                        reinv = ganancia_neta * (reinvest / 100.0)
                        divs += (ganancia_neta - reinv)
                        cap_activo += reinv
                    else: cap_activo += ganancia_neta
                    
                    tipo = 'DYNAMIC_WIN' if ganancia_neta > 0 else 'DYNAMIC_LOSS'
                    registro_trades.append({'Fecha': fecha, 'Tipo': tipo, 'Precio': row['Close'], 'Ganancia_$': ganancia_neta})
                    en_pos = False

            # 🛠️ CORRECCIÓN 3: EJECUCIÓN CON DESFASE REAL (NEXT OPEN)
            if not en_pos and row.get('Signal_Buy', False):
                # Validamos que exista una vela siguiente para ejecutar
                if i + 1 < len(df_sim):
                    precio_ent = df_sim['Open'].iloc[i+1] # Compra en la apertura de la siguiente vela
                    fecha_ent = df_sim.index[i+1]
                    costo_entrada = cap_activo * com_pct if "TRINITY" in strat else cap_ini * com_pct
                    cap_activo -= costo_entrada
                    en_pos = True
                    registro_trades.append({'Fecha': fecha_ent, 'Tipo': 'ENTRY', 'Precio': precio_ent, 'Ganancia_$': -costo_entrada})

            curva_capital[i] = (cap_activo + divs) if "TRINITY" in strat else cap_activo
            
        return curva_capital, divs, cap_activo, registro_trades

    # --- 6. OPTIMIZADOR IA ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Central de Inteligencia")
    if st.sidebar.button("🤖 Auto-Ajustar Parámetros IA", type="primary"):
        with st.spinner('IA Escaneando ADN Matemático...'):
            best_profit = -999999
            best_params = {}
            for _ in range(100):
                t_tp = round(random.uniform(1.0, 8.0), 1)
                t_sl = round(random.uniform(0.5, 3.0), 1)
                t_whale = round(random.uniform(1.5, 3.5), 1)
                t_radar = round(random.uniform(0.5, 3.0), 1)
                t_reinvest = round(random.uniform(20, 100), -1)
                
                df_test = generar_senales(df.copy(), estrategia_activa, t_whale, t_radar, use_macro_shield, use_atr_shield)
                curva_test, _, _, _ = ejecutar_simulacion(df_test, estrategia_activa, t_tp, t_sl, capital_inicial, t_reinvest, comision_pct)
                
                if curva_test[-1] > best_profit:
                    best_profit = curva_test[-1]
                    best_params = {'tp': t_tp, 'sl': t_sl, 'whale': t_whale, 'radar': t_radar, 'reinvest': t_reinvest}
            
            st.session_state.tp_pct, st.session_state.sl_pct = float(best_params['tp']), float(best_params['sl'])
            st.session_state.whale_factor, st.session_state.radar_sens = float(best_params['whale']), float(best_params['radar'])
            if "TRINITY" in estrategia_activa: st.session_state.reinvest_pct = float(best_params['reinvest'])
            st.rerun()

    # EJECUCIÓN PRINCIPAL
    df = generar_senales(df, estrategia_activa, st.session_state.whale_factor, st.session_state.radar_sens, use_macro_shield, use_atr_shield)
    equity_curve, safe_dividends, active_capital, trades_log = ejecutar_simulacion(df, estrategia_activa, st.session_state.tp_pct, st.session_state.sl_pct, capital_inicial, st.session_state.reinvest_pct, comision_pct)
    
    df['Total_Portfolio'] = equity_curve
    df['Rentabilidad_Pct'] = ((df['Total_Portfolio'] - capital_inicial) / capital_inicial) * 100

    # --- 7. MÉTRICAS ---
    df_trades = pd.DataFrame(trades_log) if len(trades_log) > 0 else pd.DataFrame()
    total_trades, wins, losses, win_rate, profit_factor = 0, 0, 0, 0, 0
    
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

    st.markdown(f"### 📊 Auditoría Definitiva: {estrategia_activa.split(' ')[0]}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Portafolio Final Neto", f"${df['Total_Portfolio'].iloc[-1]:,.2f}", f"{df['Rentabilidad_Pct'].iloc[-1]:,.2f}% Retorno")
    
    if "TRINITY" in estrategia_activa: col2.metric("Dividendos Extraídos", f"${safe_dividends:,.2f}")
    elif "JUGGERNAUT" in estrategia_activa: col2.metric("Capital Operativo", f"${active_capital:,.2f}")
    else: col2.metric("Modo", "SQUEEZE PURO")
        
    col3.metric("Win Rate Real", f"{win_rate:.1f}%")
    col4.metric("Profit Factor", f"{profit_factor:.2f}x")
    col5.metric("Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")

    # --- 8. MOTOR GRÁFICO (CROSSHAIR INTACTO) ---
    st.markdown("---")
    st.subheader(f"📈 Mapa de Ejecución Gemelo ({id_exchange.upper()})")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.65, 0.35], specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    hovertemp_clean = "Fecha: %{x}<br>Precio: $%{y:,.4f}<extra></extra>"

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Mercado"), row=1, col=1)
    
    if "DEFCON" in estrategia_activa and 'BBU' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='Bollinger Top', hovertemplate=hovertemp_clean), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='Bollinger Bot', hovertemplate=hovertemp_clean), row=1, col=1)
    elif 'EMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200', line=dict(color='orange', width=2), hovertemplate=hovertemp_clean), row=1, col=1)

    if not df_trades.empty:
        entradas = df_trades[df_trades['Tipo'] == 'ENTRY']
        fig.add_trace(go.Scatter(x=entradas['Fecha'], y=entradas['Precio'] * 0.98, mode='markers', name='Compra (Siguiente Open)', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(color='white', width=1)), hovertemplate=hovertemp_clean), row=1, col=1)
        
        salidas = df_trades[df_trades['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
        colores_salida = ['#00FF00' if t in ['TP', 'DYNAMIC_WIN'] else '#FF0000' for t in salidas['Tipo']]
        fig.add_trace(go.Scatter(x=salidas['Fecha'], y=salidas['Precio'] * 1.02, mode='markers', name='Cierre', marker=dict(symbol='triangle-down', color=colores_salida, size=14, line=dict(color='white', width=1)), text=salidas['Tipo'], hovertemplate="Cierre: %{text}<br>Fecha: %{x}<br>Precio: $%{y:,.4f}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Total_Portfolio'], mode='lines', name='Equidad Neta ($)', line=dict(color='#00FF00', width=3), hovertemplate="Fecha: %{x}<br>Capital: $%{y:,.2f}<extra></extra>"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['Rentabilidad_Pct'], mode='lines', name='Rentabilidad (%)', line=dict(color='rgba(0,0,0,0)')), row=2, col=1, secondary_y=True)

    fig.update_xaxes(showspikes=True, spikecolor="cyan", spikesnap="cursor", spikemode="toaxis+across", spikethickness=1, spikedash="solid")
    fig.update_yaxes(showspikes=True, spikecolor="cyan", spikesnap="cursor", spikemode="toaxis+across", spikethickness=1, spikedash="solid")

    fig.update_layout(
        template='plotly_dark', height=850, xaxis_rangeslider_visible=False, 
        margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ Datos Insuficientes. Intente ampliar el Time Frame o cambiar de activo.")
