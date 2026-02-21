import streamlit as st
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import pandas as pd
import numpy as np
import random
import os
import glob
import gc
from datetime import datetime, timedelta

st.set_page_config(page_title="ROCKET PROTOCOL | Alpha Quant", layout="wide", initial_sidebar_state="expanded")

# --- MEMORIA IA INSTITUCIONAL ---
buy_rules = ['Pink_Whale_Buy', 'Lock_Bounce', 'Lock_Break', 'Defcon_Buy', 'Neon_Up', 'Therm_Bounce', 'Therm_Vacuum', 'Nuclear_Buy', 'Early_Buy', 'Rebound_Buy']
sell_rules = ['Defcon_Sell', 'Neon_Dn', 'Therm_Wall_Sell', 'Therm_Panic_Sell', 'Lock_Reject', 'Lock_Breakd', 'Nuclear_Sell', 'Early_Sell']

# INICIALIZACIÓN ESTRICTA
for r_idx in range(1, 5):
    if f'gen_r{r_idx}_b' not in st.session_state: st.session_state[f'gen_r{r_idx}_b'] = ['Nuclear_Buy']
    if f'gen_r{r_idx}_s' not in st.session_state: st.session_state[f'gen_r{r_idx}_s'] = ['Nuclear_Sell']
    if f'gen_r{r_idx}_tp' not in st.session_state: st.session_state[f'gen_r{r_idx}_tp'] = 5.0
    if f'gen_r{r_idx}_sl' not in st.session_state: st.session_state[f'gen_r{r_idx}_sl'] = 2.0

if 'gen_ado' not in st.session_state: st.session_state['gen_ado'] = 5.0  
if 'winning_dna' not in st.session_state: st.session_state['winning_dna'] = ""

for s in ["TRINITY", "JUGGERNAUT", "DEFCON"]:
    if f'sld_tp_{s}' not in st.session_state: st.session_state[f'sld_tp_{s}'] = 3.0
    if f'sld_sl_{s}' not in st.session_state: st.session_state[f'sld_sl_{s}'] = 1.5
    if f'sld_wh_{s}' not in st.session_state: st.session_state[f'sld_wh_{s}'] = 2.5
    if f'sld_rd_{s}' not in st.session_state: st.session_state[f'sld_rd_{s}'] = 1.5
    if f'sld_reinv_{s}' not in st.session_state: st.session_state[f'sld_reinv_{s}'] = 100.0

# --- 1. PANEL LATERAL ---
css_spinner = """
<style>
.loader-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; pointer-events: none; background: transparent; }
.rocket { font-size: 10rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 35px rgba(0, 255, 255, 1)); }
@keyframes spin { 0% { transform: scale(1) rotate(0deg); } 50% { transform: scale(1.2) rotate(180deg); } 100% { transform: scale(1) rotate(360deg); } }
</style>
<div class="loader-container"><div class="rocket">🚀</div></div>
"""
ph_holograma = st.empty()

st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 TV REPLICA LAB</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True): 
    st.cache_data.clear()
    gc.collect()

st.sidebar.markdown("---")
st.sidebar.info("⚡ ARRANQUE SEGURO: Binance + BTC/USDT. Cámbielo a Coinbase y HNT/USD para extraer el ADN de Helium, pero use un 'Scope Histórico' menor (Ej: 365 días).")
exchange_sel = st.sidebar.selectbox("🏦 Exchange", ["binance", "coinbase", "kraken", "kucoin"], index=0)
ticker = st.sidebar.text_input("Símbolo Exacto", value="BTC/USDT")
utc_offset = st.sidebar.number_input("🌍 Zona Horaria", value=-5.0, step=0.5)

intervalos = {
    "1 Minuto": ("1m", "1min"), "5 Minutos": ("5m", "5min"), 
    "7 Minutos": ("1m", "7min"), "13 Minutos": ("1m", "13min"), 
    "15 Minutos": ("15m", "15min"), "23 Minutos": ("1m", "23min"), 
    "30 Minutos": ("30m", "30min"), "1 Hora": ("1h", "1h"), 
    "2 Horas": ("1h", "2h"), "4 Horas": ("4h", "4h"), 
    "1 Día": ("1d", "1D"), "1 Semana": ("1d", "1W")
}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=2) 
iv_download, iv_resample = intervalos[intervalo_sel]

hoy = datetime.today().date()
limite_dias = 30 if iv_download == "1m" else 180 if iv_download == "5m" else 1500
start_date, end_date = st.sidebar.slider(f"📅 Scope Histórico (Máx {limite_dias} días para esta temp)", min_value=hoy - timedelta(days=limite_dias), max_value=hoy, value=(hoy - timedelta(days=min(60, limite_dias)), hoy), format="YYYY-MM-DD")
dias_analizados = max((end_date - start_date).days, 1)

capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=1000.0, step=100.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0

# --- 2. EXTRACCIÓN MAESTRA (WARP DRIVE) ---
@st.cache_data(ttl=3600, show_spinner="📡 WARP DRIVE: Descargando y ensamblando miles de velas. Por favor espere...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, iv_res, offset):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        
        all_ohlcv, current_ts = [], start_ts
        while current_ts < end_ts:
            ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            if len(all_ohlcv) > 150000: break
            
        if not all_ohlcv: return pd.DataFrame(), "El Exchange devolvió 0 velas. Símbolo incorrecto o fecha muy antigua."
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index + timedelta(hours=offset)
        df = df[~df.index.duplicated(keep='first')]
        
        if iv_down != iv_res: 
            df = df.resample(iv_res).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        if len(df) > 50:
            df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1, adjust=False).mean()
            df['Vol_MA_100'] = df['Volume'].rolling(window=100, min_periods=1).mean()
            df['RVol'] = df['Volume'] / df['Vol_MA_100'].replace(0, 1)
            
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = df[['High', 'Low']].max(axis=1) - df[['High', 'Low']].min(axis=1)
            df['ATR'] = tr.ewm(alpha=1/14, min_periods=1, adjust=False).mean().fillna(high_low).replace(0, 0.001)
            
            df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50.0)
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0].fillna(0.0)

            df['KC_Upper'] = df['EMA_200'] + (df['ATR'] * 1.5)
            df['KC_Lower'] = df['EMA_200'] - (df['ATR'] * 1.5)
            
            basis = df['Close'].rolling(20, min_periods=1).mean()
            dev = df['Close'].rolling(20, min_periods=1).std().replace(0, 1)
            df['BBU'] = basis + (2.0 * dev)
            df['BBL'] = basis - (2.0 * dev)

            df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
            df['BB_Delta'] = (df['BBU'] - df['BBL']).diff().fillna(0)
            df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10, min_periods=1).mean().fillna(0)
            df['Vela_Verde'] = df['Close'] > df['Open']
            df['Vela_Roja'] = df['Close'] < df['Open']
            df['Cuerpo_Vela'] = abs(df['Close'] - df['Open'])
            
            df['PL30'] = df['Low'].rolling(30, min_periods=1).min()
            df['PH30'] = df['High'].rolling(30, min_periods=1).max()
            df['PL100'] = df['Low'].rolling(100, min_periods=1).min()
            df['PH100'] = df['High'].rolling(100, min_periods=1).max()
            df['PL300'] = df['Low'].rolling(300, min_periods=1).min()
            df['PH300'] = df['High'].rolling(300, min_periods=1).max()
            
            df['Z_Score'] = (df['Close'] - basis) / dev
            rsi_ma = df['RSI'].rolling(14, min_periods=1).mean()
            df['RSI_Cross_Up'] = (df['RSI'] > rsi_ma) & (df['RSI'].shift(1).fillna(50) <= rsi_ma.shift(1).fillna(50))
            df['RSI_Cross_Dn'] = (df['RSI'] < rsi_ma) & (df['RSI'].shift(1).fillna(50) >= rsi_ma.shift(1).fillna(50))
            df['Retro_Peak'] = (df['RSI'] < 30) & (df['Close'] < df['BBL'])
            
            ap = (df['High'] + df['Low'] + df['Close']) / 3
            esa = ap.ewm(span=10, min_periods=1).mean()
            d_wt = abs(ap - esa).ewm(span=10, min_periods=1).mean().replace(0, 1)
            ci = (ap - esa) / (0.015 * d_wt)
            wt1 = ci.ewm(span=21, min_periods=1).mean()
            wt2 = wt1.rolling(4, min_periods=1).mean()
            df['WT_Cross_Up'] = (wt1 > wt2) & (wt1.shift(1).fillna(0) <= wt2.shift(1).fillna(0))
            df['WT_Cross_Dn'] = (wt1 < wt2) & (wt1.shift(1).fillna(0) >= wt2.shift(1).fillna(0))
            df['WT_Oversold'] = wt1 < -60
            df['WT_Overbought'] = wt1 > 60
            
            df['Macro_Bull'] = df['Close'] >= df['EMA_200']
            is_trend = df['ADX'] >= 25
            df['Regime'] = np.where(df['Macro_Bull'] & is_trend, 1,
                           np.where(df['Macro_Bull'] & ~is_trend, 2,
                           np.where(~df['Macro_Bull'] & is_trend, 3, 4)))
            gc.collect()

        return df, "OK"
    except Exception as e: 
        return pd.DataFrame(), str(e)

df_global, status_api = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, iv_resample, utc_offset)
if df_global.empty:
    st.error(f"🚨 ERROR API: No hay datos suficientes. Detalles del servidor CCXT: {status_api}. Use BINANCE y BTC/USDT para historial profundo, o acorte las fechas si usa altcoins recientes.")

# --- 3. MOTOR PRE-CÁLCULO TOPOLÓGICO ---
def inyectar_adn(df_sim, r_sens=1.5, w_factor=2.5):
    df_sim['Whale_Cond'] = df_sim['Cuerpo_Vela'] > (df_sim['ATR'] * 0.3)
    df_sim['Flash_Vol'] = (df_sim['RVol'] > (w_factor * 0.8)) & df_sim['Whale_Cond']
    
    df_sim['Target_Lock_Sup'] = df_sim[['PL30', 'PL100', 'PL300']].max(axis=1)
    df_sim['Target_Lock_Res'] = df_sim[['PH30', 'PH100', 'PH300']].min(axis=1)
    tol = df_sim['ATR'] * 0.5
    
    df_sim['Lock_Bounce'] = (df_sim['Low'] <= (df_sim['Target_Lock_Sup'] + tol)) & (df_sim['Close'] > df_sim['Target_Lock_Sup']) & df_sim['Vela_Verde']
    df_sim['Lock_Break'] = (df_sim['Close'] > df_sim['Target_Lock_Res']) & (df_sim['Open'] <= df_sim['Target_Lock_Res']) & df_sim['Flash_Vol'] & df_sim['Vela_Verde']
    df_sim['Lock_Reject'] = (df_sim['High'] >= (df_sim['Target_Lock_Res'] - tol)) & (df_sim['Close'] < df_sim['Target_Lock_Res']) & df_sim['Vela_Roja']
    df_sim['Lock_Breakd'] = (df_sim['Close'] < df_sim['Target_Lock_Sup']) & (df_sim['Open'] >= df_sim['Target_Lock_Sup']) & df_sim['Vela_Roja']
    
    dist_sup = (abs(df_sim['Close'] - df_sim['PL30']) / df_sim['Close']) * 100
    dist_res = (abs(df_sim['Close'] - df_sim['PH30']) / df_sim['Close']) * 100
    df_sim['Radar_Activo'] = (dist_sup <= r_sens) | (dist_res <= r_sens)

    buy_score = np.zeros(len(df_sim))
    buy_score = np.where(df_sim['Retro_Peak'] | df_sim['RSI_Cross_Up'], 30, buy_score)
    buy_score = np.where(df_sim['Retro_Peak'], 50, buy_score)
    buy_score = np.where((buy_score > 0) & df_sim['Radar_Activo'], buy_score + 25, buy_score)
    buy_score = np.where((buy_score > 0) & (df_sim['Z_Score'] < -2.0), buy_score + 15, buy_score)
    
    is_magenta = (buy_score >= 70) | df_sim['Retro_Peak']
    is_whale_icon = df_sim['Flash_Vol'] & df_sim['Vela_Verde'] & (~df_sim['Flash_Vol'].shift(1).fillna(False))
    df_sim['Pink_Whale_Buy'] = is_magenta & is_whale_icon
    
    df_sim['Neon_Up'] = df_sim['Squeeze_On'] & (df_sim['Close'] >= df_sim['BBU'] * 0.999) & df_sim['Vela_Verde']
    df_sim['Neon_Dn'] = df_sim['Squeeze_On'] & (df_sim['Close'] <= df_sim['BBL'] * 1.001) & df_sim['Vela_Roja']
    df_sim['Defcon_Buy'] = df_sim['Neon_Up'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    df_sim['Defcon_Sell'] = df_sim['Neon_Dn'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    
    scan_range = df_sim['ATR'] * 2.0
    ceil_w = np.zeros(len(df_sim))
    floor_w = np.zeros(len(df_sim))
    for p_col, w in [('PL30', 1), ('PH30', 1), ('PL100', 3), ('PH100', 3), ('PL300', 5), ('PH300', 5)]:
        p_val = df_sim[p_col].values
        c_val = df_sim['Close'].values
        ceil_w += np.where((p_val > c_val) & (p_val <= c_val + scan_range), w, 0)
        floor_w += np.where((p_val < c_val) & (p_val >= c_val - scan_range), w, 0)

    df_sim['Therm_Bounce'] = (floor_w >= 4) & df_sim['RSI_Cross_Up'] & ~(ceil_w >= 4)
    df_sim['Therm_Vacuum'] = (ceil_w <= 3) & df_sim['Neon_Up'] & ~(floor_w == 0)
    df_sim['Therm_Wall_Sell'] = (ceil_w >= 4) & df_sim['RSI_Cross_Dn']
    df_sim['Therm_Panic_Sell'] = (floor_w == 0) & df_sim['Vela_Roja']
    df_sim['Cielo_Libre'] = dist_res > (r_sens * 2) 

    df_sim['Nuclear_Buy'] = is_magenta & (df_sim['WT_Oversold'] | df_sim['WT_Cross_Up'])
    df_sim['Early_Buy'] = is_magenta
    df_sim['Nuclear_Sell'] = (df_sim['RSI'] > 70) & (df_sim['WT_Overbought'] | df_sim['WT_Cross_Dn'])
    df_sim['Early_Sell'] = (df_sim['RSI'] > 70) & df_sim['Vela_Roja']
    df_sim['Rebound_Buy'] = df_sim['RSI_Cross_Up'] & ~is_magenta
    
    return df_sim

# --- NÚCLEO FÍSICO C++ CÁLCULO DE INTERÉS COMPUESTO (TV CLONE) ---
def simular_crecimiento_exponencial(h_arr, l_arr, c_arr, o_arr, b_c, s_c, t_arr, sl_arr, cap_ini, com_pct):
    cap_act = cap_ini
    en_pos = False
    p_ent, tp_act, sl_act, pos_size, invest_amt = 0.0, 0.0, 0.0, 0.0, 0.0
    g_profit, g_loss, num_trades, max_dd, peak = 0.0, 0.0, 0, cap_ini, 0.0
    
    for i in range(len(h_arr)):
        if en_pos:
            tp_p = p_ent * (1 + tp_act/100)
            sl_p = p_ent * (1 - sl_act/100)
            
            if l_arr[i] <= sl_p:
                gross = pos_size * (1 - sl_act/100)
                net = gross * (1 - com_pct)
                profit = net - invest_amt
                cap_act += profit
                g_loss += abs(profit)
                num_trades += 1
                en_pos = False
            elif h_arr[i] >= tp_p:
                gross = pos_size * (1 + tp_act/100)
                net = gross * (1 - com_pct)
                profit = net - invest_amt
                cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1
                en_pos = False
            elif s_c[i]:
                ret = (c_arr[i] - p_ent) / p_ent
                gross = pos_size * (1 + ret)
                net = gross * (1 - com_pct)
                profit = net - invest_amt
                cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1
                en_pos = False
                
            if cap_act > peak: peak = cap_act
            if peak > 0:
                dd = (peak - cap_act) / peak * 100
                if dd > max_dd: max_dd = dd
            if cap_act <= 0: break
            
        if not en_pos and b_c[i] and i+1 < len(h_arr):
            invest_amt = cap_act
            pos_size = invest_amt * (1 - com_pct) 
            p_ent = o_arr[i+1]
            tp_act = t_arr[i]
            sl_act = sl_arr[i]
            en_pos = True
            
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    return cap_act - cap_ini, pf, num_trades, max_dd

# NÚCLEO VISUAL PARA DIBUJAR
def simular_visual(df_sim, cap_ini, reinvest, com_pct):
    registro_trades = []
    n = len(df_sim)
    curva = np.full(n, cap_ini, dtype=float)
    
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl_arr = df_sim['Active_TP'].values, df_sim['Active_SL'].values
    f_arr = df_sim.index
    
    en_pos, p_ent, tp_act, sl_act = False, 0.0, 0.0, 0.0
    cap_act = cap_ini
    divs = 0.0
    pos_size = 0.0
    invest_amt = 0.0
    
    for i in range(n):
        cierra = False
        if en_pos:
            tp_p = p_ent * (1 + tp_act/100)
            sl_p = p_ent * (1 - sl_act/100)
            
            if l_arr[i] <= sl_p:
                gross = pos_size * (1 - sl_act/100)
                net = gross * (1 - com_pct)
                profit = net - invest_amt
                
                if profit > 0:
                    reinv_amt = profit * (reinvest/100)
                    divs += (profit - reinv_amt)
                    cap_act += reinv_amt
                else:
                    cap_act += profit
                    
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'SL', 'Precio': sl_p, 'Ganancia_$': profit})
                en_pos, cierra = False, True
                
            elif h_arr[i] >= tp_p:
                gross = pos_size * (1 + tp_act/100)
                net = gross * (1 - com_pct)
                profit = net - invest_amt
                
                if profit > 0:
                    reinv_amt = profit * (reinvest/100)
                    divs += (profit - reinv_amt)
                    cap_act += reinv_amt
                else:
                    cap_act += profit
                    
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'TP', 'Precio': tp_p, 'Ganancia_$': profit})
                en_pos, cierra = False, True
                
            elif sell_arr[i]:
                ret = (c_arr[i] - p_ent) / p_ent
                gross = pos_size * (1 + ret)
                net = gross * (1 - com_pct)
                profit = net - invest_amt
                
                if profit > 0:
                    reinv_amt = profit * (reinvest/100)
                    divs += (profit - reinv_amt)
                    cap_act += reinv_amt
                else:
                    cap_act += profit
                    
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'DYN_WIN' if profit>0 else 'DYN_LOSS', 'Precio': c_arr[i], 'Ganancia_$': profit})
                en_pos, cierra = False, True

        if not en_pos and not cierra and buy_arr[i] and i+1 < n and cap_act > 0:
            invest_amt = cap_act if reinvest == 100 else cap_ini
            pos_size = invest_amt * (1 - com_pct)
            p_ent = o_arr[i+1]
            tp_act = tp_arr[i]
            sl_act = sl_arr[i]
            en_pos = True
            registro_trades.append({'Fecha': f_arr[i+1], 'Tipo': 'ENTRY', 'Precio': p_ent, 'Ganancia_$': 0})

        if en_pos and cap_act > 0:
            ret_flot = (c_arr[i] - p_ent) / p_ent
            curva[i] = cap_act + (pos_size * ret_flot) + divs
        else:
            curva[i] = cap_act + divs
            
    return curva.tolist(), divs, cap_act, registro_trades, en_pos

# --- 4. TERMINAL RENDER ---
st.title("🛡️ The Alpha Quant Terminal")
tab_tri, tab_jug, tab_def, tab_gen = st.tabs(["💠 TRINITY V357", "⚔️ JUGGERNAUT V356", "🚀 DEFCON V329", "🌌 GÉNESIS V320 (RISK-ADJUSTED)"])

def renderizar_estrategia(strat_name, tab_obj, df_base):
    with tab_obj:
        if df_base.empty:
            return

        s_id = strat_name.split()[0]
        
        if st.session_state.get(f'update_pending_{s_id}', False):
            bp = st.session_state[f'pending_bp_{s_id}']
            if s_id == "GENESIS":
                for r_idx in range(1, 5):
                    st.session_state[f'gen_r{r_idx}_b'] = bp[f'b{r_idx}']
                    st.session_state[f'gen_r{r_idx}_s'] = bp[f's{r_idx}']
                    st.session_state[f'gen_r{r_idx}_tp'] = float(round(bp[f'tp{r_idx}'], 1))
                    st.session_state[f'gen_r{r_idx}_sl'] = float(round(bp[f'sl{r_idx}'], 1))
            else:
                st.session_state[f'sld_tp_{s_id}'] = float(round(bp['tp'], 1))
                st.session_state[f'sld_sl_{s_id}'] = float(round(bp['sl'], 1))
                if s_id == "TRINITY": st.session_state[f'sld_reinv_{s_id}'] = float(bp['reinv'])
                if s_id != "DEFCON":
                    st.session_state[f'sld_wh_{s_id}'] = float(bp['wh'])
                    st.session_state[f'sld_rd_{s_id}'] = float(bp['rd'])
            st.session_state[f'update_pending_{s_id}'] = False

        # --- MÓDULO GÉNESIS ---
        if s_id == "GENESIS":
            st.markdown("### 🌌 Risk-Adjusted Matrix")
            st.info("La IA evaluará el Hold de la moneda. Si no puede ganarle en dólares brutos, le entregará la combinación que extraiga el mayor rendimiento con el menor riesgo (Drawdown) posible.")
            
            c_ia1, c_ia2 = st.columns([1, 3])
            st.session_state['gen_ado'] = c_ia1.slider("🎯 Target ADO (Trades/Día)", 0.0, 100.0, value=float(st.session_state.get('gen_ado', 5.0)), step=0.5, key="ui_gen_ado")

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown("<h5 style='color:lime;'>🟢 Bull Trend (Fuerte)</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r1_b")
                st.multiselect("Cierres", sell_rules, key="gen_r1_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r1_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r1_sl")

            with c2:
                st.markdown("<h5 style='color:yellow;'>🟡 Bull Chop (Rango)</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r2_b")
                st.multiselect("Cierres", sell_rules, key="gen_r2_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r2_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r2_sl")

            with c3:
                st.markdown("<h5 style='color:red;'>🔴 Bear Trend (Fuerte)</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r3_b")
                st.multiselect("Cierres", sell_rules, key="gen_r3_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r3_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r3_sl")

            with c4:
                st.markdown("<h5 style='color:orange;'>🟠 Bear Chop (Rango)</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r4_b")
                st.multiselect("Cierres", sell_rules, key="gen_r4_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r4_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r4_sl")

            st.markdown("---")
            if c_ia2.button("🚀 EXTRACCIÓN (Prioridad Riesgo/Beneficio)", type="primary"):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                
                df_p = inyectar_adn(df_base.copy(), 1.5, 2.5)
                h_a, l_a, c_a, o_a = df_p['High'].values, df_p['Low'].values, df_p['Close'].values, df_p['Open'].values
                regime_arr = df_p['Regime'].values
                
                b_mat = {r: df_p[r].values for r in buy_rules}
                s_mat = {r: df_p[r].values for r in sell_rules}
                
                buy_hold_ret = ((c_a[-1] - o_a[0]) / o_a[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                
                best_fit = -float('inf')
                bp = None
                
                for _ in range(3000): 
                    dna_b = [random.sample(buy_rules, random.randint(1, 5)) for _ in range(4)]
                    dna_s = [random.sample(sell_rules, random.randint(1, 5)) for _ in range(4)]
                    dna_tp = [random.uniform(2.0, 30.0) for _ in range(4)]
                    dna_sl = [random.uniform(1.0, 6.0) for _ in range(4)]
                    
                    f_buy, f_sell = np.zeros(len(df_p), dtype=bool), np.zeros(len(df_p), dtype=bool)
                    f_tp, f_sl = np.zeros(len(df_p)), np.zeros(len(df_p))
                    
                    for idx in range(4):
                        mask = (regime_arr == (idx + 1))
                        
                        r_b_cond = np.zeros(len(df_p), dtype=bool)
                        for r in dna_b[idx]: r_b_cond |= b_mat[r]
                        f_buy[mask] = r_b_cond[mask]
                        
                        r_s_cond = np.zeros(len(df_p), dtype=bool)
                        for r in dna_s[idx]: r_s_cond |= s_mat[r]
                        f_sell[mask] = r_s_cond[mask]
                        
                        f_tp[mask] = dna_tp[idx]
                        f_sl[mask] = dna_sl[idx]
                    
                    net, pf, nt, mdd = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, f_buy, f_sell, f_tp, f_sl, capital_inicial, comision_pct)
                    
                    alpha_money = net - buy_hold_money
                    actual_ado = nt / dias_analizados if dias_analizados > 0 else 0
                    target_ado = st.session_state.get('gen_ado', 0.0)
                    
                    ado_multiplier = 1.0
                    if target_ado > 0:
                        if actual_ado < target_ado: ado_multiplier = (actual_ado / target_ado) ** 2
                        
                    # 🔥 LA NUEVA CONDICIÓN DE SUPERVIVENCIA (Cero Cobardía) 🔥
                    if nt >= max(5, int(dias_analizados * (target_ado * 0.2))) and net > 0: 
                        
                        fit = ((net * (pf**2) * np.sqrt(nt)) / ((mdd ** 1.5) + 1.0)) * ado_multiplier
                        
                        if alpha_money > 0:
                            fit *= 2.0 
                            
                        if fit > best_fit:
                            best_fit = fit
                            bp = {
                                'b1': dna_b[0], 's1': dna_s[0], 'tp1': dna_tp[0], 'sl1': dna_sl[0],
                                'b2': dna_b[1], 's2': dna_s[1], 'tp2': dna_tp[1], 'sl2': dna_sl[1],
                                'b3': dna_b[2], 's3': dna_s[2], 'tp3': dna_tp[2], 'sl3': dna_sl[2],
                                'b4': dna_b[3], 's4': dna_s[3], 'tp4': dna_tp[3], 'sl4': dna_sl[3],
                                'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd
                            }
                
                ph_holograma.empty()
                if bp: 
                    st.session_state[f'update_pending_{s_id}'] = True
                    st.session_state[f'pending_bp_{s_id}'] = bp
                    
                    status_msg = f"🏆 SUPERÓ AL HOLD POR +${bp['alpha']:,.2f}" if bp['alpha'] > 0 else f"🛡️ RIESGO CONTROLADO. Hold = +${buy_hold_money:,.2f} | Bot = +${bp['net']:,.2f} (Max DD: {bp['mdd']:.2f}%)"
                    
                    dna_str = f"""🌌 THE RISK-ADJUSTED MATRIX
Net Profit: +${bp['net']:,.2f} | PF: {bp['pf']:.2f}x | Trades: {bp['nt']}
{status_msg}

// 🟢 QUAD 1: BULL TREND
Compras = {bp['b1']}
Cierres = {bp['s1']}
TP = {bp['tp1']:.1f}% | SL = {bp['sl1']:.1f}%

// 🟡 QUAD 2: BULL CHOP
Compras = {bp['b2']}
Cierres = {bp['s2']}
TP = {bp['tp2']:.1f}% | SL = {bp['sl2']:.1f}%

// 🔴 QUAD 3: BEAR TREND
Compras = {bp['b3']}
Cierres = {bp['s3']}
TP = {bp['tp3']:.1f}% | SL = {bp['sl3']:.1f}%

// 🟠 QUAD 4: BEAR CHOP
Compras = {bp['b4']}
Cierres = {bp['s4']}
TP = {bp['tp4']:.1f}% | SL = {bp['sl4']:.1f}%"""

                    st.session_state['winning_dna'] = dna_str
                    st.rerun() 
                else:
                    st.error(f"❌ La IA analizó 3000 universos. Ninguno logró producir ganancias netas. Es un entorno hostil.")

            if st.session_state.get('winning_dna') != "":
                st.success("¡ADN Extraído Exitosamente!")
                st.code(st.session_state['winning_dna'], language="text")

            df_strat = inyectar_adn(df_base.copy(), 1.5, 2.5)
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            f_tp, f_sl = np.zeros(len(df_strat)), np.zeros(len(df_strat))
            
            for idx in range(1, 5):
                mask = (df_strat['Regime'].values == idx)
                r_b_cond = np.zeros(len(df_strat), dtype=bool)
                for r in st.session_state.get(f'gen_r{idx}_b', []): r_b_cond |= df_strat[r].values
                f_buy[mask] = r_b_cond[mask]
                
                r_s_cond = np.zeros(len(df_strat), dtype=bool)
                for r in st.session_state.get(f'gen_r{idx}_s', []): r_s_cond |= df_strat[r].values
                f_sell[mask] = r_s_cond[mask]
                
                f_tp[mask] = st.session_state.get(f'gen_r{idx}_tp', 5.0)
                f_sl[mask] = st.session_state.get(f'gen_r{idx}_sl', 2.0)
                
            df_strat['Signal_Buy'] = f_buy
            df_strat['Signal_Sell'] = f_sell
            df_strat['Active_TP'] = f_tp
            df_strat['Active_SL'] = f_sl
            
            eq_curve, divs, cap_act, t_log, pos_ab = simular_visual(df_strat, capital_inicial, 100.0, comision_pct)

        # --- BLOQUES NORMALES (TRINITY/JUGG/DEFCON) ---
        else:
            with st.form(f"form_{s_id}"):
                c1, c2, c3, c4 = st.columns(4)
                st.session_state[f'sld_tp_{s_id}'] = c1.slider(f"🎯 TP Base (%)", 0.5, 15.0, value=float(st.session_state.get(f'sld_tp_{s_id}', 3.0)), step=0.1)
                st.session_state[f'sld_sl_{s_id}'] = c2.slider(f"🛑 SL (%)", 0.5, 10.0, value=float(st.session_state.get(f'sld_sl_{s_id}', 1.5)), step=0.1)
                
                mac_sh, atr_sh, d_buy, d_sell = True, True, True, True
                if s_id == "TRINITY":
                    st.session_state[f'sld_reinv_{s_id}'] = c3.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get(f'sld_reinv_{s_id}', 100.0)), step=5.0)
                    st.session_state[f'sld_wh_{s_id}'] = c4.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state.get(f'sld_wh_{s_id}', 2.5)), step=0.1)
                elif s_id == "JUGGERNAUT":
                    st.session_state[f'sld_reinv_{s_id}'] = c3.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get(f'sld_reinv_{s_id}', 100.0)), step=5.0)
                    st.session_state[f'sld_wh_{s_id}'] = c4.slider("🐋 Factor", 1.0, 5.0, value=float(st.session_state.get(f'sld_wh_{s_id}', 2.5)), step=0.1)
                else:
                    st.session_state[f'sld_reinv_{s_id}'] = c3.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get(f'sld_reinv_{s_id}', 100.0)), step=5.0)
                    d_buy = st.checkbox("Squeeze Up", value=True)
                    
                if st.form_submit_button("⚡ Aplicar"): st.rerun()

            c_ia1, c_ia2 = st.columns([1, 3])
            st.session_state[f'ado_{s_id}'] = c_ia1.slider(f"🎯 Target ADO ({s_id})", 0.0, 100.0, value=float(st.session_state.get(f'ado_{s_id}', 0.0)), step=0.5)
            
            if c_ia2.button(f"🌌 RECOCIDO CUÁNTICO ({s_id})", use_container_width=True):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                best_fit = -999999
                bp = {}
                
                df_precalc = inyectar_adn(df_base.copy(), 1.5, st.session_state.get(f'sld_wh_{s_id}', 2.5))
                if s_id == "TRINITY":
                    b_cond = df_precalc['Pink_Whale_Buy'] | df_precalc['Lock_Bounce'] | df_precalc['Defcon_Buy']
                    s_cond = df_precalc['Defcon_Sell'] | df_precalc['Therm_Wall_Sell']
                elif s_id == "JUGGERNAUT":
                    b_cond = df_precalc['Pink_Whale_Buy'] | ((df_precalc['Lock_Bounce'] | df_precalc['Defcon_Buy']) & df_precalc['Macro_Bull'])
                    s_cond = df_precalc['Defcon_Sell'] | df_precalc['Therm_Wall_Sell']
                else:
                    b_cond, s_cond = df_precalc['Defcon_Buy'], df_precalc['Defcon_Sell']

                h_arr, l_arr, c_arr, o_arr = df_precalc['High'].values, df_precalc['Low'].values, df_precalc['Close'].values, df_precalc['Open'].values
                b_c, s_c = b_cond.values, s_cond.values
                
                for _ in range(1000): 
                    rtp = round(random.uniform(1.2, 8.0), 1)
                    rsl = round(random.uniform(0.5, 3.5), 1)
                    rrv = round(random.uniform(20, 100), -1) if s_id == "TRINITY" else 100.0
                    rwh = round(random.uniform(1.5, 3.5), 1) if s_id != "DEFCON" else 2.5
                    rrd = round(random.uniform(0.5, 3.0), 1) if s_id != "DEFCON" else 1.5
                    
                    t_arr, sl_arr = np.full(len(df_precalc), rtp), np.full(len(df_precalc), rsl)
                    net, pf, nt, mdd = simular_crecimiento_exponencial(h_arr, l_arr, c_arr, o_arr, b_c, s_c, t_arr, sl_arr, capital_inicial, comision_pct)
                    
                    actual_ado = nt / dias_analizados if dias_analizados > 0 else 0
                    target_ado = st.session_state.get(f'ado_{s_id}', 0.0)
                    ado_multiplier = 1.0
                    if target_ado > 0 and actual_ado < target_ado: ado_multiplier = (actual_ado / target_ado) ** 2  

                    if nt > 0 and net > 0:
                        fit = ((net * pf * np.sqrt(nt)) / ((mdd**1.5) + 1.0)) * ado_multiplier
                        if fit > best_fit:
                            best_fit, bp = fit, {'tp':rtp, 'sl':rsl, 'reinv':rrv, 'wh':rwh, 'rd':rrd}
                
                ph_holograma.empty()
                if bp:
                    st.session_state[f'update_pending_{s_id}'] = True
                    st.session_state[f'pending_bp_{s_id}'] = bp
                    st.rerun()
                else: st.error("❌ El mercado carece de fractalidad operable.")
            
            df_strat = inyectar_adn(df_base.copy(), st.session_state.get(f'sld_rd_{s_id}', 1.5), st.session_state.get(f'sld_wh_{s_id}', 2.5))
            if s_id == "TRINITY":
                df_strat['Signal_Buy'] = df_strat['Pink_Whale_Buy'] | df_strat['Lock_Bounce'] | df_strat['Defcon_Buy']
                df_strat['Signal_Sell'] = df_strat['Defcon_Sell'] | df_strat['Therm_Wall_Sell']
                df_strat['Active_TP'], df_strat['Active_SL'] = st.session_state[f'sld_tp_{s_id}'], st.session_state[f'sld_sl_{s_id}']
            elif s_id == "JUGGERNAUT":
                macro_safe = df_strat['Macro_Bull'] if mac_sh else True
                atr_safe = ~(df_strat['Cuerpo_Vela'].shift(1).fillna(0) > (df_strat['ATR'].shift(1).fillna(0.001) * 1.5)) if atr_sh else True
                df_strat['Signal_Buy'] = df_strat['Pink_Whale_Buy'] | ((df_strat['Lock_Bounce'] | df_strat['Defcon_Buy']) & macro_safe & atr_safe)
                df_strat['Signal_Sell'] = df_strat['Defcon_Sell'] | df_strat['Therm_Wall_Sell']
                df_strat['Active_TP'], df_strat['Active_SL'] = st.session_state[f'sld_tp_{s_id}'], st.session_state[f'sld_sl_{s_id}']
            else:
                df_strat['Signal_Buy'] = df_strat['Defcon_Buy'] if d_buy else False
                df_strat['Signal_Sell'] = df_strat['Defcon_Sell'] if d_sell else False
                df_strat['Active_TP'], df_strat['Active_SL'] = st.session_state[f'sld_tp_{s_id}'], st.session_state[f'sld_sl_{s_id}']
                
            eq_curve, divs, cap_act, t_log, pos_ab = simular_visual(df_strat, capital_inicial, st.session_state.get(f'sld_reinv_{s_id}', 100.0), comision_pct)

        # --- SECCIÓN COMÚN (MÉTRICAS Y GRÁFICO TRANSPARENTE) ---
        df_strat['Total_Portfolio'] = eq_curve
        ret_pct = ((eq_curve[-1] - capital_inicial) / capital_inicial) * 100

        buy_hold_ret = ((df_strat['Close'].iloc[-1] - df_strat['Open'].iloc[0]) / df_strat['Open'].iloc[0]) * 100
        alpha_pct = ret_pct - buy_hold_ret

        dftr = pd.DataFrame(t_log)
        tt, wr, pf_val, ado_act = 0, 0.0, 0.0, 0.0
        if not dftr.empty:
            exs = dftr[dftr['Tipo'].isin(['TP', 'SL', 'DYN_WIN', 'DYN_LOSS'])]
            tt = len(exs)
            ado_act = tt / dias_analizados if dias_analizados > 0 else 0
            if tt > 0:
                ws = len(exs[exs['Tipo'].isin(['TP', 'DYN_WIN'])])
                wr = (ws / tt) * 100
                gpp = exs[exs['Ganancia_$'] > 0]['Ganancia_$'].sum()
                gll = abs(exs[exs['Ganancia_$'] < 0]['Ganancia_$'].sum())
                pf_val = gpp / gll if gll > 0 else float('inf')
        
        mdd = abs((((pd.Series(eq_curve) - pd.Series(eq_curve).cummax()) / pd.Series(eq_curve).cummax()) * 100).min())

        st.markdown(f"### 📊 Auditoría: {s_id}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Portafolio Neto", f"${eq_curve[-1]:,.2f}", f"{ret_pct:.2f}%")
        
        c2.metric("ALPHA (vs Hold)", f"{alpha_pct:.2f}%", f"Hold: {buy_hold_ret:.2f}%", delta_color="normal" if alpha_pct > 0 else "inverse")
        
        c3.metric("Win Rate", f"{wr:.1f}%")
        c4.metric("Profit Factor", f"{pf_val:.2f}x")
        c5.metric("Max Drawdown", f"{mdd:.2f}%", delta_color="inverse")
        c6.metric("ADO ⚡", f"{ado_act:.2f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_200'], mode='lines', name='EMA 200', line=dict(color='orange', width=2)), row=1, col=1)

        if not dftr.empty:
            ents = dftr[dftr['Tipo'] == 'ENTRY']
            fig.add_trace(go.Scatter(x=ents['Fecha'], y=ents['Precio'], mode='markers', name='COMPRA', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=2, color='white')), hovertemplate="COMPRA<br>Precio: $%{y:,.4f}<extra></extra>"), row=1, col=1)
            wins = dftr[dftr['Tipo'].isin(['TP', 'DYN_WIN'])]
            fig.add_trace(go.Scatter(x=wins['Fecha'], y=wins['Precio'], mode='markers', name='WIN', marker=dict(symbol='triangle-down', color='#00FF00', size=14, line=dict(width=2, color='white')), text=wins['Tipo'], hovertemplate="%{text}<br>Precio: $%{y:,.4f}<extra></extra>"), row=1, col=1)
            loss = dftr[dftr['Tipo'].isin(['SL', 'DYN_LOSS'])]
            fig.add_trace(go.Scatter(x=loss['Fecha'], y=loss['Precio'], mode='markers', name='LOSS', marker=dict(symbol='triangle-down', color='#FF0000', size=14, line=dict(width=2, color='white')), text=loss['Tipo'], hovertemplate="%{text}<br>Precio: $%{y:,.4f}<extra></extra>"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Total_Portfolio'], mode='lines', name='Equidad', line=dict(color='#00FF00', width=3)), row=2, col=1)

        fig.update_yaxes(side="right")
        fig.update_layout(template='plotly_dark', height=750, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), hovermode="closest", dragmode="pan")
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True}, key=f"chart_{s_id}")

renderizar_estrategia("TRINITY V357", tab_tri, df_global)
renderizar_estrategia("JUGGERNAUT V356", tab_jug, df_global)
renderizar_estrategia("DEFCON V329", tab_def, df_global)
renderizar_estrategia("GENESIS V320", tab_gen, df_global)
