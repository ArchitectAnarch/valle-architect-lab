import streamlit as st
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import pandas as pd
import numpy as np
import random
import time
import gc
from datetime import datetime, timedelta

# --- MOTOR DE HIPER-VELOCIDAD ---
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

st.set_page_config(page_title="ROCKET PROTOCOL | Alpha Quant", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🧠 CATÁLOGOS Y ARQUITECTURA DE ESTADO
# ==========================================
base_b = ['Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 'Commander_Buy', 'Lev_Buy']
base_s = ['Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 'Commander_Sell', 'Lev_Sell']

rocket_b = ['Trinity_Buy', 'Jugg_Buy', 'Defcon_Buy', 'Lock_Buy', 'Thermal_Buy', 'Climax_Buy', 'Ping_Buy', 'Squeeze_Buy', 'Lev_Buy', 'Commander_Buy']
rocket_s = ['Trinity_Sell', 'Jugg_Sell', 'Defcon_Sell', 'Lock_Sell', 'Thermal_Sell', 'Climax_Sell', 'Ping_Sell', 'Squeeze_Sell', 'Lev_Sell', 'Commander_Sell']

estrategias = ["ALL_FORCES", "TRINITY", "JUGGERNAUT", "DEFCON", "TARGET_LOCK", "THERMAL", "PINK_CLIMAX", "PING_PONG", "NEON_SQUEEZE", "COMMANDER", "GENESIS", "ROCKET"]
macro_opts = ["All-Weather", "Bull Only (Precio > EMA 200)", "Bear Only (Precio < EMA 200)"]
vol_opts = ["All-Weather", "Trend (ADX Alto)", "Range (ADX Bajo)"]

tab_id_map = {
    "🌟 ALL FORCES (MATRIX)": "ALL_FORCES", "💠 TRINITY": "TRINITY", "⚔️ JUGGERNAUT": "JUGGERNAUT", "🚀 DEFCON": "DEFCON",
    "🎯 TARGET_LOCK": "TARGET_LOCK", "🌡️ THERMAL": "THERMAL", "🌸 PINK_CLIMAX": "PINK_CLIMAX",
    "🏓 PING_PONG": "PING_PONG", "🐛 NEON_SQUEEZE": "NEON_SQUEEZE", "👑 COMMANDER": "COMMANDER",
    "🌌 GENESIS": "GENESIS", "👑 ROCKET": "ROCKET"
}

# ==========================================
# 🧬 THE DNA VAULT (Zero-Crash Protocol)
# ==========================================
# Bóveda sagrada que nunca se destruye en la recarga
for s_id in estrategias:
    if f'opt_status_{s_id}' not in st.session_state: 
        st.session_state[f'opt_status_{s_id}'] = False
        
    if f'champion_{s_id}' not in st.session_state:
        if s_id == "ALL_FORCES":
            st.session_state[f'champion_{s_id}'] = {'b_team': ['Commander_Buy', 'Squeeze_Buy', 'Ping_Buy'], 's_team': ['Commander_Sell', 'Squeeze_Sell'], 'macro': "All-Weather", 'vol': "All-Weather", 'tp': 50.0, 'sl': 5.0, 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')}
        elif s_id in ["GENESIS", "ROCKET"]:
            v = {'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')}
            for r_idx in range(1, 5): v.update({f'r{r_idx}_b': ['Squeeze_Buy'], f'r{r_idx}_s': ['Squeeze_Sell'], f'r{r_idx}_tp': 50.0, f'r{r_idx}_sl': 5.0})
            st.session_state[f'champion_{s_id}'] = v
        else:
            st.session_state[f'champion_{s_id}'] = {'tp': 50.0, 'sl': 5.0, 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')}

def save_champion(s_id, bp):
    vault = st.session_state[f'champion_{s_id}']
    vault['fit'] = bp['fit']
    for k in bp.keys():
        if k in vault: vault[k] = bp[k]

def wipe_ui_cache():
    for key in list(st.session_state.keys()):
        if key.startswith("ui_"): del st.session_state[key]

css_spinner = """
<style>
.loader-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.85); padding: 35px; border-radius: 20px; border: 2px solid cyan; box-shadow: 0 0 30px cyan;}
.rocket { font-size: 7rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 15px cyan); }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.prog-text { color: cyan; font-size: 1.6rem; font-weight: bold; margin-top: 15px; text-shadow: 0 0 5px cyan;}
.hud-text { color: lime; font-size: 1.2rem; margin-top: 8px; font-family: monospace; }
</style>
"""
ph_holograma = st.empty()

# ==========================================
# 🌍 SIDEBAR E INFRAESTRUCTURA
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 TRUTH ENGINE V107.0</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True): 
    st.cache_data.clear()
    for s in estrategias: 
        st.session_state[f'opt_status_{s}'] = False
        if f'champion_{s}' in st.session_state: del st.session_state[f'champion_{s}']
    wipe_ui_cache()
    gc.collect()
    st.rerun()

st.sidebar.markdown("---")
exchange_sel = st.sidebar.selectbox("🏦 Exchange", ["coinbase", "kucoin", "kraken", "binance"], index=0)
ticker = st.sidebar.text_input("Símbolo Exacto", value="HNT/USD")
utc_offset = st.sidebar.number_input("🌍 Zona Horaria", value=-5.0, step=0.5)

intervalos = {"1 Minuto": "1m", "5 Minutos": "5m", "7 Minutos": "7m", "13 Minutos": "13m", "15 Minutos": "15m", "23 Minutos": "23m", "30 Minutos": "30m", "45 Minutos": "45m", "1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d"}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=6) 
iv_download = intervalos[intervalo_sel]

hoy = datetime.today().date()
is_micro = iv_download in ["1m", "5m", "7m", "13m", "23m", "45m"]
limite_dias = 45 if is_micro else 1500
start_date, end_date = st.sidebar.slider("📅 Scope Histórico", min_value=hoy - timedelta(days=limite_dias), max_value=hoy, value=(hoy - timedelta(days=min(1500, limite_dias)), hoy), format="YYYY-MM-DD")

capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=1000.0, step=100.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center; color: lime;'>🤖 PILOTO AUTOMÁTICO</h3>", unsafe_allow_html=True)
global_epochs = st.sidebar.slider("Épocas de Evolución (x3000)", 1, 50, 10, help="50 Épocas = 150,000 modelos procesados a hiper-velocidad.")

@st.cache_data(ttl=3600, show_spinner="📡 Sintetizando Malla Tensorial (Quantum Pre-calc)...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, offset):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        base_tf, resample_rule = iv_down, None
        
        if iv_down == '7m': base_tf, resample_rule = '1m', '7T'
        elif iv_down == '13m': base_tf, resample_rule = '1m', '13T'
        elif iv_down == '23m': base_tf, resample_rule = '1m', '23T'
        elif iv_down == '45m': base_tf, resample_rule = '15m', '45T'
        elif iv_down == '4h' and exchange_id.lower() == 'coinbase': base_tf, resample_rule = '1h', '4H'

        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        all_ohlcv, current_ts = [], start_ts
        fetch_limit = 720 if exchange_id == 'kraken' else 300 if exchange_id == 'coinbase' else 1000
        
        while current_ts < end_ts:
            try: ohlcv = ex_class.fetch_ohlcv(sym, base_tf, since=current_ts, limit=fetch_limit)
            except Exception: time.sleep(1); continue
            if not ohlcv or len(ohlcv) == 0: break
            if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                ohlcv = [candle for candle in ohlcv if candle[0] > all_ohlcv[-1][0]]
                if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            if ohlcv[-1][0] <= current_ts: break 
            current_ts = ohlcv[-1][0] + 1
            if len(all_ohlcv) > 100000: break
            
        if not all_ohlcv: return pd.DataFrame(), f"El Exchange devolvió 0 velas."
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        if resample_rule: df = df.resample(resample_rule).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        df.index = df.index + timedelta(hours=offset)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < 50: return pd.DataFrame(), f"❌ Se sintetizaron solo {len(df)} velas de {iv_down}."
            
        df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1, adjust=False).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, min_periods=1, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1, adjust=False).mean()
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(50).sum() / df['Volume'].rolling(50).sum().replace(0, 1) 
        df['Vol_MA_100'] = df['Volume'].rolling(window=100, min_periods=1).mean()
        df['RVol'] = df['Volume'] / df['Vol_MA_100'].replace(0, 1)
        
        high_low = df['High'] - df['Low']
        tr = df[['High', 'Low']].max(axis=1) - df[['High', 'Low']].min(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, min_periods=1, adjust=False).mean().fillna(high_low).replace(0, 0.001)
        
        df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50.0)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0].fillna(0.0)
        
        df['Basis'] = df['Close'].rolling(20, min_periods=1).mean()
        dev = df['Close'].rolling(20, min_periods=1).std(ddof=0).replace(0, 1) 
        df['BBU'] = df['Basis'] + (2.0 * dev)
        df['BBL'] = df['Basis'] - (2.0 * dev)
        df['BB_Width'] = (df['BBU'] - df['BBL']) / df['Basis'].replace(0, 1)
        df['BB_Width_Avg'] = df['BB_Width'].rolling(20, min_periods=1).mean()
        
        kc_basis = df['Close'].rolling(20, min_periods=1).mean()
        df['KC_Upper'] = kc_basis + (df['ATR'] * 1.5)
        df['KC_Lower'] = kc_basis - (df['ATR'] * 1.5)
        df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
        df['BB_Delta'] = (df['BBU'] - df['BBL']).diff().fillna(0)
        
        df['Vela_Verde'] = df['Close'] > df['Open']
        df['Vela_Roja'] = df['Close'] < df['Open']
        df['body_size'] = abs(df['Close'] - df['Open']).replace(0, 0.0001)
        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['is_falling_knife'] = (df['Open'].shift(1) - df['Close'].shift(1)) > (df['ATR'].shift(1) * 1.5)
        
        df['PL30'] = df['Low'].shift(1).rolling(30, min_periods=1).min()
        df['PH30'] = df['High'].shift(1).rolling(30, min_periods=1).max()
        df['PL100'] = df['Low'].shift(1).rolling(100, min_periods=1).min()
        df['PH100'] = df['High'].shift(1).rolling(100, min_periods=1).max()
        df['PL300'] = df['Low'].shift(1).rolling(300, min_periods=1).min()
        df['PH300'] = df['High'].shift(1).rolling(300, min_periods=1).max()
        df['Target_Lock_Sup'] = df[['PL30', 'PL100', 'PL300']].max(axis=1)
        df['Target_Lock_Res'] = df[['PH30', 'PH100', 'PH300']].min(axis=1)
        
        df['RSI_Cross_Up'] = (df['RSI'] > df['RSI'].shift(1).fillna(50))
        df['RSI_Cross_Dn'] = (df['RSI'] < df['RSI'].shift(1).fillna(50))
        df['Macro_Bull'] = df['Close'] >= df['EMA_200']
        
        c_val = df['Close'].values
        df['dist_sup'] = (c_val - df['Target_Lock_Sup'].values) / c_val * 100
        df['dist_res'] = (df['Target_Lock_Res'].values - c_val) / c_val * 100
        
        # 🔥 PRE-CÁLCULO DEL MURO TÉRMICO (Evita 150k ciclos redundantes)
        sr_val = df['ATR'].values * 2.0
        ceil_w, floor_w = np.zeros(len(df)), np.zeros(len(df))
        for p_col, w in [('PL30', 1), ('PH30', 1), ('PL100', 3), ('PH100', 3), ('PL300', 5), ('PH300', 5)]:
            p_val = df[p_col].values
            ceil_w += np.where((p_val > c_val) & (p_val <= c_val + sr_val), w, 0)
            floor_w += np.where((p_val < c_val) & (p_val >= c_val - sr_val), w, 0)
        df['ceil_w'] = ceil_w
        df['floor_w'] = floor_w
        
        gc.collect()
        return df, "OK"
    except Exception as e: 
        return pd.DataFrame(), str(e)

df_global, status_api = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, utc_offset)

if not df_global.empty:
    dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)
    st.sidebar.success(f"📥 MATRIZ LISTA: {len(df_global)} velas ({dias_reales} días).")
else:
    dias_reales = 1
    st.error(status_api)
    st.stop()

# ==========================================
# 🔥 EVALUACIÓN PURE NUMPY (V107 - 1000x Speed) 🔥
# ==========================================
@njit(fastmath=True)
def simular_crecimiento_exponencial(h_arr, l_arr, c_arr, o_arr, b_c, s_c, t_arr, sl_arr, cap_ini, com_pct, reinvest_pct):
    cap_act = cap_ini
    divs, en_pos = 0.0, False
    p_ent, tp_act, sl_act, pos_size, invest_amt = 0.0, 0.0, 0.0, 0.0, 0.0
    g_profit, g_loss, num_trades, max_dd, peak = 0.0, 0.0, 0, 0.0, cap_ini
    
    for i in range(len(h_arr)):
        if en_pos:
            tp_p = p_ent * (1.0 + tp_act/100.0)
            sl_p = p_ent * (1.0 - sl_act/100.0)
            
            if l_arr[i] <= sl_p:
                gross = pos_size * (1.0 - sl_act/100.0)
                net = gross - (gross * com_pct)
                profit = net - invest_amt
                if profit > 0:
                    reinv = profit * (reinvest_pct / 100.0)
                    divs += (profit - reinv)
                    cap_act += reinv
                else: cap_act += profit
                g_loss += abs(profit)
                num_trades += 1
                en_pos = False
            elif h_arr[i] >= tp_p:
                gross = pos_size * (1.0 + tp_act/100.0)
                net = gross - (gross * com_pct)
                profit = net - invest_amt
                if profit > 0:
                    reinv = profit * (reinvest_pct / 100.0)
                    divs += (profit - reinv)
                    cap_act += reinv
                else: cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1
                en_pos = False
            elif s_c[i]:
                ret = (c_arr[i] - p_ent) / p_ent
                gross = pos_size * (1.0 + ret)
                net = gross - (gross * com_pct)
                profit = net - invest_amt
                if profit > 0:
                    reinv = profit * (reinvest_pct / 100.0)
                    divs += (profit - reinv)
                    cap_act += reinv
                else: cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1
                en_pos = False
                
            total_equity = cap_act + divs
            if total_equity > peak: peak = total_equity
            if peak > 0:
                dd = (peak - total_equity) / peak * 100.0
                if dd > max_dd: max_dd = dd
            if cap_act <= 0: break
            
        if not en_pos and b_c[i] and i+1 < len(h_arr):
            invest_amt = cap_act if reinvest_pct == 100.0 else cap_ini
            if invest_amt > cap_act: invest_amt = cap_act 
            comm_in = invest_amt * com_pct
            pos_size = invest_amt - comm_in 
            p_ent = o_arr[i+1]
            tp_act = t_arr[i]
            sl_act = sl_arr[i]
            en_pos = True
            
    total_net = (cap_act + divs) - cap_ini
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    return total_net, pf, num_trades, max_dd

def simular_visual(df_sim, cap_ini, reinvest, com_pct):
    registro_trades = []
    n = len(df_sim)
    curva = np.full(n, cap_ini, dtype=float)
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl_arr = df_sim['Active_TP'].values, df_sim['Active_SL'].values
    f_arr = df_sim.index
    
    en_pos, p_ent, tp_act, sl_act = False, 0.0, 0.0, 0.0
    cap_act, divs, pos_size, invest_amt, total_comms = cap_ini, 0.0, 0.0, 0.0, 0.0
    
    for i in range(n):
        cierra = False
        if en_pos:
            tp_p = p_ent * (1 + tp_act/100)
            sl_p = p_ent * (1 - sl_act/100)
            if l_arr[i] <= sl_p:
                gross = pos_size * (1 - sl_act/100)
                comm_out = gross * com_pct
                total_comms += comm_out
                net = gross - comm_out
                profit = net - invest_amt
                if profit > 0:
                    reinv_amt = profit * (reinvest/100)
                    divs += (profit - reinv_amt)
                    cap_act += reinv_amt
                else: cap_act += profit
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'SL', 'Precio': sl_p, 'Ganancia_$': profit})
                en_pos, cierra = False, True
            elif h_arr[i] >= tp_p:
                gross = pos_size * (1 + tp_act/100)
                comm_out = gross * com_pct
                total_comms += comm_out
                net = gross - comm_out
                profit = net - invest_amt
                if profit > 0:
                    reinv_amt = profit * (reinvest/100)
                    divs += (profit - reinv_amt)
                    cap_act += reinv_amt
                else: cap_act += profit
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'TP', 'Precio': tp_p, 'Ganancia_$': profit})
                en_pos, cierra = False, True
            elif sell_arr[i]:
                ret = (c_arr[i] - p_ent) / p_ent
                gross = pos_size * (1 + ret)
                comm_out = gross * com_pct
                total_comms += comm_out
                net = gross - comm_out
                profit = net - invest_amt
                if profit > 0:
                    reinv_amt = profit * (reinvest/100)
                    divs += (profit - reinv_amt)
                    cap_act += reinv_amt
                else: cap_act += profit
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'DYN_WIN' if profit>0 else 'DYN_LOSS', 'Precio': c_arr[i], 'Ganancia_$': profit})
                en_pos, cierra = False, True

        if not en_pos and not cierra and buy_arr[i] and i+1 < n and cap_act > 0:
            invest_amt = cap_act if reinvest == 100 else cap_ini
            if invest_amt > cap_act: invest_amt = cap_act
            comm_in = invest_amt * com_pct
            total_comms += comm_in
            pos_size = invest_amt - comm_in
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
            
    return curva.tolist(), divs, cap_act, registro_trades, en_pos, total_comms

def optimizar_ia_tracker(s_id, df_base, cap_ini, com_pct, reinv_q, buy_hold_money, epochs=1):
    best_fit = st.session_state[f'champion_{s_id}'].get('fit', -float('inf'))
    best_net_live = 0.0
    best_pf_live = 0.0
    bp = None
    tp_min, tp_max = 5.0, 150.0 
    
    iters = 3000 * epochs
    chunks = min(iters, 20) 
    chunk_size = iters // chunks
    start_time = time.time()
    
    # 🏎️ Variables Extraídas a la Memoria RAM (Evita Pandas overhead en el loop)
    a_c = df_base['Close'].values
    a_o = df_base['Open'].values
    a_h = df_base['High'].values
    a_l = df_base['Low'].values
    a_rsi = df_base['RSI'].values
    a_adx = df_base['ADX'].values
    a_bbl = df_base['BBL'].values
    a_bbu = df_base['BBU'].values
    a_ema50 = df_base['EMA_50'].values
    a_ema200 = df_base['EMA_200'].values
    a_rvol = df_base['RVol'].values
    a_vv = df_base['Vela_Verde'].values
    a_vr = df_base['Vela_Roja'].values
    a_rcu = df_base['RSI_Cross_Up'].values
    a_rcd = df_base['RSI_Cross_Dn'].values
    a_bw = df_base['BB_Width'].values
    a_bwa_s1 = df_base['BB_Width_Avg'].shift(1).fillna(-1.0).values
    a_sqz_s1 = df_base['Squeeze_On'].shift(1).fillna(False).values
    a_lw = df_base['lower_wick'].values
    a_bs = df_base['body_size'].values
    a_dsup = df_base['dist_sup'].values
    a_dres = df_base['dist_res'].values
    a_tres = df_base['Target_Lock_Res'].values
    a_cw = df_base['ceil_w'].values
    a_fw = df_base['floor_w'].values
    a_mb = df_base['Macro_Bull'].values
    a_fk = df_base['is_falling_knife'].values
    a_c_s1 = df_base['Close'].shift(1).fillna(0.0).values
    n_len = len(a_c)

    for c in range(chunks):
        for _ in range(chunk_size): 
            rtp = round(random.uniform(tp_min, tp_max), 1)
            rsl = round(random.uniform(1.0, 20.0), 1)
            r_hitbox = round(random.uniform(0.5, 3.0), 1)   
            r_therm  = float(random.randint(3, 8))          
            r_adx    = float(random.randint(15, 35))        
            r_whale  = round(random.uniform(1.5, 4.0), 1)   
            
            s_dict = {}
            s_dict['Ping_Buy'] = (a_adx < r_adx) & (a_c < a_bbl) & a_vv
            s_dict['Ping_Sell'] = (a_c > a_bbu) | (a_rsi > 70)
            s_dict['Squeeze_Buy'] = (a_bw < a_bwa_s1) & (a_c > a_bbu) & a_vv & (a_rsi < 60)
            s_dict['Squeeze_Sell'] = (a_c < a_ema50)
            t_buy = (a_fw >= r_therm) & a_vv & a_rcu
            t_sell = (a_cw >= r_therm) & a_vr & a_rcd
            s_dict['Thermal_Buy'] = t_buy
            s_dict['Thermal_Sell'] = t_sell
            c_buy = (a_rvol > r_whale) & (a_lw > (a_bs * 2.0)) & (a_rsi < 35) & a_vv
            s_dict['Climax_Buy'] = c_buy
            s_dict['Climax_Sell'] = (a_rsi > 80)
            l_buy = (a_dsup < r_hitbox) & a_vv & a_rcu
            s_dict['Lock_Buy'] = l_buy
            s_dict['Lock_Sell'] = (a_dres < r_hitbox) | (a_h >= a_tres)
            s_dict['Defcon_Buy'] = a_sqz_s1 & (a_c > a_bbu) & (a_adx > r_adx)
            s_dict['Defcon_Sell'] = (a_c < a_ema50)
            s_dict['Jugg_Buy'] = a_mb & (a_c > a_ema50) & (a_c_s1 < a_ema50) & a_vv & ~a_fk
            s_dict['Jugg_Sell'] = (a_c < a_ema50)
            s_dict['Trinity_Buy'] = a_mb & (a_rsi < 35) & a_vv & ~a_fk
            s_dict['Trinity_Sell'] = (a_rsi > 75) | (a_c < a_ema200)
            s_dict['Lev_Buy'] = a_mb & a_rcu & (a_rsi < 45)
            s_dict['Lev_Sell'] = (a_c < a_ema200)
            s_dict['Commander_Buy'] = c_buy | t_buy | l_buy
            s_dict['Commander_Sell'] = t_sell | (a_c < a_ema50)
            
            f_buy, f_sell = np.zeros(n_len, dtype=bool), np.zeros(n_len, dtype=bool)
            f_tp, f_sl = np.zeros(n_len, dtype=np.float64), np.zeros(n_len, dtype=np.float64)

            if s_id == "ALL_FORCES":
                dna_b_team = random.sample(base_b, random.randint(1, len(base_b)))
                dna_s_team = random.sample(base_s, random.randint(1, len(base_s)))
                dna_macro = "All-Weather" if random.random() < 0.6 else random.choice(["Bull Only (Precio > EMA 200)", "Bear Only (Precio < EMA 200)"])
                dna_vol = "All-Weather" if random.random() < 0.6 else random.choice(["Trend (ADX Alto)", "Range (ADX Bajo)"])
                
                macro_mask = np.ones(n_len, dtype=bool)
                if dna_macro == "Bull Only (Precio > EMA 200)": macro_mask = a_mb
                elif dna_macro == "Bear Only (Precio < EMA 200)": macro_mask = ~a_mb
                
                vol_mask = np.ones(n_len, dtype=bool)
                if dna_vol == "Trend (ADX Alto)": vol_mask = (a_adx >= r_adx)
                elif dna_vol == "Range (ADX Bajo)": vol_mask = (a_adx < r_adx)
                
                global_mask = macro_mask & vol_mask
                
                for r in dna_b_team: f_buy |= s_dict[r]
                f_buy &= global_mask
                for r in dna_s_team: f_sell |= s_dict[r]
                
                f_tp = np.full(n_len, float(rtp))
                f_sl = np.full(n_len, float(rsl))
                
            elif s_id in ["GENESIS", "ROCKET"]:
                regime_arr = np.where(a_mb & (a_adx >= r_adx), 1, np.where(a_mb & (a_adx < r_adx), 2, np.where(~a_mb & (a_adx >= r_adx), 3, 4)))
                b_opts = base_b if s_id != "ROCKET" else rocket_b
                s_opts = base_s if s_id != "ROCKET" else rocket_s
                
                dna_b = [random.sample(b_opts, random.randint(1, 2)) for _ in range(4)]
                dna_s = [random.sample(s_opts, random.randint(1, 2)) for _ in range(4)]
                dna_tp = [random.uniform(tp_min, tp_max) for _ in range(4)]
                dna_sl = [random.uniform(2.0, 20.0) for _ in range(4)]
                
                for idx in range(4):
                    mask = (regime_arr == (idx + 1))
                    if not mask.any(): continue
                    r_b_cond = np.zeros(n_len, dtype=bool)
                    for r in dna_b[idx]: r_b_cond |= s_dict[r]
                    f_buy[mask] = r_b_cond[mask]
                    r_s_cond = np.zeros(n_len, dtype=bool)
                    for r in dna_s[idx]: r_s_cond |= s_dict[r]
                    f_sell[mask] = r_s_cond[mask]
                    f_tp[mask] = dna_tp[idx]
                    f_sl[mask] = dna_sl[idx]
            else:
                if s_id == "TRINITY": f_buy, f_sell = s_dict['Trinity_Buy'], s_dict['Trinity_Sell']
                elif s_id == "JUGGERNAUT": f_buy, f_sell = s_dict['Jugg_Buy'], s_dict['Jugg_Sell']
                elif s_id == "DEFCON": f_buy, f_sell = s_dict['Defcon_Buy'], s_dict['Defcon_Sell']
                elif s_id == "TARGET_LOCK": f_buy, f_sell = s_dict['Lock_Buy'], s_dict['Lock_Sell']
                elif s_id == "THERMAL": f_buy, f_sell = s_dict['Thermal_Buy'], s_dict['Thermal_Sell']
                elif s_id == "PINK_CLIMAX": f_buy, f_sell = s_dict['Climax_Buy'], s_dict['Climax_Sell']
                elif s_id == "PING_PONG": f_buy, f_sell = s_dict['Ping_Buy'], s_dict['Ping_Sell']
                elif s_id == "NEON_SQUEEZE": f_buy, f_sell = s_dict['Squeeze_Buy'], s_dict['Squeeze_Sell']
                elif s_id == "COMMANDER": f_buy, f_sell = s_dict['Commander_Buy'], s_dict['Commander_Sell']
                f_tp = np.full(n_len, float(rtp))
                f_sl = np.full(n_len, float(rsl))

            b_c_arr, s_c_arr = np.asarray(f_buy, dtype=bool), np.asarray(f_sell, dtype=bool)
            t_arr, sl_arr = np.asarray(f_tp, dtype=np.float64), np.asarray(f_sl, dtype=np.float64)

            net, pf, nt, mdd = simular_crecimiento_exponencial(a_h, a_l, a_c, a_o, b_c_arr, s_c_arr, t_arr, sl_arr, float(cap_ini), float(com_pct), float(reinv_q))
            alpha_money = net - buy_hold_money
            
            # 🔥 V107.0 FITNESS: Respeta Francotiradores y Penaliza Ceros 🔥
            if nt >= 1: 
                if net > 0: 
                    trade_penalty = 1.0 if nt >= 5 else (float(nt) / 5.0) 
                    fit = (net ** 1.2) * (pf ** 0.4) * np.log1p(nt) * trade_penalty / ((mdd ** 0.6) + 1.0)
                    if alpha_money > 0: fit *= 1.5 
                else: 
                    fit = net * ((mdd ** 0.5) + 1.0) / (pf + 0.001)
            else:
                fit = -999999.0 
                
            if fit > best_fit:
                best_fit = fit
                if net > best_net_live: best_net_live, best_pf_live = net, pf
                    
                if s_id == "ALL_FORCES":
                    bp = {'b_team': dna_b_team, 's_team': dna_s_team, 'macro': dna_macro, 'vol': dna_vol, 'tp': rtp, 'sl': rsl, 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'fit': fit}
                elif s_id in ["GENESIS", "ROCKET"]:
                    bp = {'r1_b': dna_b[0], 'r1_s': dna_s[0], 'r1_tp': dna_tp[0], 'r1_sl': dna_sl[0], 'r2_b': dna_b[1], 'r2_s': dna_s[1], 'r2_tp': dna_tp[1], 'r2_sl': dna_sl[1], 'r3_b': dna_b[2], 'r3_s': dna_s[2], 'r3_tp': dna_tp[2], 'r3_sl': dna_sl[2], 'r4_b': dna_b[3], 'r4_s': dna_s[3], 'r4_tp': dna_tp[3], 'r4_sl': dna_sl[3], 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'fit': fit}
                else:
                    bp = {'tp': rtp, 'sl': rsl, 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'fit': fit}
        
        elapsed = time.time() - start_time
        pct_done = int(((c + 1) / chunks) * 100)
        combos = (c + 1) * chunk_size
        eta = (elapsed / (c + 1)) * (chunks - c - 1)
        
        dyn_spinner = f"""
        <style>
        .loader-container {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.9); padding: 35px; border-radius: 20px; border: 2px solid cyan; box-shadow: 0 0 40px cyan;}}
        .rocket {{ font-size: 8rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 20px cyan); }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .prog-text {{ color: cyan; font-size: 1.8rem; font-weight: bold; margin-top: 15px; text-shadow: 0 0 5px cyan;}}
        .hud-text {{ color: lime; font-size: 1.3rem; margin-top: 8px; font-family: monospace; }}
        </style>
        <div class="loader-container">
            <div class="rocket">🚀</div>
            <div class="prog-text">DEEP MINE: {s_id}</div>
            <div class="hud-text" style="color: white;">Progreso: {pct_done}%</div>
            <div class="hud-text" style="color: white;">Combos (V107 Numpy): {combos:,}</div>
            <div class="hud-text" style="color: #00FF00; font-weight: bold; font-size: 1.5rem; margin-top: 15px;">🏆 Hallazgo Actual: ${best_net_live:.2f} | {best_pf_live:.1f}x</div>
            <div class="hud-text" style="color: yellow; margin-top: 15px;">ETA: {eta:.1f} segs</div>
        </div>
        """
        ph_holograma.markdown(dyn_spinner, unsafe_allow_html=True)
        
    return bp

def inyectar_adn_df(df_sim, hitbox, therm_w, adx_th, whale_f):
    """Calcula las señales en el dataframe para renderizado y métricas visuales"""
    sr_val = df_sim['ATR'].values * 2.0
    c_val = df_sim['Close'].values
    ceil_w, floor_w = np.zeros(len(df_sim)), np.zeros(len(df_sim))
    for p_col, w in [('PL30', 1), ('PH30', 1), ('PL100', 3), ('PH100', 3), ('PL300', 5), ('PH300', 5)]:
        p_val = df_sim[p_col].values
        ceil_w += np.where((p_val > c_val) & (p_val <= c_val + sr_val), w, 0)
        floor_w += np.where((p_val < c_val) & (p_val >= c_val - sr_val), w, 0)
        
    df_sim['floor_w'] = floor_w
    df_sim['ceil_w'] = ceil_w

    df_sim['Regime'] = np.where(df_sim['Macro_Bull'] & (df_sim['ADX'] >= adx_th), 1, np.where(df_sim['Macro_Bull'] & (df_sim['ADX'] < adx_th), 2, np.where(~df_sim['Macro_Bull'] & (df_sim['ADX'] >= adx_th), 3, 4)))
    df_sim['Ping_Buy'] = (df_sim['ADX'] < adx_th) & (df_sim['Close'] < df_sim['BBL']) & df_sim['Vela_Verde']
    df_sim['Ping_Sell'] = (df_sim['Close'] > df_sim['BBU']) | (df_sim['RSI'] > 70)
    df_sim['Neon_Up'] = (df_sim['BB_Width'] < df_sim['BB_Width_Avg'].shift(1).fillna(-1.0)) & (df_sim['Close'] > df_sim['BBU']) & df_sim['Vela_Verde'] & (df_sim['RSI'] < 60)
    df_sim['Squeeze_Buy'] = df_sim['Neon_Up']
    df_sim['Squeeze_Sell'] = (df_sim['Close'] < df_sim['EMA_50'])
    df_sim['Thermal_Buy'] = (df_sim['floor_w'] >= therm_w) & df_sim['Vela_Verde'] & df_sim['RSI_Cross_Up']
    df_sim['Thermal_Sell'] = (df_sim['ceil_w'] >= therm_w) & df_sim['Vela_Roja'] & df_sim['RSI_Cross_Dn']
    df_sim['Climax_Buy'] = (df_sim['RVol'] > whale_f) & (df_sim['lower_wick'] > (df_sim['body_size'] * 2.0)) & (df_sim['RSI'] < 35) & df_sim['Vela_Verde']
    df_sim['Climax_Sell'] = (df_sim['RSI'] > 80)
    df_sim['Lock_Buy'] = (df_sim['dist_sup'] < hitbox) & df_sim['Vela_Verde'] & df_sim['RSI_Cross_Up']
    df_sim['Lock_Sell'] = (df_sim['dist_res'] < hitbox) | (df_sim['High'] >= df_sim['Target_Lock_Res'])
    df_sim['Defcon_Buy'] = df_sim['Squeeze_On'].shift(1).fillna(False) & (df_sim['Close'] > df_sim['BBU']) & (df_sim['ADX'] > adx_th)
    df_sim['Defcon_Sell'] = (df_sim['Close'] < df_sim['EMA_50'])
    df_sim['Jugg_Buy'] = df_sim['Macro_Bull'] & (df_sim['Close'] > df_sim['EMA_50']) & (df_sim['Close'].shift(1) < df_sim['EMA_50']) & df_sim['Vela_Verde'] & ~df_sim['is_falling_knife']
    df_sim['Jugg_Sell'] = (df_sim['Close'] < df_sim['EMA_50'])
    df_sim['Trinity_Buy'] = df_sim['Macro_Bull'] & (df_sim['RSI'] < 35) & df_sim['Vela_Verde'] & ~df_sim['is_falling_knife']
    df_sim['Trinity_Sell'] = (df_sim['RSI'] > 75) | (df_sim['Close'] < df_sim['EMA_200'])
    df_sim['Lev_Buy'] = df_sim['Macro_Bull'] & df_sim['RSI_Cross_Up'] & (df_sim['RSI'] < 45)
    df_sim['Lev_Sell'] = (df_sim['Close'] < df_sim['EMA_200'])
    df_sim['Commander_Buy'] = df_sim['Climax_Buy'] | df_sim['Thermal_Buy'] | df_sim['Lock_Buy']
    df_sim['Commander_Sell'] = df_sim['Thermal_Sell'] | (df_sim['Close'] < df_sim['EMA_50'])
    return df_sim

st.title("🛡️ The Omni-Brain Lab")
tabs = st.tabs(["🌟 ALL FORCES (MATRIX)", "💠 TRINITY", "⚔️ JUGGERNAUT", "🚀 DEFCON", "🎯 TARGET_LOCK", "🌡️ THERMAL", "🌸 PINK_CLIMAX", "🏓 PING_PONG", "🐛 NEON_SQUEEZE", "👑 COMMANDER", "🌌 GENESIS", "👑 ROCKET"])

for idx, tab_name in enumerate(tab_id_map.keys()):
    with tabs[idx]:
        if df_global.empty: continue
        s_id = tab_id_map[tab_name]
        is_opt = st.session_state.get(f'opt_status_{s_id}', False)
        opt_badge = "<span style='color: lime; border: 1px solid lime; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem;'>✅ IA OPTIMIZADA</span>" if is_opt else "<span style='color: gray; border: 1px solid gray; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem;'>➖ NO OPTIMIZADA</span>"
        vault = st.session_state[f'champion_{s_id}']

        if s_id == "ALL_FORCES":
            st.markdown(f"### 🌟 ALL FORCES ALGO (Global Matrix) {opt_badge}", unsafe_allow_html=True)
            st.info("El Director Supremo. Libertad total de combinación. Ejecución 1000x Pure Numpy con Auto-Heal inquebrantable.")
            
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            c_ado = st.session_state.get(f'ui_{s_id}_ado', vault['ado'])
            c_reinv = st.session_state.get(f'ui_{s_id}_reinv', vault['reinv'])
            n_ado = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(c_ado), key=f"w_ado_{s_id}", step=0.5)
            n_reinv = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(c_reinv), key=f"w_reinv_{s_id}", step=5.0)

            with st.expander("⚙️ Calibración del ADN Base"):
                c_adv1, c_adv2 = st.columns(2)
                c_hit = st.session_state.get(f'ui_{s_id}_hitbox', vault['hitbox'])
                c_thm = st.session_state.get(f'ui_{s_id}_therm_w', vault['therm_w'])
                n_hit = c_adv1.slider("🎯 Target Lock Hitbox (%)", 0.5, 3.0, value=float(c_hit), key=f"w_hit_{s_id}", step=0.1)
                n_thm = c_adv2.slider("🌡️ Thermal Wall Weight", 3.0, 8.0, value=float(c_thm), key=f"w_tw_{s_id}", step=1.0)
                
                c_f1, c_f2 = st.columns(2)
                c_adx = st.session_state.get(f'ui_{s_id}_adx_th', vault['adx_th'])
                c_wha = st.session_state.get(f'ui_{s_id}_whale_f', vault['whale_f'])
                n_adx = c_f1.slider("🚀 Defcon/Ping ADX Threshold", 15.0, 35.0, value=float(c_adx), key=f"w_adx_{s_id}", step=1.0)
                n_wha = c_f2.slider("🐋 Climax Whale Factor", 1.5, 4.0, value=float(c_wha), key=f"w_wh_{s_id}", step=0.1)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                c_mac = st.session_state.get(f'ui_{s_id}_macro', vault['macro'])
                c_btm = [x for x in st.session_state.get(f'ui_{s_id}_b_team', vault['b_team']) if x in base_b]
                c_tp = st.session_state.get(f'ui_{s_id}_tp', vault['tp'])
                n_mac = st.selectbox("Filtro Macro (Tendencia Larga)", macro_opts, index=macro_opts.index(c_mac) if c_mac in macro_opts else 0, key=f"w_mac_{s_id}")
                n_btm = st.multiselect("Fuerzas de Ataque (Compras)", base_b, default=c_btm, key=f"w_btm_{s_id}")
                n_tp = st.slider("TP del Escuadrón %", 0.5, 150.0, value=float(c_tp), key=f"w_tp_{s_id}", step=0.5)
            with c2:
                c_vol = st.session_state.get(f'ui_{s_id}_vol', vault['vol'])
                c_stm = [x for x in st.session_state.get(f'ui_{s_id}_s_team', vault['s_team']) if x in base_s]
                c_sl = st.session_state.get(f'ui_{s_id}_sl', vault['sl'])
                n_vol = st.selectbox("Filtro Volatilidad (Fuerza ADX)", vol_opts, index=vol_opts.index(c_vol) if c_vol in vol_opts else 0, key=f"w_vol_{s_id}")
                n_stm = st.multiselect("Fuerzas de Retirada (Ventas)", base_s, default=c_stm, key=f"w_stm_{s_id}")
                n_sl = st.slider("SL del Escuadrón %", 0.5, 25.0, value=float(c_sl), key=f"w_sl_{s_id}", step=0.5)

            if c_ia3.button(f"🚀 DEEP MINE INDIVIDUAL ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                
                # Optimiza leyendo desde el estado de la UI (Para que la IA sepa qué superar)
                bp = optimizar_ia_tracker(s_id, df_global, capital_inicial, comision_pct, n_reinv, n_ado, dias_reales, buy_hold_money, epochs=global_epochs, cur_fit=vault['fit'])
                
                if bp: 
                    save_champion(s_id, bp)
                    st.session_state[f'opt_status_{s_id}'] = True
                    st.success("👑 ¡Evolución Exitosa! El ADN ha mutado a una forma superior.")
                else:
                    st.warning("🛡️ Se evaluaron decenas de miles de cruces. El setup actual sigue siendo insuperable.")
                time.sleep(2)
                ph_holograma.empty()
                wipe_ui_cache()
                st.rerun() 

            # Generación de la gráfica con lo que el usuario está viendo AHORA (new_ variables)
            df_strat = inyectar_adn_df(df_global.copy(), n_hit, n_thm, n_adx, n_wha)
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            m_mask = np.ones(len(df_strat), dtype=bool)
            if n_mac == "Bull Only (Precio > EMA 200)": m_mask = df_strat['Macro_Bull'].values
            elif n_mac == "Bear Only (Precio < EMA 200)": m_mask = ~df_strat['Macro_Bull'].values
            v_mask = np.ones(len(df_strat), dtype=bool)
            if n_vol == "Trend (ADX Alto)": v_mask = df_strat['ADX'].values >= n_adx
            elif n_vol == "Range (ADX Bajo)": v_mask = df_strat['ADX'].values < n_adx
            for r in n_btm: 
                if r in df_strat.columns: f_buy |= df_strat[r].values
            f_buy &= (m_mask & v_mask)
            for r in n_stm: 
                if r in df_strat.columns: f_sell |= df_strat[r].values
                
            df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
            df_strat['Active_TP'], df_strat['Active_SL'] = np.full(len(df_strat), n_tp), np.full(len(df_strat), n_sl)
            eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, capital_inicial, n_reinv, comision_pct)

        elif s_id in ["GENESIS", "ROCKET"]:
            st.markdown(f"### {'🌌 GÉNESIS (The Matrix)' if s_id == 'GENESIS' else '👑 ROCKET PROTOCOL (The Matrix)'} {opt_badge}", unsafe_allow_html=True)
            
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            c_ado = st.session_state.get(f'ui_{s_id}_ado', vault['ado'])
            c_reinv = st.session_state.get(f'ui_{s_id}_reinv', vault['reinv'])
            n_ado = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(c_ado), key=f"w_ado_{s_id}", step=0.5)
            n_reinv = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(c_reinv), key=f"w_reinv_{s_id}", step=5.0)

            with st.expander("⚙️ Calibración del ADN Base"):
                c_adv1, c_adv2 = st.columns(2)
                c_hit = st.session_state.get(f'ui_{s_id}_hitbox', vault['hitbox'])
                c_thm = st.session_state.get(f'ui_{s_id}_therm_w', vault['therm_w'])
                n_hit = c_adv1.slider("🎯 Target Lock Hitbox (%)", 0.5, 3.0, value=float(c_hit), key=f"w_hit_{s_id}", step=0.1)
                n_thm = c_adv2.slider("🌡️ Thermal Wall Weight", 3.0, 8.0, value=float(c_thm), key=f"w_tw_{s_id}", step=1.0)
                
                c_f1, c_f2 = st.columns(2)
                c_adx = st.session_state.get(f'ui_{s_id}_adx_th', vault['adx_th'])
                c_wha = st.session_state.get(f'ui_{s_id}_whale_f', vault['whale_f'])
                n_adx = c_f1.slider("🚀 Defcon/Ping ADX Threshold", 15.0, 35.0, value=float(c_adx), key=f"w_adx_{s_id}", step=1.0)
                n_wha = c_f2.slider("🐋 Climax Whale Factor", 1.5, 4.0, value=float(c_wha), key=f"w_wh_{s_id}", step=0.1)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            opts_b = base_b if s_id == "GENESIS" else rocket_b 
            opts_s = base_s if s_id == "GENESIS" else rocket_s
            
            n_quad = {}
            with c1:
                st.markdown("<h5 style='color:lime;'>🟢 Bull Trend</h5>", unsafe_allow_html=True)
                cb = [x for x in st.session_state.get(f'ui_{s_id}_r1_b', vault['r1_b']) if x in opts_b]
                cs = [x for x in st.session_state.get(f'ui_{s_id}_r1_s', vault['r1_s']) if x in opts_s]
                ct = st.session_state.get(f'ui_{s_id}_r1_tp', vault['r1_tp'])
                csl = st.session_state.get(f'ui_{s_id}_r1_sl', vault['r1_sl'])
                n_quad['r1_b'] = st.multiselect("Compras (B/T)", opts_b, default=cb, key=f"w_r1b_{s_id}")
                n_quad['r1_s'] = st.multiselect("Cierres (B/T)", opts_s, default=cs, key=f"w_r1s_{s_id}")
                n_quad['r1_tp'] = st.slider("TP % (B/T)", 0.5, 150.0, value=float(ct), key=f"w_r1tp_{s_id}", step=0.5)
                n_quad['r1_sl'] = st.slider("SL % (B/T)", 0.5, 25.0, value=float(csl), key=f"w_r1sl_{s_id}", step=0.5)
            with c2:
                st.markdown("<h5 style='color:yellow;'>🟡 Bull Chop</h5>", unsafe_allow_html=True)
                cb = [x for x in st.session_state.get(f'ui_{s_id}_r2_b', vault['r2_b']) if x in opts_b]
                cs = [x for x in st.session_state.get(f'ui_{s_id}_r2_s', vault['r2_s']) if x in opts_s]
                ct = st.session_state.get(f'ui_{s_id}_r2_tp', vault['r2_tp'])
                csl = st.session_state.get(f'ui_{s_id}_r2_sl', vault['r2_sl'])
                n_quad['r2_b'] = st.multiselect("Compras (B/C)", opts_b, default=cb, key=f"w_r2b_{s_id}")
                n_quad['r2_s'] = st.multiselect("Cierres (B/C)", opts_s, default=cs, key=f"w_r2s_{s_id}")
                n_quad['r2_tp'] = st.slider("TP % (B/C)", 0.5, 150.0, value=float(ct), key=f"w_r2tp_{s_id}", step=0.5)
                n_quad['r2_sl'] = st.slider("SL % (B/C)", 0.5, 25.0, value=float(csl), key=f"w_r2sl_{s_id}", step=0.5)
            with c3:
                st.markdown("<h5 style='color:red;'>🔴 Bear Trend</h5>", unsafe_allow_html=True)
                cb = [x for x in st.session_state.get(f'ui_{s_id}_r3_b', vault['r3_b']) if x in opts_b]
                cs = [x for x in st.session_state.get(f'ui_{s_id}_r3_s', vault['r3_s']) if x in opts_s]
                ct = st.session_state.get(f'ui_{s_id}_r3_tp', vault['r3_tp'])
                csl = st.session_state.get(f'ui_{s_id}_r3_sl', vault['r3_sl'])
                n_quad['r3_b'] = st.multiselect("Compras (Be/T)", opts_b, default=cb, key=f"w_r3b_{s_id}")
                n_quad['r3_s'] = st.multiselect("Cierres (Be/T)", opts_s, default=cs, key=f"w_r3s_{s_id}")
                n_quad['r3_tp'] = st.slider("TP % (Be/T)", 0.5, 150.0, value=float(ct), key=f"w_r3tp_{s_id}", step=0.5)
                n_quad['r3_sl'] = st.slider("SL % (Be/T)", 0.5, 25.0, value=float(csl), key=f"w_r3sl_{s_id}", step=0.5)
            with c4:
                st.markdown("<h5 style='color:orange;'>🟠 Bear Chop</h5>", unsafe_allow_html=True)
                cb = [x for x in st.session_state.get(f'ui_{s_id}_r4_b', vault['r4_b']) if x in opts_b]
                cs = [x for x in st.session_state.get(f'ui_{s_id}_r4_s', vault['r4_s']) if x in opts_s]
                ct = st.session_state.get(f'ui_{s_id}_r4_tp', vault['r4_tp'])
                csl = st.session_state.get(f'ui_{s_id}_r4_sl', vault['r4_sl'])
                n_quad['r4_b'] = st.multiselect("Compras (Be/C)", opts_b, default=cb, key=f"w_r4b_{s_id}")
                n_quad['r4_s'] = st.multiselect("Cierres (Be/C)", opts_s, default=cs, key=f"w_r4s_{s_id}")
                n_quad['r4_tp'] = st.slider("TP % (Be/C)", 0.5, 150.0, value=float(ct), key=f"w_r4tp_{s_id}", step=0.5)
                n_quad['r4_sl'] = st.slider("SL % (Be/C)", 0.5, 25.0, value=float(csl), key=f"w_r4sl_{s_id}", step=0.5)

            if c_ia3.button(f"🚀 DEEP MINE INDIVIDUAL ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                
                bp = optimizar_ia_tracker(s_id, df_global, capital_inicial, comision_pct, n_reinv, n_ado, dias_reales, buy_hold_money, epochs=global_epochs, is_meta=True, cur_fit=vault['fit'])
                
                if bp: 
                    save_champion(s_id, bp)
                    st.session_state[f'opt_status_{s_id}'] = True
                    st.success("👑 ¡Evolución Exitosa! El ADN ha mutado a una forma superior.")
                else:
                    st.warning("🛡️ Se evaluaron decenas de miles de cruces. El setup actual sigue siendo insuperable.")
                time.sleep(2)
                ph_holograma.empty()
                wipe_ui_cache()
                st.rerun() 

            df_strat = inyectar_adn_df(df_global.copy(), n_hit, n_thm, n_adx, n_wha)
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            f_tp, f_sl = np.zeros(len(df_strat)), np.zeros(len(df_strat))
            for idx_q in range(1, 5):
                mask = (df_strat['Regime'].values == idx_q)
                r_b_cond = np.zeros(len(df_strat), dtype=bool)
                for r in n_quad[f'r{idx_q}_b']: 
                    if r in df_strat.columns: r_b_cond |= df_strat[r].values
                f_buy[mask] = r_b_cond[mask]
                r_s_cond = np.zeros(len(df_strat), dtype=bool)
                for r in n_quad[f'r{idx_q}_s']: 
                    if r in df_strat.columns: r_s_cond |= df_strat[r].values
                f_sell[mask] = r_s_cond[mask]
                f_tp[mask] = n_quad[f'r{idx_q}_tp']
                f_sl[mask] = n_quad[f'r{idx_q}_sl']
                
            df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
            df_strat['Active_TP'], df_strat['Active_SL'] = f_tp, f_sl
            eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, capital_inicial, n_reinv, comision_pct)

        else:
            st.markdown(f"### ⚙️ {s_id} (Truth Engine) {opt_badge}", unsafe_allow_html=True)
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            
            c_ado = st.session_state.get(f'ui_{s_id}_ado', vault['ado'])
            c_reinv = st.session_state.get(f'ui_{s_id}_reinv', vault['reinv'])
            n_ado = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(c_ado), key=f"w_ado_{s_id}", step=0.5)
            n_reinv = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(c_reinv), key=f"w_reinv_{s_id}", step=5.0)

            if c_ia3.button(f"🚀 DEEP MINE INDIVIDUAL ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                
                bp = optimizar_ia_tracker(s_id, df_global, capital_inicial, comision_pct, n_reinv, n_ado, dias_reales, buy_hold_money, epochs=global_epochs, cur_fit=vault['fit'])
                if bp:
                    save_champion(s_id, bp)
                    st.session_state[f'opt_status_{s_id}'] = True
                    st.success("👑 ¡Evolución Exitosa! Nuevo récord encontrado.")
                else:
                    st.warning("🛡️ Ningún escenario superó la genética actual. Se mantuvo la corona.")
                time.sleep(2)
                ph_holograma.empty()
                wipe_ui_cache()
                st.rerun()

            with st.expander("🛠️ Ajuste Manual de Parámetros"):
                c1, c2, c3, c4 = st.columns(4)
                c_tp = st.session_state.get(f'ui_{s_id}_tp', vault['tp'])
                c_sl = st.session_state.get(f'ui_{s_id}_sl', vault['sl'])
                c_hit = st.session_state.get(f'ui_{s_id}_hitbox', vault['hitbox'])
                c_thm = st.session_state.get(f'ui_{s_id}_therm_w', vault['therm_w'])
                n_tp = c1.slider("🎯 TP Base (%)", 0.5, 150.0, value=float(c_tp), key=f"w_tp_{s_id}", step=0.1)
                n_sl = c2.slider("🛑 SL (%)", 0.5, 25.0, value=float(c_sl), key=f"w_sl_{s_id}", step=0.1)
                n_hit = c3.slider("🎯 Target Lock Hitbox (%)", 0.5, 3.0, value=float(c_hit), key=f"w_hit_{s_id}", step=0.1)
                n_thm = c4.slider("🌡️ Thermal Wall Weight", 3.0, 8.0, value=float(c_thm), key=f"w_tw_{s_id}", step=1.0)
                
                c_f1, c_f2 = st.columns(2)
                c_adx = st.session_state.get(f'ui_{s_id}_adx_th', vault['adx_th'])
                c_wha = st.session_state.get(f'ui_{s_id}_whale_f', vault['whale_f'])
                n_adx = c_f1.slider("🚀 Defcon/Ping ADX Threshold", 15.0, 35.0, value=float(c_adx), key=f"w_adx_{s_id}", step=1.0)
                n_wha = c_f2.slider("🐋 Climax Whale Factor", 1.5, 4.0, value=float(c_wha), key=f"w_wh_{s_id}", step=0.1)

            df_strat = inyectar_adn_df(df_global.copy(), n_hit, n_thm, n_adx, n_wha)
            b_c, s_c = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            
            if s_id == "TRINITY": b_c, s_c = df_strat['Trinity_Buy'].values, df_strat['Trinity_Sell'].values
            elif s_id == "JUGGERNAUT": b_c, s_c = df_strat['Jugg_Buy'].values, df_strat['Jugg_Sell'].values
            elif s_id == "DEFCON": b_c, s_c = df_strat['Defcon_Buy'].values, df_strat['Defcon_Sell'].values
            elif s_id == "TARGET_LOCK": b_c, s_c = df_strat['Lock_Buy'].values, df_strat['Lock_Sell'].values
            elif s_id == "THERMAL": b_c, s_c = df_strat['Thermal_Buy'].values, df_strat['Thermal_Sell'].values
            elif s_id == "PINK_CLIMAX": b_c, s_c = df_strat['Climax_Buy'].values, df_strat['Climax_Sell'].values
            elif s_id == "PING_PONG": b_c, s_c = df_strat['Ping_Buy'].values, df_strat['Ping_Sell'].values
            elif s_id == "NEON_SQUEEZE": b_c, s_c = df_strat['Squeeze_Buy'].values, df_strat['Squeeze_Sell'].values
            elif s_id == "COMMANDER": b_c, s_c = df_strat['Commander_Buy'].values, df_strat['Commander_Sell'].values
                
            df_strat['Signal_Buy'], df_strat['Signal_Sell'] = b_c, s_c
            df_strat['Active_TP'] = np.full(len(df_strat), n_tp)
            df_strat['Active_SL'] = np.full(len(df_strat), n_sl)
            eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, capital_inicial, n_reinv, comision_pct)

        # --- SECCIÓN COMÚN (MÉTRICAS Y BLOCK NOTE) ---
        df_strat['Total_Portfolio'] = eq_curve
        ret_pct = ((eq_curve[-1] - capital_inicial) / capital_inicial) * 100
        buy_hold_ret = ((df_strat['Close'].iloc[-1] - df_strat['Open'].iloc[0]) / df_strat['Open'].iloc[0]) * 100
        alpha_pct = ret_pct - buy_hold_ret

        dftr = pd.DataFrame(t_log)
        tt, wr, pf_val = 0, 0.0, 0.0
        if not dftr.empty:
            exs = dftr[dftr['Tipo'].isin(['TP', 'SL', 'DYN_WIN', 'DYN_LOSS'])]
            tt = len(exs)
            if tt > 0:
                ws = len(exs[exs['Tipo'].isin(['TP', 'DYN_WIN'])])
                wr = (ws / tt) * 100
                gpp = exs[exs['Ganancia_$'] > 0]['Ganancia_$'].sum()
                gll = abs(exs[exs['Ganancia_$'] < 0]['Ganancia_$'].sum())
                pf_val = gpp / gll if gll > 0 else float('inf')
        
        mdd = abs((((pd.Series(eq_curve) - pd.Series(eq_curve).cummax()) / pd.Series(eq_curve).cummax()) * 100).min())
        ado_val = tt / dias_reales if dias_reales > 0 else 0.0

        st.markdown(f"### 📊 Auditoría: {s_id}")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Portafolio Neto", f"${eq_curve[-1]:,.2f}", f"{ret_pct:.2f}%")
        c2.metric("ALPHA (vs Hold)", f"{alpha_pct:.2f}%", f"Hold: {buy_hold_ret:.2f}%", delta_color="normal" if alpha_pct > 0 else "inverse")
        c3.metric("Trades Totales", f"{tt}", f"ADO: {ado_val:.2f}/día", delta_color="off")
        c4.metric("Win Rate", f"{wr:.1f}%")
        c5.metric("Profit Factor", f"{pf_val:.2f}x")
        c6.metric("Max Drawdown", f"{mdd:.2f}%", delta_color="inverse")
        c7.metric("Comisiones", f"${total_comms:,.2f}", delta_color="inverse")

        # 📝 BLOCK NOTE INDIVIDUAL (LEE DIRECTO DE LA INTERFAZ)
        with st.expander("📝 BLOCK NOTE INDIVIDUAL (CÓDIGO PINE SCRIPT)", expanded=False):
            b_note = f"// ⚔️ {s_id} {'[✅ Optimizada]' if is_opt else '[➖ No Optimizada]'}\n"
            b_note += f"// Net Profit: ${eq_curve[-1]-capital_inicial:,.2f} ({ret_pct:.2f}%)\n"
            b_note += f"// ALPHA vs Hold: {alpha_pct:.2f}%\n"
            b_note += f"// Trades: {tt} | ADO: {ado_val:.2f} | PF: {pf_val:.2f}x | MDD: {mdd:.2f}%\n"
            b_note += f"// --------------------------------------------------\n"
            
            if s_id == "ALL_FORCES":
                b_note += f"// ⚙️ ADN BASE (Variables Universales)\n"
                b_note += f"hitbox_pct = {n_hit}%\n"
                b_note += f"therm_wall = {n_thm}\n"
                b_note += f"adx_trend = {n_adx}\n"
                b_note += f"whale_factor = {n_wha}x\n\n"
                b_note += f"// 🌍 ENTORNO MACRO Y VOLATILIDAD\n"
                b_note += f"Filtro Macro = {n_mac}\n"
                b_note += f"Filtro Volatilidad = {n_vol}\n\n"
                b_note += f"// 🔫 STRIKE TEAM (Global)\nCompras = {n_btm}\nCierres = {n_stm}\nTP = {n_tp:.1f}% | SL = {n_sl:.1f}%\n"
            elif s_id in ["GENESIS", "ROCKET"]:
                b_note += f"// ⚙️ ADN BASE (Variables Universales)\n"
                b_note += f"hitbox_pct = {n_hit}%\n"
                b_note += f"therm_wall = {n_thm}\n"
                b_note += f"adx_trend = {n_adx}\n"
                b_note += f"whale_factor = {n_wha}x\n\n"
                b_note += f"// 🟢 QUAD 1: BULL TREND (EMA 200+ | ADX >= Trend)\nquad1_b = {n_quad['r1_b']}\nquad1_s = {n_quad['r1_s']}\nquad1_tp = {n_quad['r1_tp']:.1f}%\nquad1_sl = {n_quad['r1_sl']:.1f}%\n\n"
                b_note += f"// 🟡 QUAD 2: BULL CHOP (EMA 200+ | ADX < Trend)\nquad2_b = {n_quad['r2_b']}\nquad2_s = {n_quad['r2_s']}\nquad2_tp = {n_quad['r2_tp']:.1f}%\nquad2_sl = {n_quad['r2_sl']:.1f}%\n\n"
                b_note += f"// 🔴 QUAD 3: BEAR TREND (EMA 200- | ADX >= Trend)\nquad3_b = {n_quad['r3_b']}\nquad3_s = {n_quad['r3_s']}\nquad3_tp = {n_quad['r3_tp']:.1f}%\nquad3_sl = {n_quad['r3_sl']:.1f}%\n\n"
                b_note += f"// 🟠 QUAD 4: BEAR CHOP (EMA 200- | ADX < Trend)\nquad4_b = {n_quad['r4_b']}\nquad4_s = {n_quad['r4_s']}\nquad4_tp = {n_quad['r4_tp']:.1f}%\nquad4_sl = {n_quad['r4_sl']:.1f}%\n"
            else:
                b_note += f"// ⚙️ ADN & RISK\n"
                b_note += f"tp_pct = {n_tp}%\n"
                b_note += f"sl_pct = {n_sl}%\n"
                b_note += f"hitbox_pct = {n_hit}%\n"
                b_note += f"therm_wall = {n_thm}\n"
                b_note += f"adx_trend = {n_adx}\n"
                b_note += f"whale_factor = {n_wha}x\n"
            st.code(b_note, language="text")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_50'], mode='lines', name='EMA 50 (Trend)', line=dict(color='yellow', width=2)), row=1, col=1)

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
