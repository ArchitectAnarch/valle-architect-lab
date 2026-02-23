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

# --- MOTOR DE HIPER-VELOCIDAD (NUMBA JIT COMPILER) ---
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

st.set_page_config(page_title="ROCKET PROTOCOL | Alpha Quant", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🧠 MEMORIA GLOBAL Y CATÁLOGOS
# ==========================================
base_b = ['Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 'Commander_Buy', 'Lev_Buy']
base_s = ['Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 'Commander_Sell', 'Lev_Sell']

rocket_b = ['Trinity_Buy', 'Jugg_Buy', 'Defcon_Buy', 'Lock_Buy', 'Thermal_Buy', 'Climax_Buy', 'Ping_Buy', 'Squeeze_Buy', 'Lev_Buy', 'Commander_Buy']
rocket_s = ['Trinity_Sell', 'Jugg_Sell', 'Defcon_Sell', 'Lock_Sell', 'Thermal_Sell', 'Climax_Sell', 'Ping_Sell', 'Squeeze_Sell', 'Lev_Sell', 'Commander_Sell']

estrategias = ["ALL_FORCES", "TRINITY", "JUGGERNAUT", "DEFCON", "TARGET_LOCK", "THERMAL", "PINK_CLIMAX", "PING_PONG", "NEON_SQUEEZE", "COMMANDER", "GENESIS", "ROCKET"]

tab_id_map = {
    "🌟 ALL FORCES (MATRIX)": "ALL_FORCES", "💠 TRINITY": "TRINITY", "⚔️ JUGGERNAUT": "JUGGERNAUT", "🚀 DEFCON": "DEFCON",
    "🎯 TARGET_LOCK": "TARGET_LOCK", "🌡️ THERMAL": "THERMAL", "🌸 PINK_CLIMAX": "PINK_CLIMAX",
    "🏓 PING_PONG": "PING_PONG", "🐛 NEON_SQUEEZE": "NEON_SQUEEZE", "👑 COMMANDER": "COMMANDER",
    "🌌 GENESIS": "GENESIS", "👑 ROCKET": "ROCKET"
}

# ==========================================
# 🧬 THE DNA VAULT (AUTO-HEAL & ESTADO INMUTABLE)
# ==========================================
# 1. AUTO-HEAL: Destruye el ADN corrupto si existe un choque de versiones
for s_id in estrategias:
    if f'champion_{s_id}' in st.session_state:
        vault = st.session_state[f'champion_{s_id}']
        if s_id in ["ALL_FORCES", "GENESIS", "ROCKET"] and 'r1_b' not in vault:
            del st.session_state[f'champion_{s_id}']
        elif s_id not in ["ALL_FORCES", "GENESIS", "ROCKET"] and 'hitbox' not in vault:
            del st.session_state[f'champion_{s_id}']

# 2. INICIALIZAR BÓVEDA
for s_id in estrategias:
    if f'opt_status_{s_id}' not in st.session_state: st.session_state[f'opt_status_{s_id}'] = False
    if f'champion_{s_id}' not in st.session_state:
        if s_id in ["ALL_FORCES", "GENESIS", "ROCKET"]:
            vault = {'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')}
            for r_idx in range(1, 5):
                vault.update({f'r{r_idx}_b': ['Squeeze_Buy'], f'r{r_idx}_s': ['Squeeze_Sell'], f'r{r_idx}_tp': 50.0, f'r{r_idx}_sl': 5.0})
            st.session_state[f'champion_{s_id}'] = vault
        else:
            st.session_state[f'champion_{s_id}'] = {'tp': 50.0, 'sl': 5.0, 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')}

def save_champion(s_id, bp):
    vault = st.session_state[f'champion_{s_id}']
    vault['fit'] = bp['fit']
    if s_id in ["ALL_FORCES", "GENESIS", "ROCKET"]:
        for k in ['hitbox', 'therm_w', 'adx_th', 'whale_f']: vault[k] = bp[k]
        for r_idx in range(1, 5):
            for k in ['b', 's', 'tp', 'sl']: vault[f'r{r_idx}_{k}'] = bp[f'{k}{r_idx}']
    else:
        for k in ['tp', 'sl', 'hitbox', 'therm_w', 'adx_th', 'whale_f']: vault[k] = bp[k]

def wipe_ui_cache():
    for key in list(st.session_state.keys()):
        if key.startswith("ui_"): del st.session_state[key]

ph_holograma = st.empty()

# ==========================================
# 🌍 SIDEBAR: INFRAESTRUCTURA Y PILOTO AUTOMÁTICO
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 TRUTH ENGINE V101.0</h2>", unsafe_allow_html=True)
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
global_epochs = st.sidebar.slider("Épocas de Evolución (x3000)", 1, 50, 10, help="1 Época = 3,000 combinaciones. 50 Épocas = 150,000 combinaciones procesadas sin interrupción.")

@st.cache_data(ttl=3600, show_spinner="📡 Sintetizando Malla Tensorial...")
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
# 🔥 IDENTIDADES Y LÓGICAS PURAS 🔥
# ==========================================
def inyectar_adn(df_sim, hitbox, therm_w, adx_th, whale_f):
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

@njit(fastmath=True)
def simular_crecimiento_exponencial(h_arr, l_arr, c_arr, o_arr, b_c, s_c, t_arr, sl_arr, cap_ini, com_pct, reinvest_pct):
    cap_act = cap_ini
    divs, en_pos = 0.0, False
    p_ent, tp_act, sl_act, pos_size, invest_amt = 0.0, 0.0, 0.0, 0.0, 0.0
    g_profit, g_loss, num_trades, max_dd, peak, total_comms = 0.0, 0.0, 0, 0.0, cap_ini, 0.0
    
    for i in range(len(h_arr)):
        if en_pos:
            tp_p = p_ent * (1.0 + tp_act/100.0)
            sl_p = p_ent * (1.0 - sl_act/100.0)
            
            if l_arr[i] <= sl_p:
                gross = pos_size * (1.0 - sl_act/100.0)
                comm_out = gross * com_pct
                total_comms += comm_out
                net = gross - comm_out
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
                comm_out = gross * com_pct
                total_comms += comm_out
                net = gross - comm_out
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
                comm_out = gross * com_pct
                total_comms += comm_out
                net = gross - comm_out
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
            total_comms += comm_in
            pos_size = invest_amt - comm_in 
            p_ent = o_arr[i+1]
            tp_act = t_arr[i]
            sl_act = sl_arr[i]
            en_pos = True
            
    total_net = (cap_act + divs) - cap_ini
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    return total_net, pf, num_trades, max_dd, total_comms

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

# 🧠 RUTINA DE DEEP MINE (V101.0 - AUTO PILOT MATRIX) 🧠
def optimizar_ia_tracker(s_id, df_base, cap_ini, com_pct, reinv_q, target_ado, dias_reales, buy_hold_money, epochs=1, is_meta=False):
    best_fit = -float('inf')
    bp = None
    tp_min, tp_max = (5.0, 150.0) 
    
    iters = 3000 * epochs
    chunks = 20 if epochs >= 5 else 10
    chunk_size = iters // chunks
    start_time = time.time()

    for c in range(chunks):
        for _ in range(chunk_size): 
            rtp = round(random.uniform(tp_min, tp_max), 1)
            rsl = round(random.uniform(1.0, 20.0), 1)
            
            r_hitbox = round(random.uniform(0.5, 3.0), 1)   
            r_therm  = float(random.randint(3, 8))          
            r_adx    = float(random.randint(15, 35))        
            r_whale  = round(random.uniform(1.5, 4.0), 1)   
            
            df_precalc = inyectar_adn(df_base.copy(), r_hitbox, r_therm, r_adx, r_whale)
            h_a = np.asarray(df_precalc['High'].values, dtype=np.float64)
            l_a = np.asarray(df_precalc['Low'].values, dtype=np.float64)
            c_a = np.asarray(df_precalc['Close'].values, dtype=np.float64)
            o_a = np.asarray(df_precalc['Open'].values, dtype=np.float64)
            
            if is_meta:
                # 🔥 THE MATRIX: Todos los metas (incluyendo ALL FORCES) usan cuadrantes ahora
                b_mat_opts = base_b if s_id != "ROCKET" else rocket_b
                s_mat_opts = base_s if s_id != "ROCKET" else rocket_s
                
                max_b = len(b_mat_opts) if s_id == "ALL_FORCES" else 2
                max_s = len(s_mat_opts) if s_id == "ALL_FORCES" else 2
                
                dna_b = [random.sample(b_mat_opts, random.randint(1, max_b)) for _ in range(4)]
                dna_s = [random.sample(s_mat_opts, random.randint(1, max_s)) for _ in range(4)]
                dna_tp = [random.uniform(tp_min, tp_max) for _ in range(4)]
                dna_sl = [random.uniform(2.0, 20.0) for _ in range(4)]
                
                f_buy, f_sell = np.zeros(len(df_precalc), dtype=bool), np.zeros(len(df_precalc), dtype=bool)
                f_tp, f_sl = np.zeros(len(df_precalc), dtype=np.float64), np.zeros(len(df_precalc), dtype=np.float64)
                regime_arr = df_precalc['Regime'].values
                for idx in range(4):
                    mask = (regime_arr == (idx + 1))
                    r_b_cond = np.zeros(len(df_precalc), dtype=bool)
                    for r in dna_b[idx]: r_b_cond |= df_precalc[r].values
                    f_buy[mask] = r_b_cond[mask]
                    r_s_cond = np.zeros(len(df_precalc), dtype=bool)
                    for r in dna_s[idx]: r_s_cond |= df_precalc[r].values
                    f_sell[mask] = r_s_cond[mask]
                    f_tp[mask] = dna_tp[idx]
                    f_sl[mask] = dna_sl[idx]
                b_c_arr = np.asarray(f_buy, dtype=bool)
                s_c_arr = np.asarray(f_sell, dtype=bool)
                t_arr = np.asarray(f_tp, dtype=np.float64)
                sl_arr = np.asarray(f_sl, dtype=np.float64)

            else:
                b_c, s_c = np.zeros(len(df_precalc), dtype=bool), np.zeros(len(df_precalc), dtype=bool)
                if s_id == "TRINITY": b_c, s_c = df_precalc['Trinity_Buy'].values, df_precalc['Trinity_Sell'].values
                elif s_id == "JUGGERNAUT": b_c, s_c = df_precalc['Jugg_Buy'].values, df_precalc['Jugg_Sell'].values
                elif s_id == "DEFCON": b_c, s_c = df_precalc['Defcon_Buy'].values, df_precalc['Defcon_Sell'].values
                elif s_id == "TARGET_LOCK": b_c, s_c = df_precalc['Lock_Buy'].values, df_precalc['Lock_Sell'].values
                elif s_id == "THERMAL": b_c, s_c = df_precalc['Thermal_Buy'].values, df_precalc['Thermal_Sell'].values
                elif s_id == "PINK_CLIMAX": b_c, s_c = df_precalc['Climax_Buy'].values, df_precalc['Climax_Sell'].values
                elif s_id == "PING_PONG": b_c, s_c = df_precalc['Ping_Buy'].values, df_precalc['Ping_Sell'].values
                elif s_id == "NEON_SQUEEZE": b_c, s_c = df_precalc['Squeeze_Buy'].values, df_precalc['Squeeze_Sell'].values
                elif s_id == "COMMANDER": b_c, s_c = df_precalc['Commander_Buy'].values, df_precalc['Commander_Sell'].values
                b_c_arr = np.asarray(b_c, dtype=bool)
                s_c_arr = np.asarray(s_c, dtype=bool)
                t_arr = np.asarray(np.full(len(df_precalc), float(rtp)), dtype=np.float64)
                sl_arr = np.asarray(np.full(len(df_precalc), float(rsl)), dtype=np.float64)

            net, pf, nt, mdd, comms = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, b_c_arr, s_c_arr, t_arr, sl_arr, float(cap_ini), float(com_pct), float(reinv_q))
            alpha_money = net - buy_hold_money
            
            # 🔥 V100.0 FITNESS: Respeta Francotiradores y Penaliza Inactividad 🔥
            if nt >= 1: 
                if net > 0: 
                    # Relajamos el castigo a solo 5 trades (Respeta a los Francotiradores)
                    trade_penalty = 1.0 if nt >= 5 else (float(nt) / 5.0) 
                    fit = (net ** 1.2) * (pf ** 0.5) * np.log1p(nt) * trade_penalty / ((mdd ** 0.6) + 1.0)
                    if alpha_money > 0: fit *= 1.5 
                else: 
                    fit = net * ((mdd ** 0.5) + 1.0) / (pf + 0.001)
            else:
                fit = -999999.0 
                
            if fit > best_fit:
                best_fit = fit
                if is_meta:
                    bp = {'r1_b': dna_b[0], 'r1_s': dna_s[0], 'r1_tp': dna_tp[0], 'r1_sl': dna_sl[0], 'r2_b': dna_b[1], 'r2_s': dna_s[1], 'r2_tp': dna_tp[1], 'r2_sl': dna_sl[1], 'r3_b': dna_b[2], 'r3_s': dna_s[2], 'r3_tp': dna_tp[2], 'r3_sl': dna_sl[2], 'r4_b': dna_b[3], 'r4_s': dna_s[3], 'r4_tp': dna_tp[3], 'r4_sl': dna_sl[3], 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'comms': comms, 'fit': fit}
                else:
                    bp = {'tp': rtp, 'sl': rsl, 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'reinv': reinv_q, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'comms': comms, 'fit': fit}
        
        elapsed = time.time() - start_time
        pct_done = int(((c + 1) / chunks) * 100)
        combos = (c + 1) * chunk_size
        eta = (elapsed / (c + 1)) * (chunks - c - 1)
        
        dyn_spinner = f"""
        <style>
        .loader-container {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.85); padding: 35px; border-radius: 20px; border: 2px solid cyan; box-shadow: 0 0 40px cyan;}}
        .rocket {{ font-size: 8rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 20px cyan); }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .prog-text {{ color: cyan; font-size: 1.8rem; font-weight: bold; margin-top: 15px; text-shadow: 0 0 5px cyan;}}
        .hud-text {{ color: lime; font-size: 1.3rem; margin-top: 8px; font-family: monospace; }}
        </style>
        <div class="loader-container">
            <div class="rocket">🚀</div>
            <div class="prog-text">PILOTO AUTOMÁTICO: {s_id}</div>
            <div class="hud-text">Progreso: {pct_done}%</div>
            <div class="hud-text">Combos Evaluados: {combos:,}</div>
            <div class="hud-text" style="color: yellow;">ETA: {eta:.1f} segs</div>
        </div>
        """
        ph_holograma.markdown(dyn_spinner, unsafe_allow_html=True)
        
    return bp

# 📋 REPORTE UNIVERSAL DIRECTO DE LA BÓVEDA 📋
def generar_reporte_universal(df_base, cap_ini, com_pct):
    res_str = f"📋 **REPORTE UNIVERSAL OMNI-BRAIN (V101.0 - THE MATRIX)**\n\n"
    res_str += f"⏱️ Temporalidad: {intervalo_sel} | 📊 Velas: {len(df_base)}\n\n"
    buy_hold_ret = ((df_base['Close'].iloc[-1] - df_base['Open'].iloc[0]) / df_base['Open'].iloc[0]) * 100
    res_str += f"📈 RENDIMIENTO DEL HOLD: **{buy_hold_ret:.2f}%**\n\n"
    
    for s_id in estrategias:
        vault = st.session_state[f'champion_{s_id}']
        df_strat = inyectar_adn(df_base.copy(), vault['hitbox'], vault['therm_w'], vault['adx_th'], vault['whale_f'])
        
        if s_id in ["ALL_FORCES", "GENESIS", "ROCKET"]:
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            t_arr, sl_arr = np.zeros(len(df_strat), dtype=np.float64), np.zeros(len(df_strat), dtype=np.float64)
            regimes = df_strat['Regime'].values
            
            for r_idx in range(1, 5):
                mask = (regimes == r_idx)
                b_cond = np.zeros(len(df_strat), dtype=bool)
                for rule in vault[f'r{r_idx}_b']:
                    if rule in df_strat.columns: b_cond |= df_strat[rule].values
                f_buy[mask] = b_cond[mask]
                
                s_cond = np.zeros(len(df_strat), dtype=bool)
                for rule in vault[f'r{r_idx}_s']:
                    if rule in df_strat.columns: s_cond |= df_strat[rule].values
                f_sell[mask] = s_cond[mask]
                
                t_arr[mask] = vault[f'r{r_idx}_tp']
                sl_arr[mask] = vault[f'r{r_idx}_sl']
                
            b_c_arr = np.asarray(f_buy, dtype=bool)
            s_c_arr = np.asarray(f_sell, dtype=bool)
            t_arr = np.asarray(t_arr, dtype=np.float64)
            sl_arr = np.asarray(sl_arr, dtype=np.float64)
            tp_val, sl_val = "Matrix (4Q)", "Matrix (4Q)"
        else:
            b_c = np.zeros(len(df_strat), dtype=bool)
            s_c = np.zeros(len(df_strat), dtype=bool)
            if s_id == "TRINITY": b_c, s_c = df_strat['Trinity_Buy'], df_strat['Trinity_Sell']
            elif s_id == "JUGGERNAUT": b_c, s_c = df_strat['Jugg_Buy'], df_strat['Jugg_Sell']
            elif s_id == "DEFCON": b_c, s_c = df_strat['Defcon_Buy'], df_strat['Defcon_Sell']
            elif s_id == "TARGET_LOCK": b_c, s_c = df_strat['Lock_Buy'], df_strat['Lock_Sell']
            elif s_id == "THERMAL": b_c, s_c = df_strat['Thermal_Buy'], df_strat['Thermal_Sell']
            elif s_id == "PINK_CLIMAX": b_c, s_c = df_strat['Climax_Buy'], df_strat['Climax_Sell']
            elif s_id == "PING_PONG": b_c, s_c = df_strat['Ping_Buy'], df_strat['Ping_Sell']
            elif s_id == "NEON_SQUEEZE": b_c, s_c = df_strat['Squeeze_Buy'], df_strat['Squeeze_Sell']
            elif s_id == "COMMANDER": b_c, s_c = df_strat['Commander_Buy'], df_strat['Commander_Sell']
                
            b_c_arr = np.asarray(b_c.values, dtype=bool)
            s_c_arr = np.asarray(s_c.values, dtype=bool)
            t_arr = np.asarray(np.full(len(df_strat), float(vault['tp'])), dtype=np.float64)
            sl_arr = np.asarray(np.full(len(df_strat), float(vault['sl'])), dtype=np.float64)
            tp_val, sl_val = f"{vault['tp']:.1f}%", f"{vault['sl']:.1f}%"
            
        h_a = np.asarray(df_strat['High'].values, dtype=np.float64)
        l_a = np.asarray(df_strat['Low'].values, dtype=np.float64)
        c_a = np.asarray(df_strat['Close'].values, dtype=np.float64)
        o_a = np.asarray(df_strat['Open'].values, dtype=np.float64)
        
        net, pf, nt, mdd, comms = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, b_c_arr, s_c_arr, t_arr, sl_arr, float(cap_ini), float(com_pct), float(vault['reinv']))
        
        ret_pct = (net / cap_ini) * 100
        alpha = ret_pct - buy_hold_ret
        
        opt_icon = "✅" if st.session_state.get(f'opt_status_{s_id}', False) else "➖"
        res_str += f"⚔️ **{s_id}** [{opt_icon} Optimizada]\nNet Profit: ${net:,.2f} ({ret_pct:.2f}%)\nALPHA vs Hold: {alpha:.2f}%\nTrades: {nt} | PF: {pf:.2f}x | MDD: {mdd:.2f}%\n⚙️ TP: {tp_val} | SL: {sl_val} | TargetLock: {vault['hitbox']}% | Muro: {vault['therm_w']} | ADX: {vault['adx_th']} | Whale: {vault['whale_f']}\n---\n"
        
    return res_str

# ==========================================
# 🏆 SCORECARD OVERALL
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center; color: gold;'>🏆 SCORECARD OVERALL</h3>", unsafe_allow_html=True)

if not df_global.empty:
    buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
    leaderboard = []
    
    for s_id in estrategias:
        vault = st.session_state[f'champion_{s_id}']
        df_strat = inyectar_adn(df_global.copy(), vault['hitbox'], vault['therm_w'], vault['adx_th'], vault['whale_f'])
        
        if s_id in ["ALL_FORCES", "GENESIS", "ROCKET"]:
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            t_arr, sl_arr = np.zeros(len(df_strat), dtype=np.float64), np.zeros(len(df_strat), dtype=np.float64)
            regimes = df_strat['Regime'].values
            for r_idx in range(1, 5):
                mask = (regimes == r_idx)
                b_cond, s_cond = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
                for rule in vault[f'r{r_idx}_b']:
                    if rule in df_strat.columns: b_cond |= df_strat[rule].values
                f_buy[mask] = b_cond[mask]
                for rule in vault[f'r{r_idx}_s']:
                    if rule in df_strat.columns: s_cond |= df_strat[rule].values
                f_sell[mask] = s_cond[mask]
                t_arr[mask] = vault[f'r{r_idx}_tp']
                sl_arr[mask] = vault[f'r{r_idx}_sl']
            b_c_arr, s_c_arr = np.asarray(f_buy, dtype=bool), np.asarray(f_sell, dtype=bool)
        else:
            b_c = np.zeros(len(df_strat), dtype=bool)
            s_c = np.zeros(len(df_strat), dtype=bool)
            if s_id == "TRINITY": b_c, s_c = df_strat['Trinity_Buy'], df_strat['Trinity_Sell']
            elif s_id == "JUGGERNAUT": b_c, s_c = df_strat['Jugg_Buy'], df_strat['Jugg_Sell']
            elif s_id == "DEFCON": b_c, s_c = df_strat['Defcon_Buy'], df_strat['Defcon_Sell']
            elif s_id == "TARGET_LOCK": b_c, s_c = df_strat['Lock_Buy'], df_strat['Lock_Sell']
            elif s_id == "THERMAL": b_c, s_c = df_strat['Thermal_Buy'], df_strat['Thermal_Sell']
            elif s_id == "PINK_CLIMAX": b_c, s_c = df_strat['Climax_Buy'], df_strat['Climax_Sell']
            elif s_id == "PING_PONG": b_c, s_c = df_strat['Ping_Buy'], df_strat['Ping_Sell']
            elif s_id == "NEON_SQUEEZE": b_c, s_c = df_strat['Squeeze_Buy'], df_strat['Squeeze_Sell']
            elif s_id == "COMMANDER": b_c, s_c = df_strat['Commander_Buy'], df_strat['Commander_Sell']
            b_c_arr, s_c_arr = np.asarray(b_c.values, dtype=bool), np.asarray(s_c.values, dtype=bool)
            t_arr = np.asarray(np.full(len(df_strat), float(vault['tp'])), dtype=np.float64)
            sl_arr = np.asarray(np.full(len(df_strat), float(vault['sl'])), dtype=np.float64)

        h_a = np.asarray(df_strat['High'].values, dtype=np.float64)
        l_a = np.asarray(df_strat['Low'].values, dtype=np.float64)
        c_a = np.asarray(df_strat['Close'].values, dtype=np.float64)
        o_a = np.asarray(df_strat['Open'].values, dtype=np.float64)
        net, pf, nt, mdd, comms = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, b_c_arr, s_c_arr, t_arr, sl_arr, float(capital_inicial), float(comision_pct), float(vault['reinv']))
        
        ret_pct = (net / capital_inicial) * 100
        ado_v = nt / dias_reales if dias_reales > 0 else 0.0
        
        if ret_pct > 0 and nt > 0: score = ret_pct * (1 + np.log1p(nt))
        else: score = ret_pct 
        leaderboard.append((s_id, ret_pct, nt, ado_v, score))
        
    leaderboard.sort(key=lambda x: x[4], reverse=True)
    st.sidebar.markdown(f"**📈 HOLD MKT:** `{buy_hold_ret:.2f}%`")
    for rank, (s_id, ret_pct, nt, ado_v, score) in enumerate(leaderboard):
        medal = "🏆" if rank == 0 else "🥈" if rank == 1 else "🥉" if rank == 2 else "🔹"
        color = "lime" if ret_pct > buy_hold_ret else ("#00FF00" if ret_pct > 0 else "red")
        st.sidebar.markdown(f"{medal} **{s_id}** <br> <span style='color:{color}; font-size:0.9rem;'>Neto: {ret_pct:.1f}% | Trades: {nt} | ADO: {ado_v:.2f} | Pts: {score:.0f}</span>", unsafe_allow_html=True)

# ==========================================
# 🧠 BOTONES MAESTROS SIDEBAR
# ==========================================
st.sidebar.markdown("---")
if st.sidebar.button(f"🧠 DEEP MINE GLOBAL ({global_epochs*3}k Combos)", type="primary", use_container_width=True):
    buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
    buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
    
    for s_id in estrategias:
        is_meta = s_id in ["ALL_FORCES", "GENESIS", "ROCKET"]
        reinv_q = st.session_state[f'champion_{s_id}']['reinv']
        t_ado = st.session_state[f'champion_{s_id}']['ado']
        
        bp = optimizar_ia_tracker(s_id, df_global, capital_inicial, comision_pct, reinv_q, t_ado, dias_reales, buy_hold_money, epochs=global_epochs, is_meta=is_meta)
        if bp:
            current_fit = st.session_state.get(f'champion_{s_id}', {}).get('fit', -float('inf'))
            if bp['fit'] > current_fit:
                save_champion(s_id, bp)
                st.session_state[f'opt_status_{s_id}'] = True
            
    wipe_ui_cache()
    ph_holograma.empty()
    st.sidebar.success("✅ ¡Forja Evolutiva Global Completada!")
    time.sleep(1)
    st.rerun()

if st.sidebar.button("📊 GENERAR REPORTE UNIVERSAL", use_container_width=True):
    with st.spinner("Escaneando la Bóveda de ADN..."):
        reporte_txt = generar_reporte_universal(df_global, capital_inicial, comision_pct)
    st.sidebar.text_area("Copia tu Reporte:", value=reporte_txt, height=400)

st.title("🛡️ The Omni-Brain Lab")
tabs = st.tabs(["🌟 ALL FORCES (MATRIX)", "💠 TRINITY", "⚔️ JUGGERNAUT", "🚀 DEFCON", "🎯 TARGET_LOCK", "🌡️ THERMAL", "🌸 PINK_CLIMAX", "🏓 PING_PONG", "🐛 NEON_SQUEEZE", "👑 COMMANDER", "🌌 GENESIS", "👑 ROCKET"])

for idx, tab_name in enumerate(tab_id_map.keys()):
    with tabs[idx]:
        if df_global.empty: continue
        s_id = tab_id_map[tab_name]
        is_opt = st.session_state.get(f'opt_status_{s_id}', False)
        opt_badge = "<span style='color: lime; border: 1px solid lime; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem;'>✅ IA OPTIMIZADA</span>" if is_opt else "<span style='color: gray; border: 1px solid gray; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem;'>➖ NO OPTIMIZADA</span>"
        vault = st.session_state[f'champion_{s_id}']

        if s_id in ["ALL_FORCES", "GENESIS", "ROCKET"]:
            st.markdown(f"### {'🌟 ALL FORCES ALGO (The Matrix)' if s_id == 'ALL_FORCES' else ('🌌 GÉNESIS' if s_id == 'GENESIS' else '👑 ROCKET PROTOCOL')} {opt_badge}", unsafe_allow_html=True)
            if s_id == "ALL_FORCES": st.info("La IA Suprema. Ahora funciona con la **Matriz de 4 Cuadrantes** y permite cruzar hasta los 10 algoritmos al mismo tiempo.")
            
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            
            new_ado = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(st.session_state.get(f'ui_ado_{s_id}', vault['ado'])), key=f"ui_ado_{s_id}", step=0.5)
            st.session_state[f'champion_{s_id}']['ado'] = new_ado
            
            new_reinv = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get(f'ui_reinv_{s_id}', vault['reinv'])), key=f"ui_reinv_{s_id}", step=5.0)
            st.session_state[f'champion_{s_id}']['reinv'] = new_reinv

            with st.expander("⚙️ Calibración del ADN Base"):
                c_adv1, c_adv2 = st.columns(2)
                new_hit = c_adv1.slider("🎯 Target Lock Hitbox (%)", 0.5, 3.0, value=float(st.session_state.get(f'ui_hitbox_{s_id}', vault['hitbox'])), key=f"ui_hitbox_{s_id}", step=0.1)
                st.session_state[f'champion_{s_id}']['hitbox'] = new_hit
                
                new_therm = c_adv2.slider("🌡️ Thermal Wall Weight", 3.0, 8.0, value=float(st.session_state.get(f'ui_therm_w_{s_id}', vault['therm_w'])), key=f"ui_therm_w_{s_id}", step=1.0)
                st.session_state[f'champion_{s_id}']['therm_w'] = new_therm
                
                c_f1, c_f2 = st.columns(2)
                new_adx = c_f1.slider("🚀 Defcon/Ping ADX Threshold", 15.0, 35.0, value=float(st.session_state.get(f'ui_adx_th_{s_id}', vault['adx_th'])), key=f"ui_adx_th_{s_id}", step=1.0)
                st.session_state[f'champion_{s_id}']['adx_th'] = new_adx
                
                new_whale = c_f2.slider("🐋 Climax Whale Factor", 1.5, 4.0, value=float(st.session_state.get(f'ui_whale_f_{s_id}', vault['whale_f'])), key=f"ui_whale_f_{s_id}", step=0.1)
                st.session_state[f'champion_{s_id}']['whale_f'] = new_whale

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            opts_b = base_b if s_id != "ROCKET" else rocket_b 
            opts_s = base_s if s_id != "ROCKET" else rocket_s
            
            with c1:
                st.markdown("<h5 style='color:lime;'>🟢 Bull Trend</h5>", unsafe_allow_html=True)
                new_b1 = st.multiselect("Compras", opts_b, default=[x for x in st.session_state.get(f'ui_r1_b_{s_id}', vault['r1_b']) if x in opts_b], key=f"ui_r1_b_{s_id}")
                st.session_state[f'champion_{s_id}']['r1_b'] = new_b1
                
                new_s1 = st.multiselect("Cierres", opts_s, default=[x for x in st.session_state.get(f'ui_r1_s_{s_id}', vault['r1_s']) if x in opts_s], key=f"ui_r1_s_{s_id}")
                st.session_state[f'champion_{s_id}']['r1_s'] = new_s1
                
                new_tp1 = st.slider("TP %", 0.5, 150.0, value=float(st.session_state.get(f'ui_r1_tp_{s_id}', vault['r1_tp'])), key=f"ui_r1_tp_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r1_tp'] = new_tp1
                
                new_sl1 = st.slider("SL %", 0.5, 25.0, value=float(st.session_state.get(f'ui_r1_sl_{s_id}', vault['r1_sl'])), key=f"ui_r1_sl_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r1_sl'] = new_sl1
            with c2:
                st.markdown("<h5 style='color:yellow;'>🟡 Bull Chop</h5>", unsafe_allow_html=True)
                new_b2 = st.multiselect("Compras", opts_b, default=[x for x in st.session_state.get(f'ui_r2_b_{s_id}', vault['r2_b']) if x in opts_b], key=f"ui_r2_b_{s_id}")
                st.session_state[f'champion_{s_id}']['r2_b'] = new_b2
                
                new_s2 = st.multiselect("Cierres", opts_s, default=[x for x in st.session_state.get(f'ui_r2_s_{s_id}', vault['r2_s']) if x in opts_s], key=f"ui_r2_s_{s_id}")
                st.session_state[f'champion_{s_id}']['r2_s'] = new_s2
                
                new_tp2 = st.slider("TP %", 0.5, 150.0, value=float(st.session_state.get(f'ui_r2_tp_{s_id}', vault['r2_tp'])), key=f"ui_r2_tp_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r2_tp'] = new_tp2
                
                new_sl2 = st.slider("SL %", 0.5, 25.0, value=float(st.session_state.get(f'ui_r2_sl_{s_id}', vault['r2_sl'])), key=f"ui_r2_sl_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r2_sl'] = new_sl2
            with c3:
                st.markdown("<h5 style='color:red;'>🔴 Bear Trend</h5>", unsafe_allow_html=True)
                new_b3 = st.multiselect("Compras", opts_b, default=[x for x in st.session_state.get(f'ui_r3_b_{s_id}', vault['r3_b']) if x in opts_b], key=f"ui_r3_b_{s_id}")
                st.session_state[f'champion_{s_id}']['r3_b'] = new_b3
                
                new_s3 = st.multiselect("Cierres", opts_s, default=[x for x in st.session_state.get(f'ui_r3_s_{s_id}', vault['r3_s']) if x in opts_s], key=f"ui_r3_s_{s_id}")
                st.session_state[f'champion_{s_id}']['r3_s'] = new_s3
                
                new_tp3 = st.slider("TP %", 0.5, 150.0, value=float(st.session_state.get(f'ui_r3_tp_{s_id}', vault['r3_tp'])), key=f"ui_r3_tp_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r3_tp'] = new_tp3
                
                new_sl3 = st.slider("SL %", 0.5, 25.0, value=float(st.session_state.get(f'ui_r3_sl_{s_id}', vault['r3_sl'])), key=f"ui_r3_sl_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r3_sl'] = new_sl3
            with c4:
                st.markdown("<h5 style='color:orange;'>🟠 Bear Chop</h5>", unsafe_allow_html=True)
                new_b4 = st.multiselect("Compras", opts_b, default=[x for x in st.session_state.get(f'ui_r4_b_{s_id}', vault['r4_b']) if x in opts_b], key=f"ui_r4_b_{s_id}")
                st.session_state[f'champion_{s_id}']['r4_b'] = new_b4
                
                new_s4 = st.multiselect("Cierres", opts_s, default=[x for x in st.session_state.get(f'ui_r4_s_{s_id}', vault['r4_s']) if x in opts_s], key=f"ui_r4_s_{s_id}")
                st.session_state[f'champion_{s_id}']['r4_s'] = new_s4
                
                new_tp4 = st.slider("TP %", 0.5, 150.0, value=float(st.session_state.get(f'ui_r4_tp_{s_id}', vault['r4_tp'])), key=f"ui_r4_tp_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r4_tp'] = new_tp4
                
                new_sl4 = st.slider("SL %", 0.5, 25.0, value=float(st.session_state.get(f'ui_r4_sl_{s_id}', vault['r4_sl'])), key=f"ui_r4_sl_{s_id}", step=0.5)
                st.session_state[f'champion_{s_id}']['r4_sl'] = new_sl4

            if c_ia3.button(f"🚀 DEEP MINE INDIVIDUAL ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                
                bp = optimizar_ia_tracker(s_id, df_global, capital_inicial, comision_pct, st.session_state[f'champion_{s_id}']['reinv'], st.session_state[f'champion_{s_id}']['ado'], dias_reales, buy_hold_money, epochs=global_epochs, is_meta=True)
                
                if bp: 
                    if bp['fit'] > vault['fit']:
                        save_champion(s_id, bp)
                        st.session_state[f'opt_status_{s_id}'] = True
                        st.success("👑 ¡Evolución Exitosa! Se encontró un ADN superior.")
                    else:
                        st.warning("🛡️ El Piloto Automático no pudo superar la genética actual. Se retuvo el ADN campeón.")
                    time.sleep(2)
                ph_holograma.empty()
                wipe_ui_cache()
                st.rerun() 

            df_strat = inyectar_adn(df_global.copy(), new_hit, new_therm, new_adx, new_whale)
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            f_tp, f_sl = np.zeros(len(df_strat)), np.zeros(len(df_strat))
            for idx_q, (b_lst, s_lst, t_v, s_v) in enumerate([(new_b1, new_s1, new_tp1, new_sl1), (new_b2, new_s2, new_tp2, new_sl2), (new_b3, new_s3, new_tp3, new_sl3), (new_b4, new_s4, new_tp4, new_sl4)]):
                mask = (df_strat['Regime'].values == (idx_q + 1))
                r_b_cond = np.zeros(len(df_strat), dtype=bool)
                for r in b_lst: r_b_cond |= df_strat[r].values
                f_buy[mask] = r_b_cond[mask]
                r_s_cond = np.zeros(len(df_strat), dtype=bool)
                for r in s_lst: r_s_cond |= df_strat[r].values
                f_sell[mask] = r_s_cond[mask]
                f_tp[mask] = t_v
                f_sl[mask] = s_v
                
            df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
            df_strat['Active_TP'], df_strat['Active_SL'] = f_tp, f_sl
            eq_curve, divs, cap_act, t_log, pos_ab, total_comms = simular_visual(df_strat, capital_inicial, new_reinv, comision_pct)

        else:
            st.markdown(f"### ⚙️ {s_id} (Truth Engine) {opt_badge}", unsafe_allow_html=True)
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            
            new_ado = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(st.session_state.get(f'ui_ado_{s_id}', vault['ado'])), key=f"ui_ado_{s_id}", step=0.5)
            st.session_state[f'champion_{s_id}']['ado'] = new_ado
            
            new_reinv = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get(f'ui_reinv_{s_id}', vault['reinv'])), key=f"ui_reinv_{s_id}", step=5.0)
            st.session_state[f'champion_{s_id}']['reinv'] = new_reinv

            if c_ia3.button(f"🚀 DEEP MINE INDIVIDUAL ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                
                bp = optimizar_ia_tracker(s_id, df_global, capital_inicial, comision_pct, st.session_state[f'champion_{s_id}']['reinv'], st.session_state[f'champion_{s_id}']['ado'], dias_reales, buy_hold_money, epochs=global_epochs)
                if bp:
                    if bp['fit'] > vault['fit']:
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
                new_tp = c1.slider("🎯 TP Base (%)", 0.5, 150.0, value=float(st.session_state.get(f'ui_tp_{s_id}', vault['tp'])), key=f"ui_tp_{s_id}", step=0.1)
                st.session_state[f'champion_{s_id}']['tp'] = new_tp
                
                new_sl = c2.slider("🛑 SL (%)", 0.5, 25.0, value=float(st.session_state.get(f'ui_sl_{s_id}', vault['sl'])), key=f"ui_sl_{s_id}", step=0.1)
                st.session_state[f'champion_{s_id}']['sl'] = new_sl
                
                new_hitbox = c3.slider("🎯 Target Lock Hitbox (%)", 0.5, 3.0, value=float(st.session_state.get(f'ui_hitbox_{s_id}', vault['hitbox'])), key=f"ui_hitbox_{s_id}", step=0.1)
                st.session_state[f'champion_{s_id}']['hitbox'] = new_hitbox
                
                new_therm = c4.slider("🌡️ Thermal Wall Weight", 3.0, 8.0, value=float(st.session_state.get(f'ui_therm_w_{s_id}', vault['therm_w'])), key=f"ui_therm_w_{s_id}", step=1.0)
                st.session_state[f'champion_{s_id}']['therm_w'] = new_therm
                
                c_f1, c_f2 = st.columns(2)
                new_adx = c_f1.slider("🚀 Defcon/Ping ADX Threshold", 15.0, 35.0, value=float(st.session_state.get(f'ui_adx_th_{s_id}', vault['adx_th'])), key=f"ui_adx_th_{s_id}", step=1.0)
                st.session_state[f'champion_{s_id}']['adx_th'] = new_adx
                
                new_whale = c_f2.slider("🐋 Climax Whale Factor", 1.5, 4.0, value=float(st.session_state.get(f'ui_whale_f_{s_id}', vault['whale_f'])), key=f"ui_whale_f_{s_id}", step=0.1)
                st.session_state[f'champion_{s_id}']['whale_f'] = new_whale

            df_strat = inyectar_adn(df_global.copy(), new_hitbox, new_therm, new_adx, new_whale)
            b_c, s_c = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            
            if s_id == "TRINITY": b_c, s_c = df_strat['Trinity_Buy'], df_strat['Trinity_Sell']
            elif s_id == "JUGGERNAUT": b_c, s_c = df_strat['Jugg_Buy'], df_strat['Jugg_Sell']
            elif s_id == "DEFCON": b_c, s_c = df_strat['Defcon_Buy'], df_strat['Defcon_Sell']
            elif s_id == "TARGET_LOCK": b_c, s_c = df_strat['Lock_Buy'], df_strat['Lock_Sell']
            elif s_id == "THERMAL": b_c, s_c = df_strat['Thermal_Buy'], df_strat['Thermal_Sell']
            elif s_id == "PINK_CLIMAX": b_c, s_c = df_strat['Climax_Buy'], df_strat['Climax_Sell']
            elif s_id == "PING_PONG": b_c, s_c = df_strat['Ping_Buy'], df_strat['Ping_Sell']
            elif s_id == "NEON_SQUEEZE": b_c, s_c = df_strat['Squeeze_Buy'], df_strat['Squeeze_Sell']
            elif s_id == "COMMANDER": b_c, s_c = df_strat['Commander_Buy'], df_strat['Commander_Sell']
                
            df_strat['Signal_Buy'], df_strat['Signal_Sell'] = b_c, s_c
            df_strat['Active_TP'] = new_tp
            df_strat['Active_SL'] = new_sl
            eq_curve, divs, cap_act, t_log, pos_ab, total_comms = simular_visual(df_strat, capital_inicial, new_reinv, comision_pct)

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

        # 📝 BLOCK NOTE INDIVIDUAL (CÓDIGO PINE SCRIPT)
        with st.expander("📝 BLOCK NOTE INDIVIDUAL (CÓDIGO PINE SCRIPT)", expanded=False):
            b_note = f"// ⚔️ {s_id} {'[✅ Optimizada]' if is_opt else '[➖ No Optimizada]'}\n"
            b_note += f"// Net Profit: ${eq_curve[-1]-capital_inicial:,.2f} ({ret_pct:.2f}%)\n"
            b_note += f"// ALPHA vs Hold: {alpha_pct:.2f}%\n"
            b_note += f"// Trades: {tt} | ADO: {ado_val:.2f} | PF: {pf_val:.2f}x | MDD: {mdd:.2f}%\n"
            b_note += f"// --------------------------------------------------\n"
            
            if s_id in ["ALL_FORCES", "GENESIS", "ROCKET"]:
                b_note += f"// ⚙️ ADN BASE (Variables Universales)\n"
                b_note += f"hitbox_pct = {vault['hitbox']}\n"
                b_note += f"therm_wall = {vault['therm_w']}\n"
                b_note += f"adx_trend = {vault['adx_th']}\n"
                b_note += f"whale_factor = {vault['whale_f']}\n\n"
                b_note += f"// 🟢 QUAD 1: BULL TREND (EMA 200+ | ADX >= Trend)\nquad1_b = {vault['r1_b']}\nquad1_s = {vault['r1_s']}\nquad1_tp = {vault['r1_tp']:.1f}\nquad1_sl = {vault['r1_sl']:.1f}\n\n"
                b_note += f"// 🟡 QUAD 2: BULL CHOP (EMA 200+ | ADX < Trend)\nquad2_b = {vault['r2_b']}\nquad2_s = {vault['r2_s']}\nquad2_tp = {vault['r2_tp']:.1f}\nquad2_sl = {vault['r2_sl']:.1f}\n\n"
                b_note += f"// 🔴 QUAD 3: BEAR TREND (EMA 200- | ADX >= Trend)\nquad3_b = {vault['r3_b']}\nquad3_s = {vault['r3_s']}\nquad3_tp = {vault['r3_tp']:.1f}\nquad3_sl = {vault['r3_sl']:.1f}\n\n"
                b_note += f"// 🟠 QUAD 4: BEAR CHOP (EMA 200- | ADX < Trend)\nquad4_b = {vault['r4_b']}\nquad4_s = {vault['r4_s']}\nquad4_tp = {vault['r4_tp']:.1f}\nquad4_sl = {vault['r4_sl']:.1f}\n"
            else:
                b_note += f"// ⚙️ ADN & RISK\n"
                b_note += f"tp_pct = {vault['tp']}%\n"
                b_note += f"sl_pct = {vault['sl']}%\n"
                b_note += f"hitbox_pct = {vault['hitbox']}%\n"
                b_note += f"therm_wall = {vault['therm_w']}\n"
                b_note += f"adx_trend = {vault['adx_th']}\n"
                b_note += f"whale_factor = {vault['whale_f']}\n"
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
