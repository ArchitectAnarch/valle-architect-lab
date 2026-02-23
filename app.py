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

# --- MOTOR DE HIPER-VELOCIDAD (NUMBA JIT) ---
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

st.set_page_config(page_title="ROCKET PROTOCOL | Omni-Forge", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🧠 UTILIDADES NUMPY (SHIFT RÁPIDO)
# ==========================================
def npshift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def npshift_bool(arr, num, fill_value=False):
    result = np.empty_like(arr, dtype=bool)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

# ==========================================
# 🧠 CATÁLOGOS Y ARQUITECTURA DE ESTADO
# ==========================================
base_b = ['Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 'Commander_Buy', 'Lev_Buy']
base_s = ['Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 'Commander_Sell', 'Lev_Sell']
rocket_b = ['Trinity_Buy', 'Jugg_Buy', 'Defcon_Buy', 'Lock_Buy', 'Thermal_Buy', 'Climax_Buy', 'Ping_Buy', 'Squeeze_Buy', 'Lev_Buy', 'Commander_Buy']
rocket_s = ['Trinity_Sell', 'Jugg_Sell', 'Defcon_Sell', 'Lock_Sell', 'Thermal_Sell', 'Climax_Sell', 'Ping_Sell', 'Squeeze_Sell', 'Lev_Sell', 'Commander_Sell']
quadrix_b = ['Q_Pink_Whale_Buy', 'Q_Lock_Bounce', 'Q_Lock_Break', 'Q_Neon_Up', 'Q_Defcon_Buy', 'Q_Therm_Bounce', 'Q_Therm_Vacuum', 'Q_Nuclear_Buy', 'Q_Early_Buy', 'Q_Rebound_Buy']
quadrix_s = ['Q_Lock_Reject', 'Q_Lock_Breakd', 'Q_Neon_Dn', 'Q_Defcon_Sell', 'Q_Therm_Wall_Sell', 'Q_Therm_Panic_Sell', 'Q_Nuclear_Sell', 'Q_Early_Sell']

estrategias = ["ROCKET_ULTRA", "ROCKET_COMMANDER", "QUADRIX", "JUGGERNAUT", "ALL_FORCES", "GENESIS", "ROCKET", "TRINITY", "DEFCON", "TARGET_LOCK", "THERMAL", "PINK_CLIMAX", "PING_PONG", "NEON_SQUEEZE", "COMMANDER"]
macro_opts = ["All-Weather", "Bull Only (Precio > EMA 200)", "Bear Only (Precio < EMA 200)"]
vol_opts = ["All-Weather", "Trend (ADX Alto)", "Range (ADX Bajo)"]

tab_id_map = {
    "👑 ROCKET ULTRA V55": "ROCKET_ULTRA",
    "🚀 ROCKET COMMANDER": "ROCKET_COMMANDER",
    "🌌 QUADRIX": "QUADRIX", 
    "⚔️ JUGGERNAUT V356": "JUGGERNAUT", 
    "🌟 ALL FORCES": "ALL_FORCES", 
    "🌌 GENESIS": "GENESIS", 
    "👑 ROCKET": "ROCKET", 
    "💠 TRINITY": "TRINITY", 
    "🚀 DEFCON": "DEFCON", 
    "🎯 TARGET_LOCK": "TARGET_LOCK", 
    "🌡️ THERMAL": "THERMAL", 
    "🌸 PINK_CLIMAX": "PINK_CLIMAX", 
    "🏓 PING_PONG": "PING_PONG", 
    "🐛 NEON_SQUEEZE": "NEON_SQUEEZE", 
    "👑 COMMANDER": "COMMANDER"
}

pine_map = {
    'Ping_Buy': 'ping_b', 'Ping_Sell': 'ping_s', 'Squeeze_Buy': 'squeeze_b', 'Squeeze_Sell': 'squeeze_s',
    'Thermal_Buy': 'therm_b', 'Thermal_Sell': 'therm_s', 'Climax_Buy': 'climax_b', 'Climax_Sell': 'climax_s',
    'Lock_Buy': 'lock_b', 'Lock_Sell': 'lock_s', 'Defcon_Buy': 'defcon_b', 'Defcon_Sell': 'defcon_s',
    'Jugg_Buy': 'jugg_b', 'Jugg_Sell': 'jugg_s', 'Trinity_Buy': 'trinity_b', 'Trinity_Sell': 'trinity_s',
    'Lev_Buy': 'lev_b', 'Lev_Sell': 'lev_s', 'Commander_Buy': 'commander_b', 'Commander_Sell': 'commander_s',
    'Q_Pink_Whale_Buy': 'r_Pink_Whale_Buy', 'Q_Lock_Bounce': 'r_Lock_Bounce', 'Q_Lock_Break': 'r_Lock_Break',
    'Q_Neon_Up': 'r_Neon_Up', 'Q_Defcon_Buy': 'r_Defcon_Buy', 'Q_Therm_Bounce': 'r_Therm_Bounce',
    'Q_Therm_Vacuum': 'r_Therm_Vacuum', 'Q_Nuclear_Buy': 'r_Nuclear_Buy', 'Q_Early_Buy': 'r_Early_Buy',
    'Q_Rebound_Buy': 'r_Rebound_Buy', 'Q_Lock_Reject': 'r_Lock_Reject', 'Q_Lock_Breakd': 'r_Lock_Breakd',
    'Q_Neon_Dn': 'r_Neon_Dn', 'Q_Defcon_Sell': 'r_Defcon_Sell', 'Q_Therm_Wall_Sell': 'r_Therm_Wall_Sell',
    'Q_Therm_Panic_Sell': 'r_Therm_Panic_Sell', 'Q_Nuclear_Sell': 'r_Nuclear_Sell', 'Q_Early_Sell': 'r_Early_Sell'
}

# ==========================================
# 🧬 THE DNA VAULT
# ==========================================
for s_id in estrategias:
    if f'opt_status_{s_id}' not in st.session_state: st.session_state[f'opt_status_{s_id}'] = False
    if f'champion_{s_id}' not in st.session_state:
        if s_id == "ALL_FORCES":
            st.session_state[f'champion_{s_id}'] = {'b_team': ['Commander_Buy', 'Squeeze_Buy'], 's_team': ['Commander_Sell'], 'macro': "All-Weather", 'vol': "All-Weather", 'tp': 20.0, 'sl': 5.0, 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 4.0, 'reinv': 0.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0}
        elif s_id in ["GENESIS", "ROCKET", "QUADRIX"]:
            v = {'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 4.0, 'reinv': 0.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0}
            opts_b = quadrix_b if s_id == "QUADRIX" else base_b
            opts_s = quadrix_s if s_id == "QUADRIX" else base_s
            for r_idx in range(1, 5): v.update({f'r{r_idx}_b': [opts_b[0]], f'r{r_idx}_s': [opts_s[0]], f'r{r_idx}_tp': 20.0, f'r{r_idx}_sl': 5.0})
            st.session_state[f'champion_{s_id}'] = v
        else:
            st.session_state[f'champion_{s_id}'] = {'tp': 20.0, 'sl': 5.0, 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 4.0, 'reinv': 0.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0}

def save_champion(s_id, bp):
    if bp is None: return
    vault = st.session_state[f'champion_{s_id}']
    vault['fit'] = bp['fit']
    for k in bp.keys():
        if k in vault: vault[k] = bp[k]
    # Guardamos net y winrate para el leaderboard
    vault['net'] = bp.get('net', 0.0)
    vault['winrate'] = bp.get('winrate', 0.0)

def wipe_ui_cache():
    for key in list(st.session_state.keys()):
        if key.startswith("ui_"): del st.session_state[key]

# ==========================================
# 🌍 SIDEBAR E INFRAESTRUCTURA
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 OMNI-FORGE V116.0</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True): 
    st.cache_data.clear()
    for s in estrategias: 
        st.session_state[f'opt_status_{s}'] = False
        if f'champion_{s}' in st.session_state: del st.session_state[f'champion_{s}']
    wipe_ui_cache()
    gc.collect()
    st.rerun()

# 🛑 BOTÓN DE PÁNICO 🛑
st.sidebar.markdown("---")
if st.sidebar.button("🛑 ABORTAR OPTIMIZACIÓN", use_container_width=True):
    st.session_state['abort_opt'] = True
    st.rerun()

st.sidebar.markdown("---")
exchange_sel = st.sidebar.selectbox("🏦 Exchange", ["coinbase", "kucoin", "kraken", "binance"], index=0)
ticker = st.sidebar.text_input("Símbolo Exacto", value="SD/USDC")
utc_offset = st.sidebar.number_input("🌍 Zona Horaria", value=-5.0, step=0.5)
intervalos = {"1 Minuto": "1m", "5 Minutos": "5m", "15 Minutos": "15m", "30 Minutos": "30m", "1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d"}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=2) 
iv_download = intervalos[intervalo_sel]
hoy = datetime.today().date()
is_micro = iv_download in ["1m", "5m", "15m", "30m"]
start_date, end_date = st.sidebar.slider("📅 Scope Histórico", min_value=hoy - timedelta(days=45 if is_micro else 1500), max_value=hoy, value=(hoy - timedelta(days=45 if is_micro else 1500), hoy), format="YYYY-MM-DD")
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=1000.0, step=100.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center; color: lime;'>🤖 SUPERCOMPUTADORA</h3>", unsafe_allow_html=True)
global_epochs = st.sidebar.slider("Épocas de Evolución (x3000)", 1, 1000, 50)

@st.cache_data(ttl=3600, show_spinner="📡 Construyendo Geometría Fractal & WaveTrend (V116)...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, offset):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        all_ohlcv, current_ts, error_count = [], start_ts, 0
        
        while current_ts < end_ts:
            try: 
                ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=1000)
                error_count = 0 
            except Exception as e: 
                error_count += 1
                if error_count >= 3: return pd.DataFrame(), f"❌ ERROR: Exchange rechazó símbolo. {e}"
                time.sleep(1); continue
            if not ohlcv or len(ohlcv) == 0: break
            if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                ohlcv = [c for c in ohlcv if c[0] > all_ohlcv[-1][0]]
                if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            if len(all_ohlcv) > 100000: break
            
        if not all_ohlcv: return pd.DataFrame(), f"El Exchange devolvió 0 velas."
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index + timedelta(hours=offset)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < 50: return pd.DataFrame(), f"❌ Solo {len(df)} velas."
            
        df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1, adjust=False).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, min_periods=1, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, min_periods=1, adjust=False).mean()
        df['Vol_MA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Vol_MA_100'] = df['Volume'].rolling(window=100, min_periods=1).mean()
        df['RVol'] = df['Volume'] / df['Vol_MA_100'].replace(0, 1)
        df['High_Vol'] = df['Volume'] > df['Vol_MA_20']
        
        tr = df[['High', 'Low']].max(axis=1) - df[['High', 'Low']].min(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, min_periods=1, adjust=False).mean().fillna(df['High']-df['Low']).replace(0, 0.001)
        df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50.0)
        df['RSI_MA'] = df['RSI'].rolling(14, min_periods=1).mean()
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0].fillna(0.0)
        
        # WAVETREND (QUADRIX & COMMANDER)
        ap = (df['High'] + df['Low'] + df['Close']) / 3.0
        esa = ap.ewm(span=10, adjust=False).mean()
        d_wt = (ap - esa).abs().ewm(span=10, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d_wt.replace(0, 1))
        df['WT1'] = ci.ewm(span=21, adjust=False).mean()
        df['WT2'] = df['WT1'].rolling(4, min_periods=1).mean()
        
        df['Basis'] = df['Close'].rolling(20, min_periods=1).mean()
        dev = df['Close'].rolling(20, min_periods=1).std(ddof=0).replace(0, 1) 
        df['BBU'] = df['Basis'] + (2.0 * dev)
        df['BBL'] = df['Basis'] - (2.0 * dev)
        df['BB_Width'] = (df['BBU'] - df['BBL']) / df['Basis'].replace(0, 1)
        df['BB_Width_Avg'] = df['BB_Width'].rolling(20, min_periods=1).mean()
        df['BB_Delta'] = df['BB_Width'] - df['BB_Width'].shift(1).fillna(0)
        df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10, min_periods=1).mean()
        
        kc_basis = df['Close'].rolling(20, min_periods=1).mean()
        df['KC_Upper'] = kc_basis + (df['ATR'] * 1.5)
        df['KC_Lower'] = kc_basis - (df['ATR'] * 1.5)
        df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
        
        df['Z_Score'] = (df['Close'] - df['Basis']) / dev.replace(0, 1)
        df['RSI_BB_Basis'] = df['RSI'].rolling(14, min_periods=1).mean()
        df['RSI_BB_Dev'] = df['RSI'].rolling(14, min_periods=1).std(ddof=0).replace(0,1) * 2.0
        
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
        
        df['RSI_Cross_Up'] = (df['RSI'] > df['RSI_MA']) & (df['RSI'].shift(1) <= df['RSI_MA'].shift(1))
        df['RSI_Cross_Dn'] = (df['RSI'] < df['RSI_MA']) & (df['RSI'].shift(1) >= df['RSI_MA'].shift(1))
        df['Macro_Bull'] = df['Close'] >= df['EMA_200']
        
        # PING PONG SLOPE
        x = np.arange(5)
        df['PP_Slope'] = df['Close'].rolling(5).apply(lambda y: np.polyfit(x, y, 1)[0] if len(y.dropna()) == 5 else np.nan, raw=True)

        c_val = df['Close'].values
        df['dist_sup'] = (c_val - df['Target_Lock_Sup'].values) / c_val * 100
        df['dist_res'] = (df['Target_Lock_Res'].values - c_val) / c_val * 100
        
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
    except Exception as e: return pd.DataFrame(), f"❌ ERROR FATAL GENERAL: {str(e)}"

df_global, status_api = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, utc_offset)

if not df_global.empty:
    dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)
    st.sidebar.success(f"📥 MATRIZ LISTA: {len(df_global)} velas ({dias_reales} días).")
else:
    st.error(status_api)
    st.stop()

# ==========================================
# 🏆 SCOREBOARD (LEADERBOARD) UNIVERSAL
# ==========================================
st.markdown("<h3 style='text-align: center; color: #FFD700;'>🏆 SALÓN DE LA FAMA (Leaderboard)</h3>", unsafe_allow_html=True)
leaderboard_data = []
for s in estrategias:
    v = st.session_state.get(f'champion_{s}', {})
    fit = v.get('fit', -float('inf'))
    if fit != -float('inf'):
        leaderboard_data.append({"Estrategia": s, "Puntaje": f"{fit:,.0f}", "Rentabilidad": f"{v.get('net', 0)/capital_inicial*100:.2f}%", "WinRate": f"{v.get('winrate', 0):.1f}%"})
if leaderboard_data:
    st.table(pd.DataFrame(leaderboard_data))
else:
    st.info("La bóveda está vacía. Inicie una Forja individual o Global para registrar a los campeones.")
st.markdown("---")

# ==========================================
# 🔥 PURE NUMPY BACKEND (V116 - COMMANDER & ULTRA) 🔥
# ==========================================
a_c = df_global['Close'].values
a_o = df_global['Open'].values
a_h = df_global['High'].values
a_l = df_global['Low'].values
a_rsi = df_global['RSI'].values
a_rsi_ma = df_global['RSI_MA'].values
a_adx = df_global['ADX'].values
a_bbl = df_global['BBL'].values
a_bbu = df_global['BBU'].values
a_bw = df_global['BB_Width'].values
a_bwa_s1 = npshift(df_global['BB_Width_Avg'].values, 1, -1.0)
a_wt1 = df_global['WT1'].values
a_wt2 = df_global['WT2'].values
a_ema50 = df_global['EMA_50'].values
a_ema200 = df_global['EMA_200'].values
a_atr = df_global['ATR'].values
a_rvol = df_global['RVol'].values
a_hvol = df_global['High_Vol'].values
a_vv = df_global['Vela_Verde'].values
a_vr = df_global['Vela_Roja'].values
a_rcu = df_global['RSI_Cross_Up'].values
a_rcd = df_global['RSI_Cross_Dn'].values
a_sqz_on = df_global['Squeeze_On'].values
a_bb_delta = df_global['BB_Delta'].values
a_bb_delta_avg = df_global['BB_Delta_Avg'].values
a_zscore = df_global['Z_Score'].values
a_rsi_bb_b = df_global['RSI_BB_Basis'].values
a_rsi_bb_d = df_global['RSI_BB_Dev'].values
a_lw = df_global['lower_wick'].values
a_bs = df_global['body_size'].values
a_tsup = df_global['Target_Lock_Sup'].values
a_tres = df_global['Target_Lock_Res'].values
a_dsup = df_global['dist_sup'].values
a_dres = df_global['dist_res'].values
a_cw = df_global['ceil_w'].values
a_fw = df_global['floor_w'].values
a_mb = df_global['Macro_Bull'].values
a_fk = df_global['is_falling_knife'].values
a_pp_slope = df_global['PP_Slope'].values

a_c_s1 = npshift(a_c, 1, 0.0)
a_o_s1 = npshift(a_o, 1, 0.0)
a_l_s1 = npshift(a_l, 1, 0.0)
a_l_s5 = npshift(a_l, 5, 0.0)
a_rsi_s1 = npshift(a_rsi, 1, 50.0)
a_rsi_s5 = npshift(a_rsi, 5, 50.0)
a_sqz_s1 = npshift_bool(a_sqz_on, 1, False)
a_wt1_s1 = npshift(a_wt1, 1, 0.0)
a_wt2_s1 = npshift(a_wt2, 1, 0.0)
a_pp_slope_s1 = npshift(a_pp_slope, 1, 0.0)

def calcular_señales_numpy(hitbox, therm_w, adx_th, whale_f):
    n_len = len(a_c)
    s_dict = {}
    
    trinity_safe = a_mb & ~a_fk
    neon_up = a_sqz_on & (a_c >= a_bbu * 0.999) & a_vv
    neon_dn = a_sqz_on & (a_c <= a_bbl * 1.001) & a_vr
    
    defcon_level = np.full(n_len, 5)
    m4 = neon_up | neon_dn
    defcon_level[m4] = 4
    m3 = m4 & (a_bb_delta > 0)
    defcon_level[m3] = 3
    m2 = m3 & (a_bb_delta > a_bb_delta_avg) & (a_adx > adx_th)
    defcon_level[m2] = 2
    m1 = m2 & (a_bb_delta > a_bb_delta_avg * 1.5) & (a_adx > adx_th + 5) & (a_rvol > 1.2)
    defcon_level[m1] = 1

    cond_defcon_buy = (defcon_level <= 2) & neon_up
    cond_defcon_sell = (defcon_level <= 2) & neon_dn

    is_abyss = a_fw == 0
    is_hard_wall = a_cw >= therm_w
    cond_therm_buy_bounce = (a_fw >= therm_w) & a_rcu & ~is_hard_wall
    cond_therm_buy_vacuum = (a_cw <= 3) & neon_up & ~is_abyss
    cond_therm_sell_wall = (a_cw >= therm_w) & a_rcd
    cond_therm_sell_panic = is_abyss & a_vr

    tol = a_atr * 0.5
    is_grav_sup = a_dsup < hitbox
    is_grav_res = a_dres < hitbox
    
    cross_up_res = (a_c > a_tres) & (a_c_s1 <= npshift(a_tres, 1, 0))
    cross_dn_sup = (a_c < a_tsup) & (a_c_s1 >= npshift(a_tsup, 1, 0))

    cond_lock_buy_bounce = is_grav_sup & (a_l <= a_tsup + tol) & (a_c > a_tsup) & a_vv
    cond_lock_buy_break = is_grav_res & cross_up_res & a_hvol & a_vv
    cond_lock_sell_reject = is_grav_res & (a_h >= a_tres - tol) & (a_c < a_tres) & a_vr
    cond_lock_sell_breakd = is_grav_sup & cross_dn_sup & a_vr

    flash_vol = (a_rvol > whale_f * 0.8) & (np.abs(a_c - a_o) > a_atr * 0.3)
    whale_buy = flash_vol & a_vv
    whale_sell = flash_vol & a_vr
    whale_memory = whale_buy | npshift_bool(whale_buy, 1) | npshift_bool(whale_buy, 2) | whale_sell | npshift_bool(whale_sell, 1) | npshift_bool(whale_sell, 2)
    is_whale_icon = whale_buy & ~npshift_bool(whale_buy, 1)

    rsi_vel = a_rsi - a_rsi_s1
    pre_pump = ((a_h > a_bbu) | (rsi_vel > 5)) & flash_vol & a_vv
    pump_memory = pre_pump | npshift_bool(pre_pump, 1) | npshift_bool(pre_pump, 2)
    pre_dump = ((a_l < a_bbl) | (rsi_vel < -5)) & flash_vol & a_vr
    dump_memory = pre_dump | npshift_bool(pre_dump, 1) | npshift_bool(pre_dump, 2)

    retro_peak = (a_rsi < 30) & (a_c < a_bbl)
    retro_peak_sell = (a_rsi > 70) & (a_c > a_bbu)
    k_break_up = (a_rsi > (a_rsi_bb_b + a_rsi_bb_d)) & (a_rsi_s1 <= npshift(a_rsi_bb_b + a_rsi_bb_d, 1))
    support_buy = is_grav_sup & a_rcu
    support_sell = is_grav_res & a_rcd
    div_bull = (a_l_s1 < a_l_s5) & (a_rsi_s1 > a_rsi_s5) & (a_rsi < 35)
    div_bear = (a_h_s1 > npshift(a_h, 5, 0)) & (a_rsi_s1 < a_rsi_s5) & (a_rsi > 65)

    buy_score = np.zeros(n_len)
    base_mask = retro_peak | k_break_up | support_buy | div_bull
    buy_score = np.where(base_mask & retro_peak, 50.0, np.where(base_mask & ~retro_peak, 30.0, buy_score))
    buy_score += np.where(is_grav_sup, 25.0, 0.0)
    buy_score += np.where(whale_memory, 20.0, 0.0)
    buy_score += np.where(pump_memory, 15.0, 0.0)
    buy_score += np.where(div_bull, 15.0, 0.0)
    buy_score += np.where(k_break_up & ~retro_peak, 15.0, 0.0)
    buy_score += np.where(a_zscore < -2.0, 15.0, 0.0)
    buy_score = np.where(buy_score > 99, 99.0, buy_score)

    sell_score = np.zeros(n_len)
    base_mask_s = retro_peak_sell | a_rcd | support_sell | div_bear
    sell_score = np.where(base_mask_s & retro_peak_sell, 50.0, np.where(base_mask_s & ~retro_peak_sell, 30.0, sell_score))
    sell_score += np.where(is_grav_res, 25.0, 0.0)
    sell_score += np.where(whale_memory, 20.0, 0.0)
    sell_score += np.where(dump_memory, 15.0, 0.0)
    sell_score += np.where(div_bear, 15.0, 0.0)
    sell_score += np.where(a_rcd & ~retro_peak_sell, 15.0, 0.0)
    sell_score += np.where(a_zscore > 2.0, 15.0, 0.0)
    sell_score = np.where(sell_score > 99, 99.0, sell_score)

    is_magenta = (buy_score >= 70) | retro_peak
    is_magenta_sell = (sell_score >= 70) | retro_peak_sell
    cond_pink_whale_buy = is_magenta & is_whale_icon

    wt_cross_up = (a_wt1 > a_wt2) & (a_wt1_s1 <= a_wt2_s1)
    wt_cross_dn = (a_wt1 < a_wt2) & (a_wt1_s1 >= a_wt2_s1)
    wt_oversold = a_wt1 < -60
    wt_overbought = a_wt1 > 60

    s_dict['Q_Pink_Whale_Buy'] = cond_pink_whale_buy
    s_dict['Q_Lock_Bounce'] = cond_lock_buy_bounce
    s_dict['Q_Lock_Break'] = cond_lock_buy_break
    s_dict['Q_Neon_Up'] = neon_up
    s_dict['Q_Defcon_Buy'] = cond_defcon_buy
    s_dict['Q_Therm_Bounce'] = cond_therm_buy_bounce
    s_dict['Q_Therm_Vacuum'] = cond_therm_buy_vacuum
    s_dict['Q_Nuclear_Buy'] = is_magenta & (wt_oversold | wt_cross_up)
    s_dict['Q_Early_Buy'] = is_magenta
    s_dict['Q_Rebound_Buy'] = a_rcu & ~is_magenta

    s_dict['Q_Lock_Reject'] = cond_lock_sell_reject
    s_dict['Q_Lock_Breakd'] = cond_lock_sell_breakd
    s_dict['Q_Neon_Dn'] = neon_dn
    s_dict['Q_Defcon_Sell'] = cond_defcon_sell
    s_dict['Q_Therm_Wall_Sell'] = cond_therm_sell_wall
    s_dict['Q_Therm_Panic_Sell'] = cond_therm_sell_panic
    s_dict['Q_Nuclear_Sell'] = (a_rsi > 70) & (wt_overbought | wt_cross_dn)
    s_dict['Q_Early_Sell'] = (a_rsi > 70) & a_vr

    # JUGGERNAUT Y BÁSICOS
    s_dict['JUGGERNAUT_BUY_V356'] = (trinity_safe & (cond_defcon_buy | cond_therm_buy_bounce | cond_therm_buy_vacuum | cond_lock_buy_bounce | cond_lock_buy_break)) | cond_pink_whale_buy
    s_dict['JUGGERNAUT_SELL_V356'] = cond_defcon_sell | cond_therm_sell_wall | cond_therm_sell_panic | cond_lock_sell_reject | cond_lock_sell_breakd

    s_dict['Ping_Buy'] = (a_adx < adx_th) & (a_c < a_bbl) & a_vv
    s_dict['Ping_Sell'] = (a_c > a_bbu) | (a_rsi > 70)
    s_dict['Squeeze_Buy'] = (a_bw < a_bwa_s1) & (a_c > a_bbu) & a_vv & (a_rsi < 60)
    s_dict['Squeeze_Sell'] = (a_c < a_ema50)
    s_dict['Thermal_Buy'] = cond_therm_buy_bounce
    s_dict['Thermal_Sell'] = cond_therm_sell_wall
    s_dict['Climax_Buy'] = cond_pink_whale_buy
    s_dict['Climax_Sell'] = (a_rsi > 80)
    s_dict['Lock_Buy'] = cond_lock_buy_bounce
    s_dict['Lock_Sell'] = cond_lock_sell_reject
    s_dict['Defcon_Buy'] = cond_defcon_buy
    s_dict['Defcon_Sell'] = cond_defcon_sell
    s_dict['Jugg_Buy'] = a_mb & (a_c > a_ema50) & (a_c_s1 < npshift(a_ema50,1)) & a_vv & ~a_fk
    s_dict['Jugg_Sell'] = (a_c < a_ema50)
    s_dict['Trinity_Buy'] = a_mb & (a_rsi < 35) & a_vv & ~a_fk
    s_dict['Trinity_Sell'] = (a_rsi > 75) | (a_c < a_ema200)
    s_dict['Lev_Buy'] = (a_mb & a_rcu & (a_rsi < 45))
    s_dict['Lev_Sell'] = (a_c < a_ema200)
    s_dict['Commander_Buy'] = cond_pink_whale_buy | cond_therm_buy_bounce | cond_lock_buy_bounce
    s_dict['Commander_Sell'] = cond_therm_sell_wall | (a_c < a_ema50)

    # 🚀 ROCKET COMMANDER ULTRA LÓGICA 🚀
    matrix_active = is_grav_sup | (a_fw >= 3)
    final_wick_req = np.where(matrix_active, 0.15, np.where(a_adx < 40, 0.4, 0.5))
    final_vol_req = np.where(matrix_active, 1.2, np.where(a_adx < 40, 1.5, 1.8))
    wick_rej_buy = a_lw > (a_bs * final_wick_req)
    wick_rej_sell = a_upper_wick = (a_h - np.maximum(a_o, a_c)) > (a_bs * final_wick_req)
    vol_stop_chk = a_rvol > final_vol_req
    
    climax_buy_cmdr = is_magenta & (wick_rej_buy | vol_stop_chk) & (a_c > a_o)
    climax_sell_cmdr = is_magenta_sell & (wick_rej_sell | vol_stop_chk)
    ping_buy_cmdr = (a_pp_slope > 0) & (a_pp_slope_s1 <= 0) & matrix_active & (a_c > a_o)
    ping_sell_cmdr = (a_pp_slope < 0) & (a_pp_slope_s1 >= 0) & matrix_active
    
    s_dict['RC_Buy_Q1'] = climax_buy_cmdr | ping_buy_cmdr | s_dict['Trinity_Buy'] | s_dict['Jugg_Buy']
    s_dict['RC_Sell_Q1'] = ping_sell_cmdr | s_dict['Trinity_Sell'] | s_dict['Squeeze_Sell']
    s_dict['RC_Buy_Q2'] = s_dict['Thermal_Buy'] | climax_buy_cmdr | ping_buy_cmdr
    s_dict['RC_Sell_Q2'] = s_dict['Defcon_Sell'] | s_dict['Lock_Sell'] | s_dict['Squeeze_Sell']
    s_dict['RC_Buy_Q3'] = cond_pink_whale_buy | (s_dict['Thermal_Buy'] & aegis_safe) | s_dict['Defcon_Buy'] | s_dict['Lev_Buy']
    s_dict['RC_Sell_Q3'] = climax_sell_cmdr | ping_sell_cmdr | s_dict['Lev_Sell']
    s_dict['RC_Buy_Q4'] = ping_buy_cmdr | s_dict['Defcon_Buy'] | s_dict['Lock_Buy']
    s_dict['RC_Sell_Q4'] = s_dict['Defcon_Sell'] | s_dict['Lev_Sell']
    
    regime = np.where(a_mb & (a_adx >= adx_th), 1, np.where(a_mb & (a_adx < adx_th), 2, np.where(~a_mb & (a_adx >= adx_th), 3, 4)))
    return s_dict, regime

@njit(fastmath=True)
def simular_crecimiento_exponencial(h_arr, l_arr, c_arr, o_arr, b_c, s_c, t_arr, sl_arr, cap_ini, com_pct, reinvest_pct):
    cap_act = cap_ini
    divs, en_pos, p_ent, tp_act, sl_act, pos_size, invest_amt = 0.0, False, 0.0, 0.0, 0.0, 0.0, 0.0
    g_profit, g_loss, num_trades, max_dd, peak = 0.0, 0.0, 0, 0.0, cap_ini
    for i in range(len(h_arr)):
        if en_pos:
            tp_p = p_ent * (1.0 + tp_act/100.0)
            sl_p = p_ent * (1.0 - sl_act/100.0)
            if l_arr[i] <= sl_p:
                gross = pos_size * (1.0 - sl_act/100.0)
                net = gross - (gross * com_pct)
                profit = net - invest_amt
                if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                else: cap_act += profit
                g_loss += abs(profit); num_trades += 1; en_pos = False
            elif h_arr[i] >= tp_p:
                gross = pos_size * (1.0 + tp_act/100.0)
                net = gross - (gross * com_pct)
                profit = net - invest_amt
                if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                else: cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1; en_pos = False
            elif s_c[i]:
                ret = (c_arr[i] - p_ent) / p_ent
                gross = pos_size * (1.0 + ret)
                net = gross - (gross * com_pct)
                profit = net - invest_amt
                if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                else: cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1; en_pos = False
            total_equity = cap_act + divs
            if total_equity > peak: peak = total_equity
            if peak > 0: dd = (peak - total_equity) / peak * 100.0; max_dd = max(max_dd, dd)
            if cap_act <= 0: break
        if not en_pos and b_c[i] and i+1 < len(h_arr):
            invest_amt = cap_act if reinvest_pct == 100.0 else cap_ini
            if invest_amt > cap_act: invest_amt = cap_act 
            comm_in = invest_amt * com_pct
            pos_size = invest_amt - comm_in 
            p_ent = o_arr[i+1]; tp_act = t_arr[i]; sl_act = sl_arr[i]; en_pos = True
    return (cap_act + divs) - cap_ini, g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0), num_trades, max_dd

def simular_visual(df_sim, cap_ini, reinvest, com_pct):
    registro_trades = []
    n = len(df_sim)
    curva = np.full(n, cap_ini, dtype=float)
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl_arr = df_sim['Active_TP'].values, df_sim['Active_SL'].values
    f_arr = df_sim.index
    en_pos, p_ent, tp_act, sl_act, cap_act, divs, pos_size, invest_amt, total_comms = False, 0.0, 0.0, 0.0, cap_ini, 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        cierra = False
        if en_pos:
            tp_p = p_ent * (1 + tp_act/100)
            sl_p = p_ent * (1 - sl_act/100)
            if l_arr[i] <= sl_p:
                gross = pos_size * (1 - sl_act/100)
                comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                if profit > 0: reinv_amt = profit * (reinvest/100); divs += (profit - reinv_amt); cap_act += reinv_amt
                else: cap_act += profit
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'SL', 'Precio': sl_p, 'Ganancia_$': profit}); en_pos, cierra = False, True
            elif h_arr[i] >= tp_p:
                gross = pos_size * (1 + tp_act/100)
                comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                if profit > 0: reinv_amt = profit * (reinvest/100); divs += (profit - reinv_amt); cap_act += reinv_amt
                else: cap_act += profit
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'TP', 'Precio': tp_p, 'Ganancia_$': profit}); en_pos, cierra = False, True
            elif sell_arr[i]:
                ret = (c_arr[i] - p_ent) / p_ent
                gross = pos_size * (1 + ret)
                comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                if profit > 0: reinv_amt = profit * (reinvest/100); divs += (profit - reinv_amt); cap_act += reinv_amt
                else: cap_act += profit
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'DYN_WIN' if profit>0 else 'DYN_LOSS', 'Precio': c_arr[i], 'Ganancia_$': profit}); en_pos, cierra = False, True
        if not en_pos and not cierra and buy_arr[i] and i+1 < n and cap_act > 0:
            invest_amt = cap_act if reinvest == 100 else cap_ini
            if invest_amt > cap_act: invest_amt = cap_act
            comm_in = invest_amt * com_pct; total_comms += comm_in; pos_size = invest_amt - comm_in
            p_ent = o_arr[i+1]; tp_act = tp_arr[i]; sl_act = sl_arr[i]; en_pos = True
            registro_trades.append({'Fecha': f_arr[i+1], 'Tipo': 'ENTRY', 'Precio': p_ent, 'Ganancia_$': 0})
        if en_pos and cap_act > 0: curva[i] = cap_act + (pos_size * ((c_arr[i] - p_ent) / p_ent)) + divs
        else: curva[i] = cap_act + divs
    return curva.tolist(), divs, cap_act, registro_trades, en_pos, total_comms

# 🧠 RUTINA DE DEEP MINE (V116.0 - OMNI-FORGE) 🧠
def optimizar_ia_tracker(s_id, cap_ini, com_pct, reinv_q, target_ado, dias_reales, buy_hold_money, epochs=1, cur_fit=-float('inf')):
    st.session_state['abort_opt'] = False
    best_fit, best_net_live, best_pf_live, best_nt_live, bp = cur_fit, 0.0, 0.0, 0, None
    tp_min, tp_max = 0.5, 40.0 
    iters = 3000 * epochs
    chunks = min(iters, 20)
    chunk_size = iters // chunks
    start_time = time.time()
    n_len = len(a_c)
    target_nt = max(1.0, target_ado * dias_reales)

    f_buy = np.empty(n_len, dtype=bool)
    f_sell = np.empty(n_len, dtype=bool)
    f_tp = np.empty(n_len, dtype=np.float64)
    f_sl = np.empty(n_len, dtype=np.float64)
    macro_mask = np.empty(n_len, dtype=bool)
    vol_mask = np.empty(n_len, dtype=bool)

    for c in range(chunks):
        if st.session_state.get('abort_opt', False): 
            st.warning("🛑 OPTIMIZACIÓN ABORTADA. Guardando el mejor hallazgo hasta el momento...")
            break

        for _ in range(chunk_size): 
            f_buy.fill(False)
            f_sell.fill(False)
            rtp = round(random.uniform(tp_min, tp_max), 1)
            rsl = round(random.uniform(0.5, 20.0), 1)
            r_hitbox = round(random.uniform(0.5, 3.0), 1)   
            r_therm  = float(random.randint(3, 8))          
            r_adx    = float(random.randint(15, 35))        
            r_whale  = round(random.uniform(1.5, 4.0), 1)   
            
            s_dict, regime_arr = calcular_señales_numpy(r_hitbox, r_therm, r_adx, r_whale)
            
            if s_id in ["ROCKET_ULTRA", "ROCKET_COMMANDER"]:
                f_buy[:] = s_dict['RC_Buy_Q1'] & (regime_arr == 1) | s_dict['RC_Buy_Q2'] & (regime_arr == 2) | s_dict['RC_Buy_Q3'] & (regime_arr == 3) | s_dict['RC_Buy_Q4'] & (regime_arr == 4)
                f_sell[:] = s_dict['RC_Sell_Q1'] & (regime_arr == 1) | s_dict['RC_Sell_Q2'] & (regime_arr == 2) | s_dict['RC_Sell_Q3'] & (regime_arr == 3) | s_dict['RC_Sell_Q4'] & (regime_arr == 4)
                f_tp.fill(rtp); f_sl.fill(rsl)
                
            elif s_id == "JUGGERNAUT":
                f_buy[:] = s_dict['JUGGERNAUT_BUY_V356']
                f_sell[:] = s_dict['JUGGERNAUT_SELL_V356']
                f_tp.fill(rtp); f_sl.fill(rsl)
                
            elif s_id == "ALL_FORCES":
                dna_b_team = random.sample(base_b, random.randint(1, len(base_b)))
                dna_s_team = random.sample(base_s, random.randint(1, len(base_s)))
                dna_macro = "All-Weather" if random.random() < 0.6 else random.choice(["Bull Only (Precio > EMA 200)", "Bear Only (Precio < EMA 200)"])
                dna_vol = "All-Weather" if random.random() < 0.6 else random.choice(["Trend (ADX Alto)", "Range (ADX Bajo)"])
                if dna_macro == "Bull Only (Precio > EMA 200)": macro_mask[:] = a_mb
                elif dna_macro == "Bear Only (Precio < EMA 200)": macro_mask[:] = ~a_mb
                else: macro_mask.fill(True)
                if dna_vol == "Trend (ADX Alto)": vol_mask[:] = (a_adx >= r_adx)
                elif dna_vol == "Range (ADX Bajo)": vol_mask[:] = (a_adx < r_adx)
                else: vol_mask.fill(True)
                for r in dna_b_team: f_buy |= s_dict[r]
                f_buy &= macro_mask; f_buy &= vol_mask
                for r in dna_s_team: f_sell |= s_dict[r]
                f_tp.fill(rtp); f_sl.fill(rsl)
                
            elif s_id in ["GENESIS", "ROCKET", "QUADRIX"]:
                opts_b = quadrix_b if s_id == "QUADRIX" else rocket_b if s_id == "ROCKET" else base_b
                opts_s = quadrix_s if s_id == "QUADRIX" else rocket_s if s_id == "ROCKET" else base_s
                dna_b = [random.sample(opts_b, random.randint(1, 2)) for _ in range(4)]
                dna_s = [random.sample(opts_s, random.randint(1, 2)) for _ in range(4)]
                dna_tp = [random.uniform(tp_min, tp_max) for _ in range(4)]
                dna_sl = [random.uniform(0.5, 20.0) for _ in range(4)]
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
                try: f_buy[:], f_sell[:] = s_dict[f"{s_id.split('_')[0].capitalize()}_Buy"], s_dict[f"{s_id.split('_')[0].capitalize()}_Sell"]
                except:
                    if s_id == "TARGET_LOCK": f_buy[:], f_sell[:] = s_dict["Lock_Buy"], s_dict["Lock_Sell"]
                    elif s_id == "NEON_SQUEEZE": f_buy[:], f_sell[:] = s_dict["Squeeze_Buy"], s_dict["Squeeze_Sell"]
                    elif s_id == "PINK_CLIMAX": f_buy[:], f_sell[:] = s_dict["Climax_Buy"], s_dict["Climax_Sell"]
                f_tp.fill(rtp); f_sl.fill(rsl)

            net, pf, nt, mdd = simular_crecimiento_exponencial(a_h, a_l, a_c, a_o, f_buy, f_sell, f_tp, f_sl, float(cap_ini), float(com_pct), float(reinv_q))
            alpha_money = net - buy_hold_money
            
            if nt >= 1: 
                if net > 0: 
                    ado_ratio = float(nt) / target_nt
                    trade_penalty = ado_ratio ** 3 if ado_ratio < 0.3 else ado_ratio ** 1.5 if ado_ratio < 0.8 else np.sqrt(ado_ratio) 
                    fit = (net ** 1.5) * (pf ** 0.5) * trade_penalty / ((mdd ** 0.5) + 1.0)
                    if alpha_money > 0: fit *= 1.5 
                else: 
                    fit = net * ((mdd ** 0.5) + 1.0) / (pf + 0.001)
            else:
                fit = -999999.0 
                
            if fit > best_fit:
                best_fit = fit
                best_net_live, best_pf_live, best_nt_live = net, pf, nt
                winrate_live = 0.0 # Placeholder rápido, se calcula real en UI
                
                if s_id in ["ROCKET_ULTRA", "ROCKET_COMMANDER"]:
                    bp = {'tp': rtp, 'sl': rsl, 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'fit': fit, 'net': net, 'winrate': winrate_live}
                elif s_id == "ALL_FORCES":
                    bp = {'b_team': dna_b_team, 's_team': dna_s_team, 'macro': dna_macro, 'vol': dna_vol, 'tp': rtp, 'sl': rsl, 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'fit': fit, 'net': net, 'winrate': winrate_live}
                elif s_id in ["GENESIS", "ROCKET", "QUADRIX"]:
                    bp = {'r1_b': dna_b[0], 'r1_s': dna_s[0], 'r1_tp': dna_tp[0], 'r1_sl': dna_sl[0], 'r2_b': dna_b[1], 'r2_s': dna_s[1], 'r2_tp': dna_tp[1], 'r2_sl': dna_sl[1], 'r3_b': dna_b[2], 'r3_s': dna_s[2], 'r3_tp': dna_tp[2], 'r3_sl': dna_sl[2], 'r4_b': dna_b[3], 'r4_s': dna_s[3], 'r4_tp': dna_tp[3], 'r4_sl': dna_sl[3], 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'fit': fit, 'net': net, 'winrate': winrate_live}
                else:
                    bp = {'tp': rtp, 'sl': rsl, 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'fit': fit, 'net': net, 'winrate': winrate_live}
                st.session_state[f'temp_bp_{s_id}'] = bp 
        
        elapsed = time.time() - start_time
        pct_done = int(((c + 1) / chunks) * 100)
        combos = (c + 1) * chunk_size
        eta = (elapsed / (c + 1)) * (chunks - c - 1)
        
        ph_holograma.markdown(f"""
        <style>
        .loader-container {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.95); padding: 35px; border-radius: 20px; border: 2px solid #FF00FF; box-shadow: 0 0 50px #FF00FF;}}
        .rocket {{ font-size: 8rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 20px #FF00FF); }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .prog-text {{ color: #FF00FF; font-size: 1.8rem; font-weight: bold; margin-top: 15px; text-shadow: 0 0 5px #FF00FF;}}
        .hud-text {{ color: lime; font-size: 1.3rem; margin-top: 8px; font-family: monospace; }}
        </style>
        <div class="loader-container">
            <div class="rocket">🚀</div>
            <div class="prog-text">OMNI-FORGE V116: {s_id}</div>
            <div class="hud-text" style="color: white;">Progreso: {pct_done}%</div>
            <div class="hud-text" style="color: white;">Combos: {combos:,}</div>
            <div class="hud-text" style="color: #00FF00; font-weight: bold; font-size: 1.5rem; margin-top: 15px;">🏆 Hallazgo: ${best_net_live:.2f} | PF: {best_pf_live:.1f}x | Trds: {best_nt_live}</div>
            <div class="hud-text" style="color: yellow; margin-top: 15px;">ETA: {eta:.1f} segs</div>
        </div>
        """, unsafe_allow_html=True)
        
    return bp if bp else st.session_state.get(f'temp_bp_{s_id}', None)

def run_backtest_eval(s_id, cap_ini, com_pct):
    vault = st.session_state[f'champion_{s_id}']
    s_dict, regime_arr = calcular_señales_numpy(vault['hitbox'], vault['therm_w'], vault['adx_th'], vault['whale_f'])
    n_len = len(a_c)
    f_tp = np.full(n_len, float(vault.get('tp', 0.0)))
    f_sl = np.full(n_len, float(vault.get('sl', 0.0)))

    if s_id in ["ROCKET_ULTRA", "ROCKET_COMMANDER"]:
        f_buy, f_sell = np.zeros(n_len, dtype=bool), np.zeros(n_len, dtype=bool)
        f_buy[:] = s_dict['RC_Buy_Q1'] & (regime_arr == 1) | s_dict['RC_Buy_Q2'] & (regime_arr == 2) | s_dict['RC_Buy_Q3'] & (regime_arr == 3) | s_dict['RC_Buy_Q4'] & (regime_arr == 4)
        f_sell[:] = s_dict['RC_Sell_Q1'] & (regime_arr == 1) | s_dict['RC_Sell_Q2'] & (regime_arr == 2) | s_dict['RC_Sell_Q3'] & (regime_arr == 3) | s_dict['RC_Sell_Q4'] & (regime_arr == 4)
    elif s_id == "JUGGERNAUT":
        f_buy, f_sell = s_dict['JUGGERNAUT_BUY_V356'], s_dict['JUGGERNAUT_SELL_V356']
    elif s_id == "ALL_FORCES":
        f_buy, f_sell = np.zeros(n_len, dtype=bool), np.zeros(n_len, dtype=bool)
        m_mask = np.ones(n_len, dtype=bool)
        if vault['macro'] == "Bull Only (Precio > EMA 200)": m_mask = a_mb
        elif vault['macro'] == "Bear Only (Precio < EMA 200)": m_mask = ~a_mb
        v_mask = np.ones(n_len, dtype=bool)
        if vault['vol'] == "Trend (ADX Alto)": v_mask = a_adx >= vault['adx_th']
        elif vault['vol'] == "Range (ADX Bajo)": v_mask = a_adx < vault['adx_th']
        for r in vault['b_team']: f_buy |= s_dict[r]
        f_buy &= (m_mask & v_mask)
        for r in vault['s_team']: f_sell |= s_dict[r]
    elif s_id in ["GENESIS", "ROCKET", "QUADRIX"]:
        f_buy, f_sell = np.zeros(n_len, dtype=bool), np.zeros(n_len, dtype=bool)
        f_tp, f_sl = np.zeros(n_len, dtype=np.float64), np.zeros(n_len, dtype=np.float64)
        for idx_q in range(1, 5):
            mask = (regime_arr == idx_q)
            r_b_cond = np.zeros(n_len, dtype=bool)
            for r in vault[f'r{idx_q}_b']: r_b_cond |= s_dict[r]
            f_buy[mask] = r_b_cond[mask]
            r_s_cond = np.zeros(n_len, dtype=bool)
            for r in vault[f'r{idx_q}_s']: r_s_cond |= s_dict[r]
            f_sell[mask] = r_s_cond[mask]
            f_tp[mask] = vault[f'r{idx_q}_tp']
            f_sl[mask] = vault[f'r{idx_q}_sl']
    else:
        try: f_buy, f_sell = s_dict[f"{s_id.split('_')[0].capitalize()}_Buy"], s_dict[f"{s_id.split('_')[0].capitalize()}_Sell"]
        except:
            if s_id == "TARGET_LOCK": f_buy, f_sell = s_dict["Lock_Buy"], s_dict["Lock_Sell"]
            elif s_id == "NEON_SQUEEZE": f_buy, f_sell = s_dict["Squeeze_Buy"], s_dict["Squeeze_Sell"]
            elif s_id == "PINK_CLIMAX": f_buy, f_sell = s_dict["Climax_Buy"], s_dict["Climax_Sell"]

    df_strat = df_global.copy()
    df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
    df_strat['Active_TP'], df_strat['Active_SL'] = f_tp, f_sl
    eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, cap_ini, float(vault['reinv']), com_pct)
    return df_strat, eq_curve, t_log, total_comms

def generar_pine_script(s_id, vault, sym, tf):
    if s_id in ["ROCKET_ULTRA", "ROCKET_COMMANDER"]:
        return f"""// This source code is subject to the terms of the Mozilla Public License 2.0
// © Valle_Architect_Lab | AUTO-GENERATED BY OMNI-FORGE V116

//@version=5
strategy("ROCKET {s_id.split('_')[1]} V60.2 - {sym}", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.25)

// ==========================================
// 🔗 CONEXIÓN WUNDERTRADING (WEBHOOKS JSON)
// ==========================================
wt_enter_long = input.text_area(defval='{{"action": "buy"}}', title="🟢 WT: Mensaje Enter Long")
wt_exit_long  = input.text_area(defval='{{"action": "sell"}}', title="🔴 WT: Mensaje Exit Long")

// ==========================================
// ⚙️ ADN OPTIMIZADO
// ==========================================
hitbox_pct   = {vault['hitbox']}
therm_wall   = {vault['therm_w']}
adx_trend    = {vault['adx_th']}
whale_factor = {vault['whale_f']}
base_tp = {vault['tp']}
base_sl = {vault['sl']}

// SENSORES
ema50 = ta.ema(close, 50), ema200 = ta.ema(close, 200), rsi_v = ta.rsi(close, 14)
atr_v = ta.atr(14), vol_ma = ta.sma(volume, 100), rvol = volume / (vol_ma == 0 ? 1 : vol_ma)
[di_p, di_m, adx_val] = ta.dmi(14, 14)

basis = ta.sma(close, 20), dev = 2.0 * ta.stdev(close, 20)
bbu = basis + dev, bbl = basis - dev
kc_basis = ta.sma(close, 20), kc_u = kc_basis + (atr_v * 1.5), kc_l = kc_basis - (atr_v * 1.5)
squeeze_on = (bbu < kc_u) and (bbl > kc_l)
bb_delta = (bbu - bbl) - nz((bbu[1] - bbl[1]), 0)
bb_delta_avg = ta.sma(bb_delta, 10)
pp_slope = ta.linreg(close, 5, 0) - ta.linreg(close, 5, 1)

pl100 = ta.lowest(low[1], 100), ph100 = ta.highest(high[1], 100)
dist_sup = (close - pl100) / close * 100
dist_res = (ph100 - close) / close * 100
is_grav_sup = dist_sup < hitbox_pct
is_grav_res = dist_res < hitbox_pct
is_falling_knife = (open[1] - close[1]) > (atr_v[1] * 1.5)

vela_verde = close > open, vela_roja = close < open
flash_vol = rvol > (whale_factor * 0.8) and math.abs(close-open) > (atr_v * 0.3)
is_magenta = (rsi_v < 30) or ta.crossover(rsi_v, ta.sma(rsi_v, 14))
is_magenta_sell = (rsi_v > 70) or ta.crossunder(rsi_v, ta.sma(rsi_v, 14))
cond_pink_whale_buy = is_magenta and flash_vol and vela_verde and not flash_vol[1]

// ARMAS COMANDANTE
matrix_active = is_grav_sup 
final_wick_req = matrix_active ? 0.15 : (adx_val < 40 ? 0.4 : 0.5)
final_vol_req  = matrix_active ? 1.2 : (adx_val < 40 ? 1.5 : 1.8) 
wick_rej_buy = (math.min(open, close) - low) > (math.abs(close-open) * final_wick_req)
wick_rej_sell = (high - math.max(open, close)) > (math.abs(close-open) * final_wick_req)
vol_stop_chk = rvol > final_vol_req

climax_buy_cmdr = is_magenta and (wick_rej_buy or vol_stop_chk) and vela_verde
climax_sell_cmdr = is_magenta_sell and (wick_rej_sell or vol_stop_chk)
ping_buy_cmdr = (pp_slope > 0) and nz(pp_slope[1] <= 0) and matrix_active and vela_verde
ping_sell_cmdr = (pp_slope < 0) and nz(pp_slope[1] >= 0) and matrix_active

// REGIMEN
macro_bull = close > ema200
int quad = 0
if macro_bull and adx_val >= adx_trend
    quad := 1
else if macro_bull and adx_val < adx_trend
    quad := 2
else if not macro_bull and adx_val >= adx_trend
    quad := 3
else
    quad := 4

bool final_buy = false, bool final_sell = false
if quad == 1
    final_buy := climax_buy_cmdr or ping_buy_cmdr, final_sell := ping_sell_cmdr
if quad == 2
    final_buy := climax_buy_cmdr or ping_buy_cmdr, final_sell := ping_sell_cmdr
if quad == 3
    final_buy := cond_pink_whale_buy, final_sell := climax_sell_cmdr
if quad == 4
    final_buy := ping_buy_cmdr, final_sell := ping_sell_cmdr

if final_buy and strategy.position_size == 0
    strategy.entry("ROCKET_LONG", strategy.long, alert_message=wt_enter_long)

if strategy.position_size > 0 
    if final_sell
        strategy.close("ROCKET_LONG", comment="Dyn_Exit", alert_message=wt_exit_long)
    entry_price = strategy.opentrades.entry_price(strategy.opentrades - 1)
    target_price = entry_price * (1 + (base_tp / 100))
    stop_price = entry_price * (1 - (base_sl / 100))
    strategy.exit("EXIT", "ROCKET_LONG", limit=target_price, stop=stop_price, alert_message=wt_exit_long)

plot(ema200, "EMA 200", color=macro_bull ? color.new(#00FF00, 60) : color.new(#FF0000, 60), linewidth=1)
plotchar(final_buy, title="COMPRA", char="🚀", location=location.belowbar, color=color.aqua, size=size.small)
"""
    if s_id == "JUGGERNAUT":
        return f"""// This source code is subject to the terms of the Mozilla Public License 2.0
// © Valle_Architect_Lab | AUTO-GENERATED BY OMNI-FORGE V116

//@version=5
strategy("VALLE ARCHITECT [JUGGERNAUT V356] - {sym}", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.25)

// ==========================================
// 🔗 CONEXIÓN WUNDERTRADING (WEBHOOKS JSON)
// ==========================================
wt_enter_long = input.text_area(defval='{{"action": "buy"}}', title="🟢 WT: Mensaje Enter Long")
wt_exit_long  = input.text_area(defval='{{"action": "sell"}}', title="🔴 WT: Mensaje Exit Long")

// ==========================================
// ⚙️ ADN OPTIMIZADO (Valores Cuánticos Hallados)
// ==========================================
hitbox_pct   = {vault['hitbox']}
therm_wall   = {vault['therm_w']}
adx_trend    = {vault['adx_th']}
whale_factor = {vault['whale_f']}
tp_pct = {vault['tp']}
sl_pct = {vault['sl']}

grp_shield = "🛡️ ESCUDO AEGIS"
use_macro_shield = input.bool(true, "Bloquear bajo EMA 200", group=grp_shield)
use_knife_shield = input.bool(true, "Bloquear Cuchillos Cayendo", group=grp_shield)

// MOTORES MATEMÁTICOS
atr_val = ta.atr(14), vol_ma = ta.sma(volume, 20), high_vol = volume > vol_ma
vol_ma_long = ta.sma(volume, 100), rvol = volume / (vol_ma_long == 0 ? 1 : vol_ma_long)
rsi_v = ta.rsi(close, 14), rsi_ma = ta.sma(rsi_v, 14) 
rsi_cross_up = ta.crossover(rsi_v, rsi_ma), rsi_cross_down = ta.crossunder(rsi_v, rsi_ma)
[di_p, di_m, adx_val] = ta.dmi(14, 14)
[bb_mid, bb_top, bb_bot] = ta.bb(close, 20, 2.0), [kc_m, kc_u, kc_l] = ta.kc(close, 20, 1.5)

pl_1 = ta.pivotlow(low, 30, 3), ph_1 = ta.pivothigh(high, 30, 3)
pl_2 = ta.pivotlow(low, 100, 5), ph_2 = ta.pivothigh(high, 100, 5)
pl_3 = ta.pivotlow(low, 300, 5), ph_3 = ta.pivothigh(high, 300, 5)

scan_range = atr_val * 2.0
floor_w = 0, ceil_w = 0
floor_w += (pl_1 < close and pl_1 >= close - scan_range) ? 1 : 0
floor_w += (ph_1 < close and ph_1 >= close - scan_range) ? 1 : 0
floor_w += (pl_2 < close and pl_2 >= close - scan_range) ? 3 : 0
floor_w += (ph_2 < close and ph_2 >= close - scan_range) ? 3 : 0
floor_w += (pl_3 < close and pl_3 >= close - scan_range) ? 5 : 0
floor_w += (ph_3 < close and ph_3 >= close - scan_range) ? 5 : 0

ceil_w += (pl_1 > close and pl_1 <= close + scan_range) ? 1 : 0
ceil_w += (ph_1 > close and ph_1 <= close + scan_range) ? 1 : 0
ceil_w += (pl_2 > close and pl_2 <= close + scan_range) ? 3 : 0
ceil_w += (ph_2 > close and ph_2 <= close + scan_range) ? 3 : 0
ceil_w += (pl_3 > close and pl_3 <= close + scan_range) ? 5 : 0
ceil_w += (ph_3 > close and ph_3 <= close + scan_range) ? 5 : 0

is_gravity_zone = false, target_lock_price = na, max_grav_weight = 0
if floor_w >= 3
    is_gravity_zone := true, target_lock_price := close - atr_val
if ceil_w >= 3
    is_gravity_zone := true, target_lock_price := close + atr_val

macro_ema = ta.ema(close, 200), is_macro_safe = close > macro_ema
is_falling_knife = (open[1] - close[1]) > (atr_val[1] * 1.5)
trinity_safe = true
if use_macro_shield and not is_macro_safe
    trinity_safe := false
if use_knife_shield and is_falling_knife
    trinity_safe := false

squeeze_on = (bb_top < kc_u) and (bb_bot > kc_l)
neon_break_up = squeeze_on and (close >= bb_top * 0.999) and (close > open)
neon_break_dn = squeeze_on and (close <= bb_bot * 1.001) and (close < open)
bb_delta = (bb_top - bb_bot) - nz((bb_top[1] - bb_bot[1]), 0)
bb_delta_avg = ta.sma(bb_delta, 10)

defcon_level = 5 
if neon_break_up or neon_break_dn
    defcon_level := 4
    if bb_delta > 0
        defcon_level := 3
    if bb_delta > bb_delta_avg and adx_val > adx_trend
        defcon_level := 2
    if bb_delta > (bb_delta_avg * 1.5) and adx_val > (adx_trend + 5) and rvol > 1.2
        defcon_level := 1

cond_defcon_buy  = (defcon_level <= 2) and neon_break_up
cond_defcon_sell = (defcon_level <= 2) and neon_break_dn

cond_therm_buy_bounce = (floor_w >= therm_wall) and rsi_cross_up and (ceil_w < therm_wall)
cond_therm_buy_vacuum = (ceil_w <= 3) and neon_break_up and (floor_w > 0)
cond_therm_sell_wall  = (ceil_w >= therm_wall) and rsi_cross_down
cond_therm_sell_panic = (floor_w == 0) and (close < open)

tolerance = atr_val * 0.5
cond_lock_buy_bounce = is_gravity_zone and (low <= target_lock_price + tolerance) and (close > target_lock_price) and (close > open)
cond_lock_buy_break  = is_gravity_zone and ta.crossover(close, target_lock_price) and high_vol and (close > open)
cond_lock_sell_reject = is_gravity_zone and (high >= target_lock_price - tolerance) and (close < target_lock_price) and (close < open)
cond_lock_sell_breakd = is_gravity_zone and ta.crossunder(close, target_lock_price) and (close < open)

flash_vol = rvol > (whale_factor * 0.8) and math.abs(close-open) > (atr_val * 0.3)
whale_buy = flash_vol and close > open
is_whale_icon = whale_buy and not nz(whale_buy[1], false)
buy_score = (rsi_v < 30 and close < bb_bot) ? 50.0 : 30.0
is_magenta_candle = buy_score >= 50
cond_pink_whale_buy = is_magenta_candle and is_whale_icon

do_buy = false, do_sell = false
if trinity_safe
    if cond_defcon_buy or cond_therm_buy_bounce or cond_therm_buy_vacuum or cond_lock_buy_bounce or cond_lock_buy_break
        do_buy := true
if cond_pink_whale_buy
    do_buy := true
if cond_defcon_sell or cond_therm_sell_wall or cond_therm_sell_panic or cond_lock_sell_reject or cond_lock_sell_breakd
    do_sell := true

if do_buy and strategy.position_size == 0
    strategy.entry("JUGG_LONG", strategy.long, alert_message=wt_enter_long)

if strategy.position_size > 0 
    if do_sell
        strategy.close("JUGG_LONG", comment="V356_Exit", alert_message=wt_exit_long)
    entry_price = strategy.opentrades.entry_price(strategy.opentrades - 1)
    target_price = entry_price * (1 + (tp_pct / 100))
    stop_price = entry_price * (1 - (sl_pct / 100))
    strategy.exit("EXIT", "JUGG_LONG", limit=target_price, stop=stop_price, alert_message=wt_exit_long)

plot(macro_ema, "AEGIS Line (EMA200)", color=is_macro_safe ? color.new(#00FF00, 60) : color.new(#FF0000, 60), linewidth=1)
plotchar(do_buy, title="COMPRA", char="🚀", location=location.belowbar, color=color.aqua, size=size.small)
"""

    ps = f"""// This source code is subject to the terms of the Mozilla Public License 2.0
// © Valle_Architect_Lab | AUTO-GENERATED BY OMNI-FORGE V116

//@version=5
strategy("{s_id} MATRIX - {sym} [{tf}]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_value=0.25)

// ==========================================
// 🔗 CONEXIÓN WUNDERTRADING (WEBHOOKS JSON)
// ==========================================
wt_enter_long = input.text_area(defval='{{"action": "buy"}}', title="🟢 WT: Mensaje Enter Long (Compra)")
wt_exit_long  = input.text_area(defval='{{"action": "sell"}}', title="🔴 WT: Mensaje Exit Long (Venta/Cierre)")

// ==========================================
// ⚙️ ADN BASE (Variables Universales)
// ==========================================
hitbox_pct   = {vault['hitbox']}
therm_wall   = {vault['therm_w']}
adx_trend    = {vault['adx_th']}
whale_factor = {vault['whale_f']}
"""
    if s_id not in ["GENESIS", "ROCKET", "QUADRIX", "ALL_FORCES"]:
        ps += f"\nactive_tp = {vault['tp']} / 100.0\nactive_sl = {vault['sl']} / 100.0\n"

    ps += """
// ==========================================
// 📡 SENSORES CUÁNTICOS
// ==========================================
ema50  = ta.ema(close, 50), ema200 = ta.ema(close, 200), rsi = ta.rsi(close, 14)
atr = ta.atr(14)
body_size = math.abs(close - open)
lower_wick = math.min(open, close) - low
is_falling_knife = (open[1] - close[1]) > (atr[1] * 1.5)
[di_plus, di_minus, adx] = ta.dmi(14, 14)
vol_ma100 = ta.sma(volume, 100)
rvol = vol_ma100 > 0 ? volume / vol_ma100 : 1

// WaveTrend para Quadrix
ap = hlc3 
esa = ta.ema(ap, 10)
d_wt = ta.ema(math.abs(ap - esa), 10)
ci = (ap - esa) / (0.015 * (d_wt == 0 ? 1 : d_wt))
wt1 = ta.ema(ci, 21)
wt2 = ta.sma(wt1, 4)

basis = ta.sma(close, 20), dev = 2.0 * ta.stdev(close, 20)
bbu = basis + dev, bbl = basis - dev
bb_width = (bbu - bbl) / basis, bb_width_avg = ta.sma(bb_width, 20)
kc_basis = ta.sma(close, 20), kc_upper = kc_basis + (atr * 1.5), kc_lower = kc_basis - (atr * 1.5)
squeeze_on = (bbu < kc_upper) and (bbl > kc_lower)

pl30 = ta.lowest(low[1], 30), ph30 = ta.highest(high[1], 30)
pl100 = ta.lowest(low[1], 100), ph100 = ta.highest(high[1], 100)
pl300 = ta.lowest(low[1], 300), ph300 = ta.highest(high[1], 300)

target_lock_sup = math.max(pl30, pl100, pl300)
target_lock_res = math.min(ph30, ph100, ph300)
dist_sup = (close - target_lock_sup) / close * 100
dist_res = (target_lock_res - close) / close * 100

sr_val = atr * 2.0
floor_w = 0
floor_w += (pl30  < close and pl30  >= close - sr_val) ? 1 : 0
floor_w += (ph30  < close and ph30  >= close - sr_val) ? 1 : 0
floor_w += (pl100 < close and pl100 >= close - sr_val) ? 3 : 0
floor_w += (ph100 < close and ph100 >= close - sr_val) ? 3 : 0
floor_w += (pl300 < close and pl300 >= close - sr_val) ? 5 : 0
floor_w += (ph300 < close and ph300 >= close - sr_val) ? 5 : 0

ceil_w = 0
ceil_w += (pl30  > close and pl30  <= close + sr_val) ? 1 : 0
ceil_w += (ph30  > close and ph30  <= close + sr_val) ? 1 : 0
ceil_w += (pl100 > close and pl100 <= close + sr_val) ? 3 : 0
ceil_w += (ph100 > close and ph100 <= close + sr_val) ? 3 : 0
ceil_w += (pl300 > close and pl300 <= close + sr_val) ? 5 : 0
ceil_w += (ph300 > close and ph300 <= close + sr_val) ? 5 : 0

// ==========================================
// ⚔️ INVENTARIO DE ARMAS (Señales Puras)
// ==========================================
vela_verde = close > open
vela_roja = close < open
rsi_cross_up = rsi > nz(rsi[1], 50)
rsi_cross_dn = rsi < nz(rsi[1], 50)
macro_bull = close >= ema200

ping_b = (adx < adx_trend) and (close < bbl) and vela_verde
ping_s = (close > bbu) or (rsi > 70)
neon_up = (bb_width < nz(bb_width_avg[1], -1.0)) and (close > bbu) and vela_verde and (rsi < 60)
squeeze_b = neon_up
squeeze_s = (close < ema50)
therm_b = (floor_w >= therm_wall) and vela_verde and rsi_cross_up
therm_s = (ceil_w >= therm_wall) and vela_roja and rsi_cross_dn
climax_b = (rvol > whale_factor) and (lower_wick > (body_size * 2.0)) and (rsi < 35) and vela_verde
climax_s = (rsi > 80)
lock_b = (dist_sup < hitbox_pct) and vela_verde and rsi_cross_up
lock_s = (dist_res < hitbox_pct) or (high >= target_lock_res)
defcon_b = nz((bbu[1] < (ta.sma(close[1],20) + atr[1]*1.5)) and (bbl[1] > (ta.sma(close[1],20) - atr[1]*1.5)), false) and (close > bbu) and (adx > adx_trend)
defcon_s = (close < ema50)
jugg_b = macro_bull and (close > ema50) and nz(close[1] < ema50[1], false) and vela_verde and not is_falling_knife
jugg_s = (close < ema50)
trinity_b = macro_bull and (rsi < 35) and vela_verde and not is_falling_knife
trinity_s = (rsi > 75) or (close < ema200)
lev_b = macro_bull and rsi_cross_up and (rsi < 45)
lev_s = (close < ema200)
commander_b = climax_b or therm_b or lock_b
commander_s = therm_s or (close < ema50)

// Armas Quadrix
r_Neon_Up = neon_up
r_Neon_Dn = (bb_width < nz(bb_width_avg[1], -1.0)) and (close <= bbl * 1.001) and vela_roja
r_Therm_Bounce = therm_b
r_Therm_Vacuum = (ceil_w <= 3) and r_Neon_Up and not (floor_w == 0)
r_Therm_Wall_Sell = therm_s
r_Therm_Panic_Sell = (floor_w == 0) and vela_roja
r_Lock_Bounce = (low <= target_lock_sup + (atr * 0.5)) and (close > target_lock_sup) and vela_verde
r_Lock_Break = (close > target_lock_res) and (open <= target_lock_res) and (rvol > whale_factor * 0.8) and vela_verde
r_Lock_Reject = (high >= target_lock_res - (atr * 0.5)) and (close < target_lock_res) and vela_roja
r_Lock_Breakd = (close < target_lock_sup) and (open >= target_lock_sup) and vela_roja
r_Defcon_Buy = r_Neon_Up and ((bbu-bbl)-nz(bbu[1]-bbl[1],0) > ta.sma((bbu-bbl)-nz(bbu[1]-bbl[1],0), 10)) and (adx > 20)
r_Defcon_Sell = r_Neon_Dn and ((bbu-bbl)-nz(bbu[1]-bbl[1],0) > ta.sma((bbu-bbl)-nz(bbu[1]-bbl[1],0), 10)) and (adx > 20)
is_magenta = rsi < 30 or rsi_cross_up
r_Pink_Whale_Buy = is_magenta and (rvol > whale_factor) and vela_verde
r_Nuclear_Buy = is_magenta and (wt1 < -60 or ta.crossover(wt1, wt2))
r_Early_Buy = is_magenta
r_Nuclear_Sell = (rsi > 70) and (wt1 > 60 or ta.crossunder(wt1, wt2))
r_Early_Sell = (rsi > 70) and vela_roja
r_Rebound_Buy = rsi_cross_up and not is_magenta
"""

    if s_id in ["GENESIS", "ROCKET", "QUADRIX"]:
        ps += """
// ==========================================
// 🌍 DETERMINACIÓN DEL RÉGIMEN (CUADRANTE)
// ==========================================
int regime = 0
if macro_bull and (adx >= adx_trend)
    regime := 1
else if macro_bull and (adx < adx_trend)
    regime := 2
else if not macro_bull and (adx >= adx_trend)
    regime := 3
else
    regime := 4

bool signal_buy = false
bool signal_sell = false
float active_tp = 0.0
float active_sl = 0.0
"""
        for r in range(1, 5):
            b_cond = " or ".join([pine_map[x] for x in vault[f'r{r}_b']]) if vault[f'r{r}_b'] else "false"
            s_cond = " or ".join([pine_map[x] for x in vault[f'r{r}_s']]) if vault[f'r{r}_s'] else "false"
            ps += f"\nif regime == {r}\n    signal_buy := {b_cond}\n    signal_sell := {s_cond}\n    active_tp := {vault[f'r{r}_tp']} / 100.0\n    active_sl := {vault[f'r{r}_sl']} / 100.0\n"

    elif s_id == "ALL_FORCES":
        m_cond = "macro_bull" if vault['macro'] == "Bull Only (Precio > EMA 200)" else "not macro_bull" if vault['macro'] == "Bear Only (Precio < EMA 200)" else "true"
        v_cond = "(adx >= adx_trend)" if vault['vol'] == "Trend (ADX Alto)" else "(adx < adx_trend)" if vault['vol'] == "Range (ADX Bajo)" else "true"
        b_cond = " or ".join([pine_map[x] for x in vault['b_team']]) if vault['b_team'] else "false"
        s_cond = " or ".join([pine_map[x] for x in vault['s_team']]) if vault['s_team'] else "false"
        ps += f"\nbool signal_buy = ({b_cond}) and {m_cond} and {v_cond}\nbool signal_sell = {s_cond}\nfloat active_tp = {vault['tp']} / 100.0\nfloat active_sl = {vault['sl']} / 100.0\n"
    else:
        try: b_cond, s_cond = pine_map[f"{s_id.split('_')[0].capitalize()}_Buy"], pine_map[f"{s_id.split('_')[0].capitalize()}_Sell"]
        except:
            if s_id == "TARGET_LOCK": b_cond, s_cond = pine_map["Lock_Buy"], pine_map["Lock_Sell"]
            elif s_id == "NEON_SQUEEZE": b_cond, s_cond = pine_map["Squeeze_Buy"], pine_map["Squeeze_Sell"]
            elif s_id == "PINK_CLIMAX": b_cond, s_cond = pine_map["Climax_Buy"], pine_map["Climax_Sell"]
            else: b_cond, s_cond = "false", "false"
        
        ps += f"""
bool signal_buy = {b_cond}
bool signal_sell = {s_cond}
"""

    ps += """
// ==========================================
// 🚀 EJECUCIÓN DEL TRADE (CON WUNDERTRADING)
// ==========================================
if signal_buy and strategy.position_size == 0
    strategy.entry("In", strategy.long, alert_message=wt_enter_long)

if signal_sell and strategy.position_size > 0
    strategy.close("In", comment="Dyn_Exit", alert_message=wt_exit_long)

if strategy.position_size > 0
    entry_price = strategy.opentrades.entry_price(strategy.opentrades - 1)
    target_price = entry_price * (1 + active_tp)
    stop_price = entry_price * (1 - active_sl)
    strategy.exit("TP/SL", "In", limit=target_price, stop=stop_price, alert_message=wt_exit_long)

plot(ema50, color=color.yellow, title="EMA 50")
plot(ema200, color=color.white, title="EMA 200")
plotchar(signal_buy, title="COMPRA", char="🚀", location=location.belowbar, color=color.aqua, size=size.small)
"""
    if s_id in ["GENESIS", "ROCKET", "QUADRIX"]:
        ps += 'bgcolor(regime == 1 ? color.new(color.green, 90) : regime == 2 ? color.new(color.yellow, 90) : regime == 3 ? color.new(color.red, 90) : color.new(color.orange, 90), title="Regime Matrix")\n'
    
    return ps

def generar_reporte_universal(cap_ini, com_pct):
    res_str = f"📋 **REPORTE OMNI-FORGE V116**\n\n"
    res_str += f"⏱️ Temporalidad: {intervalo_sel} | 📊 Velas: {len(df_global)}\n\n"
    buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
    res_str += f"📈 RENDIMIENTO DEL HOLD: **{buy_hold_ret:.2f}%**\n\n"
    
    for s_id in estrategias:
        _, eq_curve, t_log, _ = run_backtest_eval(s_id, cap_ini, com_pct)
        net = eq_curve[-1] - cap_ini
        ret_pct = (net / cap_ini) * 100
        alpha = ret_pct - buy_hold_ret
        nt = len([x for x in t_log if x['Tipo'] in ['TP', 'SL', 'DYN_WIN', 'DYN_LOSS']])
        opt_icon = "✅" if st.session_state.get(f'opt_status_{s_id}', False) else "➖"
        res_str += f"⚔️ **{s_id}** [{opt_icon}]\nNet Profit: ${net:,.2f} ({ret_pct:.2f}%)\nALPHA vs Hold: {alpha:.2f}%\nTrades: {nt}\n---\n"
    return res_str

if not df_global.empty:
    if st.sidebar.button(f"🧠 DEEP MINE GLOBAL ({global_epochs*3}k Combos)", type="primary", use_container_width=True):
        buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
        buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
        for s_id in estrategias:
            v = st.session_state[f'champion_{s_id}']
            bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, v['reinv'], v['ado'], dias_reales, buy_hold_money, epochs=global_epochs, cur_fit=v['fit'])
            if bp:
                save_champion(s_id, bp)
                st.session_state[f'opt_status_{s_id}'] = True
        wipe_ui_cache(); ph_holograma.empty(); st.sidebar.success("✅ ¡Forja Evolutiva Global Completada!"); time.sleep(1); st.rerun()

    if st.sidebar.button("📊 GENERAR REPORTE UNIVERSAL", use_container_width=True):
        st.sidebar.text_area("Copia tu Reporte:", value=generar_reporte_universal(capital_inicial, comision_pct), height=400)

tabs = st.tabs(list(tab_id_map.keys()))

for idx, tab_name in enumerate(tab_id_map.keys()):
    with tabs[idx]:
        if df_global.empty: continue
        s_id = tab_id_map[tab_name]
        is_opt = st.session_state.get(f'opt_status_{s_id}', False)
        opt_badge = "<span style='color: lime; border: 1px solid lime; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem;'>✅ IA OPTIMIZADA</span>" if is_opt else "<span style='color: gray; border: 1px solid gray; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem;'>➖ NO OPTIMIZADA</span>"
        vault = st.session_state[f'champion_{s_id}']

        if s_id in ["ROCKET_ULTRA", "ROCKET_COMMANDER"]:
            st.markdown(f"### {tab_name} {opt_badge}", unsafe_allow_html=True)
            st.info("Los motores maestros originales en su estado puro. Conectados a The Omni-Forge para exportación a TradingView.")
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state[f'champion_{s_id}']['ado'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(vault['ado']), key=f"ui_{s_id}_ado_w", step=0.5)
            st.session_state[f'champion_{s_id}']['reinv'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(vault['reinv']), key=f"ui_{s_id}_reinv_w", step=5.0)
            if c_ia3.button(f"🚀 FORJAR BOT ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, vault['reinv'], vault['ado'], dias_reales, capital_inicial * (buy_hold_ret / 100.0), epochs=global_epochs, cur_fit=vault['fit'])
                if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True; st.success("👑 ¡Bot Forjado con Éxito!")
                else: st.warning("🛡️ Ningún escenario superó al actual.")
                time.sleep(2); ph_holograma.empty(); wipe_ui_cache(); st.rerun()

        elif s_id == "JUGGERNAUT":
            st.markdown(f"### ⚔️ JUGGERNAUT V356 (Original Core) {opt_badge}", unsafe_allow_html=True)
            st.info("El algoritmo original de Valle Architect. Posee su propia matemática interna. La IA ajustará las variables para su código Pine Script V356.")
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state[f'champion_{s_id}']['ado'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(vault['ado']), key=f"ui_{s_id}_ado_w", step=0.5)
            st.session_state[f'champion_{s_id}']['reinv'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(vault['reinv']), key=f"ui_{s_id}_reinv_w", step=5.0)
            if c_ia3.button(f"🚀 FORJAR BOT V356 ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, vault['reinv'], vault['ado'], dias_reales, capital_inicial * (buy_hold_ret / 100.0), epochs=global_epochs, cur_fit=vault['fit'])
                if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True; st.success("👑 ¡Juggernaut Optimizado!")
                else: st.warning("🛡️ Ningún escenario superó al actual.")
                time.sleep(2); ph_holograma.empty(); wipe_ui_cache(); st.rerun()

        elif s_id == "ALL_FORCES":
            st.markdown(f"### 🌟 ALL FORCES ALGO (Global Matrix) {opt_badge}", unsafe_allow_html=True)
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state[f'champion_{s_id}']['ado'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(vault['ado']), key=f"ui_{s_id}_ado_w", step=0.5)
            st.session_state[f'champion_{s_id}']['reinv'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(vault['reinv']), key=f"ui_{s_id}_reinv_w", step=5.0)
            if c_ia3.button(f"🚀 FORJAR BOT ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, vault['reinv'], vault['ado'], dias_reales, capital_inicial * (buy_hold_ret / 100.0), epochs=global_epochs, cur_fit=vault['fit'])
                if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True; st.success("👑 ¡Bot Forjado con Éxito!")
                else: st.warning("🛡️ Ningún escenario superó al actual.")
                time.sleep(2); ph_holograma.empty(); wipe_ui_cache(); st.rerun() 

        elif s_id in ["GENESIS", "ROCKET", "QUADRIX"]:
            st.markdown(f"### {'🌌 QUADRIX (Adaptative Apex)' if s_id == 'QUADRIX' else '🌌 GÉNESIS (The Matrix)' if s_id == 'GENESIS' else '👑 ROCKET PROTOCOL (The Matrix)'} {opt_badge}", unsafe_allow_html=True)
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state[f'champion_{s_id}']['ado'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(vault['ado']), key=f"ui_{s_id}_ado_w", step=0.5)
            st.session_state[f'champion_{s_id}']['reinv'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(vault['reinv']), key=f"ui_{s_id}_reinv_w", step=5.0)
            if c_ia3.button(f"🚀 FORJAR BOT MATRIX ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, vault['reinv'], vault['ado'], dias_reales, capital_inicial * (buy_hold_ret / 100.0), epochs=global_epochs, cur_fit=vault['fit'])
                if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True; st.success("👑 ¡Bot Forjado con Éxito!")
                else: st.warning("🛡️ Ningún escenario superó al actual.")
                time.sleep(2); ph_holograma.empty(); wipe_ui_cache(); st.rerun() 

        else:
            st.markdown(f"### ⚙️ {s_id} (Truth Engine) {opt_badge}", unsafe_allow_html=True)
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state[f'champion_{s_id}']['ado'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(vault['ado']), key=f"ui_{s_id}_ado_w", step=0.5)
            st.session_state[f'champion_{s_id}']['reinv'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(vault['reinv']), key=f"ui_{s_id}_reinv_w", step=5.0)
            if c_ia3.button(f"🚀 FORJAR BOT ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, vault['reinv'], vault['ado'], dias_reales, capital_inicial * (buy_hold_ret / 100.0), epochs=global_epochs, cur_fit=vault['fit'])
                if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True; st.success("👑 ¡Bot Forjado con Éxito!")
                else: st.warning("🛡️ Ningún escenario superó al actual.")
                time.sleep(2); ph_holograma.empty(); wipe_ui_cache(); st.rerun()

        # --- DIBUJADO DE GRÁFICO ---
        df_strat, eq_curve, t_log, total_comms = run_backtest_eval(s_id, capital_inicial, comision_pct)
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

        # 🤖 GENERADOR DINÁMICO DE PINE SCRIPT 
        with st.expander("📝 EXPORTAR A TRADINGVIEW (PINE SCRIPT GENERATOR)", expanded=False):
            st.info("Este código se genera automáticamente con el ADN ganador de la estrategia. Está listo para copiar, pegar y conectarse a WunderTrading mediante Webhooks.")
            pine_code = generar_pine_script(s_id, vault, ticker.split('/')[0], iv_download)
            st.code(pine_code, language="pine")

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
