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
# 🧠 MEMORIA GLOBAL Y ESTRATEGIAS BASE
# ==========================================
base_b = ['Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 'Commander_Buy', 'Lev_Buy']
base_s = ['Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 'Commander_Sell', 'Lev_Sell']

rocket_b = ['Trinity_Buy', 'Jugg_Buy', 'Defcon_Buy', 'Lock_Buy', 'Thermal_Buy', 'Climax_Buy', 'Ping_Buy', 'Squeeze_Buy', 'Lev_Buy', 'Commander_Buy']
rocket_s = ['Trinity_Sell', 'Jugg_Sell', 'Defcon_Sell', 'Lock_Sell', 'Thermal_Sell', 'Climax_Sell', 'Ping_Sell', 'Squeeze_Sell', 'Lev_Sell', 'Commander_Sell']

estrategias = ["ALL_FORCES", "TRINITY", "JUGGERNAUT", "DEFCON", "TARGET_LOCK", "THERMAL", "PINK_CLIMAX", "PING_PONG", "NEON_SQUEEZE", "COMMANDER", "GENESIS", "ROCKET"]

tab_id_map = {
    "🌟 ALL FORCES": "ALL_FORCES", "💠 TRINITY": "TRINITY", "⚔️ JUGGERNAUT": "JUGGERNAUT", "🚀 DEFCON": "DEFCON",
    "🎯 TARGET_LOCK": "TARGET_LOCK", "🌡️ THERMAL": "THERMAL", "🌸 PINK_CLIMAX": "PINK_CLIMAX",
    "🏓 PING_PONG": "PING_PONG", "🐛 NEON_SQUEEZE": "NEON_SQUEEZE", "👑 COMMANDER": "COMMANDER",
    "🌌 GENESIS": "GENESIS", "👑 ROCKET": "ROCKET"
}

macro_opts = ["All-Weather", "Bull Only (Precio > EMA 200)", "Bear Only (Precio < EMA 200)"]
vol_opts = ["All-Weather", "Trend (ADX > 25)", "Range (ADX < 25)"]

# ==========================================
# 🧬 THE DNA VAULT (MEMORIA EVOLUTIVA INMUTABLE)
# ==========================================
for s_id in estrategias:
    if f'champion_{s_id}' not in st.session_state:
        st.session_state[f'opt_status_{s_id}'] = False
        if s_id == "ALL_FORCES":
            st.session_state[f'champion_{s_id}'] = {
                'b_team': ['Commander_Buy', 'Squeeze_Buy', 'Ping_Buy'],
                's_team': ['Commander_Sell', 'Squeeze_Sell'],
                'macro': "All-Weather", 'vol': "All-Weather",
                'tp': 50.0, 'sl': 5.0, 'wh': 2.5, 'rd': 1.5,
                'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')
            }
        elif s_id in ["GENESIS", "ROCKET"]:
            vault = {'wh': 2.5, 'rd': 1.5, 'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')}
            for r_idx in range(1, 5):
                vault[f'r{r_idx}_b'] = ['Squeeze_Buy']
                vault[f'r{r_idx}_s'] = ['Squeeze_Sell']
                vault[f'r{r_idx}_tp'] = 50.0
                vault[f'r{r_idx}_sl'] = 5.0
            st.session_state[f'champion_{s_id}'] = vault
        else:
            st.session_state[f'champion_{s_id}'] = {
                'tp': 50.0, 'sl': 5.0, 'wh': 2.5, 'rd': 1.5,
                'ado': 100.0, 'reinv': 0.0, 'fit': -float('inf')
            }
        # Sincronizar UI inicial
        for k, v in st.session_state[f'champion_{s_id}'].items():
            if k != 'fit': st.session_state[f'w_{k}_{s_id}'] = v

def save_champion(s_id, bp):
    vault = st.session_state[f'champion_{s_id}']
    vault['fit'] = bp['fit']
    if s_id == "ALL_FORCES":
        for k in ['b_team', 's_team', 'macro', 'vol', 'tp', 'sl', 'wh', 'rd']: vault[k] = bp[k]
    elif s_id in ["GENESIS", "ROCKET"]:
        vault['wh'], vault['rd'] = bp['wh'], bp['rd']
        for r_idx in range(1, 5):
            for k in ['b', 's', 'tp', 'sl']: vault[f'r{r_idx}_{k}'] = bp[f'{k}{r_idx}']
    else:
        for k in ['tp', 'sl', 'wh', 'rd']: vault[k] = bp[k]
        
    for k, v in vault.items():
        if k != 'fit': st.session_state[f'w_{k}_{s_id}'] = v

def restore_champion_to_widgets(s_id):
    vault = st.session_state[f'champion_{s_id}']
    for k, v in vault.items():
        if k != 'fit': st.session_state[f'w_{k}_{s_id}'] = v

css_spinner = """
<style>
.loader-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.8); padding: 30px; border-radius: 20px; border: 2px solid cyan; }
.rocket { font-size: 8rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 20px cyan); }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.prog-text { color: cyan; font-size: 1.5rem; font-weight: bold; margin-top: 10px; }
.hud-text { color: #00FF00; font-size: 1.1rem; margin-top: 5px; font-family: monospace; }
</style>
"""
ph_holograma = st.empty()

# ==========================================
# 🌍 SIDEBAR E INFRAESTRUCTURA
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 TRUTH ENGINE V94.0</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True): 
    st.cache_data.clear()
    for s in estrategias: 
        st.session_state[f'opt_status_{s}'] = False
        if f'champion_{s}' in st.session_state: del st.session_state[f'champion_{s}']
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

@st.cache_data(ttl=3600, show_spinner="📡 Sintetizando Topología Fractal...")
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
            
        # Motores Matemáticos de TradingView
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
        df['RSI_MA'] = df['RSI'].rolling(14, min_periods=1).mean()
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0].fillna(0.0)
        
        df['Basis'] = df['Close'].rolling(20, min_periods=1).mean()
        dev = df['Close'].rolling(20, min_periods=1).std(ddof=0).replace(0, 1) 
        df['BBU'] = df['Basis'] + (2.0 * dev)
        df['BBL'] = df['Basis'] - (2.0 * dev)
        df['BB_Width'] = (df['BBU'] - df['BBL']) / df['Basis'].replace(0, 1)
        df['BB_Width_Avg'] = df['BB_Width'].rolling(20, min_periods=1).mean()
        df['Z_Score'] = (df['Close'] - df['Basis']) / dev
        
        kc_basis = df['Close'].rolling(20, min_periods=1).mean()
        df['KC_Upper'] = kc_basis + (df['ATR'] * 1.5)
        df['KC_Lower'] = kc_basis - (df['ATR'] * 1.5)
        df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
        df['BB_Delta'] = (df['BBU'] - df['BBL']).diff().fillna(0)
        df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10, min_periods=1).mean()
        
        df['Vela_Verde'] = df['Close'] > df['Open']
        df['Vela_Roja'] = df['Close'] < df['Open']
        df['body_size'] = abs(df['Close'] - df['Open']).replace(0, 0.0001)
        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['is_falling_knife'] = (df['Open'].shift(1) - df['Close'].shift(1)) > (df['ATR'].shift(1) * 1.5)
        df['Macro_Safe'] = df['Close'] > df['EMA_200']
        
        # Wavetrend
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10, adjust=False).mean()
        d_wt = (abs(ap - esa)).ewm(span=10, adjust=False).mean().replace(0, 1)
        ci = (ap - esa) / (0.015 * d_wt)
        df['WT1'] = ci.ewm(span=21, adjust=False).mean()
        
        # Malla (Soportes/Resistencias Base)
        df['PL30'] = df['Low'].shift(1).rolling(30, min_periods=1).min()
        df['PH30'] = df['High'].shift(1).rolling(30, min_periods=1).max()
        df['PL100'] = df['Low'].shift(1).rolling(100, min_periods=1).min()
        df['PH100'] = df['High'].shift(1).rolling(100, min_periods=1).max()
        df['PL300'] = df['Low'].shift(1).rolling(300, min_periods=1).min()
        df['PH300'] = df['High'].shift(1).rolling(300, min_periods=1).max()
        
        df['RSI_Cross_Up'] = (df['RSI'] > df['RSI_MA']) & (df['RSI'].shift(1).fillna(50) <= df['RSI_MA'].shift(1).fillna(50))
        df['RSI_Cross_Dn'] = (df['RSI'] < df['RSI_MA']) & (df['RSI'].shift(1).fillna(50) >= df['RSI_MA'].shift(1).fillna(50))
        df['Momentum'] = df['Close'] - df['Close'].shift(2).fillna(df['Close'])
        df['Macro_Bull'] = df['Close'] >= df['EMA_200']
        df['PP_Slope'] = ta.linreg(df['Close'], 5, 0) - ta.linreg(df['Close'], 5, 1)
        
        is_trend = df['ADX'] >= 25
        df['Regime'] = np.where(df['Macro_Bull'] & is_trend, 1, np.where(df['Macro_Bull'] & ~is_trend, 2, np.where(~df['Macro_Bull'] & is_trend, 3, 4)))
        
        gc.collect()
        return df, "OK"
    except Exception as e: 
        return pd.DataFrame(), str(e)

df_global, status_api = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, utc_offset)

if not df_global.empty:
    dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)
    st.sidebar.success(f"📥 MATRIZ LISTA: {len(df_global)} velas sintetizadas ({dias_reales} días).")
else:
    st.error(status_api)
    st.stop()

# ==========================================
# 🔥 IDENTIDADES RESTAURADAS (PINE SCRIPT EMULATOR) 🔥
# ==========================================
def inyectar_adn(df_sim, r_sens=1.5, w_factor=2.5):
    # 1. RECÁLCULO DINÁMICO DE RADAR Y TERMÓMETRO (Requieren variables de entrada)
    scan_range_therm = df_sim['ATR'] * 2.0
    c_val = df_sim['Close'].values
    sr_val = scan_range_therm.values
    ceil_w, floor_w = np.zeros(len(df_sim)), np.zeros(len(df_sim))
    
    for p_col, w in [('PL30', 1), ('PH30', 1), ('PL100', 3), ('PH100', 3), ('PL300', 5), ('PH300', 5)]:
        p_val = df_sim[p_col].values
        ceil_w += np.where((p_val > c_val) & (p_val <= c_val + sr_val), w, 0)
        floor_w += np.where((p_val < c_val) & (p_val >= c_val - sr_val), w, 0)
    
    df_sim['ceil_w'] = ceil_w
    df_sim['floor_w'] = floor_w
    
    # Target Lock Radar
    target_lock_sup = df_sim[['PL30', 'PL100', 'PL300']].max(axis=1)
    target_lock_res = df_sim[['PH30', 'PH100', 'PH300']].min(axis=1)
    df_sim['Target_Lock_Sup'] = target_lock_sup
    df_sim['Target_Lock_Res'] = target_lock_res
    
    dist_sup = abs(df_sim['Close'] - target_lock_sup) / df_sim['Close'] * 100
    df_sim['is_gravity_zone'] = dist_sup < r_sens

    # 🏓 1. PING PONG (VWAP Reversion)
    df_sim['Ping_Buy'] = (df_sim['PP_Slope'] > 0) & (df_sim['PP_Slope'].shift(1).fillna(0) <= 0) & df_sim['is_gravity_zone'] & df_sim['Vela_Verde']
    df_sim['Ping_Sell'] = (df_sim['PP_Slope'] < 0) & (df_sim['PP_Slope'].shift(1).fillna(0) >= 0) & df_sim['is_gravity_zone']

    # 🐛 2. NEON SQUEEZE (Defcon Engine V329)
    df_sim['Neon_Up'] = df_sim['Squeeze_On'].shift(1).fillna(False) & (df_sim['Close'] >= df_sim['BB_Top'] * 0.999) & df_sim['Vela_Verde']
    df_sim['Neon_Dn'] = df_sim['Squeeze_On'].shift(1).fillna(False) & (df_sim['Close'] <= df_sim['BB_Bot'] * 1.001) & df_sim['Vela_Roja']
    df_sim['Squeeze_Buy'] = df_sim['Neon_Up']
    df_sim['Squeeze_Sell'] = df_sim['Neon_Dn'] | (df_sim['Close'] < df_sim['EMA_50'])

    # 🚀 3. DEFCON (Furia Expansiva)
    defcon_level = np.where(df_sim['Neon_Up'] | df_sim['Neon_Dn'], 4, 5)
    defcon_level = np.where(df_sim['BB_Delta'] > 0, 3, defcon_level)
    defcon_level = np.where((df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20), 2, defcon_level)
    defcon_level = np.where((df_sim['BB_Delta'] > df_sim['BB_Delta_Avg'] * 1.5) & (df_sim['ADX'] > 25) & (df_sim['RVol'] > 1.2), 1, defcon_level)
    df_sim['Defcon_Buy'] = (defcon_level <= 2) & df_sim['Neon_Up']
    df_sim['Defcon_Sell'] = (defcon_level <= 2) & df_sim['Neon_Dn']

    # 🌡️ 4. THERMAL (Termómetro Muros Nivel 3/4)
    df_sim['Thermal_Buy'] = (df_sim['floor_w'] >= 4) & df_sim['RSI_Cross_Up'] & ~(df_sim['ceil_w'] >= 4)
    df_sim['Thermal_Sell'] = (df_sim['ceil_w'] >= 4) & df_sim['RSI_Cross_Dn']

    # 🎯 5. TARGET LOCK (Hitbox Activa)
    tol = df_sim['ATR'] * 0.5
    df_sim['Lock_Buy'] = df_sim['is_gravity_zone'] & (df_sim['Low'] <= target_lock_sup + tol) & (df_sim['Close'] > target_lock_sup) & df_sim['Vela_Verde']
    df_sim['Lock_Sell'] = df_sim['is_gravity_zone'] & (df_sim['High'] >= target_lock_res - tol) & (df_sim['Close'] < target_lock_res) & df_sim['Vela_Roja']

    # 🌸 6. PINK CLIMAX (Pánico Absoluto: Score > 70)
    retro_peak = (df_sim['RSI'] < 30) & (df_sim['Close'] < df_sim['BB_Bot'])
    buy_score = np.where(retro_peak, 50.0, np.where(df_sim['RSI_Cross_Up'], 30.0, 0.0))
    buy_score = np.where(df_sim['is_gravity_zone'], buy_score + 25.0, buy_score)
    buy_score = np.where(df_sim['Z_Score'] < -2.0, buy_score + 15.0, buy_score)
    is_magenta = (buy_score >= 70) | retro_peak
    
    flash_vol = (df_sim['RVol'] > (w_factor * 0.8)) & (df_sim['body_size'] > df_sim['ATR'] * 0.3)
    whale_buy = flash_vol & df_sim['Vela_Verde']
    is_whale_icon = whale_buy & ~whale_buy.shift(1).fillna(False)
    
    df_sim['Climax_Buy'] = is_magenta & is_whale_icon
    df_sim['Climax_Sell'] = (df_sim['RSI'] > 75)

    # ⚔️ 7. JUGGERNAUT (Aegis Shield - Sin Cuchillos)
    aegis_safe = df_sim['Macro_Safe'] & ~df_sim['is_falling_knife']
    df_sim['Jugg_Buy'] = (df_sim['Defcon_Buy'] | df_sim['Lock_Buy']) & aegis_safe
    df_sim['Jugg_Sell'] = df_sim['Defcon_Sell'] | (df_sim['Close'] < df_sim['EMA_50'])

    # 👑 8. TRINITY (Juggernaut + Pink Whale)
    df_sim['Trinity_Buy'] = df_sim['Climax_Buy'] | df_sim['Jugg_Buy']
    df_sim['Trinity_Sell'] = df_sim['Defcon_Sell'] | df_sim['Thermal_Sell']
    
    # 🐉 9. LEVIATHAN
    df_sim['Lev_Buy'] = df_sim['Macro_Bull'] & df_sim['RSI_Cross_Up'] & (df_sim['RSI'].shift(1).rolling(3).min() < 50)
    df_sim['Lev_Sell'] = ~df_sim['Macro_Bull']

    # 🎖️ 10. COMMANDER (Filtro Anti-Ruido)
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

# 🧠 RUTINA DE OPTIMIZACIÓN (PROTOCOLO DE CODICIA) 🧠
def optimizar_ia_tracker(s_id, df_base, cap_ini, com_pct, reinv_q, target_ado, dias_reales, buy_hold_money, is_meta=False):
    best_fit = -float('inf')
    bp = None
    tp_min, tp_max = (10.0, 150.0) 
    
    iters = 3000 if s_id == "ALL_FORCES" else (1500 if is_meta else 2000)
    chunks = 10
    chunk_size = iters // chunks
    start_time = time.time()

    for c in range(chunks):
        for _ in range(chunk_size): 
            rtp = round(random.uniform(tp_min, tp_max), 1)
            rsl = round(random.uniform(1.0, 20.0), 1)
            rwh = round(random.uniform(1.5, 3.5), 1)
            rrd = round(random.uniform(0.5, 3.5), 1)
            
            df_precalc = inyectar_adn(df_base.copy(), r_sens=rrd, w_factor=rwh)
            h_a = np.asarray(df_precalc['High'].values, dtype=np.float64)
            l_a = np.asarray(df_precalc['Low'].values, dtype=np.float64)
            c_a = np.asarray(df_precalc['Close'].values, dtype=np.float64)
            o_a = np.asarray(df_precalc['Open'].values, dtype=np.float64)
            
            if s_id == "ALL_FORCES":
                dna_b_team = random.sample(base_b, random.randint(3, len(base_b)))
                dna_s_team = random.sample(base_s, random.randint(2, len(base_s)))
                
                dna_macro = "All-Weather" if random.random() < 0.6 else random.choice(["Bull Only (Precio > EMA 200)", "Bear Only (Precio < EMA 200)"])
                dna_vol = "All-Weather" if random.random() < 0.6 else random.choice(["Trend (ADX > 25)", "Range (ADX < 25)"])
                
                macro_mask = np.ones(len(df_precalc), dtype=bool)
                if dna_macro == "Bull Only (Precio > EMA 200)": macro_mask = df_precalc['Macro_Bull'].values
                elif dna_macro == "Bear Only (Precio < EMA 200)": macro_mask = ~df_precalc['Macro_Bull'].values
                
                vol_mask = np.ones(len(df_precalc), dtype=bool)
                if dna_vol == "Trend (ADX > 25)": vol_mask = df_precalc['ADX'].values >= 25
                elif dna_vol == "Range (ADX < 25)": vol_mask = df_precalc['ADX'].values < 25
                
                global_mask = macro_mask & vol_mask
                
                f_buy = np.zeros(len(df_precalc), dtype=bool)
                for r in dna_b_team: f_buy |= df_precalc[r].values
                f_buy &= global_mask
                
                f_sell = np.zeros(len(df_precalc), dtype=bool)
                for r in dna_s_team: f_sell |= df_precalc[r].values
                
                b_c_arr = np.asarray(f_buy, dtype=bool)
                s_c_arr = np.asarray(f_sell, dtype=bool)
                t_arr = np.asarray(np.full(len(df_precalc), float(rtp)), dtype=np.float64)
                sl_arr = np.asarray(np.full(len(df_precalc), float(rsl)), dtype=np.float64)
                
            elif is_meta:
                b_mat_opts = base_b if s_id == "GENESIS" else rocket_b
                s_mat_opts = base_s if s_id == "GENESIS" else rocket_s
                
                dna_b = [random.sample(b_mat_opts, random.randint(1, 2)) for _ in range(4)]
                dna_s = [random.sample(s_mat_opts, random.randint(1, 2)) for _ in range(4)]
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
            
            # 🔥 V94.0 FITNESS: EL PROTOCOLO DE CODICIA SUAVIZADO 🔥
            if nt >= 1: 
                if net > 0: 
                    # Relajamos el castigo a 8 operaciones en lugar de 12/15 para dar más oportunidades en temporalidades altas
                    trade_penalty = np.sqrt(float(nt)) if nt >= 8 else (float(nt) / 8.0)
                    min_profit_threshold = cap_ini * 0.05
                    if net < min_profit_threshold: net_score = net * 0.5 
                    else: net_score = net ** 1.2 
                    
                    fit = net_score * (pf ** 0.5) * trade_penalty / ((mdd ** 0.5) + 1.0)
                    if alpha_money > 0: fit *= 1.5 
                else: 
                    fit = net * ((mdd ** 0.5) + 1.0) / (pf + 0.001)
                    if alpha_money > 0: fit /= 1.5 
                    
                if fit > best_fit:
                    best_fit = fit
                    if s_id == "ALL_FORCES":
                        bp = {'b_team': dna_b_team, 's_team': dna_s_team, 'macro': dna_macro, 'vol': dna_vol, 'tp': rtp, 'sl': rsl, 'wh': rwh, 'rd': rrd, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'comms': comms, 'fit': fit}
                    elif is_meta:
                        bp = {'b1': dna_b[0], 's1': dna_s[0], 'tp1': dna_tp[0], 'sl1': dna_sl[0], 'b2': dna_b[1], 's2': dna_s[1], 'tp2': dna_tp[1], 'sl2': dna_sl[1], 'b3': dna_b[2], 's3': dna_s[2], 'tp3': dna_tp[2], 'sl3': dna_sl[2], 'b4': dna_b[3], 's4': dna_s[3], 'tp4': dna_tp[3], 'sl4': dna_sl[3], 'wh': rwh, 'rd': rrd, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'comms': comms, 'fit': fit}
                    else:
                        bp = {'tp': rtp, 'sl': rsl, 'wh': rwh, 'rd': rrd, 'reinv': reinv_q, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'comms': comms, 'fit': fit}
        
        elapsed = time.time() - start_time
        pct_done = int(((c + 1) / chunks) * 100)
        combos = (c + 1) * chunk_size
        eta = (elapsed / (c + 1)) * (chunks - c - 1)
        
        dyn_spinner = f"""
        <style>
        .loader-container {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.8); padding: 30px; border-radius: 20px; border: 2px solid cyan; }}
        .rocket {{ font-size: 6rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 15px cyan); }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .prog-text {{ color: cyan; font-size: 1.5rem; font-weight: bold; margin-top: 10px; }}
        .hud-text {{ color: #00FF00; font-size: 1.1rem; margin-top: 5px; font-family: monospace; }}
        </style>
        <div class="loader-container">
            <div class="rocket">🚀</div>
            <div class="prog-text">FORJANDO IA: {s_id}</div>
            <div class="hud-text">Progreso: {pct_done}%</div>
            <div class="hud-text">Combos Evaluados: {combos:,}</div>
            <div class="hud-text" style="color: yellow;">ETA: {eta:.1f} segs</div>
        </div>
        """
        ph_holograma.markdown(dyn_spinner, unsafe_allow_html=True)
        
    return bp

# 📋 REPORTE UNIVERSAL 📋
def generar_reporte_universal(df_base, cap_ini, com_pct):
    res_str = f"📋 **REPORTE UNIVERSAL OMNI-BRAIN (V94.0)**\n\n"
    res_str += f"⏱️ Temporalidad: {intervalo_sel} | 📊 Velas: {len(df_base)}\n\n"
    buy_hold_ret = ((df_base['Close'].iloc[-1] - df_base['Open'].iloc[0]) / df_base['Open'].iloc[0]) * 100
    res_str += f"📈 RENDIMIENTO DEL HOLD: **{buy_hold_ret:.2f}%**\n\n"
    
    for s_id in estrategias:
        wh_val = st.session_state.get(f'w_wh_{s_id}' if s_id not in ["GENESIS","ROCKET","ALL_FORCES"] else f'w_wh_{"gen" if s_id=="GENESIS" else "roc" if s_id=="ROCKET" else "allf"}', 2.5)
        rd_val = st.session_state.get(f'w_rd_{s_id}' if s_id not in ["GENESIS","ROCKET","ALL_FORCES"] else f'w_rd_{"gen" if s_id=="GENESIS" else "roc" if s_id=="ROCKET" else "allf"}', 1.5)
        df_strat = inyectar_adn(df_base.copy(), rd_val, wh_val)
        
        if s_id == "ALL_FORCES":
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            b_team = st.session_state.get('w_b_team_ALL_FORCES', [])
            s_team = st.session_state.get('w_s_team_ALL_FORCES', [])
            m_filt = st.session_state.get('w_macro_ALL_FORCES', "All-Weather")
            v_filt = st.session_state.get('w_vol_ALL_FORCES', "All-Weather")
            
            macro_mask = np.ones(len(df_strat), dtype=bool)
            if m_filt == "Bull Only (Precio > EMA 200)": macro_mask = df_strat['Macro_Bull'].values
            elif m_filt == "Bear Only (Precio < EMA 200)": macro_mask = ~df_strat['Macro_Bull'].values
            
            vol_mask = np.ones(len(df_strat), dtype=bool)
            if v_filt == "Trend (ADX > 25)": vol_mask = df_strat['ADX'].values >= 25
            elif v_filt == "Range (ADX < 25)": vol_mask = df_strat['ADX'].values < 25
            
            for r in b_team:
                if r in df_strat.columns: f_buy |= df_strat[r].values
            f_buy &= (macro_mask & vol_mask)
            
            for r in s_team:
                if r in df_strat.columns: f_sell |= df_strat[r].values
                
            b_c_arr, s_c_arr = np.asarray(f_buy, dtype=bool), np.asarray(f_sell, dtype=bool)
            t_arr = np.asarray(np.full(len(df_strat), st.session_state.get('w_tp_ALL_FORCES', 50.0)), dtype=np.float64)
            sl_arr = np.asarray(np.full
