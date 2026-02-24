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
ph_holograma = st.empty()

# ==========================================
# 🧠 UTILIDADES NUMPY Y RÉPLICAS TV
# ==========================================
def npshift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0: result[:num] = fill_value; result[num:] = arr[:-num]
    elif num < 0: result[num:] = fill_value; result[:num] = arr[-num:]
    else: result[:] = arr
    return result

def npshift_bool(arr, num, fill_value=False):
    result = np.empty_like(arr, dtype=bool)
    if num > 0: result[:num] = fill_value; result[num:] = arr[:-num]
    elif num < 0: result[num:] = fill_value; result[:num] = arr[-num:]
    else: result[:] = arr
    return result

def get_tv_pivot(series, left, right, is_high=True):
    window = left + right + 1
    roll = series.rolling(window).max() if is_high else series.rolling(window).min()
    is_pivot = series.shift(right) == roll
    return series.shift(right).where(is_pivot, np.nan).ffill()

# ==========================================
# 🧠 CATÁLOGOS Y DOCTRINAS TÁCTICAS
# ==========================================
estrategias = [
    "ROCKET_ULTRA", "ROCKET_COMMANDER", "APEX_HYBRID", "MERCENARY",
    "QUADRIX", "JUGGERNAUT", "GENESIS", "ROCKET", "ALL_FORCES", 
    "TRINITY", "DEFCON", "TARGET_LOCK", "THERMAL", "PINK_CLIMAX", "PING_PONG", "NEON_SQUEEZE", "COMMANDER"
]

tab_id_map = {
    "👑 ROCKET ULTRA": "ROCKET_ULTRA", "🚀 ROCKET COMMANDER": "ROCKET_COMMANDER",
    "⚡ APEX ABSOLUTO": "APEX_HYBRID", "🔫 MERCENARY": "MERCENARY",
    "🌌 QUADRIX": "QUADRIX", "⚔️ JUGGERNAUT V356": "JUGGERNAUT", 
    "🌌 GENESIS": "GENESIS", "👑 ROCKET": "ROCKET", "🌟 ALL FORCES": "ALL_FORCES",
    "💠 TRINITY": "TRINITY", "🚀 DEFCON": "DEFCON", "🎯 TARGET_LOCK": "TARGET_LOCK", 
    "🌡️ THERMAL": "THERMAL", "🌸 PINK_CLIMAX": "PINK_CLIMAX", "🏓 PING_PONG": "PING_PONG", 
    "🐛 NEON_SQUEEZE": "NEON_SQUEEZE", "👑 COMMANDER": "COMMANDER"
}

doctrinas = {
    "ROCKET_ULTRA": "Cazador Adaptativo (V55). Interpola temporalidades (desde 1m hasta 1D) para mutar sus parámetros. Usa Trailing Stop dinámico rastreando el precio máximo histórico (Highest Price) de la operación.",
    "ROCKET_COMMANDER": "El Almirante (V60.2). Cruza el Radar de Gravedad con Osciladores WaveTrend. Detecta anomalías 'Magenta' filtrando el ruido con exigencias estrictas de volumen y mechas (Wick Rejection).",
    "APEX_HYBRID": "El Depredador Absoluto (V337). Combina la precisión del Escudo Aegis y Target Lock con el poder explosivo del motor Defcon. Ignora caídas a menos que detecte una Vela Rosa.",
    "MERCENARY": "Francotirador de Alta Frecuencia (1.1). Entra y sale rápido basándose en micro-tendencias (ADX), compresión de Bandas de Bollinger y cierres implacables bajo la EMA 50.",
    "QUADRIX": "Matriz Cuádruple. Combina el oscilador WaveTrend (WT1/WT2) con Z-Score y regresiones lineales para detectar reversiones exactas en el precio.",
    "JUGGERNAUT": "El Tanque Blindado (V356). Su Escudo Aegis bloquea compras en caídas libres (>1.5 ATR). Suma puntajes (hasta 99%) considerando Muros Térmicos y divergencias.",
    "GENESIS": "La Matriz Original. Analiza 4 Cuadrantes (Alcista/Bajista cruzado con Rango/Tendencia) y asigna un equipo de algoritmos para cada clima del mercado.",
    "ROCKET": "Variante Agresiva de la Matriz. Prioriza armas de ruptura de volatilidad (Squeeze, Defcon, Climax) buscando explosiones direccionales.",
    "ALL_FORCES": "El Enjambre. Pone a todos los algoritmos base a operar al mismo tiempo bajo un filtro macro global. Busca maximizar el ADO (Trades por día).",
    "TRINITY": "Gatillo de Reversión. Compra cuando el precio cae fuerte pero el RSI marca sobreventa profunda (< 35) y rechaza la caída.",
    "DEFCON": "Buscador de Squeeze (V329). Opera cuando las Bandas de Bollinger se comprimen dentro del Canal de Keltner y el precio rompe con furia alcista.",
    "TARGET_LOCK": "Radar Gravitacional (V332). Detecta niveles históricos de soporte/resistencia y opera rebotes usando una tolerancia matemática del ATR.",
    "THERMAL": "Termómetro de Muros (V331). Si el 'Suelo' pesa más de 4, asume que es irrompible y entra en la operación al primer cruce de RSI.",
    "PINK_CLIMAX": "Cazador de Ballenas. Dispara solo cuando detecta un volumen relativo masivo (RVol extremo) acompañado de una mecha inferior gigante y RSI estrangulado.",
    "PING_PONG": "Física de Regresión Lineal. Usa álgebra para calcular la pendiente de los últimos 5 cierres. Entra y sale rebotando dentro del canal.",
    "NEON_SQUEEZE": "Expansión Ligera. Caza rupturas de volatilidad comparando el ancho de las Bandas de Bollinger actuales contra su promedio histórico.",
    "COMMANDER": "Infantería Pesada. Agrupa Climax, Thermal y Target Lock en un solo escuadrón."
}

base_b = ['Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 'Commander_Buy', 'Lev_Buy']
base_s = ['Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 'Commander_Sell', 'Lev_Sell']
quadrix_b = ['Q_Pink_Whale_Buy', 'Q_Lock_Bounce', 'Q_Lock_Break', 'Q_Neon_Up', 'Q_Defcon_Buy', 'Q_Therm_Bounce', 'Q_Therm_Vacuum', 'Q_Nuclear_Buy', 'Q_Early_Buy', 'Q_Rebound_Buy']
quadrix_s = ['Q_Lock_Reject', 'Q_Lock_Breakd', 'Q_Neon_Dn', 'Q_Defcon_Sell', 'Q_Therm_Wall_Sell', 'Q_Therm_Panic_Sell', 'Q_Nuclear_Sell', 'Q_Early_Sell']

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

for s_id in estrategias:
    if f'opt_status_{s_id}' not in st.session_state: st.session_state[f'opt_status_{s_id}'] = False
    if f'champion_{s_id}' not in st.session_state:
        if s_id == "ALL_FORCES": st.session_state[f'champion_{s_id}'] = {'b_team': ['Commander_Buy'], 's_team': ['Commander_Sell'], 'macro': "All-Weather", 'vol': "All-Weather", 'tp': 20.0, 'sl': 5.0, 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 'whale_f': 2.5, 'ado': 4.0, 'reinv': 0.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0}
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
    vault['net'] = bp.get('net', 0.0)
    vault['winrate'] = bp.get('winrate', 0.0)

def wipe_ui_cache():
    for key in list(st.session_state.keys()):
        if key.startswith("ui_"): del st.session_state[key]

# ==========================================
# 🌍 SIDEBAR E INFRAESTRUCTURA
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 OMNI-FORGE V123.0</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True, key="btn_purge"): 
    st.cache_data.clear(); wipe_ui_cache(); gc.collect(); st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🛑 ABORTAR OPTIMIZACIÓN", use_container_width=True, key="btn_abort"):
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

if st.sidebar.button(f"🧠 DEEP MINE GLOBAL ({global_epochs*3}k)", type="primary", use_container_width=True, key="btn_global"):
    st.session_state['run_global'] = True
    st.rerun()

@st.cache_data(ttl=3600, show_spinner="📡 Construyendo Geometría Fractal & WaveTrend (V123)...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, offset):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        all_ohlcv, current_ts, error_count = [], start_ts, 0
        while current_ts < end_ts:
            try: ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=1000); error_count = 0 
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
        
        ap = (df['High'] + df['Low'] + df['Close']) / 3.0
        esa = ap.ewm(span=10, adjust=False).mean()
        d_wt = (ap - esa).abs().ewm(span=10, adjust=False).mean()
        df['WT1'] = ((ap - esa) / (0.015 * d_wt.replace(0, 1))).ewm(span=21, adjust=False).mean()
        df['WT2'] = df['WT1'].rolling(4, min_periods=1).mean()
        
        df['Basis'] = df['Close'].rolling(20, min_periods=1).mean()
        dev = df['Close'].rolling(20, min_periods=1).std(ddof=0).replace(0, 1) 
        df['BBU'] = df['Basis'] + (2.0 * dev); df['BBL'] = df['Basis'] - (2.0 * dev)
        df['BB_Width'] = (df['BBU'] - df['BBL']) / df['Basis'].replace(0, 1)
        df['BB_Width_Avg'] = df['BB_Width'].rolling(20, min_periods=1).mean()
        df['BB_Delta'] = df['BB_Width'] - df['BB_Width'].shift(1).fillna(0)
        df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10, min_periods=1).mean()
        
        kc_basis = df['Close'].rolling(20, min_periods=1).mean()
        df['KC_Upper'] = kc_basis + (df['ATR'] * 1.5); df['KC_Lower'] = kc_basis - (df['ATR'] * 1.5)
        df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
        df['Z_Score'] = (df['Close'] - df['Basis']) / dev.replace(0, 1)
        df['RSI_BB_Basis'] = df['RSI'].rolling(14, min_periods=1).mean()
        df['RSI_BB_Dev'] = df['RSI'].rolling(14, min_periods=1).std(ddof=0).replace(0,1) * 2.0
        
        df['Vela_Verde'] = df['Close'] > df['Open']; df['Vela_Roja'] = df['Close'] < df['Open']
        df['body_size'] = abs(df['Close'] - df['Open']).replace(0, 0.0001)
        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['is_falling_knife'] = (df['Open'].shift(1) - df['Close'].shift(1)) > (df['ATR'].shift(1) * 1.5)
        
        df['PL30_P'] = get_tv_pivot(df['Low'], 30, 3, False)
        df['PH30_P'] = get_tv_pivot(df['High'], 30, 3, True)
        df['PL100_P'] = get_tv_pivot(df['Low'], 100, 5, False)
        df['PH100_P'] = get_tv_pivot(df['High'], 100, 5, True)
        df['PL300_P'] = get_tv_pivot(df['Low'], 300, 5, False)
        df['PH300_P'] = get_tv_pivot(df['High'], 300, 5, True)
        
        df['PL30_L'] = df['Low'].shift(1).rolling(30, min_periods=1).min()
        df['PH30_L'] = df['High'].shift(1).rolling(30, min_periods=1).max()
        df['PL100_L'] = df['Low'].shift(1).rolling(100, min_periods=1).min()
        df['PH100_L'] = df['High'].shift(1).rolling(100, min_periods=1).max()
        df['PL300_L'] = df['Low'].shift(1).rolling(300, min_periods=1).min()
        df['PH300_L'] = df['High'].shift(1).rolling(300, min_periods=1).max()

        df['RSI_Cross_Up'] = (df['RSI'] > df['RSI_MA']) & (df['RSI'].shift(1) <= df['RSI_MA'].shift(1))
        df['RSI_Cross_Dn'] = (df['RSI'] < df['RSI_MA']) & (df['RSI'].shift(1) >= df['RSI_MA'].shift(1))
        df['Macro_Bull'] = df['Close'] >= df['EMA_200']
        df['PP_Slope'] = (2*df['Close'] + df['Close'].shift(1) - df['Close'].shift(3) - 2*df['Close'].shift(4)) / 10.0
        gc.collect()
        return df, "OK"
    except Exception as e: return pd.DataFrame(), f"❌ ERROR FATAL GENERAL: {str(e)}"

df_global, status_api = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, utc_offset)
if not df_global.empty: dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)
else: st.error(status_api); st.stop()

# ==========================================
# 📊 REPORTE UNIVERSAL DIRECTO
# ==========================================
def generar_reporte_universal(cap_ini, com_pct):
    res_str = f"📋 **REPORTE OMNI-FORGE V123.0**\n\n"
    res_str += f"⏱️ Temporalidad: {intervalo_sel} | 📊 Velas: {len(df_global)}\n\n"
    buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
    res_str += f"📈 RENDIMIENTO DEL HOLD: **{buy_hold_ret:.2f}%**\n\n"
    for s_id in estrategias:
        v = st.session_state.get(f'champion_{s_id}', {})
        opt_icon = "✅" if st.session_state.get(f'opt_status_{s_id}', False) else "➖"
        res_str += f"⚔️ **{s_id}** [{opt_icon}]\nNet Profit: ${v.get('net',0):,.2f} \nWin Rate: {v.get('winrate',0):.1f}%\n---\n"
    return res_str

if st.sidebar.button("📊 GENERAR REPORTE UNIVERSAL", use_container_width=True, key="btn_univ_report"):
    st.sidebar.text_area("Copia tu Reporte:", value=generar_reporte_universal(capital_inicial, comision_pct), height=400)

# ==========================================
# 🏆 SCOREBOARD (LEADERBOARD) UNIVERSAL
# ==========================================
st.markdown("<h3 style='text-align: center; color: #FFD700;'>🏆 SALÓN DE LA FAMA (Clasificación por Rentabilidad Neta)</h3>", unsafe_allow_html=True)
leaderboard_data = []
for s in estrategias:
    v = st.session_state.get(f'champion_{s}', {})
    fit = v.get('fit', -float('inf'))
    if fit != -float('inf'):
        net_val = v.get('net', 0)
        leaderboard_data.append({"Estrategia": s, "Neto_Num": net_val, "Rentabilidad Neta": f"${net_val:,.2f} ({net_val/capital_inicial*100:.2f}%)", "WinRate": f"{v.get('winrate', 0):.1f}%", "Puntaje IA (Riesgo)": f"{fit:,.0f}"})
if leaderboard_data:
    # 🏅 Ordenamiento Estricto por Ganancia ($)
    leaderboard_data.sort(key=lambda x: x['Neto_Num'], reverse=True)
    for item in leaderboard_data: del item['Neto_Num'] # Removemos la columna auxiliar
    st.table(pd.DataFrame(leaderboard_data))
else: st.info("La bóveda está vacía. Inicie una Forja individual o Global para registrar a los campeones.")
st.markdown("---")

# ==========================================
# 🔥 PURE NUMPY BACKEND (V123.0) 🔥
# ==========================================
a_c = df_global['Close'].values; a_o = df_global['Open'].values; a_h = df_global['High'].values; a_l = df_global['Low'].values
a_rsi = df_global['RSI'].values; a_rsi_ma = df_global['RSI_MA'].values; a_adx = df_global['ADX'].values
a_bbl = df_global['BBL'].values; a_bbu = df_global['BBU'].values; a_bw = df_global['BB_Width'].values
a_bwa_s1 = npshift(df_global['BB_Width_Avg'].values, 1, -1.0)
a_wt1 = df_global['WT1'].values; a_wt2 = df_global['WT2'].values
a_ema50 = df_global['EMA_50'].values; a_ema200 = df_global['EMA_200'].values; a_atr = df_global['ATR'].values
a_rvol = df_global['RVol'].values; a_hvol = df_global['High_Vol'].values
a_vv = df_global['Vela_Verde'].values; a_vr = df_global['Vela_Roja'].values
a_rcu = df_global['RSI_Cross_Up'].values; a_rcd = df_global['RSI_Cross_Dn'].values
a_sqz_on = df_global['Squeeze_On'].values; a_bb_delta = df_global['BB_Delta'].values; a_bb_delta_avg = df_global['BB_Delta_Avg'].values
a_zscore = df_global['Z_Score'].values; a_rsi_bb_b = df_global['RSI_BB_Basis'].values; a_rsi_bb_d = df_global['RSI_BB_Dev'].values
a_lw = df_global['lower_wick'].values; a_bs = df_global['body_size'].values
a_mb = df_global['Macro_Bull'].values; a_fk = df_global['is_falling_knife'].values
a_pp_slope = df_global['PP_Slope'].fillna(0).values

a_pl30_p = df_global['PL30_P'].fillna(0).values; a_ph30_p = df_global['PH30_P'].fillna(99999).values
a_pl100_p = df_global['PL100_P'].fillna(0).values; a_ph100_p = df_global['PH100_P'].fillna(99999).values
a_pl300_p = df_global['PL300_P'].fillna(0).values; a_ph300_p = df_global['PH300_P'].fillna(99999).values
a_pl30_l = df_global['PL30_L'].fillna(0).values; a_ph30_l = df_global['PH30_L'].fillna(99999).values
a_pl100_l = df_global['PL100_L'].fillna(0).values; a_ph100_l = df_global['PH100_L'].fillna(99999).values
a_pl300_l = df_global['PL300_L'].fillna(0).values; a_ph300_l = df_global['PH300_L'].fillna(99999).values

a_c_s1 = npshift(a_c, 1, 0.0); a_o_s1 = npshift(a_o, 1, 0.0); a_l_s1 = npshift(a_l, 1, 0.0); a_l_s5 = npshift(a_l, 5, 0.0)
a_rsi_s1 = npshift(a_rsi, 1, 50.0); a_rsi_s5 = npshift(a_rsi, 5, 50.0)
a_wt1_s1 = npshift(a_wt1, 1, 0.0); a_wt2_s1 = npshift(a_wt2, 1, 0.0)
a_pp_slope_s1 = npshift(a_pp_slope, 1, 0.0)

def calcular_señales_numpy(s_id, hitbox, therm_w, adx_th, whale_f):
    n_len = len(a_c); s_dict = {}
    if s_id in ["ROCKET_ULTRA", "MERCENARY", "ALL_FORCES", "GENESIS", "ROCKET", "QUADRIX"]:
        a_tsup = np.maximum(a_pl30_l, np.maximum(a_pl100_l, a_pl300_l)); a_tres = np.minimum(a_ph30_l, np.minimum(a_ph100_l, a_ph300_l))
        pl30, ph30, pl100, ph100, pl300, ph300 = a_pl30_l, a_ph30_l, a_pl100_l, a_ph100_l, a_pl300_l, a_ph300_l
    else:
        a_tsup = np.maximum(a_pl30_p, np.maximum(a_pl100_p, a_pl300_p)); a_tres = np.minimum(a_ph30_p, np.minimum(a_ph100_p, a_ph300_p))
        pl30, ph30, pl100, ph100, pl300, ph300 = a_pl30_p, a_ph30_p, a_pl100_p, a_ph100_p, a_pl30_p, a_ph300_p

    a_dsup = np.abs(a_c - a_tsup) / a_c * 100; a_dres = np.abs(a_c - a_tres) / a_c * 100
    sr_val = a_atr * 2.0
    ceil_w = np.where((ph30 > a_c) & (ph30 <= a_c + sr_val), 1, 0) + np.where((pl30 > a_c) & (pl30 <= a_c + sr_val), 1, 0) + np.where((ph100 > a_c) & (ph100 <= a_c + sr_val), 3, 0) + np.where((pl100 > a_c) & (pl100 <= a_c + sr_val), 3, 0) + np.where((ph300 > a_c) & (ph300 <= a_c + sr_val), 5, 0) + np.where((pl300 > a_c) & (pl300 <= a_c + sr_val), 5, 0)
    floor_w = np.where((ph30 < a_c) & (ph30 >= a_c - sr_val), 1, 0) + np.where((pl30 < a_c) & (pl30 >= a_c - sr_val), 1, 0) + np.where((ph100 < a_c) & (ph100 >= a_c - sr_val), 3, 0) + np.where((pl100 < a_c) & (pl100 >= a_c - sr_val), 3, 0) + np.where((ph300 < a_c) & (ph300 >= a_c - sr_val), 5, 0) + np.where((pl300 < a_c) & (pl300 >= a_c - sr_val), 5, 0)

    trinity_safe = a_mb & ~a_fk
    neon_up = a_sqz_on & (a_c >= a_bbu * 0.999) & a_vv; neon_dn = a_sqz_on & (a_c <= a_bbl * 1.001) & a_vr
    defcon_level = np.full(n_len, 5); m4 = neon_up | neon_dn; defcon_level[m4] = 4; m3 = m4 & (a_bb_delta > 0); defcon_level[m3] = 3; m2 = m3 & (a_bb_delta > a_bb_delta_avg) & (a_adx > adx_th); defcon_level[m2] = 2; m1 = m2 & (a_bb_delta > a_bb_delta_avg * 1.5) & (a_adx > adx_th + 5) & (a_rvol > 1.2); defcon_level[m1] = 1

    cond_defcon_buy = (defcon_level <= 2) & neon_up; cond_defcon_sell = (defcon_level <= 2) & neon_dn
    is_abyss = floor_w == 0; is_hard_wall = ceil_w >= therm_w
    cond_therm_buy_bounce = (floor_w >= therm_w) & a_rcu & ~is_hard_wall; cond_therm_buy_vacuum = (ceil_w <= 3) & neon_up & ~is_abyss
    cond_therm_sell_wall = (ceil_w >= therm_w) & a_rcd; cond_therm_sell_panic = is_abyss & a_vr

    tol = a_atr * 0.5; is_grav_sup = a_dsup < hitbox; is_grav_res = a_dres < hitbox
    cross_up_res = (a_c > a_tres) & (a_c_s1 <= npshift(a_tres, 1, 0)); cross_dn_sup = (a_c < a_tsup) & (a_c_s1 >= npshift(a_tsup, 1, 0))
    cond_lock_buy_bounce = is_grav_sup & (a_l <= a_tsup + tol) & (a_c > a_tsup) & a_vv
    cond_lock_buy_break = is_grav_res & cross_up_res & a_hvol & a_vv
    cond_lock_sell_reject = is_grav_res & (a_h >= a_tres - tol) & (a_c < a_tres) & a_vr
    cond_lock_sell_breakd = is_grav_sup & cross_dn_sup & a_vr

    flash_vol = (a_rvol > whale_f * 0.8) & (np.abs(a_c - a_o) > a_atr * 0.3)
    whale_buy = flash_vol & a_vv; whale_sell = flash_vol & a_vr
    whale_memory = whale_buy | npshift_bool(whale_buy, 1) | npshift_bool(whale_buy, 2) | whale_sell | npshift_bool(whale_sell, 1) | npshift_bool(whale_sell, 2)
    is_whale_icon = whale_buy & ~npshift_bool(whale_buy, 1)

    rsi_vel = a_rsi - a_rsi_s1
    pre_pump = ((a_h > a_bbu) | (rsi_vel > 5)) & flash_vol & a_vv; pump_memory = pre_pump | npshift_bool(pre_pump, 1) | npshift_bool(pre_pump, 2)
    pre_dump = ((a_l < a_bbl) | (rsi_vel < -5)) & flash_vol & a_vr; dump_memory = pre_dump | npshift_bool(pre_dump, 1) | npshift_bool(pre_dump, 2)

    retro_peak = (a_rsi < 30) & (a_c < a_bbl); retro_peak_sell = (a_rsi > 70) & (a_c > a_bbu)
    k_break_up = (a_rsi > (a_rsi_bb_b + a_rsi_bb_d)) & (a_rsi_s1 <= npshift(a_rsi_bb_b + a_rsi_bb_d, 1))
    support_buy = is_grav_sup & a_rcu; support_sell = is_grav_res & a_rcd
    div_bull = (a_l_s1 < a_l_s5) & (a_rsi_s1 > a_rsi_s5) & (a_rsi < 35); div_bear = (npshift(a_h, 1, 0) > npshift(a_h, 5, 0)) & (a_rsi_s1 < a_rsi_s5) & (a_rsi > 65)

    buy_score = np.zeros(n_len); base_mask = retro_peak | k_break_up | support_buy | div_bull
    buy_score = np.where(base_mask & retro_peak, 50.0, np.where(base_mask & ~retro_peak, 30.0, buy_score))
    buy_score += np.where(is_grav_sup, 25.0, 0.0); buy_score += np.where(whale_memory, 20.0, 0.0); buy_score += np.where(pump_memory, 15.0, 0.0); buy_score += np.where(div_bull, 15.0, 0.0); buy_score += np.where(k_break_up & ~retro_peak, 15.0, 0.0); buy_score += np.where(a_zscore < -2.0, 15.0, 0.0)
    buy_score = np.where(buy_score > 99, 99.0, buy_score)

    sell_score = np.zeros(n_len); base_mask_s = retro_peak_sell | a_rcd | support_sell | div_bear
    sell_score = np.where(base_mask_s & retro_peak_sell, 50.0, np.where(base_mask_s & ~retro_peak_sell, 30.0, sell_score))
    sell_score += np.where(is_grav_res, 25.0, 0.0); sell_score += np.where(whale_memory, 20.0, 0.0); sell_score += np.where(dump_memory, 15.0, 0.0); sell_score += np.where(div_bear, 15.0, 0.0); sell_score += np.where(a_rcd & ~retro_peak_sell, 15.0, 0.0); sell_score += np.where(a_zscore > 2.0, 15.0, 0.0)
    sell_score = np.where(sell_score > 99, 99.0, sell_score)

    is_magenta = (buy_score >= 70) | retro_peak; is_magenta_sell = (sell_score >= 70) | retro_peak_sell
    cond_pink_whale_buy = is_magenta & is_whale_icon

    wt_cross_up = (a_wt1 > a_wt2) & (a_wt1_s1 <= a_wt2_s1); wt_cross_dn = (a_wt1 < a_wt2) & (a_wt1_s1 >= a_wt2_s1)
    wt_oversold = a_wt1 < -60; wt_overbought = a_wt1 > 60

    s_dict['Ping_Buy'] = (a_adx < adx_th) & (a_c < a_bbl) & a_vv; s_dict['Ping_Sell'] = (a_c > a_bbu) | (a_rsi > 70)
    s_dict['Squeeze_Buy'] = neon_up; s_dict['Squeeze_Sell'] = (a_c < a_ema50)
    s_dict['Thermal_Buy'] = cond_therm_buy_bounce; s_dict['Thermal_Sell'] = cond_therm_sell_wall
    s_dict['Climax_Buy'] = cond_pink_whale_buy; s_dict['Climax_Sell'] = (a_rsi > 80)
    s_dict['Lock_Buy'] = cond_lock_buy_bounce; s_dict['Lock_Sell'] = cond_lock_sell_reject
    s_dict['Defcon_Buy'] = cond_defcon_buy; s_dict['Defcon_Sell'] = cond_defcon_sell
    s_dict['Jugg_Buy'] = a_mb & (a_c > a_ema50) & (a_c_s1 < npshift(a_ema50,1)) & a_vv & ~a_fk; s_dict['Jugg_Sell'] = (a_c < a_ema50)
    s_dict['Trinity_Buy'] = a_mb & (a_rsi < 35) & a_vv & ~a_fk; s_dict['Trinity_Sell'] = (a_rsi > 75) | (a_c < a_ema200)
    s_dict['Lev_Buy'] = a_mb & a_rcu & (a_rsi < 45); s_dict['Lev_Sell'] = (a_c < a_ema200)
    s_dict['Commander_Buy'] = cond_pink_whale_buy | cond_therm_buy_bounce | cond_lock_buy_bounce; s_dict['Commander_Sell'] = cond_therm_sell_wall | (a_c < a_ema50)

    s_dict['Q_Pink_Whale_Buy'] = cond_pink_whale_buy; s_dict['Q_Lock_Bounce'] = cond_lock_buy_bounce; s_dict['Q_Lock_Break'] = cond_lock_buy_break
    s_dict['Q_Neon_Up'] = neon_up; s_dict['Q_Defcon_Buy'] = cond_defcon_buy; s_dict['Q_Therm_Bounce'] = cond_therm_buy_bounce; s_dict['Q_Therm_Vacuum'] = cond_therm_buy_vacuum
    s_dict['Q_Nuclear_Buy'] = is_magenta & (wt_oversold | wt_cross_up); s_dict['Q_Early_Buy'] = is_magenta; s_dict['Q_Rebound_Buy'] = a_rcu & ~is_magenta
    s_dict['Q_Lock_Reject'] = cond_lock_sell_reject; s_dict['Q_Lock_Breakd'] = cond_lock_sell_breakd; s_dict['Q_Neon_Dn'] = neon_dn
    s_dict['Q_Defcon_Sell'] = cond_defcon_sell; s_dict['Q_Therm_Wall_Sell'] = cond_therm_sell_wall; s_dict['Q_Therm_Panic_Sell'] = cond_therm_sell_panic
    s_dict['Q_Nuclear_Sell'] = (a_rsi > 70) & (wt_overbought | wt_cross_dn); s_dict['Q_Early_Sell'] = (a_rsi > 70) & a_vr

    s_dict['MERC_PING'] = s_dict['Ping_Buy']; s_dict['MERC_JUGG'] = s_dict['Jugg_Buy']; s_dict['MERC_CLIM'] = (a_rvol > whale_f) & (a_lw > (a_bs * 2.0)) & (a_rsi < 35) & a_vv
    s_dict['MERC_SELL'] = (a_c < a_ema50) | (a_c < a_ema200)

    s_dict['APEX_BUY'] = cond_pink_whale_buy | (trinity_safe & (cond_defcon_buy | cond_lock_buy_bounce | cond_lock_buy_break))
    s_dict['APEX_SELL'] = cond_defcon_sell | cond_lock_sell_reject | cond_lock_sell_breakd
    s_dict['JUGGERNAUT_BUY_V356'] = (trinity_safe & (cond_defcon_buy | cond_therm_buy_bounce | cond_therm_buy_vacuum | cond_lock_buy_bounce | cond_lock_buy_break)) | cond_pink_whale_buy
    s_dict['JUGGERNAUT_SELL_V356'] = cond_defcon_sell | cond_therm_sell_wall | cond_therm_sell_panic | cond_lock_sell_reject | cond_lock_sell_breakd

    matrix_active = is_grav_sup | (floor_w >= 3)
    final_wick_req = np.where(matrix_active, 0.15, np.where(a_adx < 40, 0.4, 0.5)); final_vol_req = np.where(matrix_active, 1.2, np.where(a_adx < 40, 1.5, 1.8))
    wick_rej_buy = a_lw > (a_bs * final_wick_req); wick_rej_sell = (a_h - np.maximum(a_o, a_c)) > (a_bs * final_wick_req); vol_stop_chk = a_rvol > final_vol_req
    climax_buy_cmdr = is_magenta & (wick_rej_buy | vol_stop_chk) & (a_c > a_o); climax_sell_cmdr = is_magenta_sell & (wick_rej_sell | vol_stop_chk)
    ping_buy_cmdr = (a_pp_slope > 0) & (a_pp_slope_s1 <= 0) & matrix_active & (a_c > a_o); ping_sell_cmdr = (a_pp_slope < 0) & (a_pp_slope_s1 >= 0) & matrix_active
    
    s_dict['RC_Buy_Q1'] = climax_buy_cmdr | ping_buy_cmdr | cond_pink_whale_buy; s_dict['RC_Sell_Q1'] = ping_sell_cmdr | cond_defcon_sell
    s_dict['RC_Buy_Q2'] = cond_therm_buy_bounce | climax_buy_cmdr | ping_buy_cmdr; s_dict['RC_Sell_Q2'] = cond_defcon_sell | cond_lock_sell_reject
    s_dict['RC_Buy_Q3'] = cond_pink_whale_buy | cond_defcon_buy; s_dict['RC_Sell_Q3'] = climax_sell_cmdr | ping_sell_cmdr 
    s_dict['RC_Buy_Q4'] = ping_buy_cmdr | cond_defcon_buy | cond_lock_buy_bounce; s_dict['RC_Sell_Q4'] = cond_defcon_sell | cond_therm_sell_panic

    regime = np.where(a_mb & (a_adx >= adx_th), 1, np.where(a_mb & (a_adx < adx_th), 2, np.where(~a_mb & (a_adx >= adx_th), 3, 4)))
    return s_dict, regime

@njit(fastmath=True)
def simular_crecimiento_exponencial(h_arr, l_arr, c_arr, o_arr, b_c, s_c, t_arr, sl_arr, cap_ini, com_pct, reinvest_pct):
    cap_act, divs, en_pos, p_ent, tp_act, sl_act, pos_size, invest_amt = cap_ini, 0.0, False, 0.0, 0.0, 0.0, 0.0, 0.0
    g_profit, g_loss, num_trades, max_dd, peak = 0.0, 0.0, 0, 0.0, cap_ini
    for i in range(len(h_arr)):
        if en_pos:
            tp_p = p_ent * (1.0 + tp_act/100.0); sl_p = p_ent * (1.0 - sl_act/100.0)
            if l_arr[i] <= sl_p:
                gross = pos_size * (1.0 - sl_act/100.0); net = gross - (gross * com_pct); profit = net - invest_amt
                if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                else: cap_act += profit
                g_loss += abs(profit); num_trades += 1; en_pos = False
            elif h_arr[i] >= tp_p:
                gross = pos_size * (1.0 + tp_act/100.0); net = gross - (gross * com_pct); profit = net - invest_amt
                if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                else: cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1; en_pos = False
            elif s_c[i]:
                ret = (c_arr[i] - p_ent) / p_ent; gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
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
            comm_in = invest_amt * com_pct; pos_size = invest_amt - comm_in 
            p_ent = o_arr[i+1]; tp_act = t_arr[i]; sl_act = sl_arr[i]; en_pos = True
    return (cap_act + divs) - cap_ini, g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0), num_trades, max_dd

def simular_visual(df_sim, cap_ini, reinvest, com_pct):
    registro_trades = []
    n = len(df_sim)
    curva = np.full(n, cap_ini, dtype=float)
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl
