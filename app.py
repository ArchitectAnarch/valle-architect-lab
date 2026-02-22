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
import time
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

# --- MEMORIA IA INSTITUCIONAL ---
buy_rules = ['Pink_Whale_Buy', 'Lock_Bounce', 'Lock_Break', 'Defcon_Buy', 'Neon_Up', 'Therm_Bounce', 'Therm_Vacuum', 'Nuclear_Buy', 'Early_Buy', 'Rebound_Buy', 'Pink_Climax_Buy', 'Ping_Pong_Buy', 'Commander_Buy']
sell_rules = ['Defcon_Sell', 'Neon_Dn', 'Therm_Wall_Sell', 'Therm_Panic_Sell', 'Lock_Reject', 'Lock_Breakd', 'Nuclear_Sell', 'Early_Sell', 'Pink_Climax_Sell', 'Ping_Pong_Sell', 'Commander_Sell']

rocket_b = ['Trinity_Buy', 'Jugg_Buy', 'Defcon_Buy_Sig', 'Lock_Buy', 'Thermal_Buy', 'Climax_Buy', 'Ping_Buy', 'Squeeze_Buy', 'Lev_Buy', 'Commander_Buy']
rocket_s = ['Trinity_Sell', 'Jugg_Sell', 'Defcon_Sell_Sig', 'Lock_Sell', 'Thermal_Sell', 'Climax_Sell', 'Ping_Sell', 'Squeeze_Sell', 'Lev_Sell', 'Commander_Sell']

estrategias = ["TRINITY", "JUGGERNAUT", "DEFCON", "TARGET_LOCK", "THERMAL", "PINK_CLIMAX", "PING_PONG", "NEON_SQUEEZE", "COMMANDER", "GENESIS", "ROCKET"]

for r_idx in range(1, 5):
    if f'gen_r{r_idx}_b' not in st.session_state: st.session_state[f'gen_r{r_idx}_b'] = ['Neon_Up']
    if f'gen_r{r_idx}_s' not in st.session_state: st.session_state[f'gen_r{r_idx}_s'] = ['Neon_Dn']
    if f'gen_r{r_idx}_tp' not in st.session_state: st.session_state[f'gen_r{r_idx}_tp'] = 10.0
    if f'gen_r{r_idx}_sl' not in st.session_state: st.session_state[f'gen_r{r_idx}_sl'] = 3.0
    
    if f'roc_r{r_idx}_b' not in st.session_state: st.session_state[f'roc_r{r_idx}_b'] = ['Commander_Buy']
    if f'roc_r{r_idx}_s' not in st.session_state: st.session_state[f'roc_r{r_idx}_s'] = ['Commander_Sell']
    if f'roc_r{r_idx}_tp' not in st.session_state: st.session_state[f'roc_r{r_idx}_tp'] = 10.0
    if f'roc_r{r_idx}_sl' not in st.session_state: st.session_state[f'roc_r{r_idx}_sl'] = 3.0

for s in estrategias:
    if f'dna_{s}' not in st.session_state: st.session_state[f'dna_{s}'] = ""
    if f'ado_{s}' not in st.session_state: st.session_state[f'ado_{s}'] = 5.0 
    if f'sld_tp_{s}' not in st.session_state: st.session_state[f'sld_tp_{s}'] = 10.0
    if f'sld_sl_{s}' not in st.session_state: st.session_state[f'sld_sl_{s}'] = 3.0
    if f'sld_wh_{s}' not in st.session_state: st.session_state[f'sld_wh_{s}'] = 2.5
    if f'sld_rd_{s}' not in st.session_state: st.session_state[f'sld_rd_{s}'] = 1.5
    if f'sld_reinv_{s}' not in st.session_state: st.session_state[f'sld_reinv_{s}'] = 100.0

css_spinner = """
<style>
.loader-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; pointer-events: none; background: transparent; }
.rocket { font-size: 10rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 35px rgba(0, 255, 255, 1)); }
@keyframes spin { 0% { transform: scale(1) rotate(0deg); } 50% { transform: scale(1.2) rotate(180deg); } 100% { transform: scale(1) rotate(360deg); } }
</style>
<div class="loader-container"><div class="rocket">🚀</div></div>
"""
ph_holograma = st.empty()

st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 ROCKET PROTOCOL V67.0</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True): 
    st.cache_data.clear()
    gc.collect()

st.sidebar.markdown("---")
exchange_sel = st.sidebar.selectbox("🏦 Exchange", ["coinbase", "kucoin", "kraken", "binance"], index=0)
ticker = st.sidebar.text_input("Símbolo Exacto", value="HNT/USD")
utc_offset = st.sidebar.number_input("🌍 Zona Horaria", value=-5.0, step=0.5)

intervalos = {
    "1 Minuto": "1m", "5 Minutos": "5m", "7 Minutos": "7m", "13 Minutos": "13m", 
    "15 Minutos": "15m", "23 Minutos": "23m", "30 Minutos": "30m", "45 Minutos": "45m", 
    "1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d"
}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=6) 
iv_download = intervalos[intervalo_sel]

hoy = datetime.today().date()
is_micro = iv_download in ["1m", "5m", "7m", "13m", "23m", "45m"]
limite_dias = 45 if is_micro else 1500
start_date, end_date = st.sidebar.slider("📅 Scope Histórico", min_value=hoy - timedelta(days=limite_dias), max_value=hoy, value=(hoy - timedelta(days=min(1500, limite_dias)), hoy), format="YYYY-MM-DD")

capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=1000.0, step=100.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0

@st.cache_data(ttl=3600, show_spinner="📡 Sintetizando Velas y Lógicas Avanzadas...")
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
            try:
                ohlcv = ex_class.fetch_ohlcv(sym, base_tf, since=current_ts, limit=fetch_limit)
            except Exception:
                time.sleep(1)
                continue
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
        
        if resample_rule:
            df = df.resample(resample_rule).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            
        df.index = df.index + timedelta(hours=offset)
        df = df[~df.index.duplicated(keep='first')]
        
        if len(df) < 50:
            return pd.DataFrame(), f"❌ Se sintetizaron solo {len(df)} velas de {iv_down}. La IA requiere mínimo 50. Amplíe los días."
            
        # 🔥 CÁLCULOS MATEMÁTICOS BASE 🔥
        df['EMA_200'] = df['Close'].ewm(span=200, min_periods=1, adjust=False).mean()
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
        
        kc_basis = df['Close'].rolling(20, min_periods=1).mean()
        df['KC_Upper'] = kc_basis + (df['ATR'] * 1.5)
        df['KC_Lower'] = kc_basis - (df['ATR'] * 1.5)
        df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
        df['BB_Delta'] = (df['BBU'] - df['BBL']).diff().fillna(0)
        df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10, min_periods=1).mean().fillna(0)
        
        df['Vela_Verde'] = df['Close'] > df['Open']
        df['Vela_Roja'] = df['Close'] < df['Open']
        df['body_size'] = abs(df['Close'] - df['Open']).replace(0, 0.0001)
        df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['is_falling_knife'] = (df['Open'].shift(1) - df['Close'].shift(1)) > (df['ATR'].shift(1) * 1.5)
        
        # 🔥 RED DE RADARES 360° 🔥
        df['PL30'] = df['Low'].shift(1).rolling(30, min_periods=1).min()
        df['PH30'] = df['High'].shift(1).rolling(30, min_periods=1).max()
        df['PL100'] = df['Low'].shift(1).rolling(100, min_periods=1).min()
        df['PH100'] = df['High'].shift(1).rolling(100, min_periods=1).max()
        df['PL300'] = df['Low'].shift(1).rolling(300, min_periods=1).min()
        df['PH300'] = df['High'].shift(1).rolling(300, min_periods=1).max()
        
        df['Target_Lock_Sup'] = df[['PL30', 'PL100', 'PL300']].max(axis=1)
        df['Target_Lock_Res'] = df[['PH30', 'PH100', 'PH300']].min(axis=1)
        
        scan_range = df['ATR'] * 2.0
        c_val = df['Close'].values
        sr_val = scan_range.values
        ceil_w, floor_w = np.zeros(len(df)), np.zeros(len(df))
        for p_col, w in [('PL30', 1), ('PH30', 1), ('PL100', 3), ('PH100', 3), ('PL300', 5), ('PH300', 5)]:
            p_val = df[p_col].values
            ceil_w += np.where((p_val > c_val) & (p_val <= c_val + sr_val), w, 0)
            floor_w += np.where((p_val < c_val) & (p_val >= c_val - sr_val), w, 0)
        df['ceil_w'] = ceil_w
        df['floor_w'] = floor_w
        
        df['RSI_MA'] = df['RSI'].rolling(14, min_periods=1).mean()
        df['RSI_Cross_Up'] = (df['RSI'] > df['RSI_MA']) & (df['RSI'].shift(1).fillna(50) <= df['RSI_MA'].shift(1).fillna(50))
        df['RSI_Cross_Dn'] = (df['RSI'] < df['RSI_MA']) & (df['RSI'].shift(1).fillna(50) >= df['RSI_MA'].shift(1).fillna(50))
        
        df['PP_Slope'] = ta.linreg(df['Close'], 5, 0) - ta.linreg(df['Close'], 5, 1)
        df['Macro_Bull'] = df['Close'] >= df['EMA_200']
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

# 🔥 LA CIRUGÍA ALGORÍTMICA PROFUNDA (V67.0 - DIAMOND HANDS) 🔥
def inyectar_adn(df_sim, r_sens=1.5, w_factor=2.5):
    
    # 🏓 1. PING PONG (Lógica Invertida Corregida: Reversión al VWAP)
    # Compra abajo en pánico (pendiente subiendo) y vende arriba en euforia.
    df_sim['Ping_Buy'] = (df_sim['PP_Slope'] > 0) & (df_sim['PP_Slope'].shift(1).fillna(0) <= 0) & (df_sim['Close'] < df_sim['VWAP']) & (df_sim['RSI'] < 45)
    df_sim['Ping_Sell'] = (df_sim['PP_Slope'] < 0) & (df_sim['PP_Slope'].shift(1).fillna(0) >= 0) & (df_sim['Close'] > df_sim['VWAP']) & (df_sim['RSI'] > 60)

    # 🌸 2. PINK CLIMAX (Filtro de Venta Ampliado - Deja Correr las Ganancias)
    df_sim['Climax_Buy'] = (df_sim['RVol'] > 1.2) & (df_sim['lower_wick'] >= (df_sim['body_size'] * 0.8)) & (df_sim['RSI'] < 50) & df_sim['Vela_Verde']
    # Solo vende si la euforia es masiva (>70 RSI) o rechazo bajista gigante
    df_sim['Climax_Sell'] = (df_sim['RSI'] > 70) | ((df_sim['upper_wick'] >= (df_sim['body_size'] * 1.5)) & (df_sim['RVol'] > 1.5) & df_sim['Vela_Roja'])

    # 🌡️ 3. THERMAL (Visión de Muros Reales)
    df_sim['Therm_Bounce'] = (df_sim['floor_w'] >= 1) & (df_sim['RSI'] > df_sim['RSI'].shift(1)) & (df_sim['RSI'] < 60) & df_sim['Vela_Verde']
    dist_next_res = (df_sim['Target_Lock_Res'] - df_sim['Close']) / df_sim['Close'] * 100
    df_sim['Therm_Vacuum'] = (df_sim['Close'] > df_sim['Target_Lock_Res']) & (dist_next_res > 2.0) & df_sim['Vela_Verde']
    df_sim['Thermal_Buy'] = df_sim['Therm_Bounce'] | df_sim['Therm_Vacuum']
    # Solo vende en techos fuertes o sobrecompra extrema
    df_sim['Thermal_Sell'] = (df_sim['ceil_w'] >= 4) | (df_sim['RSI'] > 75) 

    # 🎯 4. TARGET LOCK (Filtro de Euforia Falsa)
    df_sim['Lock_Bounce'] = (df_sim['Low'] <= (df_sim['Target_Lock_Sup'] + df_sim['ATR'])) & (df_sim['Close'] > df_sim['Target_Lock_Sup']) & df_sim['Vela_Verde']
    # Exige que el RSI cruce al alza (impulso confirmado)
    df_sim['Lock_Buy'] = df_sim['Lock_Bounce'] & df_sim['RSI_Cross_Up'] & (df_sim['RSI'] < 65)
    df_sim['Lock_Reject'] = (df_sim['High'] >= (df_sim['Target_Lock_Res'] - df_sim['ATR'])) & (df_sim['Close'] < df_sim['Target_Lock_Res']) & df_sim['Vela_Roja']
    # No vende si el RSI aún tiene espacio
    df_sim['Lock_Sell'] = df_sim['Lock_Reject'] & (df_sim['RSI'] > 55)

    # 🐛 5. NEON SQUEEZE (Salida Holgada por Base Bollinger)
    df_sim['BB_Contraction'] = df_sim['BB_Width'] < df_sim['BB_Width_Avg']
    df_sim['Neon_Up'] = df_sim['Squeeze_On'].shift(1).fillna(False) & (df_sim['Close'] > df_sim['BBU']) & df_sim['Vela_Verde'] & (df_sim['RVol'] >= 1.0) & (df_sim['ADX'] > 15)
    df_sim['Squeeze_Buy'] = df_sim['Neon_Up']
    # Vende solo cuando el precio cae de la media de 20
    df_sim['Squeeze_Sell'] = (df_sim['Close'] < df_sim['Basis']) | (df_sim['RSI'] > 80)

    # 🚀 6. DEFCON (Entrada Temprana, Salida Holgada)
    df_sim['Defcon_Buy_Sig'] = df_sim['Neon_Up'] & (df_sim['BB_Delta'] > 0) & (df_sim['RVol'] > 1.0)
    df_sim['Defcon_Sell_Sig'] = (df_sim['Close'] < df_sim['Basis']) | (df_sim['RSI'] > 80)

    # ⚔️ 7. JUGGERNAUT (Protegido por Aegis, Salida por Base)
    df_sim['Pink_Whale_Buy'] = df_sim['Climax_Buy'] & (df_sim['RVol'] > (w_factor * 0.8))
    df_sim['aegis_safe'] = ~df_sim['is_falling_knife']
    df_sim['Jugg_Buy'] = (df_sim['Defcon_Buy_Sig'] | df_sim['Pink_Whale_Buy'] | df_sim['Lock_Buy']) & df_sim['aegis_safe']
    # Abandona la lenta EMA 50 y usa la Base de Bollinger para capturar tendencias completas
    df_sim['Jugg_Sell'] = (df_sim['Close'] < df_sim['Basis']) | (df_sim['RSI'] > 75)

    # 👑 TRINITY Y COMMANDER
    df_sim['Trinity_Buy'] = df_sim['Pink_Whale_Buy'] | (df_sim['Lock_Buy'] & df_sim['aegis_safe']) | (df_sim['Defcon_Buy_Sig'] & df_sim['aegis_safe'])
    df_sim['Trinity_Sell'] = df_sim['Defcon_Sell_Sig'] | df_sim['Thermal_Sell']
    df_sim['Lev_Buy'] = df_sim['Macro_Bull'] & df_sim['RSI_Cross_Up'] & (df_sim['RSI'].shift(1).fillna(50) < 50)
    df_sim['Lev_Sell'] = (df_sim['Close'] < df_sim['EMA_200'])
    
    df_sim['confirmacion_alcista'] = (df_sim['Close'] > df_sim['High'].shift(1)) & df_sim['Vela_Verde']
    df_sim['Commander_Buy'] = df_sim['Pink_Whale_Buy'] | ((df_sim['Lock_Buy'] | df_sim['Thermal_Buy']) & df_sim['aegis_safe']) | (df_sim['Climax_Buy'] & df_sim['confirmacion_alcista'])
    df_sim['Commander_Sell'] = df_sim['Defcon_Sell_Sig'] | df_sim['Thermal_Sell'] | df_sim['Climax_Sell'] | df_sim['Ping_Sell']

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

st.title("🛡️ The Omni-Brain Lab")
tabs = st.tabs(["💠 TRINITY", "⚔️ JUGGERNAUT", "🚀 DEFCON", "🎯 TARGET_LOCK", "🌡️ THERMAL", "🌸 PINK_CLIMAX", "🏓 PING_PONG", "🐛 NEON_SQUEEZE", "👑 COMMANDER", "🌌 GENESIS", "👑 ROCKET"])

tab_id_map = {
    "💠 TRINITY": "TRINITY", "⚔️ JUGGERNAUT": "JUGGERNAUT", "🚀 DEFCON": "DEFCON",
    "🎯 TARGET_LOCK": "TARGET_LOCK", "🌡️ THERMAL": "THERMAL", "🌸 PINK_CLIMAX": "PINK_CLIMAX",
    "🏓 PING_PONG": "PING_PONG", "🐛 NEON_SQUEEZE": "NEON_SQUEEZE", "👑 COMMANDER": "COMMANDER",
    "🌌 GENESIS": "GENESIS", "👑 ROCKET": "ROCKET"
}

def optimizar_ia(s_id, df_base, cap_ini, com_pct, reinv_q, target_ado, dias_reales, buy_hold_money):
    best_fit = -float('inf')
    bp = None
    tp_min, tp_max = (1.5, 20.0) if target_ado > 2 else (3.0, 40.0)
    
    for _ in range(2000): 
        rtp = round(random.uniform(tp_min, tp_max), 1)
        rsl = round(random.uniform(1.0, 8.0), 1)
        rwh = round(random.uniform(1.5, 3.5), 1)
        rrd = round(random.uniform(0.5, 3.5), 1)
        
        df_precalc = inyectar_adn(df_base.copy(), r_sens=rrd, w_factor=rwh)
        h_a, l_a, c_a, o_a = df_precalc['High'].values, df_precalc['Low'].values, df_precalc['Close'].values, df_precalc['Open'].values
        
        if s_id == "TRINITY":
            b_c, s_c = df_precalc['Trinity_Buy'].values, df_precalc['Trinity_Sell'].values
        elif s_id == "JUGGERNAUT":
            b_c, s_c = df_precalc['Jugg_Buy'].values, df_precalc['Jugg_Sell'].values
        elif s_id == "DEFCON":
            b_c, s_c = df_precalc['Defcon_Buy_Sig'].values, df_precalc['Defcon_Sell_Sig'].values
        elif s_id == "TARGET_LOCK":
            b_c, s_c = df_precalc['Lock_Buy'].values, df_precalc['Lock_Sell'].values
        elif s_id == "THERMAL":
            b_c, s_c = df_precalc['Thermal_Buy'].values, df_precalc['Thermal_Sell'].values
        elif s_id == "PINK_CLIMAX":
            b_c, s_c = df_precalc['Climax_Buy'].values, df_precalc['Climax_Sell'].values
        elif s_id == "PING_PONG":
            b_c, s_c = df_precalc['Ping_Buy'].values, df_precalc['Ping_Sell'].values
        elif s_id == "NEON_SQUEEZE":
            b_c, s_c = df_precalc['Squeeze_Buy'].values, df_precalc['Squeeze_Sell'].values
        elif s_id == "COMMANDER":
            b_c, s_c = df_precalc['Commander_Buy'].values, df_precalc['Commander_Sell'].values
            
        t_arr, sl_arr = np.full(len(df_precalc), float(rtp)), np.full(len(df_precalc), float(rsl))
        net, pf, nt, mdd, comms = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, b_c, s_c, t_arr, sl_arr, float(cap_ini), float(com_pct), float(reinv_q))
        
        actual_ado = nt / dias_reales if dias_reales > 0 else 0
        ado_multiplier = 1.0
        if target_ado > 0 and actual_ado < target_ado: ado_multiplier = (actual_ado / target_ado) ** 2  

        alpha_money = net - buy_hold_money
        
        if nt >= 1: 
            if net > 0: fit = (net * (pf**2) * np.sqrt(nt)) / ((mdd ** 1.5) + 1.0) * ado_multiplier
            else: fit = net * ((mdd ** 1.5) + 1.0) / (pf + 0.001)

            if alpha_money > 0: fit *= 2.0 
                
            if fit > best_fit:
                best_fit = fit
                bp = {'tp':rtp, 'sl':rsl, 'wh':rwh, 'rd':rrd, 'reinv':reinv_q, 'net':net, 'pf':pf, 'nt':nt, 'alpha':alpha_money, 'mdd':mdd, 'comms':comms}
    return bp

for idx, tab_name in enumerate(tab_id_map.keys()):
    with tabs[idx]:
        if df_global.empty: continue
        s_id = tab_id_map[tab_name]
        
        if st.session_state.get(f'update_pending_{s_id}', False):
            bp = st.session_state[f'pending_bp_{s_id}']
            if s_id == "GENESIS":
                for r_idx in range(1, 5):
                    st.session_state[f'gen_r{r_idx}_b'] = bp[f'b{r_idx}']
                    st.session_state[f'gen_r{r_idx}_s'] = bp[f's{r_idx}']
                    st.session_state[f'gen_r{r_idx}_tp'] = float(round(bp[f'tp{r_idx}'], 1))
                    st.session_state[f'gen_r{r_idx}_sl'] = float(round(bp[f'sl{r_idx}'], 1))
                st.session_state['gen_wh'] = float(round(bp['wh'], 1))
                st.session_state['gen_rd'] = float(round(bp['rd'], 1))
            elif s_id == "ROCKET":
                for r_idx in range(1, 5):
                    st.session_state[f'roc_r{r_idx}_b'] = bp[f'b{r_idx}']
                    st.session_state[f'roc_r{r_idx}_s'] = bp[f's{r_idx}']
                    st.session_state[f'roc_r{r_idx}_tp'] = float(round(bp[f'tp{r_idx}'], 1))
                    st.session_state[f'roc_r{r_idx}_sl'] = float(round(bp[f'sl{r_idx}'], 1))
                st.session_state['gen_wh'] = float(round(bp['wh'], 1))
                st.session_state['gen_rd'] = float(round(bp['rd'], 1))
            else:
                st.session_state[f'sld_tp_{s_id}'] = float(round(bp['tp'], 1))
                st.session_state[f'sld_sl_{s_id}'] = float(round(bp['sl'], 1))
                st.session_state[f'sld_reinv_{s_id}'] = float(bp['reinv'])
                st.session_state[f'sld_wh_{s_id}'] = float(round(bp['wh'], 1))
                st.session_state[f'sld_rd_{s_id}'] = float(round(bp['rd'], 1))
            st.session_state[f'update_pending_{s_id}'] = False

        if s_id == "ROCKET":
            st.markdown("### 👑 ROCKET PROTOCOL (El Comandante Supremo)")
            st.info("La Inteligencia Artificial asigna Estrategias Completas a cada cuadrante del mercado.")
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state['ado_ROCKET'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(st.session_state.get('ado_ROCKET', 5.0)), step=0.5, key="ui_ado_roc")
            st.session_state['sld_reinv_ROCKET'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get('sld_reinv_ROCKET', 100.0)), step=5.0, key="ui_reinv_roc")

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("<h5 style='color:lime;'>🟢 Bull Trend</h5>", unsafe_allow_html=True)
                st.multiselect("Estrategia Compra", rocket_b, key="roc_r1_b")
                st.multiselect("Estrategia Cierre", rocket_s, key="roc_r1_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="roc_r1_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="roc_r1_sl")
            with c2:
                st.markdown("<h5 style='color:yellow;'>🟡 Bull Chop</h5>", unsafe_allow_html=True)
                st.multiselect("Estrategia Compra", rocket_b, key="roc_r2_b")
                st.multiselect("Estrategia Cierre", rocket_s, key="roc_r2_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="roc_r2_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="roc_r2_sl")
            with c3:
                st.markdown("<h5 style='color:red;'>🔴 Bear Trend</h5>", unsafe_allow_html=True)
                st.multiselect("Estrategia Compra", rocket_b, key="roc_r3_b")
                st.multiselect("Estrategia Cierre", rocket_s, key="roc_r3_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="roc_r3_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="roc_r3_sl")
            with c4:
                st.markdown("<h5 style='color:orange;'>🟠 Bear Chop</h5>", unsafe_allow_html=True)
                st.multiselect("Estrategia Compra", rocket_b, key="roc_r4_b")
                st.multiselect("Estrategia Cierre", rocket_s, key="roc_r4_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="roc_r4_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="roc_r4_sl")

            if c_ia3.button("🚀 INICIAR ENSAMBLAJE", type="primary", key="btn_roc"):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                
                best_fit = -float('inf')
                bp = None
                reinv_q = st.session_state.get('sld_reinv_ROCKET', 100.0)
                t_ado = st.session_state.get('ado_ROCKET', 5.0)
                
                for _ in range(2000): 
                    rwh = round(random.uniform(1.5, 4.0), 1)
                    rrd = round(random.uniform(0.5, 3.0), 1)
                    df_p = inyectar_adn(df_global.copy(), r_sens=rrd, w_factor=rwh)
                    h_a, l_a, c_a, o_a = df_p['High'].values, df_p['Low'].values, df_p['Close'].values, df_p['Open'].values
                    regime_arr = df_p['Regime'].values
                    
                    b_mat = {r: df_p[r].values for r in rocket_b}
                    s_mat = {r: df_p[r].values for r in rocket_s}

                    dna_b = [random.sample(rocket_b, random.randint(1, 2)) for _ in range(4)]
                    dna_s = [random.sample(rocket_s, random.randint(1, 2)) for _ in range(4)]
                    dna_tp = [random.uniform(0.5, 20.0) for _ in range(4)]
                    dna_sl = [random.uniform(0.5, 8.0) for _ in range(4)]
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
                    
                    t_arr, sl_arr = np.full(len(df_p), 0.0), np.full(len(df_p), 0.0)
                    for i in range(len(df_p)): t_arr[i] = f_tp[i]; sl_arr[i] = f_sl[i]
                    
                    net, pf, nt, mdd, comms = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, f_buy, f_sell, t_arr, sl_arr, float(capital_inicial), float(comision_pct), float(reinv_q))
                    alpha_money = net - buy_hold_money
                    actual_ado = nt / dias_reales if dias_reales > 0 else 0
                    ado_multiplier = 1.0
                    if t_ado > 0 and actual_ado < t_ado: ado_multiplier = (actual_ado / t_ado) ** 2  

                    if nt >= 1:
                        if net > 0: fit = (net * (pf**2) * np.sqrt(nt)) / ((mdd ** 1.5) + 1.0) * ado_multiplier
                        else: fit = net * ((mdd ** 1.5) + 1.0) / (pf + 0.001)

                        if fit > best_fit:
                            best_fit = fit
                            bp = {'b1': dna_b[0], 's1': dna_s[0], 'tp1': dna_tp[0], 'sl1': dna_sl[0], 'b2': dna_b[1], 's2': dna_s[1], 'tp2': dna_tp[1], 'sl2': dna_sl[1], 'b3': dna_b[2], 's3': dna_s[2], 'tp3': dna_tp[2], 'sl3': dna_sl[2], 'b4': dna_b[3], 's4': dna_s[3], 'tp4': dna_tp[3], 'sl4': dna_sl[3], 'wh': rwh, 'rd': rrd, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'comms': comms}
                
                ph_holograma.empty()
                if bp: 
                    st.session_state[f'update_pending_{s_id}'] = True
                    st.session_state[f'pending_bp_{s_id}'] = bp
                    status_msg = f"🏆 ALPHA CREADO: +${bp['alpha']:,.2f}" if bp['alpha'] > 0 else f"🛡️ RIESGO CONTROLADO. Hold: +${buy_hold_money:,.2f}" if bp['net'] > 0 else f"❌ PÉRDIDA."
                    dna_str = f"👑 ROCKET PROTOCOL (META-ALGORITMO)\nNet Profit: ${bp['net']:,.2f} | PF: {bp['pf']:.2f}x | Trades: {bp['nt']} | Comisiones: ${bp['comms']:,.2f}\n{status_msg}\n\n⚙️ Factor Ballena: {bp['wh']}x | Radar: {bp['rd']}%\n\n// 🟢 QUAD 1: BULL TREND\nAsignado a: {bp['b1']} & {bp['s1']}\nTP = {bp['tp1']:.1f}% | SL = {bp['sl1']:.1f}%\n// 🟡 QUAD 2: BULL CHOP\nAsignado a: {bp['b2']} & {bp['s2']}\nTP = {bp['tp2']:.1f}% | SL = {bp['sl2']:.1f}%\n// 🔴 QUAD 3: BEAR TREND\nAsignado a: {bp['b3']} & {bp['s3']}\nTP = {bp['tp3']:.1f}% | SL = {bp['sl3']:.1f}%\n// 🟠 QUAD 4: BEAR CHOP\nAsignado a: {bp['b4']} & {bp['s4']}\nTP = {bp['tp4']:.1f}% | SL = {bp['sl4']:.1f}%"
                    st.session_state[f'dna_{s_id}'] = dna_str
                    st.rerun() 

            if st.session_state.get(f'dna_{s_id}') != "":
                st.code(st.session_state[f'dna_{s_id}'], language="text")

            df_strat = inyectar_adn(df_global.copy(), st.session_state.get('gen_rd', 1.5), st.session_state.get('gen_wh', 2.5))
            f_buy, f_sell = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            f_tp, f_sl = np.zeros(len(df_strat)), np.zeros(len(df_strat))
            for idx in range(1, 5):
                mask = (df_strat['Regime'].values == idx)
                r_b_cond = np.zeros(len(df_strat), dtype=bool)
                for r in st.session_state.get(f'roc_r{idx}_b', []): r_b_cond |= df_strat[r].values
                f_buy[mask] = r_b_cond[mask]
                r_s_cond = np.zeros(len(df_strat), dtype=bool)
                for r in st.session_state.get(f'roc_r{idx}_s', []): r_s_cond |= df_strat[r].values
                f_sell[mask] = r_s_cond[mask]
                f_tp[mask] = st.session_state.get(f'roc_r{idx}_tp', 5.0)
                f_sl[mask] = st.session_state.get(f'roc_r{idx}_sl', 2.0)
                
            df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
            df_strat['Active_TP'], df_strat['Active_SL'] = f_tp, f_sl
            eq_curve, divs, cap_act, t_log, pos_ab, total_comms = simular_visual(df_strat, capital_inicial, st.session_state.get('sld_reinv_ROCKET', 100.0), comision_pct)

        elif s_id == "GENESIS":
            st.markdown("### 🌌 GÉNESIS (Omni-Brain)")
            st.info("La IA halla la combinación perfecta por Cuadrante.")
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state['gen_ado'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(st.session_state.get('gen_ado', 5.0)), step=0.5, key="ui_gen_ado")
            st.session_state['gen_reinv'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get('gen_reinv', 100.0)), step=5.0, key="ui_gen_reinv")

            with st.expander("⚙️ Calibración Global"):
                c_adv1, c_adv2 = st.columns(2)
                st.session_state['gen_wh'] = c_adv1.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state.get('gen_wh', 2.5)), step=0.1)
                st.session_state['gen_rd'] = c_adv2.slider("📡 Radar Sens.", 0.5, 5.0, value=float(st.session_state.get('gen_rd', 1.5)), step=0.1)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("<h5 style='color:lime;'>🟢 Bull Trend</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r1_b")
                st.multiselect("Cierres", sell_rules, key="gen_r1_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r1_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r1_sl")
            with c2:
                st.markdown("<h5 style='color:yellow;'>🟡 Bull Chop</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r2_b")
                st.multiselect("Cierres", sell_rules, key="gen_r2_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r2_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r2_sl")
            with c3:
                st.markdown("<h5 style='color:red;'>🔴 Bear Trend</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r3_b")
                st.multiselect("Cierres", sell_rules, key="gen_r3_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r3_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r3_sl")
            with c4:
                st.markdown("<h5 style='color:orange;'>🟠 Bear Chop</h5>", unsafe_allow_html=True)
                st.multiselect("Compras", buy_rules, key="gen_r4_b")
                st.multiselect("Cierres", sell_rules, key="gen_r4_s")
                st.slider("TP %", 0.5, 30.0, step=0.5, key="gen_r4_tp")
                st.slider("SL %", 0.5, 15.0, step=0.5, key="gen_r4_sl")

            if c_ia3.button("🚀 EXTRACCIÓN CUÁNTICA", type="primary", key="btn_gen"):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                best_fit = -float('inf')
                bp = None
                reinv_q = st.session_state.get('gen_reinv', 100.0)
                t_ado = st.session_state.get('gen_ado', 5.0)
                
                for _ in range(2000): 
                    rwh = round(random.uniform(1.5, 4.0), 1)
                    rrd = round(random.uniform(0.5, 3.0), 1)
                    df_p = inyectar_adn(df_global.copy(), r_sens=rrd, w_factor=rwh)
                    h_a, l_a, c_a, o_a = df_p['High'].values, df_p['Low'].values, df_p['Close'].values, df_p['Open'].values
                    regime_arr = df_p['Regime'].values
                    b_mat = {r: df_p[r].values for r in buy_rules}
                    s_mat = {r: df_p[r].values for r in sell_rules}

                    dna_b = [random.sample(buy_rules, random.randint(1, 3)) for _ in range(4)]
                    dna_s = [random.sample(sell_rules, random.randint(1, 3)) for _ in range(4)]
                    dna_tp = [random.uniform(0.5, 20.0) for _ in range(4)]
                    dna_sl = [random.uniform(0.5, 8.0) for _ in range(4)]
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
                    
                    t_arr, sl_arr = np.full(len(df_p), 0.0), np.full(len(df_p), 0.0)
                    for i in range(len(df_p)): t_arr[i] = f_tp[i]; sl_arr[i] = f_sl[i]
                    
                    net, pf, nt, mdd, comms = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, f_buy, f_sell, t_arr, sl_arr, float(capital_inicial), float(comision_pct), float(reinv_q))
                    alpha_money = net - buy_hold_money
                    actual_ado = nt / dias_reales if dias_reales > 0 else 0
                    ado_multiplier = 1.0
                    if t_ado > 0 and actual_ado < t_ado: ado_multiplier = (actual_ado / t_ado) ** 2  

                    if nt >= 1:
                        if net > 0: fit = (net * (pf**2) * np.sqrt(nt)) / ((mdd ** 1.5) + 1.0) * ado_multiplier
                        else: fit = net * ((mdd ** 1.5) + 1.0) / (pf + 0.001)

                        if fit > best_fit:
                            best_fit = fit
                            bp = {'b1': dna_b[0], 's1': dna_s[0], 'tp1': dna_tp[0], 'sl1': dna_sl[0], 'b2': dna_b[1], 's2': dna_s[1], 'tp2': dna_tp[1], 'sl2': dna_sl[1], 'b3': dna_b[2], 's3': dna_s[2], 'tp3': dna_tp[2], 'sl3': dna_sl[2], 'b4': dna_b[3], 's4': dna_s[3], 'tp4': dna_tp[3], 'sl4': dna_sl[3], 'wh': rwh, 'rd': rrd, 'net': net, 'pf': pf, 'nt': nt, 'alpha': alpha_money, 'mdd': mdd, 'comms': comms}
                
                ph_holograma.empty()
                if bp: 
                    st.session_state[f'update_pending_{s_id}'] = True
                    st.session_state[f'pending_bp_{s_id}'] = bp
                    status_msg = f"🏆 ALPHA CREADO: +${bp['alpha']:,.2f}" if bp['alpha'] > 0 else f"🛡️ RIESGO CONTROLADO. Hold: +${buy_hold_money:,.2f}" if bp['net'] > 0 else f"❌ PÉRDIDA."
                    dna_str = f"🌌 GENESIS OMNI-BRAIN\nNet Profit: ${bp['net']:,.2f} | PF: {bp['pf']:.2f}x | Trades: {bp['nt']} | Comisiones: ${bp['comms']:,.2f}\n{status_msg}\n\n⚙️ Factor Ballena: {bp['wh']}x | Radar: {bp['rd']}%\n\n// 🟢 QUAD 1: BULL TREND\nCompras = {bp['b1']}\nCierres = {bp['s1']}\nTP = {bp['tp1']:.1f}% | SL = {bp['sl1']:.1f}%\n// 🟡 QUAD 2: BULL CHOP\nCompras = {bp['b2']}\nCierres = {bp['s2']}\nTP = {bp['tp2']:.1f}% | SL = {bp['sl2']:.1f}%\n// 🔴 QUAD 3: BEAR TREND\nCompras = {bp['b3']}\nCierres = {bp['s3']}\nTP = {bp['tp3']:.1f}% | SL = {bp['sl3']:.1f}%\n// 🟠 QUAD 4: BEAR CHOP\nCompras = {bp['b4']}\nCierres = {bp['s4']}\nTP = {bp['tp4']:.1f}% | SL = {bp['sl4']:.1f}%"
                    st.session_state[f'dna_{s_id}'] = dna_str
                    st.rerun() 

            if st.session_state.get(f'dna_{s_id}') != "":
                st.code(st.session_state[f'dna_{s_id}'], language="text")

            df_strat = inyectar_adn(df_global.copy(), st.session_state.get('gen_rd', 1.5), st.session_state.get('gen_wh', 2.5))
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
                
            df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
            df_strat['Active_TP'], df_strat['Active_SL'] = f_tp, f_sl
            eq_curve, divs, cap_act, t_log, pos_ab, total_comms = simular_visual(df_strat, capital_inicial, st.session_state.get('gen_reinv', 100.0), comision_pct)

        # --- BLOQUES ESTRATÉGICOS INDIVIDUALES ---
        else:
            st.markdown(f"### ⚙️ {s_id} (Truth Engine)")
            c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
            st.session_state[f'ado_{s_id}'] = c_ia1.slider("🎯 Target ADO", 0.0, 100.0, value=float(st.session_state.get(f'ado_{s_id}', 5.0)), step=0.5, key=f"ui_ado_{s_id}")
            st.session_state[f'sld_reinv_{s_id}'] = c_ia2.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get(f'sld_reinv_{s_id}', 100.0)), step=5.0, key=f"ui_reinv_{s_id}")

            if c_ia3.button(f"🚀 OPTIMIZACIÓN ABSOLUTA ({s_id})", type="primary", key=f"btn_ai_{s_id}"):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
                buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
                reinv_q = st.session_state.get(f'sld_reinv_{s_id}', 100.0)
                t_ado = st.session_state.get(f'ado_{s_id}', 5.0)
                
                bp = optimizar_ia(s_id, df_global, capital_inicial, comision_pct, reinv_q, t_ado, dias_reales, buy_hold_money)
                
                ph_holograma.empty()
                if bp:
                    st.session_state[f'update_pending_{s_id}'] = True
                    st.session_state[f'pending_bp_{s_id}'] = bp
                    status_msg = f"🏆 ALPHA CREADO: +${bp['alpha']:,.2f}" if bp['alpha'] > 0 else f"🛡️ RIESGO CONTROLADO. Hold: +${buy_hold_money:,.2f}" if bp['net'] > 0 else f"❌ PÉRDIDA."
                    
                    dna_str = f"⚙️ {s_id} OMNI-BRAIN\nNet Profit: ${bp['net']:,.2f} | PF: {bp['pf']:.2f}x | Trades: {bp['nt']} | Comisiones: ${bp['comms']:,.2f}\n{status_msg}\n\n⚙️ Factor Ballena: {bp['wh']}x | Radar: {bp['rd']}%\nTP = {bp['tp']:.1f}% | SL = {bp['sl']:.1f}%"
                    st.session_state[f'dna_{s_id}'] = dna_str
                    st.rerun()
                else: st.error("❌ El motor no pudo encontrar ni 1 solo trade.")

            if st.session_state.get(f'dna_{s_id}') != "":
                st.code(st.session_state[f'dna_{s_id}'], language="text")

            with st.expander("🛠️ Ajuste Manual de Parámetros"):
                c1, c2, c3, c4 = st.columns(4)
                st.session_state[f'sld_tp_{s_id}'] = c1.slider("🎯 TP Base (%)", 0.5, 50.0, value=float(st.session_state.get(f'sld_tp_{s_id}', 10.0)), step=0.1, key=f"man_tp_{s_id}")
                st.session_state[f'sld_sl_{s_id}'] = c2.slider("🛑 SL (%)", 0.5, 20.0, value=float(st.session_state.get(f'sld_sl_{s_id}', 3.0)), step=0.1, key=f"man_sl_{s_id}")
                st.session_state[f'sld_wh_{s_id}'] = c3.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state.get(f'sld_wh_{s_id}', 2.5)), step=0.1, key=f"man_wh_{s_id}")
                st.session_state[f'sld_rd_{s_id}'] = c4.slider("📡 Radar Sens.", 0.5, 5.0, value=float(st.session_state.get(f'sld_rd_{s_id}', 1.5)), step=0.1, key=f"man_rd_{s_id}")

            current_wh = st.session_state.get(f'sld_wh_{s_id}', 2.5) 
            current_rd = st.session_state.get(f'sld_rd_{s_id}', 1.5) 
            
            df_strat = inyectar_adn(df_global.copy(), current_rd, current_wh)
            if s_id == "TARGET_LOCK":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Lock_Buy'], df_strat['Lock_Sell']
            elif s_id == "THERMAL":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Thermal_Buy'], df_strat['Thermal_Sell']
            elif s_id == "TRINITY":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Trinity_Buy'], df_strat['Trinity_Sell']
            elif s_id == "JUGGERNAUT":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Jugg_Buy'], df_strat['Jugg_Sell']
            elif s_id == "DEFCON":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Defcon_Buy_Sig'], df_strat['Defcon_Sell_Sig']
            elif s_id == "PINK_CLIMAX":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Climax_Buy'], df_strat['Climax_Sell']
            elif s_id == "PING_PONG":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Ping_Buy'], df_strat['Ping_Sell']
            elif s_id == "NEON_SQUEEZE":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Squeeze_Buy'], df_strat['Squeeze_Sell']
            elif s_id == "COMMANDER":
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Commander_Buy'], df_strat['Commander_Sell']
                
            df_strat['Active_TP'] = st.session_state[f'sld_tp_{s_id}']
            df_strat['Active_SL'] = st.session_state[f'sld_sl_{s_id}']
            eq_curve, divs, cap_act, t_log, pos_ab, total_comms = simular_visual(df_strat, capital_inicial, st.session_state.get(f'sld_reinv_{s_id}', 100.0), comision_pct)

        # --- SECCIÓN COMÚN (MÉTRICAS) ---
        df_strat['Total_Portfolio'] = eq_curve
        ret_pct = ((eq_curve[-1] - capital_inicial) / capital_inicial) * 100
        buy_hold_ret = ((df_strat['Close'].iloc[-1] - df_strat['Open'].iloc[0]) / df_strat['Open'].iloc[0]) * 100
        alpha_pct = ret_pct - buy_hold_ret

        dftr = pd.DataFrame(t_log)
        tt, wr, pf_val, ado_act = 0, 0.0, 0.0, 0.0
        if not dftr.empty:
            exs = dftr[dftr['Tipo'].isin(['TP', 'SL', 'DYN_WIN', 'DYN_LOSS'])]
            tt = len(exs)
            ado_act = tt / dias_reales if dias_reales > 0 else 0
            if tt > 0:
                ws = len(exs[exs['Tipo'].isin(['TP', 'DYN_WIN'])])
                wr = (ws / tt) * 100
                gpp = exs[exs['Ganancia_$'] > 0]['Ganancia_$'].sum()
                gll = abs(exs[exs['Ganancia_$'] < 0]['Ganancia_$'].sum())
                pf_val = gpp / gll if gll > 0 else float('inf')
        
        mdd = abs((((pd.Series(eq_curve) - pd.Series(eq_curve).cummax()) / pd.Series(eq_curve).cummax()) * 100).min())

        st.markdown(f"### 📊 Auditoría: {s_id}")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Portafolio Neto", f"${eq_curve[-1]:,.2f}", f"{ret_pct:.2f}%")
        c2.metric("ALPHA (vs Hold)", f"{alpha_pct:.2f}%", f"Hold: {buy_hold_ret:.2f}%", delta_color="normal" if alpha_pct > 0 else "inverse")
        c3.metric("Trades Totales", f"{tt}")
        c4.metric("Win Rate", f"{wr:.1f}%")
        c5.metric("Profit Factor", f"{pf_val:.2f}x")
        c6.metric("Max Drawdown", f"{mdd:.2f}%", delta_color="inverse")
        c7.metric("Comisiones", f"${total_comms:,.2f}", delta_color="inverse")

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
