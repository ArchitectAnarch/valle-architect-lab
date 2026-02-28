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
import json
from datetime import datetime, timedelta

# --- MOTOR DE HIPER-VELOCIDAD (NUMBA JIT) ---
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

st.set_page_config(page_title="ROCKET PROTOCOL | Genesis Lab", layout="wide", initial_sidebar_state="expanded")
ph_holograma = st.empty()

if st.session_state.get('app_version') != 'V168':
    st.session_state.clear()
    st.session_state['app_version'] = 'V168'

# ==========================================
# 🧠 1. FUNCIONES MATEMÁTICAS C++
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

@njit(fastmath=True)
def simular_crecimiento_exponencial_ia_core(h_arr, l_arr, c_arr, o_arr, atr_arr, rsi_arr, z_arr, adx_arr, 
    b_c, s_c, w_rsi, w_z, w_adx, th_buy, th_sell, 
    atr_tp_mult, atr_sl_mult, cap_ini, com_pct, reinvest_pct, slippage_pct):
    
    cap_act = cap_ini; divs = 0.0; en_pos = False; p_ent = 0.0
    pos_size = 0.0; invest_amt = 0.0; g_profit = 0.0; g_loss = 0.0; num_trades = 0; max_dd = 0.0; peak = cap_ini
    slip_in = 1.0 + (slippage_pct / 100.0)
    slip_out = 1.0 - (slippage_pct / 100.0)
    tp_p = 0.0; sl_p = 0.0
    
    for i in range(len(h_arr)):
        if en_pos:
            if l_arr[i] <= sl_p:
                exec_p = sl_p * slip_out
                ret = (exec_p - p_ent) / p_ent
                gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
                if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                else: cap_act += profit
                g_loss += abs(profit); num_trades += 1; en_pos = False
            elif h_arr[i] >= tp_p:
                exec_p = tp_p * slip_out
                ret = (exec_p - p_ent) / p_ent
                gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
                if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                else: cap_act += profit
                if profit > 0: g_profit += profit 
                else: g_loss += abs(profit)
                num_trades += 1; en_pos = False
            else:
                score = (rsi_arr[i] * w_rsi) + (z_arr[i] * w_z) + (adx_arr[i] * w_adx)
                if s_c[i] or (score < th_sell):
                    exit_price = (o_arr[i+1] if i+1 < len(o_arr) else c_arr[i]) * slip_out
                    ret = (exit_price - p_ent) / p_ent; gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
                    if profit > 0: reinv = profit * (reinvest_pct / 100.0); divs += (profit - reinv); cap_act += reinv
                    else: cap_act += profit
                    if profit > 0: g_profit += profit 
                    else: g_loss += abs(profit)
                    num_trades += 1; en_pos = False
            
            total_equity = cap_act + divs
            if total_equity > peak: peak = total_equity
            if peak > 0: dd = (peak - total_equity) / peak * 100.0; max_dd = max(max_dd, dd)
            if cap_act <= 0: break
            
        if not en_pos and i+1 < len(h_arr):
            score = (rsi_arr[i] * w_rsi) + (z_arr[i] * w_z) + (adx_arr[i] * w_adx)
            if b_c[i] or (score > th_buy):
                invest_amt = cap_act if reinvest_pct == 100.0 else cap_ini
                if invest_amt > cap_act: invest_amt = cap_act 
                comm_in = invest_amt * com_pct; pos_size = invest_amt - comm_in 
                p_ent = o_arr[i+1] * slip_in
                current_atr = atr_arr[i]
                tp_p = p_ent + (current_atr * atr_tp_mult)
                sl_p = p_ent - (current_atr * atr_sl_mult)
                en_pos = True
                
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    return (cap_act + divs) - cap_ini, pf, num_trades, max_dd

def simular_visual(df_sim, cap_ini, reinvest, com_pct, slippage_pct=0.0):
    registro_trades = []; n = len(df_sim); curva = np.full(n, cap_ini, dtype=float)
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    atr_arr = df_sim['ATR'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl_arr = df_sim['Active_TP'].values, df_sim['Active_SL'].values
    f_arr = df_sim.index
    en_pos, p_ent, tp_act, sl_act, cap_act, divs, pos_size, invest_amt, total_comms = False, 0.0, 0.0, 0.0, cap_ini, 0.0, 0.0, 0.0, 0.0
    
    slip_in = 1.0 + (slippage_pct/100.0)
    slip_out = 1.0 - (slippage_pct/100.0)

    for i in range(n):
        cierra = False
        if en_pos:
            tp_p = p_ent + tp_act; sl_p = p_ent - sl_act
                
            if l_arr[i] <= sl_p:
                exec_p = sl_p * slip_out
                ret = (exec_p - p_ent) / p_ent; gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                if profit > 0: reinv_amt = profit * (reinvest/100); divs += (profit - reinv_amt); cap_act += reinv_amt
                else: cap_act += profit
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'SL', 'Precio': exec_p, 'Ganancia_$': profit}); en_pos, cierra = False, True
            elif h_arr[i] >= tp_p:
                exec_p = tp_p * slip_out
                ret = (exec_p - p_ent) / p_ent; gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                if profit > 0: reinv_amt = profit * (reinvest/100); divs += (profit - reinv_amt); cap_act += reinv_amt
                else: cap_act += profit
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'TP', 'Precio': exec_p, 'Ganancia_$': profit}); en_pos, cierra = False, True
            elif sell_arr[i]:
                exit_price = (o_arr[i+1] if i+1 < n else c_arr[i]) * slip_out
                ret = (exit_price - p_ent) / p_ent; gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                if profit > 0: reinv_amt = profit * (reinvest/100); divs += (profit - reinv_amt); cap_act += reinv_amt
                else: cap_act += profit
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i+1] if i+1 < n else f_arr[i], 'Tipo': 'DYN_WIN' if profit>0 else 'DYN_LOSS', 'Precio': exit_price, 'Ganancia_$': profit}); en_pos, cierra = False, True
        
        if not en_pos and not cierra and buy_arr[i] and i+1 < n and cap_act > 0:
            invest_amt = cap_act if reinvest == 100 else cap_ini
            if invest_amt > cap_act: invest_amt = cap_act
            comm_in = invest_amt * com_pct; total_comms += comm_in; pos_size = invest_amt - comm_in
            p_ent = o_arr[i+1] * slip_in
            tp_act = atr_arr[i] * float(tp_arr[i])
            sl_act = atr_arr[i] * float(sl_arr[i])
            en_pos = True
            registro_trades.append({'Fecha': f_arr[i+1], 'Tipo': 'ENTRY', 'Precio': p_ent, 'Ganancia_$': 0})
        
        if en_pos and cap_act > 0: curva[i] = cap_act + (pos_size * ((c_arr[i] - p_ent) / p_ent)) + divs
        else: curva[i] = cap_act + divs
    return curva.tolist(), divs, cap_act, registro_trades, en_pos, total_comms

# ==========================================
# 🧬 2. ARSENAL DE INDICADORES (ADN)
# ==========================================
if 'ai_algos' not in st.session_state or len(st.session_state['ai_algos']) == 0: 
    st.session_state['ai_algos'] = [f"AI_GENESIS_{random.randint(100, 999)}"]

estrategias = st.session_state['ai_algos']
tab_id_map = {f"🤖 {ai_id}": ai_id for ai_id in estrategias}

todas_las_armas_b = [
    'Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 
    'Commander_Buy', 'Lev_Buy', 'Q_Pink_Whale_Buy', 'Q_Lock_Bounce', 'Q_Lock_Break', 'Q_Neon_Up', 'Q_Defcon_Buy', 
    'Q_Therm_Bounce', 'Q_Therm_Vacuum', 'Q_Nuclear_Buy', 'Q_Early_Buy', 'Q_Rebound_Buy',
    'Wyc_Spring_Buy', 'VSA_Accum_Buy', 'Fibo_618_Buy', 'MACD_Impulse_Buy', 'Stoch_OS_Buy'
]
todas_las_armas_s = [
    'Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 
    'Commander_Sell', 'Lev_Sell', 'Q_Lock_Reject', 'Q_Lock_Breakd', 'Q_Neon_Dn', 'Q_Defcon_Sell', 'Q_Therm_Wall_Sell', 
    'Q_Therm_Panic_Sell', 'Q_Nuclear_Sell', 'Q_Early_Sell',
    'Wyc_Upthrust_Sell', 'VSA_Dist_Sell', 'Fibo_618_Sell', 'MACD_Exhaust_Sell', 'Stoch_OB_Sell'
]

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
    'Q_Therm_Panic_Sell': 'r_Therm_Panic_Sell', 'Q_Nuclear_Sell': 'r_Nuclear_Sell', 'Q_Early_Sell': 'r_Early_Sell',
    'Wyc_Spring_Buy': 'wyc_spring_buy', 'Wyc_Upthrust_Sell': 'wyc_upthrust_sell',
    'VSA_Accum_Buy': 'vsa_accum_buy', 'VSA_Dist_Sell': 'vsa_dist_sell',
    'Fibo_618_Buy': 'fibo_618_buy', 'Fibo_618_Sell': 'fibo_618_sell',
    'MACD_Impulse_Buy': 'macd_impulse_buy', 'MACD_Exhaust_Sell': 'macd_exhaust_sell',
    'Stoch_OS_Buy': 'stoch_os_buy', 'Stoch_OB_Sell': 'stoch_ob_sell'
}

for s_id in estrategias:
    if f'opt_status_{s_id}' not in st.session_state: st.session_state[f'opt_status_{s_id}'] = False
    if f'champion_{s_id}' not in st.session_state:
        st.session_state[f'champion_{s_id}'] = {
            'b_team': [random.choice(todas_las_armas_b)], 's_team': [random.choice(todas_las_armas_s)], 
            'macro': "All-Weather", 'vol': "All-Weather", 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 
            'whale_f': 2.5, 'ado': 4.0, 'reinv': 0.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0, 
            'w_rsi': 0.0, 'w_z': 0.0, 'w_adx': 0.0, 'th_buy': 99.0, 'th_sell': -99.0, 'atr_tp': 2.0, 'atr_sl': 1.0
        }

def save_champion(s_id, bp):
    if not bp: return
    vault = st.session_state.setdefault(f'champion_{s_id}', {})
    if bp.get('fit', -float('inf')) <= vault.get('fit', -float('inf')):
        return
    for k in bp.keys(): vault[k] = bp[k]

# ==========================================
# 🌍 4. SIDEBAR E INFRAESTRUCTURA
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🧬 GENESIS LAB V168</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True, key="btn_purge"): 
    st.cache_data.clear()
    keys_to_keep = ['app_version', 'ai_algos']
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep: del st.session_state[k]
    gc.collect(); st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🛑 ABORTAR RUN GLOBAL", use_container_width=True, key="btn_abort"):
    st.session_state['abort_opt'] = True
    st.session_state['global_queue'] = []
    st.session_state['run_global'] = False
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
st.sidebar.markdown("<h3 style='text-align: center; color: lime;'>🤖 CÁMARA DE MUTACIÓN</h3>", unsafe_allow_html=True)
global_epochs = st.sidebar.slider("Épocas de Evolución (x3000)", 1, 1000, 50)
target_strats = st.sidebar.multiselect("🎯 Mutantes a Forjar:", estrategias, default=estrategias)

if st.sidebar.button(f"🧠 DEEP MINE GLOBAL", type="primary", use_container_width=True, key="btn_global"):
    st.session_state['global_queue'] = target_strats.copy()
    st.session_state['abort_opt'] = False
    st.session_state['run_global'] = True
    st.rerun()

if st.sidebar.button("🤖 CREAR NUEVO MUTANTE IA", type="secondary", use_container_width=True, key="btn_mutant"):
    new_id = f"AI_MUTANT_{random.randint(100, 999)}"
    st.session_state['ai_algos'].append(new_id)
    estrategias.append(new_id)
    st.session_state[f'champion_{new_id}'] = {
        'b_team': [random.choice(todas_las_armas_b)], 's_team': [random.choice(todas_las_armas_s)], 
        'macro': "All-Weather", 'vol': "All-Weather", 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 
        'whale_f': 2.5, 'ado': 4.0, 'reinv': 0.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0, 
        'w_rsi': 0.0, 'w_z': 0.0, 'w_adx': 0.0, 'th_buy': 99.0, 'th_sell': -99.0, 'atr_tp': 2.0, 'atr_sl': 1.0
    }
    st.session_state['global_queue'] = [new_id]
    st.session_state['run_global'] = True
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center; color: #9932CC;'>🌌 DEEP FORGE (Standby)</h3>", unsafe_allow_html=True)
deep_epochs_target = st.sidebar.number_input("Objetivo Épocas Profundas", min_value=10000, max_value=10000000, value=1000000, step=10000)

if st.sidebar.button("🌌 CREAR MUTANTE PROFUNDO", type="secondary", use_container_width=True, key="btn_mutant_deep"):
    new_id = f"AI_DEEP_{random.randint(100, 999)}"
    st.session_state['ai_algos'].append(new_id)
    estrategias.append(new_id)
    st.session_state[f'champion_{new_id}'] = {
        'b_team': [random.choice(todas_las_armas_b)], 's_team': [random.choice(todas_las_armas_s)], 
        'macro': "All-Weather", 'vol': "All-Weather", 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 
        'whale_f': 2.5, 'ado': 4.0, 'reinv': 0.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0, 
        'w_rsi': 0.0, 'w_z': 0.0, 'w_adx': 0.0, 'th_buy': 99.0, 'th_sell': -99.0, 'atr_tp': 2.0, 'atr_sl': 1.0
    }
    st.session_state['deep_opt_state'] = {'s_id': new_id, 'target_epochs': deep_epochs_target, 'current_epoch': 0, 'paused': False}
    st.rerun()

deep_state = st.session_state.get('deep_opt_state', {})
if deep_state and deep_state.get('target_epochs', 0) > 0:
    st.sidebar.info(f"⚙️ Optimizando: **{deep_state['s_id']}**\nProgreso: {deep_state['current_epoch']:,} / {deep_state['target_epochs']:,} Épocas")
    if deep_state.get('paused', False):
        if st.sidebar.button("▶️ REANUDAR FORJA PROFUNDA", use_container_width=True, type="primary"):
            st.session_state['deep_opt_state']['paused'] = False
            st.rerun()
    else:
        if st.sidebar.button("⏸️ PAUSAR FORJA PROFUNDA", use_container_width=True):
            st.session_state['deep_opt_state']['paused'] = True
            st.rerun()
    
    if st.sidebar.button("⏹️ ABORTAR PROFUNDA", use_container_width=True):
        st.session_state['deep_opt_state'] = {}
        st.rerun()

def generar_reporte_universal(cap_ini, com_pct):
    res_str = f"📋 **REPORTE GENESIS LAB V168.0**\n\n"
    res_str += f"⏱️ Temporalidad: {intervalo_sel} | 📊 Ticker: {ticker}\n\n"
    for s_id in estrategias:
        v = st.session_state.get(f'champion_{s_id}', {})
        opt_icon = "✅" if st.session_state.get(f'opt_status_{s_id}', False) else "➖"
        res_str += f"🧬 **{s_id}** [{opt_icon}]\nNet Profit: ${v.get('net',0):,.2f} \nWin Rate: {v.get('winrate',0):.1f}%\n---\n"
    return res_str

st.sidebar.markdown("---")
if st.sidebar.button("📊 GENERAR REPORTE", use_container_width=True, key="btn_univ_report"):
    st.sidebar.text_area("Block Note Universal:", value=generar_reporte_universal(capital_inicial, comision_pct), height=200)

# ==========================================
# 🛑 5. EXTRACCIÓN Y WARM-UP (Sincronía con TV) 🛑
# ==========================================
@st.cache_data(ttl=3600, show_spinner="📡 Sincronizando Línea Temporal con TradingView (V168)...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, offset, is_micro):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        
        # 🔥 WARM-UP FANTASMA: Se descargan días extra para pre-calentar EMA y ATR 🔥
        warmup_days = 40 if is_micro else 150
        warmup_start = start - timedelta(days=warmup_days)
        start_ts = int(datetime.combine(warmup_start, datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        
        all_ohlcv, current_ts, error_count = [], start_ts, 0
        while current_ts < end_ts:
            try: ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=1000); error_count = 0 
            except Exception as e: 
                error_count += 1
                if error_count >= 3: return pd.DataFrame(), f"❌ ERROR: Exchange rechazó símbolo."
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
        
        macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd_df.iloc[:, 0].fillna(0)
        df['MACD_Sig'] = macd_df.iloc[:, 2].fillna(0)
        
        stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        df['Stoch_K'] = stoch_df.iloc[:, 0].fillna(50)
        df['Stoch_D'] = stoch_df.iloc[:, 1].fillna(50)

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
        
        df['PL30_L'] = df['Low'].shift(1).rolling(30, min_periods=1).min(); df['PH30_L'] = df['High'].shift(1).rolling(30, min_periods=1).max()
        df['PL100_L'] = df['Low'].shift(1).rolling(100, min_periods=1).min(); df['PH100_L'] = df['High'].shift(1).rolling(100, min_periods=1).max()
        df['PL300_L'] = df['Low'].shift(1).rolling(300, min_periods=1).min(); df['PH300_L'] = df['High'].shift(1).rolling(300, min_periods=1).max()

        df['RSI_Cross_Up'] = (df['RSI'] > df['RSI_MA']) & (df['RSI'].shift(1) <= df['RSI_MA'].shift(1))
        df['RSI_Cross_Dn'] = (df['RSI'] < df['RSI_MA']) & (df['RSI'].shift(1) >= df['RSI_MA'].shift(1))
        df['Macro_Bull'] = df['Close'] >= df['EMA_200']
        df['PP_Slope'] = (2*df['Close'] + df['Close'].shift(1) - df['Close'].shift(3) - 2*df['Close'].shift(4)) / 10.0
        
        # 🔥 WARM-UP TERMINADO: CORTAR DATOS FANTASMA PARA SINCRONÍA EXACTA 🔥
        target_start = pd.to_datetime(datetime.combine(start, datetime.min.time())) + timedelta(hours=offset)
        df = df[df.index >= target_start]

        gc.collect()
        return df, "OK"
    except Exception as e: return pd.DataFrame(), f"❌ ERROR FATAL GENERAL: {str(e)}"

df_global, status_api = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, utc_offset, is_micro)
if df_global.empty: st.error(status_api); st.stop()

dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)
st.sidebar.info(f"📊 Matrix Data: **{len(df_global):,} velas** ({dias_reales} días de Mercado Evaluado)")

a_c = df_global['Close'].values; a_o = df_global['Open'].values; a_h = df_global['High'].values; a_l = df_global['Low'].values
a_rsi = df_global['RSI'].values; a_rsi_ma = df_global['RSI_MA'].values; a_adx = df_global['ADX'].values
a_macd = df_global['MACD'].values; a_macd_sig = df_global['MACD_Sig'].values
a_stoch_k = df_global['Stoch_K'].values; a_stoch_d = df_global['Stoch_D'].values
a_bbl = df_global['BBL'].values; a_bbu = df_global['BBU'].values; a_bw = df_global['BB_Width'].values
a_bwa_s1 = npshift(df_global['BB_Width_Avg'].values, 1, -1.0)
a_wt1 = df_global['WT1'].values; a_wt2 = df_global['WT2'].values
a_ema50 = df_global['EMA_50'].values; a_ema200 = df_global['EMA_200'].values; a_atr = df_global['ATR'].values
a_rvol = df_global['RVol'].values; a_hvol = df_global['High_Vol'].values
a_vv = df_global['Vela_Verde'].values; a_vr = df_global['Vela_Roja'].values
a_rcu = df_global['RSI_Cross_Up'].values; a_rcd = df_global['RSI_Cross_Dn'].values
a_sqz_on = df_global['Squeeze_On'].values; a_bb_delta = df_global['BB_Delta'].values; a_bb_delta_avg = df_global['BB_Delta_Avg'].values
a_zscore = df_global['Z_Score'].values; a_rsi_bb_b = df_global['RSI_BB_Basis'].values; a_rsi_bb_d = df_global['RSI_BB_Dev'].values
a_lw = df_global['lower_wick'].values; a_uw = df_global['upper_wick'].values; a_bs = df_global['body_size'].values
a_mb = df_global['Macro_Bull'].values; a_fk = df_global['is_falling_knife'].values
a_pp_slope = df_global['PP_Slope'].fillna(0).values

a_pl30_l = df_global['PL30_L'].fillna(0).values; a_ph30_l = df_global['PH30_L'].fillna(99999).values
a_pl100_l = df_global['PL100_L'].fillna(0).values; a_ph100_l = df_global['PH100_L'].fillna(99999).values
a_pl300_l = df_global['PL300_L'].fillna(0).values; a_ph300_l = df_global['PH300_L'].fillna(99999).values

a_c_s1 = npshift(a_c, 1, 0.0); a_o_s1 = npshift(a_o, 1, 0.0); a_l_s1 = npshift(a_l, 1, 0.0); a_l_s5 = npshift(a_l, 5, 0.0)
a_h_s1 = npshift(a_h, 1, 0.0); a_h_s5 = npshift(a_h, 5, 0.0)
a_rsi_s1 = npshift(a_rsi, 1, 50.0); a_rsi_s5 = npshift(a_rsi, 5, 50.0)
a_wt1_s1 = npshift(a_wt1, 1, 0.0); a_wt2_s1 = npshift(a_wt2, 1, 0.0)
a_macd_s1 = npshift(a_macd, 1, 0.0)

def calcular_señales_numpy(hitbox, therm_w, adx_th, whale_f):
    n_len = len(a_c); s_dict = {}
    
    a_tsup = np.maximum(a_pl30_l, np.maximum(a_pl100_l, a_pl300_l)); a_tres = np.minimum(a_ph30_l, np.minimum(a_ph100_l, a_ph300_l))
    a_dsup = np.abs(a_c - a_tsup) / a_c * 100; a_dres = np.abs(a_c - a_tres) / a_c * 100
    sr_val = a_atr * 2.0

    ceil_w = np.where((a_ph30_l > a_c) & (a_ph30_l <= a_c + sr_val), 1, 0) + np.where((a_pl30_l > a_c) & (a_pl30_l <= a_c + sr_val), 1, 0) + np.where((a_ph100_l > a_c) & (a_ph100_l <= a_c + sr_val), 3, 0) + np.where((a_pl100_l > a_c) & (a_pl100_l <= a_c + sr_val), 3, 0) + np.where((a_ph300_l > a_c) & (a_ph300_l <= a_c + sr_val), 5, 0) + np.where((a_pl300_l > a_c) & (a_pl300_l <= a_c + sr_val), 5, 0)
    floor_w = np.where((a_ph30_l < a_c) & (a_ph30_l >= a_c - sr_val), 1, 0) + np.where((a_pl30_l < a_c) & (a_pl30_l >= a_c - sr_val), 1, 0) + np.where((a_ph100_l < a_c) & (a_ph100_l >= a_c - sr_val), 3, 0) + np.where((a_pl100_l < a_c) & (a_pl100_l >= a_c - sr_val), 3, 0) + np.where((a_ph300_l < a_c) & (a_ph300_l >= a_c - sr_val), 5, 0) + np.where((a_pl300_l < a_c) & (a_pl300_l >= a_c - sr_val), 5, 0)

    is_abyss = floor_w == 0
    is_hard_wall = ceil_w >= therm_w

    trinity_safe = a_mb & ~a_fk
    neon_up = a_sqz_on & (a_c >= a_bbu * 0.999) & a_vv; neon_dn = a_sqz_on & (a_c <= a_bbl * 1.001) & a_vr
    defcon_level = np.full(n_len, 5); m4 = neon_up | neon_dn; defcon_level[m4] = 4; m3 = m4 & (a_bb_delta > 0); defcon_level[m3] = 3; m2 = m3 & (a_bb_delta > a_bb_delta_avg) & (a_adx > adx_th); defcon_level[m2] = 2; m1 = m2 & (a_bb_delta > a_bb_delta_avg * 1.5) & (a_adx > adx_th + 5) & (a_rvol > 1.2); defcon_level[m1] = 1

    cond_defcon_buy = (defcon_level <= 2) & neon_up; cond_defcon_sell = (defcon_level <= 2) & neon_dn
    
    cond_therm_buy_bounce = (floor_w >= therm_w) & a_rcu & ~is_hard_wall
    cond_therm_buy_vacuum = (ceil_w <= 3) & neon_up & ~is_abyss
    cond_therm_sell_wall = (ceil_w >= therm_w) & a_rcd
    cond_therm_sell_panic = is_abyss & a_vr

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
    div_bull = (a_l_s1 < a_l_s5) & (a_rsi_s1 > a_rsi_s5) & (a_rsi < 35); div_bear = (a_h_s1 > a_h_s5) & (a_rsi_s1 < a_rsi_s5) & (a_rsi > 65)

    buy_score = np.zeros(n_len); base_mask = retro_peak | k_break_up | support_buy | div_bull
    buy_score = np.where(base_mask & retro_peak, 50.0, np.where(base_mask & ~retro_peak, 30.0, buy_score))
    buy_score += np.where(is_grav_sup, 25.0, 0.0); buy_score += np.where(whale_memory, 20.0, 0.0); buy_score += np.where(pump_memory, 15.0, 0.0); buy_score += np.where(div_bull, 15.0, 0.0); buy_score += np.where(k_break_up & ~retro_peak, 15.0, 0.0); buy_score += np.where(a_zscore < -2.0, 15.0, 0.0)
    
    sell_score = np.zeros(n_len); base_mask_s = retro_peak_sell | a_rcd | support_sell | div_bear
    sell_score = np.where(base_mask_s & retro_peak_sell, 50.0, np.where(base_mask_s & ~retro_peak_sell, 30.0, sell_score))
    sell_score += np.where(is_grav_res, 25.0, 0.0); sell_score += np.where(whale_memory, 20.0, 0.0); sell_score += np.where(dump_memory, 15.0, 0.0); sell_score += np.where(div_bear, 15.0, 0.0); sell_score += np.where(a_rcd & ~retro_peak_sell, 15.0, 0.0); sell_score += np.where(a_zscore > 2.0, 15.0, 0.0)

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
    s_dict['Commander_Buy'] = cond_pink_whale_buy | cond_lock_buy_bounce; s_dict['Commander_Sell'] = (a_c < a_ema50)

    s_dict['Q_Pink_Whale_Buy'] = cond_pink_whale_buy; s_dict['Q_Lock_Bounce'] = cond_lock_buy_bounce; s_dict['Q_Lock_Break'] = cond_lock_buy_break
    s_dict['Q_Neon_Up'] = neon_up; s_dict['Q_Defcon_Buy'] = cond_defcon_buy; s_dict['Q_Therm_Bounce'] = cond_therm_buy_bounce; s_dict['Q_Therm_Vacuum'] = cond_therm_buy_vacuum
    s_dict['Q_Nuclear_Buy'] = is_magenta & (wt_oversold | wt_cross_up); s_dict['Q_Early_Buy'] = is_magenta; s_dict['Q_Rebound_Buy'] = a_rcu & ~is_magenta
    s_dict['Q_Lock_Reject'] = cond_lock_sell_reject; s_dict['Q_Lock_Breakd'] = cond_lock_sell_breakd; s_dict['Q_Neon_Dn'] = neon_dn
    s_dict['Q_Defcon_Sell'] = cond_defcon_sell; s_dict['Q_Therm_Wall_Sell'] = cond_therm_sell_wall; s_dict['Q_Therm_Panic_Sell'] = cond_therm_sell_panic
    s_dict['Q_Nuclear_Sell'] = (a_rsi > 70) & (wt_overbought | wt_cross_dn); s_dict['Q_Early_Sell'] = (a_rsi > 70) & a_vr

    s_dict['Wyc_Spring_Buy'] = (a_l < a_tsup) & (a_c > a_tsup) & a_hvol
    s_dict['Wyc_Upthrust_Sell'] = (a_h > a_tres) & (a_c < a_tres) & a_hvol
    s_dict['VSA_Accum_Buy'] = (a_bs < a_atr * 0.5) & (a_lw > a_bs * 1.5) & a_hvol & a_vr
    s_dict['VSA_Dist_Sell'] = (a_bs < a_atr * 0.5) & (a_uw > a_bs * 1.5) & a_hvol & a_vv
    
    swing_range = a_ph30_l - a_pl30_l
    fib_618_b = a_ph30_l - (swing_range * 0.618)
    fib_618_s = a_pl30_l + (swing_range * 0.618)
    s_dict['Fibo_618_Buy'] = (a_l < fib_618_b) & (a_c > fib_618_b)
    s_dict['Fibo_618_Sell'] = (a_h > fib_618_s) & (a_c < fib_618_s)
    
    s_dict['MACD_Impulse_Buy'] = (a_macd > a_macd_sig) & (a_macd > 0) & (a_macd > a_macd_s1)
    s_dict['MACD_Exhaust_Sell'] = (a_macd < a_macd_sig) & (a_macd > 0) & (a_macd < a_macd_s1)
    s_dict['Stoch_OS_Buy'] = (a_stoch_k < 20) & (a_stoch_k > a_stoch_d)
    s_dict['Stoch_OB_Sell'] = (a_stoch_k > 80) & (a_stoch_k < a_stoch_d)

    s_dict['Organic_Vol'] = a_hvol
    s_dict['Organic_Squeeze'] = a_sqz_on
    s_dict['Organic_Safe'] = a_mb & ~a_fk
    s_dict['Organic_Pump'] = pump_memory
    s_dict['Organic_Dump'] = dump_memory

    return s_dict

def optimizar_ia_tracker(s_id, cap_ini, com_pct, reinv_q, target_ado, dias_reales, buy_hold_money, epochs=1, cur_net=-float('inf'), cur_fit=-float('inf'), deep_info=None):
    best_fit_live = cur_fit; best_net_live = cur_net; best_pf_live = 0.0; best_nt_live = 0; bp = None

    iters = 3000 * epochs
    chunks = min(iters, 50) 
    chunk_size = max(1, iters // chunks)
    start_time = time.time()
    n_len = len(a_c)

    f_buy = np.empty(n_len, dtype=bool); f_sell = np.empty(n_len, dtype=bool)
    default_f = np.zeros(n_len, dtype=bool)
    update_mod = int(max(1, chunks // 4))
    ones_mask = np.ones(n_len, dtype=bool)

    for c in range(chunks):
        if st.session_state.get('abort_opt', False): 
            st.warning("🛑 OPTIMIZACIÓN ABORTADA."); break

        r_hitbox = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]); r_therm = random.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        r_adx = random.choice([15.0, 20.0, 25.0, 30.0, 35.0]); r_whale = random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        s_dict = calcular_señales_numpy(r_hitbox, r_therm, r_adx, r_whale)

        m_mask_dict = {
            "Bull Only": a_mb, "Bear Only": ~a_mb, "Organic_Vol": s_dict['Organic_Vol'],
            "Organic_Squeeze": s_dict['Organic_Squeeze'], "Organic_Safe": s_dict['Organic_Safe'],
            "All-Weather": ones_mask, "Ignore": ones_mask
        }
        v_mask_dict = {
            "Trend": (a_adx >= r_adx), "Range": (a_adx < r_adx), "Organic_Pump": s_dict['Organic_Pump'],
            "Organic_Dump": s_dict['Organic_Dump'], "All-Weather": ones_mask, "Ignore": ones_mask
        }

        for _ in range(chunk_size): 
            f_buy.fill(False); f_sell.fill(False)
            
            # 🔥 OBLIGACIÓN GENÉTICA: Combinaciones masivas (5 a 12 armas) 🔥
            dna_b_team = random.sample(todas_las_armas_b, random.randint(5, 12))
            dna_s_team = random.sample(todas_las_armas_s, random.randint(5, 12))
            
            dna_macro = random.choice(["All-Weather", "Bull Only", "Bear Only", "Ignore", "Organic_Vol", "Organic_Squeeze", "Organic_Safe"])
            dna_vol = random.choice(["All-Weather", "Trend", "Range", "Ignore", "Organic_Pump", "Organic_Dump"])
            
            m_mask = m_mask_dict[dna_macro]
            v_mask = v_mask_dict[dna_vol]
            
            for r in dna_b_team: f_buy |= s_dict.get(r, default_f)
            f_buy &= (m_mask & v_mask)
            for r in dna_s_team: f_sell |= s_dict.get(r, default_f)
            
            r_w_rsi = random.uniform(-2.0, 2.0); r_w_z = random.uniform(-10.0, 10.0); r_w_adx = random.uniform(-2.0, 2.0)
            r_th_b = random.uniform(0.0, 100.0); r_th_s = random.uniform(-100.0, 0.0)
            
            r_atr_tp = round(random.uniform(0.5, 15.0), 2); r_atr_sl = round(random.uniform(1.0, 20.0), 2)
            r_reinv = float(random.choice([20.0, 50.0, 100.0, 100.0])) 
            r_ado = float(round(random.uniform(1.0, 15.0), 1))
            
            net, pf, nt, mdd = simular_crecimiento_exponencial_ia_core(
                a_h, a_l, a_c, a_o, a_atr, a_rsi, a_zscore, a_adx,
                f_buy, f_sell, r_w_rsi, r_w_z, r_w_adx, r_th_b, r_th_s,
                r_atr_tp, r_atr_sl, float(cap_ini), float(com_pct), float(r_reinv), 0.05
            )

            # 🔥 V168: CASTIGO A LA PEREZA (Calcina bots con < 5 trades o ADO < 1.0) 🔥
            if nt >= 5: 
                ado_actual = nt / max(1, dias_reales)
                if ado_actual >= 1.0 and net > 0:
                    safe_pf = min(pf, 10.0)
                    ado_factor = ado_actual ** 2.0 
                    fit_score = net * safe_pf * ado_factor

                    if fit_score > best_fit_live:
                        best_fit_live = fit_score; best_net_live = net; best_pf_live = pf; best_nt_live = nt
                        bp = {'b_team': dna_b_team, 's_team': dna_s_team, 'macro': dna_macro, 'vol': dna_vol, 'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'fit': fit_score, 'net': net, 'winrate': 0.0, 'reinv': r_reinv, 'ado': r_ado, 'w_rsi': r_w_rsi, 'w_z': r_w_z, 'w_adx': r_w_adx, 'th_buy': r_th_b, 'th_sell': r_th_s, 'atr_tp': r_atr_tp, 'atr_sl': r_atr_sl}
                
        if c == 0 or c == (chunks - 1) or c % update_mod == 0:
            elapsed = time.time() - start_time
            pct_done = int(((c + 1) / chunks) * 100); combos = (c + 1) * chunk_size; eta = (elapsed / (c + 1)) * (chunks - c - 1)
            
            if deep_info:
                macro_pct = int(((deep_info['current'] + c*(chunk_size//3000)) / deep_info['total']) * 100)
                title = f"🌌 DEEP FORGE: {s_id}"
                subtitle = f"Progreso Macro: {deep_info['current']:,} / {deep_info['total']:,} Épocas ({macro_pct}%)<br>ETA Bloque: {eta:.1f}s"
                color = "#9932CC"
            else:
                title = f"GENESIS LAB V168: {s_id}"
                subtitle = f"Progreso: {pct_done}% | ADN Probado: {combos:,}<br>ETA: {eta:.1f} segs"
                color = "#00FFFF"

            ph_holograma.markdown(f"""
            <style>
            .loader-container {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.95); padding: 35px; border-radius: 20px; border: 2px solid {color}; box-shadow: 0 0 50px {color};}}
            .rocket {{ font-size: 8rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 20px {color}); }}
            @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            </style>
            <div class="loader-container">
                <div class="rocket">🧬</div>
                <div style="color: {color}; font-size: 1.8rem; font-weight: bold; margin-top: 15px;">{title}</div>
                <div style="color: white; font-size: 1.3rem;">{subtitle}</div>
                <div style="color: #00FF00; font-weight: bold; font-size: 1.5rem; margin-top: 15px;">🏆 Mejor Neta: ${best_net_live:.2f} | PF: {best_pf_live:.1f}x</div>
            </div>
            """, unsafe_allow_html=True)
            
    return bp if bp else None

def run_backtest_eval(s_id, cap_ini, com_pct):
    vault = st.session_state.get(f'champion_{s_id}', {})
    s_dict = calcular_señales_numpy(vault.get('hitbox',1.5), vault.get('therm_w',4.0), vault.get('adx_th',25.0), vault.get('whale_f',2.5))
    
    n_len = len(a_c)
    f_tp = np.full(n_len, float(vault.get('atr_tp', 0.0))); f_sl = np.full(n_len, float(vault.get('atr_sl', 0.0)))
    f_buy = np.zeros(n_len, dtype=bool); f_sell = np.zeros(n_len, dtype=bool)
    default_f = np.zeros(n_len, dtype=bool)
    ones_mask = np.ones(n_len, dtype=bool)

    if vault.get('macro') == "Bull Only": m_mask = a_mb
    elif vault.get('macro') == "Bear Only": m_mask = ~a_mb
    elif vault.get('macro') == "Organic_Vol": m_mask = s_dict['Organic_Vol']
    elif vault.get('macro') == "Organic_Squeeze": m_mask = s_dict['Organic_Squeeze']
    elif vault.get('macro') == "Organic_Safe": m_mask = s_dict['Organic_Safe']
    else: m_mask = ones_mask

    if vault.get('vol') == "Trend": v_mask = (a_adx >= vault.get('adx_th', 25.0))
    elif vault.get('vol') == "Range": v_mask = (a_adx < vault.get('adx_th', 25.0))
    elif vault.get('vol') == "Organic_Pump": v_mask = s_dict['Organic_Pump']
    elif vault.get('vol') == "Organic_Dump": v_mask = s_dict['Organic_Dump']
    else: v_mask = ones_mask

    for r in vault.get('b_team', []): f_buy |= s_dict.get(r, default_f)
    f_buy &= (m_mask & v_mask)
    for r in vault.get('s_team', []): f_sell |= s_dict.get(r, default_f)
    
    score_arr = (a_rsi * vault.get('w_rsi', 0.0)) + (a_zscore * vault.get('w_z', 0.0)) + (a_adx * vault.get('w_adx', 0.0))
    f_buy |= (score_arr > vault.get('th_buy', 999.0))
    f_sell |= (score_arr < vault.get('th_sell', -999.0))

    df_strat = df_global.copy()
    df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
    df_strat['Active_TP'], df_strat['Active_SL'] = f_tp, f_sl
    
    eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, cap_ini, float(vault.get('reinv', 0.0)), com_pct, 0.05)
    return df_strat, eq_curve, t_log, total_comms

def generar_pine_script(s_id, vault, sym, tf, buy_pct=20, sell_pct=20):
    v_hb = vault.get('hitbox', 1.5); v_tw = vault.get('therm_w', 4.0)
    v_adx = vault.get('adx_th', 25.0); v_wf = vault.get('whale_f', 2.5)
    
    json_buy = f'{{"passphrase": "ASTRONAUTA", "action": "{{{{strategy.order.action}}}}", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {buy_pct}, "limit_price": {{{{close}}}}, "side": "🟢 COMPRA"}}'
    json_sell = f'{{"passphrase": "ASTRONAUTA", "action": "{{{{strategy.order.action}}}}", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {sell_pct}, "limit_price": {{{{close}}}}, "side": "🔴 VENTA"}}'

    ps_base = f"""//@version=5
strategy("{s_id} MATRIX - {sym} [{tf}]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_value=0.25, slippage=0)
wt_enter_long = input.text_area(defval='{json_buy}', title="🟢 WT: Mensaje Enter Long")
wt_exit_long  = input.text_area(defval='{json_sell}', title="🔴 WT: Mensaje Exit Long")

grp_time = "📅 FILTRO DE FECHA"
start_year = input.int(2020, "Año de Inicio", group=grp_time)
start_month = input.int(1, "Mes de Inicio", group=grp_time)
start_day = input.int(1, "Día de Inicio", group=grp_time)
window = time >= timestamp(syminfo.timezone, start_year, start_month, start_day, 0, 0)

hitbox_pct   = {v_hb}
therm_wall   = {v_tw}
adx_trend    = {v_adx}
whale_factor = {v_wf}
"""
    ps_indicators = """
vol_ma_20 = ta.sma(volume, 20)
vol_ma_100 = ta.sma(volume, 100)
ema50  = ta.ema(close, 50)
ema200 = ta.ema(close, 200)
rsi_v = ta.rsi(close, 14)
atr = ta.atr(14)
body_size = math.abs(close - open)
lower_wick = math.min(open, close) - low
upper_wick = high - math.max(open, close)
is_falling_knife = (open[1] - close[1]) > (atr[1] * 1.5)
[di_plus, di_minus, adx] = ta.dmi(14, 14)
rvol = volume / (vol_ma_100 == 0 ? 1 : vol_ma_100)
high_vol = volume > vol_ma_20

ap = hlc3, esa = ta.ema(ap, 10), d_wt = ta.ema(math.abs(ap - esa), 10)
wt1 = ta.ema((ap - esa) / (0.015 * (d_wt == 0 ? 1 : d_wt)), 21), wt2 = ta.sma(wt1, 4)

basis = ta.sma(close, 20)
stdev20 = ta.stdev(close, 20)
dev = 2.0 * stdev20
bbu = basis + dev
bbl = basis - dev
bb_width = (bbu - bbl) / basis
bb_width_avg = ta.sma(bb_width, 20)
bb_delta = bb_width - nz(bb_width[1], 0)
bb_delta_avg = ta.sma(bb_delta, 10)

kc_u = ta.sma(close, 20) + (atr * 1.5)
kc_l = ta.sma(close, 20) - (atr * 1.5)
squeeze_on = (bbu < kc_u) and (bbl > kc_l)

z_score = stdev20 == 0 ? 0 : (close - basis) / stdev20
rsi_bb_basis = ta.sma(rsi_v, 14), rsi_bb_dev = ta.stdev(rsi_v, 14) * 2.0

vela_verde = close > open, vela_roja = close < open
rsi_ma = ta.sma(rsi_v, 14)
rsi_cross_up = (rsi_v > rsi_ma) and (nz(rsi_v[1]) <= nz(rsi_ma[1]))
rsi_cross_dn = (rsi_v < rsi_ma) and (nz(rsi_v[1]) >= nz(rsi_ma[1]))
macro_bull = close >= ema200
trinity_safe = macro_bull and not is_falling_knife

[macdLine, signalLine, _] = ta.macd(close, 12, 26, 9)
stoch_k = ta.sma(ta.stoch(close, high, low, 14), 3)
stoch_d = ta.sma(stoch_k, 3)

local_high = ta.highest(high[1], 30)
local_low = ta.lowest(low[1], 30)
swing_range = local_high - local_low
fib_618_b = local_high - (swing_range * 0.618)
fib_618_s = local_low + (swing_range * 0.618)

low_30 = ta.lowest(low[1], 30), low_100 = ta.lowest(low[1], 100), low_300 = ta.lowest(low[1], 300)
a_tsup = math.max(nz(low_30, 0), nz(low_100, 0), nz(low_300, 0))

high_30 = ta.highest(high[1], 30), high_100 = ta.highest(high[1], 100), high_300 = ta.highest(high[1], 300)
a_tres = math.min(nz(high_30, 99999), nz(high_100, 99999), nz(high_300, 99999))

a_dsup = math.abs(close - a_tsup) / close * 100, a_dres = math.abs(close - a_tres) / close * 100
sr_val = atr * 2.0

ceil_w = 0, floor_w = 0
ceil_w += (ta.highest(high[1], 30) > close and ta.highest(high[1], 30) <= close + sr_val) ? 1 : 0
ceil_w += (ta.lowest(low[1], 30) > close and ta.lowest(low[1], 30) <= close + sr_val) ? 1 : 0
ceil_w += (ta.highest(high[1], 100) > close and ta.highest(high[1], 100) <= close + sr_val) ? 3 : 0
ceil_w += (ta.lowest(low[1], 100) > close and ta.lowest(low[1], 100) <= close + sr_val) ? 3 : 0
ceil_w += (ta.highest(high[1], 300) > close and ta.highest(high[1], 300) <= close + sr_val) ? 5 : 0
ceil_w += (ta.lowest(low[1], 300) > close and ta.lowest(low[1], 300) <= close + sr_val) ? 5 : 0

floor_w += (ta.highest(high[1], 30) < close and ta.highest(high[1], 30) >= close - sr_val) ? 1 : 0
floor_w += (ta.lowest(low[1], 30) < close and ta.lowest(low[1], 30) >= close - sr_val) ? 1 : 0
floor_w += (ta.highest(high[1], 100) < close and ta.highest(high[1], 100) >= close - sr_val) ? 3 : 0
floor_w += (ta.lowest(low[1], 100) < close and ta.lowest(low[1], 100) >= close - sr_val) ? 3 : 0
floor_w += (ta.highest(high[1], 300) < close and ta.highest(high[1], 300) >= close - sr_val) ? 5 : 0
floor_w += (ta.lowest(low[1], 300) < close and ta.lowest(low[1], 300) >= close - sr_val) ? 5 : 0

is_abyss = floor_w == 0
is_hard_wall = ceil_w >= therm_wall

neon_up = squeeze_on and (close >= bbu * 0.999) and vela_verde
neon_dn = squeeze_on and (close <= bbl * 1.001) and vela_roja
defcon_level = 5
if neon_up or neon_dn
    defcon_level := 4
    if bb_delta > 0
        defcon_level := 3
        if bb_delta > bb_delta_avg and adx > adx_trend
            defcon_level := 2
            if bb_delta > (bb_delta_avg * 1.5) and adx > (adx_trend + 5) and rvol > 1.2
                defcon_level := 1

cond_defcon_buy = defcon_level <= 2 and neon_up
cond_defcon_sell = defcon_level <= 2 and neon_dn

cond_therm_buy_bounce = (floor_w >= therm_wall) and rsi_cross_up and not is_hard_wall
cond_therm_buy_vacuum = (ceil_w <= 3) and neon_up and not is_abyss
cond_therm_sell_wall = (ceil_w >= therm_wall) and rsi_cross_dn
cond_therm_sell_panic = is_abyss and vela_roja

tol = atr * 0.5, is_grav_sup = a_dsup < hitbox_pct, is_grav_res = a_dres < hitbox_pct
cross_up_res = (close > a_tres) and (nz(close[1]) <= nz(a_tres[1]))
cross_dn_sup = (close < a_tsup) and (nz(close[1]) >= nz(a_tsup[1]))
cond_lock_buy_bounce = is_grav_sup and (low <= a_tsup + tol) and (close > a_tsup) and vela_verde
cond_lock_buy_break = is_grav_res and cross_up_res and high_vol and vela_verde
cond_lock_sell_reject = is_grav_res and (high >= a_tres - tol) and (close < a_tres) and vela_roja
cond_lock_sell_breakd = is_grav_sup and cross_dn_sup and vela_roja

flash_vol = (rvol > whale_factor * 0.8) and (body_size > atr * 0.3)
whale_buy = flash_vol and vela_verde, whale_sell = flash_vol and vela_roja
whale_memory = whale_buy or nz(whale_buy[1]) or nz(whale_buy[2]) or whale_sell or nz(whale_sell[1]) or nz(whale_sell[2])
is_whale_icon = whale_buy and not nz(whale_buy[1])
rsi_vel = rsi_v - nz(rsi_v[1])
pre_pump = (high > bbu or rsi_vel > 5) and flash_vol and vela_verde
pump_memory = pre_pump or nz(pre_pump[1]) or nz(pre_pump[2])
pre_dump = (low < bbl or rsi_vel < -5) and flash_vol and vela_roja
dump_memory = pre_dump or nz(pre_dump[1]) or nz(pre_dump[2])

retro_peak = (rsi_v < 30) and (close < bbl)
retro_peak_sell = (rsi_v > 70) and (close > bbu)
k_break_up = (rsi_v > (rsi_bb_basis + rsi_bb_dev)) and (nz(rsi_v[1]) <= (nz(rsi_bb_basis[1]) + nz(rsi_bb_dev[1])))
support_buy = is_grav_sup and rsi_cross_up
support_sell = is_grav_res and rsi_cross_dn
div_bull = nz(low[1]) < nz(low[5]) and nz(rsi_v[1]) > nz(rsi_v[5]) and (rsi_v < 35)
div_bear = nz(high[1]) > nz(high[5]) and nz(rsi_v[1]) < nz(rsi_v[5]) and (rsi_v > 65)

base_mask = retro_peak or k_break_up or support_buy or div_bull
buy_score = 0.0
buy_score := (base_mask and retro_peak) ? 50.0 : (base_mask and not retro_peak) ? 30.0 : buy_score
buy_score += is_grav_sup ? 25.0 : 0.0
buy_score += whale_memory ? 20.0 : 0.0
buy_score += pump_memory ? 15.0 : 0.0
buy_score += div_bull ? 15.0 : 0.0
buy_score += (k_break_up and not retro_peak) ? 15.0 : 0.0
buy_score += (z_score < -2.0) ? 15.0 : 0.0
buy_score := buy_score > 99 ? 99.0 : buy_score

base_mask_s = retro_peak_sell or rsi_cross_dn or support_sell or div_bear
sell_score = 0.0
sell_score := (base_mask_s and retro_peak_sell) ? 50.0 : (base_mask_s and not retro_peak_sell) ? 30.0 : sell_score
sell_score += is_grav_res ? 25.0 : 0.0
sell_score += whale_memory ? 20.0 : 0.0
sell_score += dump_memory ? 15.0 : 0.0
sell_score += div_bear ? 15.0 : 0.0
sell_score += (rsi_cross_dn and not retro_peak_sell) ? 15.0 : 0.0
sell_score += (z_score > 2.0) ? 15.0 : 0.0
sell_score := sell_score > 99 ? 99.0 : sell_score

is_magenta = (buy_score >= 70) or retro_peak
is_magenta_sell = (sell_score >= 70) or retro_peak_sell
cond_pink_whale_buy = is_magenta and is_whale_icon

wt_cross_up = (wt1 > wt2) and (nz(wt1[1]) <= nz(wt2[1]))
wt_cross_dn = (wt1 < wt2) and (nz(wt1[1]) >= nz(wt2[1]))
wt_oversold = wt1 < -60
wt_overbought = wt1 > 60

// 🔥 POOL DE ARMAS 🔥
ping_b = (adx < adx_trend) and (close < bbl) and vela_verde
ping_s = (close > bbu) or (rsi_v > 70)
squeeze_b = neon_up
squeeze_s = (close < ema50)
therm_b = cond_therm_buy_bounce
therm_s = cond_therm_sell_wall
climax_b = cond_pink_whale_buy
climax_s = (rsi_v > 80)
lock_b = cond_lock_buy_bounce
lock_s = cond_lock_sell_reject
defcon_b = cond_defcon_buy
defcon_s = cond_defcon_sell
jugg_b = macro_bull and (close > ema50) and nz(close[1]) < nz(ema50[1]) and vela_verde and not is_falling_knife
jugg_s = (close < ema50)
trinity_b = macro_bull and (rsi_v < 35) and vela_verde and not is_falling_knife
trinity_s = (rsi_v > 75) or (close < ema200)
lev_b = macro_bull and rsi_cross_up and (rsi_v < 45)
lev_s = (close < ema200)
commander_b = cond_pink_whale_buy or cond_lock_buy_bounce
commander_s = (close < ema50)

r_Pink_Whale_Buy = cond_pink_whale_buy
r_Lock_Bounce = cond_lock_buy_bounce
r_Lock_Break = cond_lock_buy_break
r_Neon_Up = neon_up
r_Defcon_Buy = cond_defcon_buy
r_Therm_Bounce = cond_therm_buy_bounce
r_Therm_Vacuum = cond_therm_buy_vacuum
r_Nuclear_Buy = is_magenta and (wt_oversold or wt_cross_up)
r_Early_Buy = is_magenta
r_Rebound_Buy = rsi_cross_up and not is_magenta
r_Lock_Reject = cond_lock_sell_reject
r_Lock_Breakd = cond_lock_sell_breakd
r_Neon_Dn = neon_dn
r_Defcon_Sell = cond_defcon_sell
r_Therm_Wall_Sell = cond_therm_sell_wall
r_Therm_Panic_Sell = cond_therm_sell_panic
r_Nuclear_Sell = (rsi_v > 70) and (wt_overbought or wt_cross_dn)
r_Early_Sell = (rsi_v > 70) and vela_roja

wyc_spring_buy = (low < a_tsup) and (close > a_tsup) and high_vol
wyc_upthrust_sell = (high > a_tres) and (close < a_tres) and high_vol
vsa_accum_buy = (body_size < atr * 0.5) and (lower_wick > body_size * 1.5) and high_vol and vela_roja
vsa_dist_sell = (body_size < atr * 0.5) and (upper_wick > body_size * 1.5) and high_vol and vela_verde
fibo_618_buy = (low < fib_618_b) and (close > fib_618_b)
fibo_618_sell = (high > fib_618_s) and (close < fib_618_s)
macd_impulse_buy = (macdLine > signalLine) and (macdLine > 0) and (macdLine > nz(macdLine[1]))
macd_exhaust_sell = (macdLine < signalLine) and (macdLine > 0) and (macdLine < nz(macdLine[1]))
stoch_os_buy = (stoch_k < 20) and (stoch_k > stoch_d)
stoch_ob_sell = (stoch_k > 80) and (stoch_k < stoch_d)
"""
    ps_logic = ""
    if vault.get('macro') == "Bull Only": m_cond = "macro_bull"
    elif vault.get('macro') == "Bear Only": m_cond = "not macro_bull"
    elif vault.get('macro') == "Organic_Vol": m_cond = "high_vol"
    elif vault.get('macro') == "Organic_Squeeze": m_cond = "squeeze_on"
    elif vault.get('macro') == "Organic_Safe": m_cond = "trinity_safe"
    else: m_cond = "true"

    if vault.get('vol') == "Trend": v_cond = "(adx >= adx_trend)"
    elif vault.get('vol') == "Range": v_cond = "(adx < adx_trend)"
    elif vault.get('vol') == "Organic_Pump": v_cond = "pump_memory"
    elif vault.get('vol') == "Organic_Dump": v_cond = "dump_memory"
    else: v_cond = "true"

    b_cond = " or ".join([pine_map.get(x, "false") for x in vault.get('b_team', [])]) if vault.get('b_team') else "false"
    s_cond = " or ".join([pine_map.get(x, "false") for x in vault.get('s_team', [])]) if vault.get('s_team') else "false"
    
    ps_logic += f"""
float w_rsi = {vault.get('w_rsi',0.0):.4f}
float w_z = {vault.get('w_z',0.0):.4f}
float w_adx = {vault.get('w_adx',0.0):.4f}
float math_score = (rsi_v * w_rsi) + (z_score * w_z) + (adx * w_adx)

bool signal_buy = (({b_cond}) and {m_cond} and {v_cond}) or (math_score > {vault.get('th_buy',999):.2f})
bool signal_sell = ({s_cond}) or (math_score < {vault.get('th_sell',-999):.2f})

float atr_tp_mult = {vault.get('atr_tp',2.0):.2f}
float atr_sl_mult = {vault.get('atr_sl',1.0):.2f}
"""
    ps_exec = """
var float my_atr = na

if signal_buy and strategy.position_size == 0 and window
    strategy.entry("In", strategy.long, alert_message=wt_enter_long)
    my_atr := atr

// 🔥 V168: LÍMITES DINÁMICOS AMARRADOS AL PRECIO REAL DE ENTRADA 🔥
if strategy.position_size > 0
    float entry_p = strategy.opentrades.entry_price(0)
    strategy.exit("TP/SL", "In", limit=entry_p + (my_atr * atr_tp_mult), stop=entry_p - (my_atr * atr_sl_mult), alert_message=wt_exit_long)

if signal_sell and strategy.position_size > 0
    strategy.close("In", comment="Dyn_Exit", alert_message=wt_exit_long)

plotshape(signal_buy, title="COMPRA", style=shape.triangleup, location=location.belowbar, color=color.aqua, size=size.tiny)
plotshape(signal_sell, title="VENTA", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.tiny)
"""
    return ps_base + ps_indicators + ps_logic + ps_exec

# ==========================================
# 🛑 7. EJECUCIÓN GLOBAL (COLA ASÍNCRONA)
# ==========================================
if 'global_queue' not in st.session_state:
    st.session_state['global_queue'] = []

if st.session_state.get('run_global', False):
    time.sleep(0.1) 
    if len(st.session_state['global_queue']) > 0:
        s_id = st.session_state['global_queue'].pop(0)
        ph_holograma.markdown(f"<div style='text-align:center; padding: 20px; background: rgba(0,0,0,0.8); border: 2px solid cyan; border-radius: 10px;'><h2 style='color:cyan;'>⚙️ Forjando ADN: {s_id}...</h2><h4 style='color:lime;'>Quedan {len(st.session_state['global_queue'])} mutantes en incubación.</h4></div>", unsafe_allow_html=True)
        time.sleep(0.1)
        
        v = st.session_state.get(f'champion_{s_id}', {})
        buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
        buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
        
        bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, float(v.get('reinv',0.0)), float(v.get('ado',4.0)), dias_reales, buy_hold_money, epochs=global_epochs, cur_net=float(v.get('net',-float('inf'))), cur_fit=float(v.get('fit',-float('inf'))))
        
        if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True
        
        st.rerun()
    else:
        st.session_state['run_global'] = False
        ph_holograma.empty()
        st.sidebar.success("✅ ¡Incubación Genética Completada!")
        time.sleep(2); st.rerun()

# ==========================================
# 🌌 8. EJECUCIÓN PROFUNDA (DEEP FORGE STANDBY)
# ==========================================
deep_state = st.session_state.get('deep_opt_state', {})
if deep_state and not deep_state.get('paused', False) and deep_state.get('current_epoch', 0) < deep_state.get('target_epochs', 0):
    time.sleep(0.1) 
    
    chunk = 1000 
    if deep_state['target_epochs'] - deep_state['current_epoch'] < chunk:
        chunk = deep_state['target_epochs'] - deep_state['current_epoch']
        
    s_id = deep_state['s_id']
    v = st.session_state.get(f'champion_{s_id}', {})
    buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
    buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
    
    deep_info = {'current': deep_state['current_epoch'], 'total': deep_state['target_epochs']}
    
    bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, float(v.get('reinv',0.0)), float(v.get('ado',4.0)), dias_reales, buy_hold_money, epochs=chunk, cur_net=float(v.get('net',-float('inf'))), cur_fit=float(v.get('fit',-float('inf'))), deep_info=deep_info)
    
    if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True
    
    st.session_state['deep_opt_state']['current_epoch'] += chunk
    
    if st.session_state['deep_opt_state']['current_epoch'] >= deep_state['target_epochs']:
        st.session_state['deep_opt_state']['paused'] = True
        ph_holograma.empty()
        st.sidebar.success(f"🌌 ¡FORJA PROFUNDA COMPLETADA PARA {s_id}!")
        time.sleep(2)
        
    st.rerun()

st.title("🛡️ GENESIS LAB - The Omni-Brain")

with st.expander("🏆 SALÓN DE LA FAMA GENÉTICA (Ordenado por Rentabilidad Neta)", expanded=False):
    leaderboard_data = []
    for s in estrategias:
        v = st.session_state.get(f'champion_{s}', {})
        fit = v.get('fit', -float('inf'))
        opt_str = "✅" if fit != -float('inf') else "➖ No Opt"
        net_val = v.get('net', 0)
        leaderboard_data.append({"Mutante": s, "Neto_Num": net_val, "Rentabilidad": f"${net_val:,.2f}", "WinRate": f"{v.get('winrate', 0):.1f}%", "Estado": opt_str})
    
    leaderboard_data.sort(key=lambda x: x['Neto_Num'], reverse=True)
    for item in leaderboard_data: del item['Neto_Num']
    st.table(pd.DataFrame(leaderboard_data))

st.markdown("<h3 style='text-align: center; color: #00FF00;'>🎡 CARRUSEL DE MUTANTES IA</h3>", unsafe_allow_html=True)
tab_names = list(tab_id_map.keys())
selected_tab_name = st.selectbox("Selecciona un Espécimen:", tab_names, index=len(tab_names)-1)

s_id = tab_id_map[selected_tab_name]
is_opt = st.session_state.get(f'opt_status_{s_id}', False)
opt_badge = "<span style='color: lime;'>✅ ADN OPTIMIZADO</span>" if is_opt else "<span style='color: gray;'>➖ ADN VIRGEN</span>"
vault = st.session_state.get(f'champion_{s_id}', {})

st.markdown(f"### {selected_tab_name} {opt_badge}", unsafe_allow_html=True)

with st.expander("🧬 VER ADN DEL MUTANTE Y ARMAS TÁCTICAS", expanded=True):
    st.markdown(f"**🟢 Escuadrón de Compra:** {', '.join(vault.get('b_team', []))}")
    st.markdown(f"**🔴 Escuadrón de Venta:** {', '.join(vault.get('s_team', []))}")
    st.markdown(f"**🌍 Clima Macro:** `{vault.get('macro', '')}` | **🌪️ Clima Volatilidad:** `{vault.get('vol', '')}`")
    st.markdown(f"**🎛️ Pesos del Perceptrón:** RSI: `{vault.get('w_rsi',0):.2f}` | Z-Score: `{vault.get('w_z',0):.2f}` | ADX: `{vault.get('w_adx',0):.2f}`")
    st.markdown(f"**📏 Gatillos Sensibles:** Buy > `{vault.get('th_buy',0):.2f}` | Sell < `{vault.get('th_sell',0):.2f}`")
    st.markdown(f"**🎯 Camaleón ATR (Toma de Ganancias):** TP: `{vault.get('atr_tp',0):.2f}x` | SL: `{vault.get('atr_sl',0):.2f}x`")

c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
st.session_state[f'champion_{s_id}']['ado'] = c_ia1.slider("🎯 Target ADO (IA Override)", 0.0, 100.0, value=float(vault.get('ado', 4.0)), key=f"ui_{s_id}_ado_w", step=0.5)
st.session_state[f'champion_{s_id}']['reinv'] = c_ia2.slider("💵 Reinversión % (IA Override)", 0.0, 100.0, value=float(vault.get('reinv', 0.0)), key=f"ui_{s_id}_reinv_w", step=5.0)

c_ps1, c_ps2 = st.columns(2)
ps_buy_pct = c_ps1.number_input("🟢 % Inversión Compra (Pine Script)", min_value=1, max_value=100, value=20, step=1, key=f"ui_{s_id}_ps_buy")
ps_sell_pct = c_ps2.number_input("🔴 % Desinversión Venta (Pine Script)", min_value=1, max_value=100, value=20, step=1, key=f"ui_{s_id}_ps_sell")

c_btn1, c_btn2 = c_ia3.columns(2)
if c_btn1.button(f"🚀 FORJAR RÁPIDO ({global_epochs*3}k)", type="primary", key=f"btn_opt_{s_id}"):
    ph_holograma.markdown(f"<div style='text-align:center; padding: 20px; background: rgba(0,0,0,0.8); border: 2px solid #FF00FF; border-radius: 10px;'><h2 style='color:#FF00FF;'>🚀 Mutando {s_id}...</h2></div>", unsafe_allow_html=True)
    time.sleep(0.1)
    buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
    bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, float(vault.get('reinv', 0.0)), float(vault.get('ado', 4.0)), dias_reales, capital_inicial * (buy_hold_ret / 100.0), epochs=global_epochs, cur_net=float(vault.get('net', -float('inf'))), cur_fit=float(vault.get('fit', -float('inf'))))
    if bp: save_champion(s_id, bp); st.session_state[f'opt_status_{s_id}'] = True; st.success("👑 ¡Mutante Forjado!")
    time.sleep(1); ph_holograma.empty(); st.rerun()

if c_btn2.button(f"🌌 ACTIVAR FORJA PROFUNDA", type="secondary", key=f"btn_deep_{s_id}"):
    st.session_state['deep_opt_state'] = {'s_id': s_id, 'target_epochs': deep_epochs_target, 'current_epoch': 0, 'paused': False}
    st.rerun()

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
        gpp = exs[exs['Ganancia_$'] > 0]['Ganancia_$'].sum(); gll = abs(exs[exs['Ganancia_$'] < 0]['Ganancia_$'].sum())
        pf_val = gpp / gll if gll > 0 else float('inf')

vault['net'] = eq_curve[-1] - capital_inicial; vault['winrate'] = wr
mdd = abs((((pd.Series(eq_curve) - pd.Series(eq_curve).cummax()) / pd.Series(eq_curve).cummax()) * 100).min())
ado_val = tt / dias_reales if dias_reales > 0 else 0.0

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Net Profit", f"${eq_curve[-1]-capital_inicial:,.2f}", f"{ret_pct:.2f}%")
c2.metric("ALPHA", f"{alpha_pct:.2f}%", delta_color="normal" if alpha_pct > 0 else "inverse")
c3.metric("Trades", f"{tt}", f"ADO: {ado_val:.2f}")
c4.metric("Win Rate", f"{wr:.1f}%")
c5.metric("Profit Factor", f"{pf_val:.2f}x")
c6.metric("Drawdown", f"{mdd:.2f}%", delta_color="inverse")
c7.metric("Comisiones", f"${total_comms:,.2f}", delta_color="inverse")

with st.expander("📝 CÓDIGO DE TRASPLANTE A TRADINGVIEW (PINE SCRIPT)", expanded=False):
    st.info("Traducción Matemática Idéntica a TradingView. Warm-Up y TP/SL Intrabarra corregidos.")
    st.code(generar_pine_script(s_id, vault, ticker.split('/')[0], iv_download, ps_buy_pct, ps_sell_pct), language="pine")

st.markdown("---")
st.info("🖱️ **TIP GRÁFICO:** Si las velas se ven aplanadas, haz **Doble Clic** dentro del gráfico o usa el botón **'Autoscale'** (Casita) en el menú de la esquina superior derecha.")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(
    x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], 
    name="Precio", increasing_line_color='cyan', decreasing_line_color='magenta'
), row=1, col=1)

fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_50'], mode='lines', name='Río Center (EMA 50)', line=dict(color='yellow', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_200'], mode='lines', name='Macro Trend (EMA 200)', line=dict(color='purple', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBU'], mode='lines', name='Squeeze Top (BBU)', line=dict(color='rgba(128,128,128,0.5)', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBL'], mode='lines', name='Squeeze Bot (BBL)', line=dict(color='rgba(128,128,128,0.5)', width=1)), row=1, col=1)

if not dftr.empty:
    ents = dftr[dftr['Tipo'] == 'ENTRY']
    fig.add_trace(go.Scatter(x=ents['Fecha'], y=ents['Precio'], mode='markers', name='COMPRA', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=2, color='white'))), row=1, col=1)
    wins = dftr[dftr['Tipo'].isin(['TP', 'DYN_WIN'])]
    fig.add_trace(go.Scatter(x=wins['Fecha'], y=wins['Precio'], mode='markers', name='WIN', marker=dict(symbol='triangle-down', color='#00FF00', size=14, line=dict(width=2, color='white'))), row=1, col=1)
    loss = dftr[dftr['Tipo'].isin(['SL', 'DYN_LOSS'])]
    fig.add_trace(go.Scatter(x=loss['Fecha'], y=loss['Precio'], mode='markers', name='LOSS', marker=dict(symbol='triangle-down', color='#FF0000', size=14, line=dict(width=2, color='white'))), row=1, col=1)

fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Total_Portfolio'], mode='lines', name='Equidad', line=dict(color='#00FF00', width=3)), row=2, col=1)

y_min_force = df_strat['Low'].min() * 0.98
y_max_force = df_strat['High'].max() * 1.02

fig.update_xaxes(fixedrange=False)
fig.update_yaxes(fixedrange=False, side="right", range=[y_min_force, y_max_force], row=1, col=1)

fig.update_layout(
    template='plotly_dark', 
    height=800, 
    xaxis_rangeslider_visible=False,
    dragmode='pan',
    hovermode='x unified',
    margin=dict(l=10, r=50, t=30, b=10)
)

st.plotly_chart(
    fig, 
    use_container_width=True, 
    key=f"chart_{s_id}", 
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
)
