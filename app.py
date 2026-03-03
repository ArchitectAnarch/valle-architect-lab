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
import os
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

# 🔥 V220: SINCRONÍA TOTAL (ANTI-GAPS Y PRECISION TRADINGVIEW) 🔥
if st.session_state.get('app_version') != 'V220':
    st.session_state.clear()
    st.session_state['app_version'] = 'V220'

# ==========================================
# 🧠 1. FUNCIONES MATEMÁTICAS BASE
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

# ==========================================
# ⚙️ 2. NÚCLEO C++ (EMULACIÓN 1:1 DE TRADINGVIEW ANTI-GAPS)
# ==========================================
@njit(fastmath=True)
def simular_crecimiento_exponencial_ia_core(h_arr, l_arr, c_arr, o_arr, atr_arr, rsi_arr, z_arr, adx_arr, 
    b_c, s_c, w_rsi, w_z, w_adx, th_buy, th_sell, 
    atr_tp_mult, atr_sl_mult, cap_ini, com_pct, invest_pct, slippage_pct, m_mask, v_mask):
    
    cap_act = cap_ini; en_pos = False; p_ent = 0.0
    pos_size = 0.0; invest_amt = 0.0; g_profit = 0.0; g_loss = 0.0; num_trades = 0; max_dd = 0.0; peak = cap_ini
    slip_in = 1.0 + (slippage_pct / 100.0)
    slip_out = 1.0 - (slippage_pct / 100.0)
    tp_p = 0.0; sl_p = 0.0
    wins = 0
    bars_in_trade = 0
    
    for i in range(len(h_arr)):
        just_closed_dyn = False
        
        if en_pos:
            bars_in_trade += 1
            cierra = False
            
            # 🔥 EVALUACIÓN INTRABAR (Activo a partir de la vela 2)
            if bars_in_trade > 1: 
                # 🔥 V220: ANTI-GAP REALITY CHECK (Igual a Pine Script) 🔥
                if l_arr[i] <= sl_p:
                    # Si el mercado abre con Gap por debajo del SL, te comes el precio de apertura.
                    exec_p = sl_p if o_arr[i] > sl_p else o_arr[i]
                    exec_p = exec_p * slip_out
                    
                    ret = (exec_p - p_ent) / p_ent
                    gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
                    cap_act += profit
                    g_loss += abs(profit); num_trades += 1; en_pos = False; cierra = True
                    if cap_act > peak: peak = cap_act
                    if peak > 0: dd = (peak - cap_act) / peak * 100.0; max_dd = max(max_dd, dd)
                
                elif h_arr[i] >= tp_p:
                    # Si el mercado abre con Gap por encima del TP, tomas la ganancia del precio de apertura.
                    exec_p = tp_p if o_arr[i] < tp_p else o_arr[i]
                    exec_p = exec_p * slip_out
                    
                    ret = (exec_p - p_ent) / p_ent
                    gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
                    cap_act += profit
                    g_profit += profit; num_trades += 1; en_pos = False; cierra = True
                    if profit > 0: wins += 1
                    if cap_act > peak: peak = cap_act
                    if peak > 0: dd = (peak - cap_act) / peak * 100.0; max_dd = max(max_dd, dd)

            # 🔥 EVALUACIÓN DE CIERRE DINÁMICO (Pine Script: strategy.close)
            if not cierra:
                score = (rsi_arr[i] * w_rsi) + (z_arr[i] * w_z) + (adx_arr[i] * w_adx)
                if s_c[i] or (score < th_sell):
                    # Se vende a la apertura de la vela de mañana, sufriendo los gaps del mercado
                    exit_price = (o_arr[i+1] if i+1 < len(o_arr) else c_arr[i]) * slip_out 
                    ret = (exit_price - p_ent) / p_ent; gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
                    cap_act += profit
                    if profit > 0: 
                        g_profit += profit; wins += 1
                    else: 
                        g_loss += abs(profit)
                    num_trades += 1; en_pos = False
                    just_closed_dyn = True
            
            if cap_act <= 0: break
            
        # 🔥 EVALUACIÓN DE ENTRADA (Bloqueado si se hizo Dyn_Exit este mismo tick)
        if not en_pos and not just_closed_dyn and i+1 < len(h_arr):
            score = (rsi_arr[i] * w_rsi) + (z_arr[i] * w_z) + (adx_arr[i] * w_adx)
            if (b_c[i] or (score > th_buy)) and m_mask[i] and v_mask[i]:
                if invest_pct > 0: invest_amt = cap_act * (invest_pct / 100.0) 
                else: invest_amt = cap_ini
                if invest_amt > cap_act: invest_amt = cap_act 
                
                comm_in = invest_amt * com_pct; pos_size = invest_amt - comm_in 
                p_ent = o_arr[i+1] * slip_in 
                
                current_atr = atr_arr[i] 
                tp_p = p_ent + (current_atr * atr_tp_mult)
                sl_p = p_ent - (current_atr * atr_sl_mult)
                
                en_pos = True
                bars_in_trade = 0 
                
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    wr = (wins / num_trades) * 100.0 if num_trades > 0 else 0.0
    return (cap_act - cap_ini), pf, num_trades, max_dd, wr

def simular_visual(df_sim, cap_ini, invest_pct, com_pct, slippage_pct=0.0):
    registro_trades = []; n = len(df_sim); curva = np.full(n, cap_ini, dtype=float)
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    atr_arr = df_sim['ATR'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl_arr = df_sim['Active_TP'].values, df_sim['Active_SL'].values
    f_arr = df_sim.index
    en_pos, p_ent, tp_p, sl_p, cap_act, pos_size, invest_amt, total_comms = False, 0.0, 0.0, 0.0, cap_ini, 0.0, 0.0, 0.0
    bars_in_trade = 0
    
    slip_in = 1.0 + (slippage_pct/100.0)
    slip_out = 1.0 - (slippage_pct/100.0)

    for i in range(n):
        cierra = False
        just_closed_dyn = False
        
        if en_pos:
            bars_in_trade += 1
            if bars_in_trade > 1:
                if l_arr[i] <= sl_p:
                    # 🔥 ANTI-GAP VISUAL 🔥
                    exec_p = sl_p if o_arr[i] > sl_p else o_arr[i]
                    exec_p = exec_p * slip_out
                    
                    ret = (exec_p - p_ent) / p_ent; gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                    cap_act += profit
                    if cap_act <= 0: cap_act = 0
                    registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'SL', 'Precio': exec_p, 'Ganancia_$': profit}); en_pos = False; cierra = True
                elif h_arr[i] >= tp_p:
                    exec_p = tp_p if o_arr[i] < tp_p else o_arr[i]
                    exec_p = exec_p * slip_out
                    
                    ret = (exec_p - p_ent) / p_ent; gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                    cap_act += profit
                    registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'TP', 'Precio': exec_p, 'Ganancia_$': profit}); en_pos = False; cierra = True
            
            if not cierra and sell_arr[i]:
                exit_price = (o_arr[i+1] if i+1 < n else c_arr[i]) * slip_out
                ret = (exit_price - p_ent) / p_ent; gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out; net = gross - comm_out; profit = net - invest_amt
                cap_act += profit
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i+1] if i+1 < n else f_arr[i], 'Tipo': 'DYN_WIN' if profit>0 else 'DYN_LOSS', 'Precio': exit_price, 'Ganancia_$': profit}); en_pos = False; just_closed_dyn = True
        
        if not en_pos and not just_closed_dyn and buy_arr[i] and i+1 < n and cap_act > 0:
            if invest_pct > 0: invest_amt = cap_act * (invest_pct / 100.0)
            else: invest_amt = cap_ini
            if invest_amt > cap_act: invest_amt = cap_act
            comm_in = invest_amt * com_pct; total_comms += comm_in; pos_size = invest_amt - comm_in
            
            p_ent = o_arr[i+1] * slip_in
            tp_act = atr_arr[i] * float(df_sim['Active_TP'].values[i])
            sl_act = atr_arr[i] * float(df_sim['Active_SL'].values[i])
            tp_p = p_ent + tp_act
            sl_p = p_ent - sl_act
            
            en_pos = True
            bars_in_trade = 0
            registro_trades.append({'Fecha': f_arr[i+1], 'Tipo': 'ENTRY', 'Precio': p_ent, 'Ganancia_$': 0})
        
        if en_pos and cap_act > 0: curva[i] = cap_act + (pos_size * ((c_arr[i] - p_ent) / p_ent))
        else: curva[i] = cap_act
    return curva.tolist(), 0.0, cap_act, registro_trades, en_pos, total_comms

def simular_monte_carlo(trades_list, cap_ini, num_simulations=1000):
    if not trades_list or len(trades_list) < 5: return None, 0.0
    rets = [t['Ganancia_$'] for t in trades_list if t['Tipo'] in ['TP', 'SL', 'DYN_WIN', 'DYN_LOSS']]
    if not rets: return None, 0.0
    rets_arr = np.array(rets)
    n_trades = len(rets_arr)
    mc_curves = np.zeros((num_simulations, n_trades + 1))
    mc_curves[:, 0] = cap_ini
    ruined_count = 0
    for i in range(num_simulations):
        np.random.shuffle(rets_arr)
        for j in range(n_trades):
            mc_curves[i, j+1] = mc_curves[i, j] + rets_arr[j]
            if mc_curves[i, j+1] <= 0:
                mc_curves[i, j+1:] = 0
                ruined_count += 1
                break
    risk_of_ruin = (ruined_count / num_simulations) * 100.0
    return mc_curves, risk_of_ruin

# ==========================================
# 🧬 3. DICCIONARIOS Y GUARDIANES
# ==========================================
todas_las_armas_b = [
    'Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 
    'Commander_Buy', 'Lev_Buy', 'Q_Pink_Whale_Buy', 'Q_Lock_Bounce', 'Q_Lock_Break', 'Q_Neon_Up', 'Q_Defcon_Buy', 
    'Q_Therm_Bounce', 'Q_Therm_Vacuum', 'Q_Nuclear_Buy', 'Q_Early_Buy', 'Q_Rebound_Buy',
    'Wyc_Spring_Buy', 'VSA_Accum_Buy', 'Fibo_618_Buy', 'MACD_Impulse_Buy', 'Stoch_OS_Buy',
    'PA_Engulfing_Buy', 'PA_Pinbar_Buy', 'PA_3_Soldiers_Buy'
]
todas_las_armas_s = [
    'Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 
    'Commander_Sell', 'Lev_Sell', 'Q_Lock_Reject', 'Q_Lock_Breakd', 'Q_Neon_Dn', 'Q_Defcon_Sell', 'Q_Therm_Wall_Sell', 
    'Q_Therm_Panic_Sell', 'Q_Nuclear_Sell', 'Q_Early_Sell',
    'Wyc_Upthrust_Sell', 'VSA_Dist_Sell', 'Fibo_618_Sell', 'MACD_Exhaust_Sell', 'Stoch_OB_Sell',
    'PA_Engulfing_Sell', 'PA_Pinbar_Sell', 'PA_3_Crows_Sell'
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
    'Stoch_OS_Buy': 'stoch_os_buy', 'Stoch_OB_Sell': 'stoch_ob_sell',
    'PA_Engulfing_Buy': 'pa_engulfing_buy', 'PA_Engulfing_Sell': 'pa_engulfing_sell',
    'PA_Pinbar_Buy': 'pa_pinbar_buy', 'PA_Pinbar_Sell': 'pa_pinbar_sell',
    'PA_3_Soldiers_Buy': 'pa_3_soldiers', 'PA_3_Crows_Sell': 'pa_3_crows'
}

if 'ai_algos' not in st.session_state or len(st.session_state['ai_algos']) == 0: 
    st.session_state['ai_algos'] = [f"AI_GENESIS_{random.randint(100, 999)}"]

estrategias = st.session_state['ai_algos']
tab_id_map = {f"🤖 {ai_id}": ai_id for ai_id in estrategias}

def get_default_dna():
    return {
        'b_team': [], 's_team': [], 
        'b_trigger': random.choice(todas_las_armas_b), 'b_confirm': random.choice(todas_las_armas_b), 'b_op': '&',
        's_trigger': random.choice(todas_las_armas_s), 's_confirm': random.choice(todas_las_armas_s), 's_op': '&',
        'macro': "All-Weather", 'vol': "All-Weather", 'hitbox': 1.5, 'therm_w': 4.0, 'adx_th': 25.0, 
        'whale_f': 2.5, 'ado': 4.0, 'reinv': 20.0, 'fit': -float('inf'), 'net': 0.0, 'winrate': 0.0, 'pf': 0.0, 'nt': 0,
        'w_rsi': 0.0, 'w_z': 0.0, 'w_adx': 0.0, 'th_buy': 99.0, 'th_sell': -99.0, 'atr_tp': 2.0, 'atr_sl': 1.0
    }

def get_safe_vault(s_id):
    vault = st.session_state.get(f'champion_{s_id}')
    if not vault or not isinstance(vault, dict):
        try:
            if os.path.exists(f"champ_{s_id}.json"):
                with open(f"champ_{s_id}.json", "r") as f:
                    data = json.load(f)
                    if data and isinstance(data, dict):
                        vault = data
        except: pass
    if not vault or not isinstance(vault, dict):
        vault = get_default_dna()
    st.session_state[f'champion_{s_id}'] = vault
    return vault

def save_champion(s_id, bp):
    if not bp or not isinstance(bp, dict): return
    vault = get_safe_vault(s_id)
    if bp.get('fit', -float('inf')) <= vault.get('fit', -float('inf')): return
    for k in bp.keys(): vault[k] = bp[k]
    st.session_state[f'champion_{s_id}'] = vault
    try:
        with open(f"champ_{s_id}.json", "w") as f:
            json.dump(vault, f)
    except: pass

for s_id in estrategias:
    if f'opt_status_{s_id}' not in st.session_state: st.session_state[f'opt_status_{s_id}'] = False
    get_safe_vault(s_id)

# ==========================================
# 🌍 4. SIDEBAR E INFRAESTRUCTURA UI
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🧬 GENESIS LAB V220</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True, key="btn_purge"): 
    st.cache_data.clear()
    keys_to_keep = ['app_version', 'ai_algos']
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep: del st.session_state[k]
    gc.collect(); st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 Usa este botón si ves un buen récord. **La IA abortará pero guardará al campeón físicamente.**")
if st.sidebar.button("🛑 ABORTAR RUN GLOBAL (Y MOSTRAR CAMPEÓN)", use_container_width=True, key="btn_abort"):
    st.session_state['abort_opt'] = True
    st.session_state['global_queue'] = []
    st.session_state['run_global'] = False
    st.session_state['deep_opt_state'] = {}
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
start_date, end_date = st.sidebar.slider("📅 Scope Histórico", min_value=hoy - timedelta(days=250 if is_micro else 1500), max_value=hoy, value=(hoy - timedelta(days=200 if is_micro else 1500), hoy), format="YYYY-MM-DD")

capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=1000.0, step=100.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.15, step=0.05) / 100.0 

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center; color: lime;'>🤖 CÁMARA DE MUTACIÓN</h3>", unsafe_allow_html=True)
global_epochs = st.sidebar.slider("Épocas de Evolución (x250)", 1, 1000, 50)
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
    get_safe_vault(new_id)
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
    get_safe_vault(new_id)
    st.session_state['abort_opt'] = False
    st.session_state['deep_opt_state'] = {'s_id': new_id, 'target_epochs': deep_epochs_target, 'current_epoch': 0, 'paused': False, 'start_time': time.time()}
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

def generar_reporte_universal(cap_ini, com_pct):
    res_str = f"📋 **REPORTE GENESIS LAB V220.0**\n\n"
    res_str += f"⏱️ Temporalidad: {intervalo_sel} | 📊 Ticker: {ticker}\n\n"
    for s_id in estrategias:
        v = get_safe_vault(s_id)
        opt_icon = "✅" if st.session_state.get(f'opt_status_{s_id}', False) else "➖"
        res_str += f"🧬 **{s_id}** [{opt_icon}]\nNet Profit: ${v.get('net',0):,.2f} \nWin Rate: {v.get('winrate',0):.1f}%\n---\n"
    return res_str

st.sidebar.markdown("---")
if st.sidebar.button("📊 GENERAR REPORTE", use_container_width=True, key="btn_univ_report"):
    st.sidebar.text_area("Block Note Universal:", value=generar_reporte_universal(capital_inicial, comision_pct), height=200)

# ==========================================
# 🛑 5. EXTRACCIÓN Y WARM-UP INSTITUCIONAL 🛑
# ==========================================
@st.cache_data(ttl=3600, show_spinner="📡 Sincronizando Línea Temporal con TradingView (V220)...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, offset, is_micro):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
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
        
        df['CHOP'] = ta.chop(df['High'], df['Low'], df['Close'], length=14).fillna(50.0)
        
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
        
        df['PA_Engulfing_Buy'] = (df['Vela_Verde']) & (df['Vela_Roja'].shift(1)) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
        df['PA_Engulfing_Sell'] = (df['Vela_Roja']) & (df['Vela_Verde'].shift(1)) & (df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))
        df['PA_Pinbar_Buy'] = (df['lower_wick'] > df['body_size'] * 2.5) & (df['upper_wick'] < df['body_size'])
        df['PA_Pinbar_Sell'] = (df['upper_wick'] > df['body_size'] * 2.5) & (df['lower_wick'] < df['body_size'])
        df['PA_3_Soldiers'] = (df['Vela_Verde']) & (df['Vela_Verde'].shift(1)) & (df['Vela_Verde'].shift(2)) & (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2))
        df['PA_3_Crows'] = (df['Vela_Roja']) & (df['Vela_Roja'].shift(1)) & (df['Vela_Roja'].shift(2)) & (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2))

        df['PL30_L'] = df['Low'].shift(1).rolling(30, min_periods=1).min(); df['PH30_L'] = df['High'].shift(1).rolling(30, min_periods=1).max()
        df['PL100_L'] = df['Low'].shift(1).rolling(100, min_periods=1).min(); df['PH100_L'] = df['High'].shift(1).rolling(100, min_periods=1).max()
        df['PL300_L'] = df['Low'].shift(1).rolling(300, min_periods=1).min(); df['PH300_L'] = df['High'].shift(1).rolling(300, min_periods=1).max()

        df['RSI_Cross_Up'] = (df['RSI'] > df['RSI_MA']) & (df['RSI'].shift(1) <= df['RSI_MA'].shift(1))
        df['RSI_Cross_Dn'] = (df['RSI'] < df['RSI_MA']) & (df['RSI'].shift(1) >= df['RSI_MA'].shift(1))
        df['Macro_Bull'] = df['Close'] >= df['EMA_200']
        df['PP_Slope'] = (2*df['Close'] + df['Close'].shift(1) - df['Close'].shift(3) - 2*df['Close'].shift(4)) / 10.0
        
        target_start = pd.to_datetime(datetime.combine(start, datetime.min.time())) + timedelta(hours=offset)
        df = df[df.index >= target_start]

        split_idx = int(len(df) * 1.0)
        df['Is_Train'] = True

        gc.collect()
        return df, "OK"
    except Exception as e: return pd.DataFrame(), f"❌ ERROR FATAL GENERAL: {str(e)}"

df_global, status_api = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, utc_offset, is_micro)
if df_global.empty: st.error(status_api); st.stop()

dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)
st.sidebar.info(f"📊 Matrix Data: **{len(df_global):,} velas** ({dias_reales} días Evaluados)")

# ==========================================
# 🧠 6. CREACIÓN DE MATRICES NUMPY
# ==========================================
a_c = df_global['Close'].values; a_o = df_global['Open'].values; a_h = df_global['High'].values; a_l = df_global['Low'].values
a_rsi = df_global['RSI'].values; a_rsi_ma = df_global['RSI_MA'].values; a_adx = df_global['ADX'].values
a_macd = df_global['MACD'].values; a_macd_sig = df_global['MACD_Sig'].values
a_stoch_k = df_global['Stoch_K'].values; a_stoch_d = df_global['Stoch_D'].values
a_chop = df_global['CHOP'].values
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

a_pa_eng_b = df_global['PA_Engulfing_Buy'].values; a_pa_eng_s = df_global['PA_Engulfing_Sell'].values
a_pa_pin_b = df_global['PA_Pinbar_Buy'].values; a_pa_pin_s = df_global['PA_Pinbar_Sell'].values
a_pa_3sol_b = df_global['PA_3_Soldiers'].values; a_pa_3cro_s = df_global['PA_3_Crows'].values

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
    whale_buy = flash_vol & a_vv
    whale_sell = flash_vol & a_vr
    whale_memory = whale_buy | npshift_bool(whale_buy, 1) | npshift_bool(whale_buy, 2) | whale_sell | npshift_bool(whale_sell, 1) | npshift_bool(whale_sell, 2)
    is_whale_icon = whale_buy & ~npshift_bool(whale_buy, 1)

    rsi_vel = a_rsi - a_rsi_s1
    pre_pump = ((a_h > a_bbu) | (rsi_vel > 5)) & flash_vol & a_vv; pump_memory = pre_pump | npshift_bool(pre_pump, 1) | npshift_bool(pre_pump, 2)
    pre_dump = ((a_l < a_bbl) | (rsi_vel < -5)) & flash_vol & a_vr; dump_memory = pre_dump | npshift_bool(pre_dump, 1) | npshift_bool(pre_dump, 2)

    retro_peak = (a_rsi < 30) & (a_c < a_bbl); retro_peak_sell = (a_rsi > 70) & (a_c > a_bbu)
    k_break_up = (a_rsi > (a_rsi_bb_b + a_rsi_bb_d)) & (a_rsi_s1 <= npshift(a_rsi_bb_b + a_rsi_bb_d, 1))
    support_buy = is_grav_sup & a_rcu
    support_sell = is_grav_res & a_rcd
    div_bull = (a_l_s1 < a_l_s5) & (a_rsi_s1 > a_rsi_s5) & (a_rsi < 35)
    div_bear = (a_h_s1 > a_h_s5) & (a_rsi_s1 < a_rsi_s5) & (a_rsi > 65)

    buy_score = np.zeros(n_len); base_mask = retro_peak | k_break_up | support_buy | div_bull
    buy_score = np.where(base_mask & retro_peak, 50.0, np.where(base_mask & ~retro_peak, 30.0, buy_score))
    buy_score += np.where(is_grav_sup, 25.0, 0.0); buy_score += np.where(whale_memory, 20.0, 0.0); buy_score += np.where(pump_memory, 15.0, 0.0); buy_score += np.where(div_bull, 15.0, 0.0); buy_score += np.where(k_break_up & ~retro_peak, 15.0, 0.0); buy_score += np.where(a_zscore < -2.0, 15.0, 0.0)
    
    sell_score = np.zeros(n_len); base_mask_s = retro_peak_sell | a_rcd | support_sell | div_bear
    sell_score = np.where(base_mask_s & retro_peak_sell, 50.0, np.where(base_mask_s & ~retro_peak_sell, 30.0, sell_score))
    sell_score += np.where(is_grav_res, 25.0, 0.0); sell_score += np.where(whale_memory, 20.0, 0.0); sell_score += np.where(dump_memory, 15.0, 0.0); sell_score += np.where(div_bear, 15.0, 0.0); sell_score += np.where(a_rcd & ~retro_peak_sell, 15.0, 0.0); sell_score += np.where(a_zscore > 2.0, 15.0, 0.0)

    is_magenta = (buy_score >= 70) | retro_peak
    is_magenta_sell = (sell_score >= 70) | retro_peak_sell
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

    s_dict['PA_Engulfing_Buy'] = a_pa_eng_b; s_dict['PA_Engulfing_Sell'] = a_pa_eng_s
    s_dict['PA_Pinbar_Buy'] = a_pa_pin_b; s_dict['PA_Pinbar_Sell'] = a_pa_pin_s
    s_dict['PA_3_Soldiers_Buy'] = a_pa_3sol_b; s_dict['PA_3_Crows_Sell'] = a_pa_3cro_s

    s_dict['Organic_Vol'] = a_hvol
    s_dict['Organic_Squeeze'] = a_sqz_on
    s_dict['Organic_Safe'] = a_mb & ~a_fk
    s_dict['Organic_Pump'] = pump_memory
    s_dict['Organic_Dump'] = dump_memory
    s_dict['Organic_Gaussian_Clean'] = a_chop < 61.8

    return s_dict

def optimizar_ia_tracker(s_id, cap_ini, com_pct, invest_pct, target_ado, dias_reales, buy_hold_money, epochs=1, cur_net=-float('inf'), cur_fit=-float('inf'), deep_info=None):
    vault = get_safe_vault(s_id)
    best_fit_live = vault.get('fit', -float('inf'))
    best_net_live = vault.get('net', -float('inf'))
    best_pf_live = vault.get('pf', 0.0)
    best_nt_live = vault.get('nt', 0)
    
    best_dna = None
    if best_fit_live != -float('inf'):
        best_dna = vault.copy()

    iters = 250 * epochs
    chunk_size = 250 
    chunks = max(1, iters // chunk_size)
    start_time = time.time()
    n_len = len(a_c)
    split_idx = n_len
    dias_entrenamiento = max(1, dias_reales)

    default_f = np.zeros(n_len, dtype=bool)
    ones_mask = np.ones(n_len, dtype=bool)

    for c in range(chunks):
        if st.session_state.get('abort_opt', False): 
            st.warning("🛑 OPTIMIZACIÓN ABORTADA. Extrayendo el campeón retenido en memoria..."); 
            break

        for _ in range(chunk_size): 
            if best_dna is not None: 
                rand_val = random.random()
                
                if rand_val < 0.50:
                    dna_b_trigger = best_dna.get('b_trigger', random.choice(todas_las_armas_b))
                    dna_b_confirm = best_dna.get('b_confirm', random.choice(todas_las_armas_b))
                    dna_b_op = best_dna.get('b_op', '&')
                    dna_s_trigger = best_dna.get('s_trigger', random.choice(todas_las_armas_s))
                    dna_s_confirm = best_dna.get('s_confirm', random.choice(todas_las_armas_s))
                    dna_s_op = best_dna.get('s_op', '&')
                    
                    if random.random() < 0.15: dna_b_trigger = random.choice(todas_las_armas_b)
                    if random.random() < 0.15: dna_s_trigger = random.choice(todas_las_armas_s)
                    if random.random() < 0.05: dna_b_op = random.choice(['&', '|'])
                    
                    dna_macro = best_dna.get('macro', 'All-Weather')
                    dna_vol = best_dna.get('vol', 'All-Weather')
                    r_hitbox = best_dna.get('hitbox', 1.5)
                    r_therm = best_dna.get('therm_w', 4.0)
                    r_adx = best_dna.get('adx_th', 25.0)
                    r_whale = best_dna.get('whale_f', 2.5)

                    if random.random() < 0.1:
                        target_env = random.choice(['hb', 'th', 'adx', 'wh'])
                        if target_env == 'hb': r_hitbox = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
                        elif target_env == 'th': r_therm = random.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                        elif target_env == 'adx': r_adx = random.choice([15.0, 20.0, 25.0, 30.0, 35.0])

                    r_w_rsi = round(best_dna.get('w_rsi', 0.0) + random.gauss(0, 0.1), 4)
                    r_w_z = round(best_dna.get('w_z', 0.0) + random.gauss(0, 0.5), 4) 
                    r_w_adx = round(best_dna.get('w_adx', 0.0) + random.gauss(0, 0.1), 4)
                    r_th_b = round(best_dna.get('th_buy', 50.0) + random.gauss(0, 2.0), 2)
                    r_th_s = round(best_dna.get('th_sell', -50.0) + random.gauss(0, 2.0), 2)
                    r_atr_tp = max(0.1, round(best_dna.get('atr_tp', 2.0) + random.gauss(0, 0.2), 2))
                    r_atr_sl = max(0.1, round(best_dna.get('atr_sl', 1.0) + random.gauss(0, 0.2), 2))
                    
                    r_w_z = max(-10.0, min(10.0, r_w_z))
                    r_th_b = max(0.0, min(100.0, r_th_b))
                    r_th_s = max(-100.0, min(0.0, r_th_s))
                
                elif rand_val < 0.85:
                    dna_b_trigger = random.choice(todas_las_armas_b) if random.random() < 0.5 else best_dna.get('b_trigger')
                    dna_b_confirm = random.choice(todas_las_armas_b) if random.random() < 0.5 else best_dna.get('b_confirm')
                    dna_b_op = random.choice(['&', '|']) if random.random() < 0.5 else best_dna.get('b_op')
                    dna_s_trigger = random.choice(todas_las_armas_s) if random.random() < 0.5 else best_dna.get('s_trigger')
                    dna_s_confirm = random.choice(todas_las_armas_s) if random.random() < 0.5 else best_dna.get('s_confirm')
                    dna_s_op = random.choice(['&', '|']) if random.random() < 0.5 else best_dna.get('s_op')
                    dna_macro = random.choice(["All-Weather", "Bull Only", "Bear Only", "Organic_Vol", "Organic_Squeeze", "Organic_Safe"]) if random.random() < 0.5 else best_dna.get('macro')
                    dna_vol = random.choice(["All-Weather", "Trend", "Range", "Organic_Pump", "Organic_Dump"]) if random.random() < 0.5 else best_dna.get('vol')
                    
                    r_hitbox = best_dna.get('hitbox', 1.5)
                    r_therm = best_dna.get('therm_w', 4.0)
                    r_adx = best_dna.get('adx_th', 25.0)
                    r_whale = best_dna.get('whale_f', 2.5)

                    r_w_rsi = round(best_dna.get('w_rsi', 0.0), 4)
                    r_w_z = round(best_dna.get('w_z', 0.0), 4)
                    r_w_adx = round(best_dna.get('w_adx', 0.0), 4)
                    r_th_b = round(best_dna.get('th_buy', 50.0), 2)
                    r_th_s = round(best_dna.get('th_sell', -50.0), 2)
                    r_atr_tp = round(best_dna.get('atr_tp', 2.0), 2)
                    r_atr_sl = round(best_dna.get('atr_sl', 1.0), 2)

                else:
                    dna_b_trigger = random.choice(todas_las_armas_b)
                    dna_b_confirm = random.choice(todas_las_armas_b)
                    dna_b_op = random.choice(['&', '|'])
                    dna_s_trigger = random.choice(todas_las_armas_s)
                    dna_s_confirm = random.choice(todas_las_armas_s)
                    dna_s_op = random.choice(['&', '|'])
                    dna_macro = random.choice(["All-Weather", "Bull Only", "Bear Only", "Ignore", "Organic_Vol", "Organic_Squeeze", "Organic_Safe", "Organic_Gaussian_Clean"])
                    dna_vol = random.choice(["All-Weather", "Trend", "Range", "Ignore", "Organic_Pump", "Organic_Dump", "Organic_Gaussian_Clean"])
                    
                    r_hitbox = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
                    r_therm = random.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                    r_adx = random.choice([15.0, 20.0, 25.0, 30.0, 35.0])
                    r_whale = random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
                    
                    r_w_rsi = round(random.uniform(-2.0, 2.0), 4)
                    r_w_z = round(random.uniform(-10.0, 10.0), 4)
                    r_w_adx = round(random.uniform(-2.0, 2.0), 4)
                    r_th_b = round(random.uniform(0.0, 100.0), 2)
                    r_th_s = round(random.uniform(-100.0, 0.0), 2)
                    r_atr_tp = round(random.uniform(0.5, 15.0), 2)
                    r_atr_sl = round(random.uniform(1.0, 10.0), 2)
            else: 
                r_hitbox = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]); r_therm = random.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                r_adx = random.choice([15.0, 20.0, 25.0, 30.0, 35.0]); r_whale = random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
                dna_b_trigger = random.choice(todas_las_armas_b); dna_b_confirm = random.choice(todas_las_armas_b); dna_b_op = random.choice(['&', '|'])
                dna_s_trigger = random.choice(todas_las_armas_s); dna_s_confirm = random.choice(todas_las_armas_s); dna_s_op = random.choice(['&', '|'])
                dna_macro = random.choice(["All-Weather", "Bull Only", "Bear Only", "Ignore", "Organic_Vol", "Organic_Squeeze", "Organic_Safe", "Organic_Gaussian_Clean"])
                dna_vol = random.choice(["All-Weather", "Trend", "Range", "Ignore", "Organic_Pump", "Organic_Dump", "Organic_Gaussian_Clean"])
                r_w_rsi = round(random.uniform(-2.0, 2.0), 4)
                r_w_z = round(random.uniform(-10.0, 10.0), 4)
                r_w_adx = round(random.uniform(-2.0, 2.0), 4)
                r_th_b = round(random.uniform(0.0, 100.0), 2)
                r_th_s = round(random.uniform(-100.0, 0.0), 2)
                r_atr_tp = round(random.uniform(0.5, 15.0), 2)
                r_atr_sl = round(random.uniform(1.0, 10.0), 2)

            s_dict = calcular_señales_numpy(r_hitbox, r_therm, r_adx, r_whale)

            m_mask_dict = {
                "Bull Only": a_mb, "Bear Only": ~a_mb, "Organic_Vol": s_dict['Organic_Vol'],
                "Organic_Squeeze": s_dict['Organic_Squeeze'], "Organic_Safe": s_dict['Organic_Safe'],
                "Organic_Gaussian_Clean": s_dict['Organic_Gaussian_Clean'],
                "All-Weather": ones_mask, "Ignore": ones_mask
            }
            v_mask_dict = {
                "Trend": (a_adx >= r_adx), "Range": (a_adx < r_adx), "Organic_Pump": s_dict['Organic_Pump'],
                "Organic_Dump": s_dict['Organic_Dump'], "Organic_Gaussian_Clean": s_dict['Organic_Gaussian_Clean'],
                "All-Weather": ones_mask, "Ignore": ones_mask
            }
            
            m_mask = m_mask_dict.get(dna_macro, ones_mask)
            v_mask = v_mask_dict.get(dna_vol, ones_mask)
            
            if dna_b_op == '&': f_buy_tactical = s_dict.get(dna_b_trigger, default_f) & s_dict.get(dna_b_confirm, default_f)
            else: f_buy_tactical = s_dict.get(dna_b_trigger, default_f) | s_dict.get(dna_b_confirm, default_f)
                
            if dna_s_op == '&': f_sell_tactical = s_dict.get(dna_s_trigger, default_f) & s_dict.get(dna_s_confirm, default_f)
            else: f_sell_tactical = s_dict.get(dna_s_trigger, default_f) | s_dict.get(dna_s_confirm, default_f)
            
            net, pf, nt, mdd, wr = simular_crecimiento_exponencial_ia_core(
                a_h[:split_idx], a_l[:split_idx], a_c[:split_idx], a_o[:split_idx], a_atr[:split_idx], 
                a_rsi[:split_idx], a_zscore[:split_idx], a_adx[:split_idx],
                f_buy_tactical[:split_idx], f_sell_tactical[:split_idx], 
                r_w_rsi, r_w_z, r_w_adx, r_th_b, r_th_s,
                r_atr_tp, r_atr_sl, float(cap_ini), float(com_pct), float(invest_pct), 0.0,
                m_mask[:split_idx], v_mask[:split_idx]
            )

            fit_score = -float('inf') 
            if nt >= 3: 
                ado_actual = nt / max(1, dias_entrenamiento)
                ado_target_safe = max(0.1, target_ado)
                
                if net > 0:
                    safe_pf = min(pf, 3.0) 
                    dd_penalty = 1.0
                    if mdd > 25.0: dd_penalty = (25.0 / mdd)
                    if wr < 40.0: dd_penalty *= (wr / 40.0)
                    
                    ado_factor = 1.0
                    if ado_actual < (ado_target_safe * 0.3): 
                        ado_factor = max(0.2, ado_actual / (ado_target_safe * 0.3))
                        
                    fit_score = net * safe_pf * dd_penalty * ado_factor
                else:
                    ado_diff = abs(ado_actual - ado_target_safe) * 5
                    fit_score = net - mdd - ado_diff
            else:
                fit_score = net - 1000.0 

            if fit_score > best_fit_live:
                best_fit_live = fit_score; best_net_live = net; best_pf_live = pf; best_nt_live = nt
                bp = {
                    'b_trigger': dna_b_trigger, 'b_confirm': dna_b_confirm, 'b_op': dna_b_op, 
                    's_trigger': dna_s_trigger, 's_confirm': dna_s_confirm, 's_op': dna_s_op, 
                    'macro': dna_macro, 'vol': dna_vol, 'hitbox': r_hitbox, 'therm_w': r_therm, 
                    'adx_th': r_adx, 'whale_f': r_whale, 'fit': fit_score, 'net': net, 'winrate': wr, 
                    'pf': pf, 'nt': nt,
                    'reinv': invest_pct, 'ado': ado_actual if nt >= 3 else 0.0, 'w_rsi': r_w_rsi, 'w_z': r_w_z, 
                    'w_adx': r_w_adx, 'th_buy': r_th_b, 'th_sell': r_th_s, 'atr_tp': r_atr_tp, 'atr_sl': r_atr_sl
                }
                best_dna = bp.copy()
                save_champion(s_id, bp)
                st.session_state[f'opt_status_{s_id}'] = True
            
        global_start = deep_info.get('start_time', start_time) if deep_info else start_time
        total_elapsed_sec = time.time() - global_start
        h, rem = divmod(total_elapsed_sec, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{int(h):02d}h:{int(m):02d}m:{int(s):02d}s"

        if deep_info:
            current_epoch_val = deep_info['current'] + (c+1)*(chunk_size)
            macro_pct = int((current_epoch_val / deep_info['total']) * 100)
            title = f"🌌 DEEP FORGE: {s_id}"
            subtitle = f"Épocas: {current_epoch_val:,} / {deep_info['total']:,} ({macro_pct}%)<br>⏱️ Tiempo: {time_str}"
            color = "#9932CC"
        else:
            pct_done = int(((c + 1) / chunks) * 100)
            combos = (c + 1) * chunk_size
            title = f"GENESIS LAB V220: {s_id}"
            subtitle = f"Progreso: {pct_done}% | ADN Probados: {combos:,}<br>⏱️ Tiempo Ejecución: {time_str}"
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
            <div style="color: #00FF00; font-weight: bold; font-size: 1.5rem; margin-top: 15px;">🏆 Récord Global (Neto Real): ${best_net_live:.2f}</div>
            <div style="color: cyan; font-size: 1.0rem;">Trades: {best_nt_live} | Win Rate: {best_pf_live:.2f}x PF | Score IA: {best_fit_live:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
            
    return best_dna

def run_backtest_eval(s_id, cap_ini, com_pct):
    vault = get_safe_vault(s_id)
    s_dict = calcular_señales_numpy(vault.get('hitbox',1.5), vault.get('therm_w',4.0), vault.get('adx_th',25.0), vault.get('whale_f',2.5))
    
    n_len = len(a_c)
    
    w_rsi = round(float(vault.get('w_rsi', 0.0)), 4)
    w_z = round(float(vault.get('w_z', 0.0)), 4)
    w_adx = round(float(vault.get('w_adx', 0.0)), 4)
    th_buy = round(float(vault.get('th_buy', 999.0)), 2)
    th_sell = round(float(vault.get('th_sell', -999.0)), 2)
    atr_tp = round(float(vault.get('atr_tp', 0.0)), 2)
    atr_sl = round(float(vault.get('atr_sl', 0.0)), 2)
    
    f_tp = np.full(n_len, atr_tp)
    f_sl = np.full(n_len, atr_sl)
    f_buy = np.zeros(n_len, dtype=bool); f_sell = np.zeros(n_len, dtype=bool)
    default_f = np.zeros(n_len, dtype=bool)
    ones_mask = np.ones(n_len, dtype=bool)

    if vault.get('macro') == "Bull Only": m_mask = a_mb
    elif vault.get('macro') == "Bear Only": m_mask = ~a_mb
    elif vault.get('macro') == "Organic_Vol": m_mask = s_dict['Organic_Vol']
    elif vault.get('macro') == "Organic_Squeeze": m_mask = s_dict['Organic_Squeeze']
    elif vault.get('macro') == "Organic_Safe": m_mask = s_dict['Organic_Safe']
    elif vault.get('macro') == "Organic_Gaussian_Clean": m_mask = s_dict['Organic_Gaussian_Clean']
    else: m_mask = ones_mask

    if vault.get('vol') == "Trend": v_mask = (a_adx >= vault.get('adx_th', 25.0))
    elif vault.get('vol') == "Range": v_mask = (a_adx < vault.get('adx_th', 25.0))
    elif vault.get('vol') == "Organic_Pump": v_mask = s_dict['Organic_Pump']
    elif vault.get('vol') == "Organic_Dump": v_mask = s_dict['Organic_Dump']
    elif vault.get('vol') == "Organic_Gaussian_Clean": v_mask = s_dict['Organic_Gaussian_Clean']
    else: v_mask = ones_mask

    t_b, op_b, c_b = vault.get('b_trigger', ''), vault.get('b_op', '&'), vault.get('b_confirm', '')
    if op_b == '&': f_buy_tactical = s_dict.get(t_b, default_f) & s_dict.get(c_b, default_f) if t_b and c_b else default_f
    else: f_buy_tactical = s_dict.get(t_b, default_f) | s_dict.get(c_b, default_f) if t_b and c_b else default_f
    
    t_s, op_s, c_s = vault.get('s_trigger', ''), vault.get('s_op', '&'), vault.get('s_confirm', '')
    if op_s == '&': f_sell_tactical = s_dict.get(t_s, default_f) & s_dict.get(c_s, default_f) if t_s and c_s else default_f
    else: f_sell_tactical = s_dict.get(t_s, default_f) | s_dict.get(c_s, default_f) if t_s and c_s else default_f
    
    score_arr = (a_rsi * w_rsi) + (a_zscore * w_z) + (a_adx * w_adx)
    
    f_buy = f_buy_tactical | (score_arr > th_buy)
    f_buy &= (m_mask & v_mask)
    f_sell = f_sell_tactical | (score_arr < th_sell)

    df_strat = df_global.copy()
    df_strat['Signal_Buy'], df_strat['Signal_Sell'] = f_buy, f_sell
    df_strat['Active_TP'], df_strat['Active_SL'] = f_tp, f_sl
    
    eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, cap_ini, float(vault.get('reinv', 20.0)), com_pct, 0.0)
    return df_strat, eq_curve, t_log, total_comms

def generar_pine_script(s_id, vault, sym, tf, buy_pct, sell_pct, com_pct, start_date_obj):
    v_hb = vault.get('hitbox', 1.5); v_tw = vault.get('therm_w', 4.0)
    v_adx = vault.get('adx_th', 25.0); v_wf = vault.get('whale_f', 2.5)
    
    json_buy = f'{{"passphrase": "ASTRONAUTA", "action": "{{{{strategy.order.action}}}}", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {buy_pct}, "limit_price": {{{{close}}}}, "side": "🟢 COMPRA"}}'
    json_sell = f'{{"passphrase": "ASTRONAUTA", "action": "{{{{strategy.order.action}}}}", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {sell_pct}, "limit_price": {{{{close}}}}, "side": "🔴 VENTA"}}'

    ps_base = f"""//@version=5
strategy("{s_id} MATRIX - {sym} [{tf}]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value={buy_pct}, commission_value={com_pct*100}, slippage=0)
wt_enter_long = input.text_area(defval='{json_buy}', title="🟢 WT: Mensaje Enter Long")
wt_exit_long  = input.text_area(defval='{json_sell}', title="🔴 WT: Mensaje Exit Long")

grp_time = "📅 FILTRO DE FECHA"
start_year = input.int({start_date_obj.year}, "Año de Inicio", group=grp_time)
start_month = input.int({start_date_obj.month}, "Mes de Inicio", group=grp_time)
start_day = input.int({start_date_obj.day}, "Día de Inicio", group=grp_time)
window = time >= timestamp(syminfo.timezone, start_year, start_month, start_day, 0, 0)

hitbox_pct   = {v_hb}
therm_wall   = {v_tw}
adx_trend    =
