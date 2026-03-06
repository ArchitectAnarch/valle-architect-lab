import streamlit as st
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# 🔥 V260: THE QUANTUM SNIPER (VOTING SYSTEM + EXTREME SPEED + TRUE GREED) 🔥
APP_VERSION = 'V260'
if st.session_state.get('app_version') != APP_VERSION:
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state['app_version'] = APP_VERSION
    st.rerun()

# ==========================================
# 💾 RECUPERACIÓN DE MEMORIA (HALL OF FAME NATIVO)
# ==========================================
if 'ai_algos' not in st.session_state or len(st.session_state['ai_algos']) == 0: 
    loaded_algos = []
    for file in os.listdir():
        if file.startswith("champ_") and file.endswith(".json"):
            loaded_algos.append(file.replace("champ_", "").replace(".json", ""))
    
    if not loaded_algos:
        loaded_algos = [f"AI_GENESIS_{random.randint(100, 999)}"]
    
    st.session_state['ai_algos'] = list(dict.fromkeys(loaded_algos))

estrategias = st.session_state['ai_algos']
tab_id_map = {f"🤖 {ai_id}": ai_id for ai_id in estrategias}

def get_default_dna():
    return {
        'b_team': random.sample(['Ping_Buy', 'Thermal_Buy'], 1), 's_team': random.sample(['Ping_Sell', 'Thermal_Sell'], 1), 
        'b_op': '&', 's_op': '&', 'macro': "All-Weather", 'vol': "All-Weather", 'hitbox': 1.5, 'therm_w': 4.0, 
        'adx_th': 25.0, 'whale_f': 2.5, 'ado': 4.0, 'reinv': 20.0, 'fit': -float('inf'), 
        'net': 0.0, 'net_is': 0.0, 'net_oos': 0.0, 'winrate': 0.0, 'pf': 0.0, 'nt': 0, 'w_rsi': 0.0, 'w_z': 0.0, 'w_adx': 0.0, 
        'th_buy': 99.0, 'th_sell': -99.0, 'atr_tp': 2.0, 'atr_sl': 1.0
    }

def get_safe_vault(s_id):
    vault = st.session_state.get(f'champion_{s_id}')
    if not vault or not isinstance(vault, dict):
        try:
            if os.path.exists(f"champ_{s_id}.json"):
                with open(f"champ_{s_id}.json", "r") as f:
                    data = json.load(f)
                    if data and isinstance(data, dict): vault = data
        except: pass
    if not vault or not isinstance(vault, dict): vault = get_default_dna()
    st.session_state[f'champion_{s_id}'] = vault
    return vault

def save_champion(s_id, bp):
    if not bp or not isinstance(bp, dict): return
    vault = get_safe_vault(s_id)
    if bp.get('fit', -float('inf')) <= vault.get('fit', -float('inf')): return
    for k in bp.keys(): vault[k] = bp[k]
    st.session_state[f'champion_{s_id}'] = vault
    try:
        with open(f"champ_{s_id}.json", "w") as f: json.dump(vault, f)
    except: pass

for s_id in estrategias:
    v = get_safe_vault(s_id)
    if f'opt_status_{s_id}' not in st.session_state: 
        st.session_state[f'opt_status_{s_id}'] = (v.get('fit', -float('inf')) != -float('inf'))

# ==========================================
# 🧠 1. FUNCIONES MATEMÁTICAS C-SPEED
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
def rma_pine(arr, length):
    alpha = 1.0 / length; out = np.full_like(arr, np.nan)
    sum_val = 0.0; count = 0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            if count < length:
                sum_val += arr[i]; count += 1
                if count == length: out[i] = sum_val / length
            else: out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
    return out

# ==========================================
# ⚙️ 2. NÚCLEO C++ (MÁQUINA DE ESTADOS ULTRA RÁPIDA)
# ==========================================
@njit(fastmath=True)
def simular_core_rapido(h_arr, l_arr, c_arr, o_arr, atr_arr, 
    f_buy, f_sell, atr_tp_mult, atr_sl_mult, cap_ini, com_pct, invest_pct, slippage_pct, is_calib):
    
    cap_act = cap_ini; en_pos = False; pending_dyn_exit = False
    p_ent = 0.0; pos_size = 0.0; invest_amt = 0.0; g_profit = 0.0; g_loss = 0.0
    num_trades = 0; max_dd = 0.0; peak = cap_ini
    slip_in = 1.0 + (slippage_pct / 100.0); slip_out = 1.0 - (slippage_pct / 100.0)
    tp_p = 0.0; sl_p = 0.0; wins = 0; bars_in_trade = 0
    
    for i in range(len(h_arr)):
        cierra = False
        
        if pending_dyn_exit and en_pos:
            exit_price = o_arr[i] * slip_out
            ret = (exit_price - p_ent) / p_ent
            gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
            cap_act += profit
            if profit > 0: wins += 1; g_profit += profit
            else: g_loss += abs(profit)
            num_trades += 1; en_pos = False; cierra = True
            if cap_act > peak: peak = cap_act
            if peak > 0: max_dd = max(max_dd, (peak - cap_act) / peak * 100.0)
            
        pending_dyn_exit = False 
        
        if en_pos and not cierra:
            bars_in_trade += 1
            if bars_in_trade >= 1: 
                hit_sl = l_arr[i] <= sl_p
                hit_tp = h_arr[i] >= tp_p
                if hit_sl and hit_tp:
                    if c_arr[i] >= o_arr[i]:
                        exec_p = sl_p if o_arr[i] > sl_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent
                    else:
                        exec_p = tp_p if o_arr[i] < tp_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent
                elif hit_sl:
                    exec_p = sl_p if o_arr[i] > sl_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent
                elif hit_tp:
                    exec_p = tp_p if o_arr[i] < tp_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent
                
                if hit_sl or hit_tp:
                    exec_p = exec_p * slip_out
                    gross = pos_size * (1.0 + ret); net = gross - (gross * com_pct); profit = net - invest_amt
                    cap_act += profit
                    if profit > 0: wins += 1; g_profit += profit
                    else: g_loss += abs(profit)
                    num_trades += 1; en_pos = False; cierra = True
                    if cap_act > peak: peak = cap_act
                    if peak > 0: max_dd = max(max_dd, (peak - cap_act) / peak * 100.0)

        if en_pos and not cierra and not is_calib:
            if f_sell[i]: pending_dyn_exit = True
                
        if cap_act <= 0: break
        
        if not en_pos and not pending_dyn_exit and i+1 < len(h_arr):
            if f_buy[i]:
                invest_amt = cap_act * (invest_pct / 100.0) if invest_pct > 0 else cap_ini
                if invest_amt > cap_act: invest_amt = cap_act 
                comm_in = invest_amt * com_pct; pos_size = invest_amt - comm_in 
                p_ent = o_arr[i+1] * slip_in 
                
                if is_calib:
                    tp_p = round(p_ent * 1.002, 5); sl_p = round(p_ent * 0.998, 5)
                else:
                    tp_p = round(p_ent + (atr_arr[i] * atr_tp_mult), 5)
                    sl_p = round(p_ent - (atr_arr[i] * atr_sl_mult), 5)
                en_pos = True; bars_in_trade = 0
                
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    wr = (wins / num_trades) * 100.0 if num_trades > 0 else 0.0
    return (cap_act - cap_ini), pf, num_trades, max_dd, wr

# ==========================================
# 📊 SIMULADOR VISUAL
# ==========================================
def simular_visual(df_sim, cap_ini, invest_pct, com_pct, slippage_pct=0.0, is_calib=False):
    registro_trades = []; n = len(df_sim); curva = np.full(n, cap_ini, dtype=float)
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    atr_arr, buy_arr, sell_arr = df_sim['ATR'].values, df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl_arr, f_arr = df_sim['Active_TP'].values, df_sim['Active_SL'].values, df_sim.index
    
    en_pos = False; pending_dyn_exit = False
    p_ent = 0.0; tp_p = 0.0; sl_p = 0.0; cap_act = cap_ini
    pos_size = 0.0; invest_amt = 0.0; total_comms = 0.0; bars_in_trade = 0
    slip_in = 1.0 + (slippage_pct/100.0); slip_out = 1.0 - (slippage_pct/100.0)

    for i in range(n):
        cierra = False
        if pending_dyn_exit and en_pos:
            exit_price = o_arr[i] * slip_out; ret = (exit_price - p_ent) / p_ent
            gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out
            profit = gross - comm_out - invest_amt; cap_act += profit
            if cap_act <= 0: cap_act = 0
            registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'DYN_WIN' if profit>0 else 'DYN_LOSS', 'Precio': exit_price, 'Ganancia_$': profit})
            en_pos = False; cierra = True
            
        pending_dyn_exit = False
        
        if en_pos and not cierra:
            bars_in_trade += 1
            if bars_in_trade >= 1:
                hit_sl = l_arr[i] <= sl_p; hit_tp = h_arr[i] >= tp_p
                if hit_sl and hit_tp:
                    if c_arr[i] >= o_arr[i]:
                        exec_p = sl_p if o_arr[i] > sl_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent; p_type = 'SL'
                    else:
                        exec_p = tp_p if o_arr[i] < tp_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent; p_type = 'TP'
                elif hit_sl:
                    exec_p = sl_p if o_arr[i] > sl_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent; p_type = 'SL'
                elif hit_tp:
                    exec_p = tp_p if o_arr[i] < tp_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent; p_type = 'TP'
                    
                if hit_sl or hit_tp:
                    exec_p = exec_p * slip_out
                    gross = pos_size * (1 + ret); comm_out = gross * com_pct; total_comms += comm_out
                    profit = gross - comm_out - invest_amt; cap_act += profit
                    if cap_act <= 0: cap_act = 0
                    final_type = 'TP' if profit > 0 else 'SL'
                    registro_trades.append({'Fecha': f_arr[i], 'Tipo': final_type, 'Precio': exec_p, 'Ganancia_$': profit})
                    en_pos = False; cierra = True
            
        if en_pos and not cierra and not is_calib:
            if sell_arr[i]: pending_dyn_exit = True
        
        if not en_pos and not pending_dyn_exit and i+1 < n and cap_act > 0:
            if buy_arr[i]:
                invest_amt = cap_act * (invest_pct / 100.0) if invest_pct > 0 else cap_ini
                if invest_amt > cap_act: invest_amt = cap_act
                comm_in = invest_amt * com_pct; total_comms += comm_in; pos_size = invest_amt - comm_in
                
                p_ent = o_arr[i+1] * slip_in
                
                if is_calib:
                    tp_p = np.round(p_ent * 1.002, 5); sl_p = np.round(p_ent * 0.998, 5)
                else:
                    tp_p = np.round(p_ent + (atr_arr[i] * float(tp_arr[i])), 5); sl_p = np.round(p_ent - (atr_arr[i] * float(sl_arr[i])), 5)
                
                en_pos = True; bars_in_trade = 0
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
                mc_curves[i, j+1:] = 0; ruined_count += 1
                break
    risk_of_ruin = (ruined_count / num_simulations) * 100.0
    return mc_curves, risk_of_ruin

def generar_radar(wr, pf, ado, ret_pct, alpha_pct, target_ado):
    fig = go.Figure()
    norm_wr = min(wr * 2, 100) 
    norm_pf = min(pf * 25, 100) 
    norm_ado = min((ado / max(0.1, target_ado)) * 100, 100) 
    norm_ret = min(max(ret_pct / 30, 0), 100) 
    norm_alpha = min(max(alpha_pct / 30, 0), 100)
    
    real_texts = [f"{wr:.1f}% WR", f"{pf:.2f}x PF", f"{ado:.2f} ADO", f"{ret_pct:.1f}% Neto", f"{alpha_pct:.1f}% Alpha", f"{wr:.1f}% WR"]
    
    fig.add_trace(go.Scatterpolar(
        r=[norm_wr, norm_pf, norm_ado, norm_ret, norm_alpha, norm_wr],
        theta=['Win Rate', 'Profit Factor', 'Trades/Día (ADO)', 'Rentabilidad', 'Alpha (vs Hold)', 'Win Rate'],
        mode='lines+markers+text', text=real_texts, textposition="top center", textfont=dict(color='white', size=11, family="Arial Black"),
        fill='toself', name='Perfil de Depredador IA', line_color='#00FFFF', fillcolor='rgba(0, 255, 255, 0.2)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=False, range=[0, 100], color='gray', gridcolor='rgba(255, 255, 255, 0.1)')), showlegend=False, template='plotly_dark', height=350, margin=dict(l=40, r=40, t=30, b=30), title=dict(text="🧬 Escáner de Poder de la IA", x=0.5, font=dict(color="cyan")))
    return fig

# ==========================================
# 🧬 3. DICCIONARIOS Y GUARDIANES
# ==========================================
todas_las_armas_b = ['Ping_Buy', 'Climax_Buy', 'Thermal_Buy', 'Lock_Buy', 'Squeeze_Buy', 'Defcon_Buy', 'Jugg_Buy', 'Trinity_Buy', 'Commander_Buy', 'Lev_Buy', 'Q_Pink_Whale_Buy', 'Q_Lock_Bounce', 'Q_Lock_Break', 'Q_Neon_Up', 'Q_Defcon_Buy', 'Q_Therm_Bounce', 'Q_Therm_Vacuum', 'Q_Nuclear_Buy', 'Q_Early_Buy', 'Q_Rebound_Buy', 'Wyc_Spring_Buy', 'VSA_Accum_Buy', 'Fibo_618_Buy', 'MACD_Impulse_Buy', 'Stoch_OS_Buy', 'PA_Engulfing_Buy', 'PA_Pinbar_Buy', 'PA_3_Soldiers_Buy']
todas_las_armas_s = ['Ping_Sell', 'Climax_Sell', 'Thermal_Sell', 'Lock_Sell', 'Squeeze_Sell', 'Defcon_Sell', 'Jugg_Sell', 'Trinity_Sell', 'Commander_Sell', 'Lev_Sell', 'Q_Lock_Reject', 'Q_Lock_Breakd', 'Q_Neon_Dn', 'Q_Defcon_Sell', 'Q_Therm_Wall_Sell', 'Q_Therm_Panic_Sell', 'Q_Nuclear_Sell', 'Q_Early_Sell', 'Wyc_Upthrust_Sell', 'VSA_Dist_Sell', 'Fibo_618_Sell', 'MACD_Exhaust_Sell', 'Stoch_OB_Sell', 'PA_Engulfing_Sell', 'PA_Pinbar_Sell', 'PA_3_Crows_Sell']

pine_map = {'Ping_Buy': 'ping_b', 'Ping_Sell': 'ping_s', 'Squeeze_Buy': 'squeeze_b', 'Squeeze_Sell': 'squeeze_s', 'Thermal_Buy': 'therm_b', 'Thermal_Sell': 'therm_s', 'Climax_Buy': 'climax_b', 'Climax_Sell': 'climax_s', 'Lock_Buy': 'lock_b', 'Lock_Sell': 'lock_s', 'Defcon_Buy': 'defcon_b', 'Defcon_Sell': 'defcon_s', 'Jugg_Buy': 'jugg_b', 'Jugg_Sell': 'jugg_s', 'Trinity_Buy': 'trinity_b', 'Trinity_Sell': 'trinity_s', 'Lev_Buy': 'lev_b', 'Lev_Sell': 'lev_s', 'Commander_Buy': 'commander_b', 'Commander_Sell': 'commander_s', 'Q_Pink_Whale_Buy': 'r_Pink_Whale_Buy', 'Q_Lock_Bounce': 'r_Lock_Bounce', 'Q_Lock_Break': 'r_Lock_Break', 'Q_Neon_Up': 'r_Neon_Up', 'Q_Defcon_Buy': 'r_Defcon_Buy', 'Q_Therm_Bounce': 'r_Therm_Bounce', 'Q_Therm_Vacuum': 'r_Therm_Vacuum', 'Q_Nuclear_Buy': 'r_Nuclear_Buy', 'Q_Early_Buy': 'r_Early_Buy', 'Q_Rebound_Buy': 'r_Rebound_Buy', 'Q_Lock_Reject': 'r_Lock_Reject', 'Q_Lock_Breakd': 'r_Lock_Breakd', 'Q_Neon_Dn': 'r_Neon_Dn', 'Q_Defcon_Sell': 'r_Defcon_Sell', 'Q_Therm_Wall_Sell': 'r_Therm_Wall_Sell', 'Q_Therm_Panic_Sell': 'r_Therm_Panic_Sell', 'Q_Nuclear_Sell': 'r_Nuclear_Sell', 'Q_Early_Sell': 'r_Early_Sell', 'Wyc_Spring_Buy': 'wyc_spring_buy', 'Wyc_Upthrust_Sell': 'wyc_upthrust_sell', 'VSA_Accum_Buy': 'vsa_accum_buy', 'VSA_Dist_Sell': 'vsa_dist_sell', 'Fibo_618_Buy': 'fibo_618_buy', 'Fibo_618_Sell': 'fibo_618_sell', 'MACD_Impulse_Buy': 'macd_impulse_buy', 'MACD_Exhaust_Sell': 'macd_exhaust_sell', 'Stoch_OS_Buy': 'stoch_os_buy', 'Stoch_OB_Sell': 'stoch_ob_sell', 'PA_Engulfing_Buy': 'pa_engulfing_buy', 'PA_Engulfing_Sell': 'pa_engulfing_sell', 'PA_Pinbar_Buy': 'pa_pinbar_buy', 'PA_Pinbar_Sell': 'pa_pinbar_sell', 'PA_3_Soldiers_Buy': 'pa_3_soldiers', 'PA_3_Crows_Sell': 'pa_3_crows'}

# ==========================================
# 🌍 4. SIDEBAR UI (MENU DESPLEGABLE)
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🧬 GENESIS LAB V260</h2>", unsafe_allow_html=True)

with st.sidebar.expander("🌍 DATOS Y EXCHANGE", expanded=True):
    exchange_sel = st.selectbox("🏦 Exchange", ["coinbase", "kucoin", "kraken", "binance"], index=0)
    ticker = st.text_input("Símbolo Exacto", value="SD/USDC")
    utc_offset = st.number_input("🌍 Zona Horaria", value=-5.0, step=0.5)
    intervalos = {"1 Minuto": "1m", "5 Minutos": "5m", "15 Minutos": "15m", "30 Minutos": "30m", "1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d"}
    intervalo_sel = st.selectbox("Temporalidad", list(intervalos.keys()), index=2) 
    iv_download = intervalos[intervalo_sel]
    hoy = datetime.today().date()
    is_micro = iv_download in ["1m", "5m", "15m", "30m"]
    st.info("⚠️ Para espejo 100% real, usa la misma fecha que en el 'Trading Range' de TV.")
    start_date, end_date = st.slider("📅 Scope (Rango de Fechas)", min_value=hoy - timedelta(days=250 if is_micro else 1500), max_value=hoy, value=(hoy - timedelta(days=200 if is_micro else 1500), hoy), format="YYYY-MM-DD")
    
    if st.button("📥 DESCARGAR MATRIX DE DATOS", use_container_width=True, type="primary"):
        st.session_state['data_params'] = {'ex': exchange_sel, 'sym': ticker, 'start': start_date, 'end': end_date, 'iv': iv_download, 'offset': utc_offset, 'micro': is_micro}
        st.rerun()

with st.sidebar.expander("💼 CAPITAL Y COMISIONES", expanded=False):
    capital_inicial = st.number_input("Capital Inicial (USD)", value=1000.0, step=100.0)
    comision_pct = st.number_input("Comisión (%)", value=0.15, step=0.05) / 100.0 
    is_calib_mode = st.checkbox("🛠️ MODO CALIBRACIÓN TV", value=False)

with st.sidebar.expander("🤖 INTELIGENCIA Y FORJA", expanded=False):
    st.markdown("<h4 style='color: #FFA500;'>🧠 IA GREED FACTOR</h4>", unsafe_allow_html=True)
    st.caption("0.0 Sniper | 0.5 Balance | 1.0 Depredador")
    greed_factor = st.slider("Nivel de Avaricia", 0.0, 1.0, 0.8, 0.1)
    
    st.markdown("---")
    global_epochs = st.slider("Épocas Rápidas (x1000)", 1, 1000, 50)
    target_strats = st.multiselect("🎯 Mutantes a Forjar:", estrategias, default=estrategias)
    if st.button(f"🧠 DEEP MINE GLOBAL", type="primary", use_container_width=True, key="btn_global"):
        st.session_state['global_queue'] = target_strats.copy(); st.session_state['abort_opt'] = False; st.session_state['run_global'] = True; st.rerun()

    if st.button("🤖 CREAR NUEVO MUTANTE IA", type="secondary", use_container_width=True, key="btn_mutant"):
        new_id = f"AI_MUTANT_{int(time.time())}_{random.randint(10, 99)}"
        if new_id not in st.session_state['ai_algos']:
            st.session_state['ai_algos'].append(new_id); get_safe_vault(new_id); st.session_state['global_queue'] = [new_id]; st.session_state['run_global'] = True; st.rerun()

    st.markdown("---")
    deep_epochs_target = st.number_input("Objetivo Épocas Profundas", min_value=10000, max_value=10000000, value=100000, step=10000)
    if st.button("🌌 CREAR MUTANTE PROFUNDO", type="secondary", use_container_width=True, key="btn_mutant_deep"):
        new_id = f"AI_DEEP_{int(time.time())}_{random.randint(10, 99)}"
        if new_id not in st.session_state['ai_algos']:
            st.session_state['ai_algos'].append(new_id); get_safe_vault(new_id); st.session_state['abort_opt'] = False
            st.session_state['deep_opt_state'] = {'s_id': new_id, 'target_epochs': deep_epochs_target, 'current_epoch': 0, 'paused': False, 'start_time': time.time()}; st.rerun()

with st.sidebar.expander("⚙️ SISTEMA", expanded=False):
    if st.button("🛑 ABORTAR RUN GLOBAL", use_container_width=True, key="btn_abort"):
        st.session_state['abort_opt'] = True; st.session_state['global_queue'] = []; st.session_state['run_global'] = False; st.rerun()
    if st.button("🔄 PURGAR MEMORIA RAM", use_container_width=True, key="btn_purge"): 
        st.cache_data.clear(); st.session_state.clear(); gc.collect(); st.rerun()

if 'data_params' not in st.session_state:
    st.info("👈 Por favor, configura los datos en el menú lateral y haz clic en **'📥 DESCARGAR MATRIX DE DATOS'** para iniciar el Laboratorio Quant.")
    st.stop()

dp = st.session_state['data_params']

# ==========================================
# 🛑 5. EXTRACCIÓN Y WARM-UP INSTITUCIONAL
# ==========================================
@st.cache_data(ttl=3600, show_spinner="📡 Sincronizando Línea Temporal con Servidores (V260)...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, offset, is_micro, version_key):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        warmup_days = 40 if is_micro else 150
        start_ts = int(datetime.combine(start - timedelta(days=warmup_days), datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        
        all_ohlcv, current_ts, error_count = [], start_ts, 0
        req_limit = 1000
        if 'coinbase' in exchange_id.lower(): req_limit = 300
        if 'kraken' in exchange_id.lower(): req_limit = 720

        while current_ts < end_ts:
            try: 
                ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=req_limit); error_count = 0 
            except Exception as e: 
                error_count += 1
                if error_count >= 3: return pd.DataFrame(), f"❌ ERROR CCXT ({exchange_id}): {str(e)}"
                time.sleep(1); continue
            if not ohlcv or len(ohlcv) == 0: break
            if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                ohlcv = [c for c in ohlcv if c[0] > all_ohlcv[-1][0]]
                if not ohlcv: break
            all_ohlcv.extend(ohlcv); current_ts = ohlcv[-1][0] + 1
            if len(all_ohlcv) > 100000: break
            
        if not all_ohlcv: return pd.DataFrame(), f"El Exchange devolvió 0 velas."
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
        df.index = df.index + timedelta(hours=offset); df = df[~df.index.duplicated(keep='first')]
        if len(df) < 50: return pd.DataFrame(), f"❌ Solo {len(df)} velas. Intenta ampliar el rango de fechas."
        
        freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'}
        pd_freq = freq_map.get(iv_down, '15min')
        df = df.resample(pd_freq).asfreq()
        df['Close'] = df['Close'].ffill()
        df['Open'] = df['Open'].fillna(df['Close'])
        df['High'] = df['High'].fillna(df['Close'])
        df['Low'] = df['Low'].fillna(df['Close'])
        df['Volume'] = df['Volume'].fillna(0)
            
        a_h, a_l, a_c, a_o = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
        
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['Vol_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_MA_100'] = df['Volume'].rolling(window=100).mean()
        df['RVol'] = df['Volume'] / np.where(df['Vol_MA_100'] == 0, 1, df['Vol_MA_100'])
        df['High_Vol'] = df['Volume'] > df['Vol_MA_20']
        
        tr = np.zeros_like(a_c); tr[0] = a_h[0] - a_l[0]
        for i in range(1, len(a_c)): tr[i] = max(a_h[i] - a_l[i], abs(a_h[i] - a_c[i-1]), abs(a_l[i] - a_c[i-1]))
        df['ATR'] = rma_pine(tr, 14); df['ATR'] = df['ATR'].fillna(df['High']-df['Low'])
        
        delta = np.zeros_like(a_c); delta[1:] = a_c[1:] - a_c[:-1]
        u = np.where(delta > 0, delta, 0.0); d = np.where(delta < 0, -delta, 0.0)
        rs_u = rma_pine(u, 14); rs_d = rma_pine(d, 14); rs = rs_u / np.where(rs_d == 0, 1e-10, rs_d)
        df['RSI'] = np.where(rs_d == 0, 100.0, 100.0 - (100.0 / (1.0 + rs))); df['RSI_MA'] = df['RSI'].rolling(14).mean()
        
        upm = np.zeros_like(a_h); upm[1:] = a_h[1:] - a_h[:-1]
        downm = np.zeros_like(a_l); downm[1:] = a_l[:-1] - a_l[1:]
        plusDM = np.where((upm > downm) & (upm > 0), upm, 0.0)
        minusDM = np.where((downm > upm) & (downm > 0), downm, 0.0)
        trur = rma_pine(tr, 14); plus = 100 * rma_pine(plusDM, 14) / trur; minus = 100 * rma_pine(minusDM, 14) / trur
        sum_dm = plus + minus; dx = 100 * np.abs(plus - minus) / np.where(sum_dm == 0, 1, sum_dm)
        df['ADX'] = rma_pine(dx, 14)
        
        sum_tr = pd.Series(tr).rolling(14).sum()
        hh_14, ll_14 = df['High'].rolling(14).max(), df['Low'].rolling(14).min()
        df['CHOP'] = 100 * np.log10(sum_tr / (hh_14 - ll_14)) / np.log10(14)
        
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Sig'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        stoch = 100 * (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
        df['Stoch_K'] = stoch.rolling(3).mean(); df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        ap = (df['High'] + df['Low'] + df['Close']) / 3.0
        esa = ap.ewm(span=10, adjust=False).mean(); d_wt = (ap - esa).abs().ewm(span=10, adjust=False).mean()
        df['WT1'] = ((ap - esa) / (0.015 * np.where(d_wt == 0, 1, d_wt))).ewm(span=21, adjust=False).mean()
        df['WT2'] = df['WT1'].rolling(4).mean()
        
        df['Basis'] = df['Close'].rolling(20).mean(); dev = df['Close'].rolling(20).std(ddof=0)
        df['BBU'] = df['Basis'] + (2.0 * dev); df['BBL'] = df['Basis'] - (2.0 * dev)
        df['BB_Width'] = (df['BBU'] - df['BBL']) / np.where(df['Basis'] == 0, 1, df['Basis'])
        df['BB_Width_Avg'] = df['BB_Width'].rolling(20).mean()
        df['BB_Delta'] = df['BB_Width'] - df['BB_Width'].shift(1).fillna(0)
        df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10).mean()
        
        df['KC_Upper'] = df['Basis'] + (df['ATR'] * 1.5); df['KC_Lower'] = df['Basis'] - (df['ATR'] * 1.5)
        df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
        df['Z_Score'] = np.where(dev == 0, 0, (df['Close'] - df['Basis']) / dev)
        df['RSI_BB_Basis'] = df['RSI'].rolling(14).mean(); df['RSI_BB_Dev'] = df['RSI'].rolling(14).std(ddof=0) * 2.0
        
        df['Vela_Verde'], df['Vela_Roja'] = df['Close'] > df['Open'], df['Close'] < df['Open']
        df['body_size'] = np.abs(df['Close'] - df['Open']) 
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
        
        target_start = pd.to_datetime(datetime.combine(start, datetime.min.time())) + timedelta(hours=offset)
        df = df[df.index >= target_start]; df['Is_Train'] = True
        gc.collect(); return df, "OK"
    except Exception as e: return pd.DataFrame(), f"❌ ERROR FATAL: {str(e)}"

df_global, status_api = cargar_matriz(dp['ex'], dp['sym'], dp['start'], dp['end'], dp['iv'], dp['offset'], dp['micro'], st.session_state['app_version'])
if df_global.empty: st.error(status_api); st.stop()
dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)

st.success(f"📊 Matrix Data Extraída: **{len(df_global):,} velas** | **{dias_reales} días**")

# ==========================================
# 🧠 6. CREACIÓN DE MATRICES NUMPY
# ==========================================
a_c, a_o, a_h, a_l = df_global['Close'].values, df_global['Open'].values, df_global['High'].values, df_global['Low'].values
a_rsi, a_rsi_ma, a_adx = df_global['RSI'].values, df_global['RSI_MA'].values, df_global['ADX'].values
a_macd, a_macd_sig, a_chop = df_global['MACD'].values, df_global['MACD_Sig'].values, df_global['CHOP'].values
a_stoch_k, a_stoch_d = df_global['Stoch_K'].values, df_global['Stoch_D'].values
a_bbl, a_bbu, a_bw = df_global['BBL'].values, df_global['BBU'].values, df_global['BB_Width'].values
a_wt1, a_wt2 = df_global['WT1'].values, df_global['WT2'].values
a_ema50, a_ema200, a_atr = df_global['EMA_50'].values, df_global['EMA_200'].values, df_global['ATR'].values
a_rvol, a_hvol = df_global['RVol'].values, df_global['High_Vol'].values
a_vv, a_vr = df_global['Vela_Verde'].values, df_global['Vela_Roja'].values
a_rcu, a_rcd = df_global['RSI_Cross_Up'].values, df_global['RSI_Cross_Dn'].values
a_sqz_on = df_global['Squeeze_On'].values
a_bb_delta, a_bb_delta_avg = df_global['BB_Delta'].values, df_global['BB_Delta_Avg'].values
a_zscore, a_rsi_bb_b, a_rsi_bb_d = df_global['Z_Score'].values, df_global['RSI_BB_Basis'].values, df_global['RSI_BB_Dev'].values
a_lw, a_uw, a_bs = df_global['lower_wick'].values, df_global['upper_wick'].values, df_global['body_size'].values
a_mb, a_fk = df_global['Macro_Bull'].values, df_global['is_falling_knife'].values

a_pa_eng_b, a_pa_eng_s = df_global['PA_Engulfing_Buy'].values, df_global['PA_Engulfing_Sell'].values
a_pa_pin_b, a_pa_pin_s = df_global['PA_Pinbar_Buy'].values, df_global['PA_Pinbar_Sell'].values
a_pa_3sol_b, a_pa_3cro_s = df_global['PA_3_Soldiers'].values, df_global['PA_3_Crows'].values

a_pl30_l, a_ph30_l = df_global['PL30_L'].fillna(0).values, df_global['PH30_L'].fillna(99999).values
a_pl100_l, a_ph100_l = df_global['PL100_L'].fillna(0).values, df_global['PH100_L'].fillna(99999).values
a_pl300_l, a_ph300_l = df_global['PL300_L'].fillna(0).values, df_global['PH300_L'].fillna(99999).values

a_c_s1, a_o_s1, a_l_s1 = npshift(a_c, 1, 0.0), npshift(a_o, 1, 0.0), npshift(a_l, 1, 0.0)
a_l_s5, a_h_s1, a_h_s5 = npshift(a_l, 5, 0.0), npshift(a_h, 1, 0.0), npshift(a_h, 5, 0.0)
a_rsi_s1, a_rsi_s5 = npshift(a_rsi, 1, 50.0), npshift(a_rsi, 5, 50.0)
a_wt1_s1, a_wt2_s1, a_macd_s1 = npshift(a_wt1, 1, 0.0), npshift(a_wt2, 1, 0.0), npshift(a_macd, 1, 0.0)

def calcular_señales_numpy(hitbox, therm_w, adx_th, whale_f):
    n_len = len(a_c); s_dict = {}
    a_tsup = np.maximum(a_pl30_l, np.maximum(a_pl100_l, a_pl300_l))
    a_tres = np.minimum(a_ph30_l, np.minimum(a_ph100_l, a_ph300_l))
    a_dsup = np.where(a_c == 0, 0, np.abs(a_c - a_tsup) / a_c * 100)
    a_dres = np.where(a_c == 0, 0, np.abs(a_c - a_tres) / a_c * 100)
    sr_val = a_atr * 2.0

    ceil_w = np.where((a_ph30_l > a_c) & (a_ph30_l <= a_c + sr_val), 1, 0) + np.where((a_pl30_l > a_c) & (a_pl30_l <= a_c + sr_val), 1, 0) + np.where((a_ph100_l > a_c) & (a_ph100_l <= a_c + sr_val), 3, 0) + np.where((a_pl100_l > a_c) & (a_pl100_l <= a_c + sr_val), 3, 0) + np.where((a_ph300_l > a_c) & (a_ph300_l <= a_c + sr_val), 5, 0) + np.where((a_pl300_l > a_c) & (a_pl300_l <= a_c + sr_val), 5, 0)
    floor_w = np.where((a_ph30_l < a_c) & (a_ph30_l >= a_c - sr_val), 1, 0) + np.where((a_pl30_l < a_c) & (a_pl30_l >= a_c - sr_val), 1, 0) + np.where((a_ph100_l < a_c) & (a_ph100_l >= a_c - sr_val), 3, 0) + np.where((a_pl100_l < a_c) & (a_pl100_l >= a_c - sr_val), 3, 0) + np.where((a_ph300_l < a_c) & (a_ph300_l >= a_c - sr_val), 5, 0) + np.where((a_pl300_l < a_c) & (a_pl300_l >= a_c - sr_val), 5, 0)

    is_abyss, is_hard_wall = floor_w == 0, ceil_w >= therm_w
    trinity_safe = a_mb & ~a_fk
    neon_up, neon_dn = a_sqz_on & (a_c >= a_bbu * 0.999) & a_vv, a_sqz_on & (a_c <= a_bbl * 1.001) & a_vr
    
    defcon_level = np.full(n_len, 5)
    m4 = neon_up | neon_dn; defcon_level[m4] = 4
    m3 = m4 & (a_bb_delta > 0); defcon_level[m3] = 3
    m2 = m3 & (a_bb_delta > a_bb_delta_avg) & (a_adx > adx_th); defcon_level[m2] = 2
    m1 = m2 & (a_bb_delta > a_bb_delta_avg * 1.5) & (a_adx > adx_th + 5) & (a_rvol > 1.2); defcon_level[m1] = 1

    cond_defcon_buy, cond_defcon_sell = (defcon_level <= 2) & neon_up, (defcon_level <= 2) & neon_dn
    cond_therm_buy_bounce, cond_therm_sell_wall = (floor_w >= therm_w) & a_rcu & ~is_hard_wall, (ceil_w >= therm_w) & a_rcd
    cond_therm_buy_vacuum, cond_therm_sell_panic = (ceil_w <= 3) & neon_up & ~is_abyss, is_abyss & a_vr

    tol = a_atr * 0.5
    is_grav_sup, is_grav_res = a_dsup < hitbox, a_dres < hitbox
    cross_up_res, cross_dn_sup = (a_c > a_tres) & (a_c_s1 <= npshift(a_tres, 1, 0)), (a_c < a_tsup) & (a_c_s1 >= npshift(a_tsup, 1, 0))
    
    cond_lock_buy_bounce, cond_lock_buy_break = is_grav_sup & (a_l <= a_tsup + tol) & (a_c > a_tsup) & a_vv, is_grav_res & cross_up_res & a_hvol & a_vv
    cond_lock_sell_reject, cond_lock_sell_breakd = is_grav_res & (a_h >= a_tres - tol) & (a_c < a_tres) & a_vr, is_grav_sup & cross_dn_sup & a_vr

    flash_vol = (a_rvol > whale_f * 0.8) & (np.abs(a_c - a_o) > a_atr * 0.3)
    whale_buy, whale_sell = flash_vol & a_vv, flash_vol & a_vr
    whale_memory = whale_buy | npshift_bool(whale_buy, 1) | npshift_bool(whale_buy, 2) | whale_sell | npshift_bool(whale_sell, 1) | npshift_bool(whale_sell, 2)
    is_whale_icon = whale_buy & ~npshift_bool(whale_buy, 1)

    rsi_vel = a_rsi - a_rsi_s1
    pre_pump = ((a_h > a_bbu) | (rsi_vel > 5)) & flash_vol & a_vv
    pump_memory = pre_pump | npshift_bool(pre_pump, 1) | npshift_bool(pre_pump, 2)
    pre_dump = ((a_l < a_bbl) | (rsi_vel < -5)) & flash_vol & a_vr
    dump_memory = pre_dump | npshift_bool(pre_dump, 1) | npshift_bool(pre_dump, 2)

    retro_peak, retro_peak_sell = (a_rsi < 30) & (a_c < a_bbl), (a_rsi > 70) & (a_c > a_bbu)
    k_break_up = (a_rsi > (a_rsi_bb_b + a_rsi_bb_d)) & (a_rsi_s1 <= npshift(a_rsi_bb_b + a_rsi_bb_d, 1))
    support_buy, support_sell = is_grav_sup & a_rcu, is_grav_res & a_rcd
    div_bull, div_bear = (a_l_s1 < a_l_s5) & (a_rsi_s1 > a_rsi_s5) & (a_rsi < 35), (a_h_s1 > a_h_s5) & (a_rsi_s1 < a_rsi_s5) & (a_rsi > 65)

    buy_score, sell_score = np.zeros(n_len), np.zeros(n_len)
    base_mask = retro_peak | k_break_up | support_buy | div_bull
    buy_score = np.where(base_mask & retro_peak, 50.0, np.where(base_mask & ~retro_peak, 30.0, buy_score))
    buy_score += np.where(is_grav_sup, 25.0, 0.0); buy_score += np.where(whale_memory, 20.0, 0.0); buy_score += np.where(pump_memory, 15.0, 0.0); buy_score += np.where(div_bull, 15.0, 0.0); buy_score += np.where(k_break_up & ~retro_peak, 15.0, 0.0); buy_score += np.where(a_zscore < -2.0, 15.0, 0.0)
    
    base_mask_s = retro_peak_sell | a_rcd | support_sell | div_bear
    sell_score = np.where(base_mask_s & retro_peak_sell, 50.0, np.where(base_mask_s & ~retro_peak_sell, 30.0, sell_score))
    sell_score += np.where(is_grav_res, 25.0, 0.0); sell_score += np.where(whale_memory, 20.0, 0.0); sell_score += np.where(dump_memory, 15.0, 0.0); sell_score += np.where(div_bear, 15.0, 0.0); sell_score += np.where(a_rcd & ~retro_peak_sell, 15.0, 0.0); sell_score += np.where(a_zscore > 2.0, 15.0, 0.0)

    is_magenta, is_magenta_sell = (buy_score >= 70) | retro_peak, (sell_score >= 70) | retro_peak_sell
    cond_pink_whale_buy = is_magenta & is_whale_icon
    wt_cross_up, wt_cross_dn = (a_wt1 > a_wt2) & (a_wt1_s1 <= a_wt2_s1), (a_wt1 < a_wt2) & (a_wt1_s1 >= a_wt2_s1)
    wt_oversold, wt_overbought = a_wt1 < -60, a_wt1 > 60

    s_dict['Ping_Buy'] = (a_adx < adx_th) & (a_c < a_bbl) & a_vv; s_dict['Ping_Sell'] = (a_c > a_bbu) | (a_rsi > 70)
    s_dict['Squeeze_Buy'] = neon_up; s_dict['Squeeze_Sell'] = (a_c < a_ema50)
    s_dict['Thermal_Buy'] = cond_therm_buy_bounce; s_dict['Thermal_Sell'] = cond_therm_sell_wall
    s_dict['Climax_Buy'] = cond_pink_whale_buy; s_dict['Climax_Sell'] = (a_rsi > 80)
    s_dict['Lock_Buy'] = cond_lock_buy_bounce; s_dict['Lock_Sell'] = cond_lock_sell_reject
    s_dict['Defcon_Buy'] = cond_defcon_buy; s_dict['Defcon_Sell'] = cond_defcon_sell
    s_dict['Jugg_Buy'] = a_mb & (a_c > a_ema50) & (a_c_s1 < npshift(a_ema50,1)) & a_vv & ~a_fk; s_dict['Jugg_Sell'] = (a_c < a_ema50)
    s_dict['Trinity_Buy'] = a_mb & (a_rsi < 35) & a_vv & ~a_fk; s_dict['Trinity_Sell'] = (a_rsi > 75) | (a_c < a_ema200)
    s_dict['Lev_Buy'] = a_mb & a_rcu & (a_rsi < 45); s_dict['Lev_Sell'] = (a_c < a_ema200)

    s_dict['Q_Pink_Whale_Buy'] = cond_pink_whale_buy; s_dict['Q_Lock_Bounce'] = cond_lock_buy_bounce; s_dict['Q_Lock_Break'] = cond_lock_buy_break; s_dict['Q_Neon_Up'] = neon_up; s_dict['Q_Defcon_Buy'] = cond_defcon_buy; s_dict['Q_Therm_Bounce'] = cond_therm_buy_bounce; s_dict['Q_Therm_Vacuum'] = cond_therm_buy_vacuum; s_dict['Q_Nuclear_Buy'] = is_magenta & (wt_oversold | wt_cross_up); s_dict['Q_Early_Buy'] = is_magenta; s_dict['Q_Rebound_Buy'] = a_rcu & ~is_magenta
    s_dict['Q_Lock_Reject'] = cond_lock_sell_reject; s_dict['Q_Lock_Breakd'] = cond_lock_sell_breakd; s_dict['Q_Neon_Dn'] = neon_dn; s_dict['Q_Defcon_Sell'] = cond_defcon_sell; s_dict['Q_Therm_Wall_Sell'] = cond_therm_sell_wall; s_dict['Q_Therm_Panic_Sell'] = cond_therm_sell_panic; s_dict['Q_Nuclear_Sell'] = (a_rsi > 70) & (wt_overbought | wt_cross_dn); s_dict['Q_Early_Sell'] = (a_rsi > 70) & a_vr

    s_dict['Wyc_Spring_Buy'] = (a_l < a_tsup) & (a_c > a_tsup) & a_hvol; s_dict['Wyc_Upthrust_Sell'] = (a_h > a_tres) & (a_c < a_tres) & a_hvol
    s_dict['VSA_Accum_Buy'] = (a_bs < a_atr * 0.5) & (a_lw > a_bs * 1.5) & a_hvol & a_vr; s_dict['VSA_Dist_Sell'] = (a_bs < a_atr * 0.5) & (a_uw > a_bs * 1.5) & a_hvol & a_vv
    
    swing_range = a_ph30_l - a_pl30_l
    fib_618_b = a_ph30_l - (swing_range * 0.618); fib_618_s = a_pl30_l + (swing_range * 0.618)
    s_dict['Fibo_618_Buy'] = (a_l < fib_618_b) & (a_c > fib_618_b); s_dict['Fibo_618_Sell'] = (a_h > fib_618_s) & (a_c < fib_618_s)
    
    s_dict['MACD_Impulse_Buy'] = (a_macd > a_macd_sig) & (a_macd > 0) & (a_macd > a_macd_s1); s_dict['MACD_Exhaust_Sell'] = (a_macd < a_macd_sig) & (a_macd > 0) & (a_macd < a_macd_s1)
    s_dict['Stoch_OS_Buy'] = (a_stoch_k < 20) & (a_stoch_k > a_stoch_d); s_dict['Stoch_OB_Sell'] = (a_stoch_k > 80) & (a_stoch_k < a_stoch_d)
    s_dict['PA_Engulfing_Buy'] = a_pa_eng_b; s_dict['PA_Engulfing_Sell'] = a_pa_eng_s; s_dict['PA_Pinbar_Buy'] = a_pa_pin_b; s_dict['PA_Pinbar_Sell'] = a_pa_pin_s
    s_dict['PA_3_Soldiers_Buy'] = a_pa_3sol_b; s_dict['PA_3_Crows_Sell'] = a_pa_3cro_s

    s_dict['Organic_Vol'] = a_hvol; s_dict['Organic_Squeeze'] = a_sqz_on; s_dict['Organic_Safe'] = a_mb & ~a_fk; s_dict['Organic_Pump'] = pump_memory; s_dict['Organic_Dump'] = dump_memory; s_dict['Organic_Gaussian_Clean'] = a_chop < 61.8

    f_calib_buy = np.zeros(n_len, dtype=bool)
    for i in range(0, n_len, 50): f_calib_buy[i] = True
    s_dict['Calibrador'] = f_calib_buy
    return s_dict

def optimizar_ia_tracker(s_id, cap_ini, com_pct, invest_pct, target_ado, dias_reales, buy_hold_money, epochs=1, cur_net=-float('inf'), cur_fit=-float('inf'), deep_info=None, greed_factor=0.8):
    vault = get_safe_vault(s_id)
    best_fit_live = vault.get('fit', -float('inf'))
    
    iters = 3000 * epochs
    chunk_size = 1000
    chunks = max(1, iters // chunk_size)
    if deep_info: chunks = min(chunks, 10) 
    
    start_time = time.time(); n_len = len(a_c)
    
    split_idx = int(n_len * 0.70) 
    dias_entrenamiento = max(1, dias_reales * 0.70)
    default_f, ones_mask = np.zeros(n_len, dtype=bool), np.ones(n_len, dtype=bool)

    for c in range(chunks):
        if st.session_state.get('abort_opt', False): break

        for _ in range(chunk_size): 
            dna_b_team = random.sample(todas_las_armas_b, random.randint(3, 8))
            dna_s_team = random.sample(todas_las_armas_s, random.randint(3, 8))
            
            dna_b_op = random.choice(['&', '|', 'vote'])
            dna_s_op = random.choice(['&', '|', 'vote'])
            
            dna_macro = random.choice(["All-Weather", "Bull Only", "Bear Only", "Ignore", "Organic_Vol", "Organic_Squeeze", "Organic_Safe", "Organic_Gaussian_Clean"])
            dna_vol = random.choice(["All-Weather", "Trend", "Range", "Ignore", "Organic_Pump", "Organic_Dump", "Organic_Gaussian_Clean"])
            
            r_hitbox = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            r_therm = random.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            r_adx = random.choice([15.0, 20.0, 25.0, 30.0, 35.0])
            r_whale = random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
            
            r_w_rsi = round(random.uniform(-2.0, 2.0), 4)
            r_w_z = round(random.uniform(-10.0, 10.0), 4)
            r_w_adx = round(random.uniform(-2.0, 2.0), 4)
            r_th_b = round(random.uniform(50.0, 100.0), 2)
            r_th_s = round(random.uniform(-100.0, -50.0), 2)
            r_atr_tp = round(random.uniform(0.5, 15.0), 2)
            r_atr_sl = round(random.uniform(1.0, 10.0), 2)

            s_dict = calcular_señales_numpy(r_hitbox, r_therm, r_adx, r_whale)

            m_mask = ones_mask if dna_macro == "Ignore" or dna_macro == "All-Weather" else (a_mb if dna_macro == "Bull Only" else (~a_mb if dna_macro == "Bear Only" else s_dict.get(dna_macro, ones_mask)))
            v_mask = ones_mask if dna_vol == "Ignore" or dna_vol == "All-Weather" else ((a_adx >= r_adx) if dna_vol == "Trend" else ((a_adx < r_adx) if dna_vol == "Range" else s_dict.get(dna_vol, ones_mask)))
            
            if dna_b_op == '&':
                f_buy_tactical = np.ones(n_len, dtype=bool)
                for r in dna_b_team: f_buy_tactical &= s_dict.get(r, default_f)
            elif dna_b_op == '|':
                f_buy_tactical = np.zeros(n_len, dtype=bool)
                for r in dna_b_team: f_buy_tactical |= s_dict.get(r, default_f)
            else:
                votes = np.zeros(n_len, dtype=int)
                for r in dna_b_team: votes += s_dict.get(r, default_f).astype(int)
                f_buy_tactical = votes >= max(1, len(dna_b_team) // 2)

            if dna_s_op == '&':
                f_sell_tactical = np.ones(n_len, dtype=bool)
                for r in dna_s_team: f_sell_tactical &= s_dict.get(r, default_f)
            elif dna_s_op == '|':
                f_sell_tactical = np.zeros(n_len, dtype=bool)
                for r in dna_s_team: f_sell_tactical |= s_dict.get(r, default_f)
            else:
                votes = np.zeros(n_len, dtype=int)
                for r in dna_s_team: votes += s_dict.get(r, default_f).astype(int)
                f_sell_tactical = votes >= max(1, len(dna_s_team) // 2)
            
            score_arr = (a_rsi * r_w_rsi) + (a_zscore * r_w_z) + (a_adx * r_w_adx)
            f_buy_final = (f_buy_tactical | (score_arr > r_th_b)) & m_mask & v_mask
            f_sell_final = f_sell_tactical | (score_arr < r_th_s)

            # 🛑 1. SIMULACIÓN IN-SAMPLE (Entrenamiento)
            net_is, pf_is, nt_is, mdd_is, wr_is = simular_core_rapido(
                a_h[:split_idx], a_l[:split_idx], a_c[:split_idx], a_o[:split_idx], a_atr[:split_idx], 
                f_buy_final[:split_idx], f_sell_final[:split_idx], 
                r_atr_tp, r_atr_sl, float(cap_ini), float(com_pct), float(invest_pct), 0.0, False
            )

            ado_actual = nt_is / max(1, dias_entrenamiento)
            fit_score = -float('inf') 
            
            if nt_is >= 3 and net_is > 0: 
                avg_trade_net_pct = (net_is / cap_ini) / nt_is * 100.0
                if avg_trade_net_pct < 0.25:
                    fit_score = net_is - 5000.0 
                else:
                    if greed_factor >= 0.7:
                        pf_mod = 1.0 if pf_is > 1.1 else 0.5
                        dd_penalty = 1.0 if mdd_is <= 60.0 else (mdd_is / 60.0)
                        fit_score = (net_is * (1.0 + np.log10(nt_is)) * pf_mod) / dd_penalty
                    elif greed_factor <= 0.3:
                        pf_mod = pf_is ** 2.0
                        wr_mod = (wr_is / 40.0) ** 2.0
                        dd_penalty = np.exp(mdd_is / 20.0)
                        fit_score = (net_is * pf_mod * wr_mod) / dd_penalty
                    else:
                        pf_mod = min(pf_is, 5.0)
                        dd_penalty = 1.0 if mdd_is <= 35.0 else (mdd_is / 35.0)
                        fit_score = (net_is * pf_mod) / dd_penalty
            elif nt_is > 0:
                fit_score = net_is - mdd_is - (abs(ado_actual - max(0.1, target_ado)) * 5)
            else:
                fit_score = net_is - 1000.0 

            # 🛑 2. SIMULACIÓN OOS & PASADA GLOBAL PARA SINCRONIZAR UI
            if fit_score > best_fit_live:
                net_oos, pf_oos, nt_oos, mdd_oos, wr_oos = simular_core_rapido(
                    a_h[split_idx:], a_l[split_idx:], a_c[split_idx:], a_o[split_idx:], a_atr[split_idx:], 
                    f_buy_final[split_idx:], f_sell_final[split_idx:], 
                    r_atr_tp, r_atr_sl, float(cap_ini), float(com_pct), float(invest_pct), 0.0, False
                )
                
                net_tot, pf_tot, nt_tot, mdd_tot, wr_tot = simular_core_rapido(
                    a_h, a_l, a_c, a_o, a_atr, f_buy_final, f_sell_final, 
                    r_atr_tp, r_atr_sl, float(cap_ini), float(com_pct), float(invest_pct), 0.0, False
                )
                
                best_fit_live = fit_score
                
                bp = {
                    'b_team': dna_b_team, 's_team': dna_s_team, 'b_op': dna_b_op, 's_op': dna_s_op,
                    'macro': dna_macro, 'vol': dna_vol, 'hitbox': r_hitbox, 'therm_w': r_therm, 
                    'adx_th': r_adx, 'whale_f': r_whale, 'fit': fit_score, 
                    'net': net_tot, 'net_is': net_is, 'net_oos': net_oos,
                    'winrate': wr_tot, 'pf': pf_tot, 'nt': nt_tot, 'reinv': invest_pct, 'ado': ado_actual, 
                    'w_rsi': r_w_rsi, 'w_z': r_w_z, 'w_adx': r_w_adx, 
                    'th_buy': r_th_b, 'th_sell': r_th_s, 'atr_tp': r_atr_tp, 'atr_sl': r_atr_sl
                }
                save_champion(s_id, bp)
                st.session_state[f'opt_status_{s_id}'] = True
            
        global_start = deep_info.get('start_time', start_time) if deep_info else start_time
        total_elapsed_sec = time.time() - global_start
        h, rem = divmod(total_elapsed_sec, 3600); m, s = divmod(rem, 60)
        time_str = f"{int(h):02d}h:{int(m):02d}m:{int(s):02d}s"
        
        vault = get_safe_vault(s_id)
        current_best_net = vault.get('net', 0.0)
        current_best_nt = vault.get('nt', 0)
        current_best_pf = vault.get('pf', 0.0)

        if deep_info:
            current_epoch_val = deep_info['current'] + (c+1)*(chunk_size); macro_pct = int((current_epoch_val / deep_info['total']) * 100)
            title = f"🌌 DEEP FORGE: {s_id}"; subtitle = f"Épocas: {current_epoch_val:,} / {deep_info['total']:,} ({macro_pct}%)<br>⏱️ Tiempo: {time_str}"; color = "#9932CC"
        else:
            pct_done = int(((c + 1) / chunks) * 100); combos = (c + 1) * chunk_size
            title = f"GENESIS LAB V260: {s_id}"; subtitle = f"Progreso: {pct_done}% | ADN Probados: {combos:,}<br>⏱️ Tiempo Ejecución: {time_str}"; color = "#00FFFF"

        html_str = f"""
        <style>
        .loader-container {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; text-align: center; background: rgba(0,0,0,0.95); padding: 35px; border-radius: 20px; border: 2px solid {color}; box-shadow: 0 0 50px {color};}}
        .rocket {{ font-size: 8rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 20px {color}); }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        </style>
        <div class="loader-container">
            <div class="rocket">🧬</div>
            <div style="color: {color}; font-size: 1.8rem; font-weight: bold; margin-top: 15px;">{title}</div>
            <div style="color: white; font-size: 1.3rem;">{subtitle}</div>
            <div style="color: #00FF00; font-weight: bold; font-size: 1.5rem; margin-top: 15px;">🏆 Récord Global: ${current_best_net:.2f}</div>
            <div style="color: cyan; font-size: 1.0rem;">Trades Totales: {current_best_nt} | Win Rate: {current_best_pf:.2f}x PF</div>
        </div>
        """
        ph_holograma.markdown(html_str, unsafe_allow_html=True)
            
    return get_safe_vault(s_id) if best_fit_live != -float('inf') else None

def run_backtest_eval(s_id, cap_ini, com_pct):
    vault = get_safe_vault(s_id)
    is_calib = st.session_state.get('is_calib_mode', False)
    
    s_dict = calcular_señales_numpy(vault.get('hitbox',1.5), vault.get('therm_w',4.0), vault.get('adx_th',25.0), vault.get('whale_f',2.5))
    n_len = len(a_c)
    
    w_rsi, w_z, w_adx = round(float(vault.get('w_rsi', 0.0)), 4), round(float(vault.get('w_z', 0.0)), 4), round(float(vault.get('w_adx', 0.0)), 4)
    th_buy, th_sell = round(float(vault.get('th_buy', 99.0)), 2), round(float(vault.get('th_sell', -999.0)), 2)
    atr_tp, atr_sl = round(float(vault.get('atr_tp', 0.0)), 2), round(float(vault.get('atr_sl', 0.0)), 2)
    
    f_tp, f_sl = np.full(n_len, atr_tp), np.full(n_len, atr_sl)
    default_f, ones_mask = np.zeros(n_len, dtype=bool), np.ones(n_len, dtype=bool)

    if is_calib:
        f_buy = s_dict['Calibrador']
        f_sell = default_f
    else:
        m_mask = ones_mask if vault.get('macro') in ["Ignore", "All-Weather"] else (a_mb if vault.get('macro') == "Bull Only" else (~a_mb if vault.get('macro') == "Bear Only" else s_dict.get(vault.get('macro'), ones_mask)))
        v_mask = ones_mask if vault.get('vol') in ["Ignore", "All-Weather"] else ((a_adx >= vault.get('adx_th', 25.0)) if vault.get('vol') == "Trend" else ((a_adx < vault.get('adx_th', 25.0)) if vault.get('vol') == "Range" else s_dict.get(vault.get('vol'), ones_mask)))

        dna_b_op = vault.get('b_op', '&')
        dna_b_team = vault.get('b_team', ['Ping_Buy'])
        if dna_b_op == '&':
            f_buy_tactical = np.ones(n_len, dtype=bool)
            for r in dna_b_team: f_buy_tactical &= s_dict.get(r, default_f)
        elif dna_b_op == '|':
            f_buy_tactical = np.zeros(n_len, dtype=bool)
            for r in dna_b_team: f_buy_tactical |= s_dict.get(r, default_f)
        else:
            votes = np.zeros(n_len, dtype=int)
            for r in dna_b_team: votes += s_dict.get(r, default_f).astype(int)
            f_buy_tactical = votes >= max(1, len(dna_b_team) // 2)

        dna_s_op = vault.get('s_op', '&')
        dna_s_team = vault.get('s_team', ['Ping_Sell'])
        if dna_s_op == '&':
            f_sell_tactical = np.ones(n_len, dtype=bool)
            for r in dna_s_team: f_sell_tactical &= s_dict.get(r, default_f)
        elif dna_s_op == '|':
            f_sell_tactical = np.zeros(n_len, dtype=bool)
            for r in dna_s_team: f_sell_tactical |= s_dict.get(r, default_f)
        else:
            votes = np.zeros(n_len, dtype=int)
            for r in dna_s_team: votes += s_dict.get(r, default_f).astype(int)
            f_sell_tactical = votes >= max(1, len(dna_s_team) // 2)
        
        score_arr = (a_rsi * w_rsi) + (a_zscore * w_z) + (a_adx * w_adx)
        f_buy = (f_buy_tactical | (score_arr > th_buy)) & (m_mask & v_mask)
        f_sell = f_sell_tactical | (score_arr < th_sell)

    df_strat = df_global.copy()
    df_strat['Signal_Buy'], df_strat['Signal_Sell'], df_strat['Active_TP'], df_strat['Active_SL'] = f_buy, f_sell, f_tp, f_sl
    eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, cap_ini, float(vault.get('reinv', 20.0)), com_pct, 0.0, is_calib)
    return df_strat, eq_curve, t_log, total_comms

def build_pine_cond(team, op):
    if not team: return "false"
    if op == '&': return " and ".join([pine_map.get(x, 'false') for x in team])
    elif op == '|': return " or ".join([pine_map.get(x, 'false') for x in team])
    else:
        vote_str = " + ".join([f"({pine_map.get(x, 'false')} ? 1 : 0)" for x in team])
        thresh = max(1, len(team) // 2)
        return f"(({vote_str}) >= {thresh})"

# 🔥 TRASPLANTE MAESTRO DE LÓGICA DE SALIDA A TODOS LOS SCRIPTS 🔥
def generar_pine_script(s_id, vault, sym, tf, buy_pct, sell_pct, com_pct, start_date_obj, is_calib=False):
    v_hb = vault.get('hitbox', 1.5); v_tw = vault.get('therm_w', 4.0); v_adx = vault.get('adx_th', 25.0); v_wf = vault.get('whale_f', 2.5)
    json_buy = f'{{"passphrase": "ASTRONAUTA", "action": "{{{{strategy.order.action}}}}", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {buy_pct}, "limit_price": {{{{close}}}}, "side": "🟢 COMPRA"}}'
    json_sell = f'{{"passphrase": "ASTRONAUTA", "action": "{{{{strategy.order.action}}}}", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {sell_pct}, "limit_price": {{{{close}}}}, "side": "🔴 VENTA"}}'

    ps_base = f"""//@version=5
strategy("{s_id} MATRIX - {sym} [{tf}]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value={buy_pct}, commission_type=strategy.commission.percent, commission_value={com_pct*100}, slippage=0, process_orders_on_close=true)

// 1. WEBHOOKS DE ALTA PRIORIDAD
wt_enter_long = input.text_area(defval='{json_buy}', title="🟢 WT: Mensaje Enter Long")
wt_exit_long  = input.text_area(defval='{json_sell}', title="🔴 WT: Mensaje Exit Long")

grp_time = "📅 FILTRO DE FECHA"
start_year = input.int({start_date_obj.year}, "Año de Inicio", group=grp_time)
start_month = input.int({start_date_obj.month}, "Mes de Inicio", group=grp_time)
start_day = input.int({start_date_obj.day}, "Día de Inicio", group=grp_time)
window = time >= timestamp(syminfo.timezone, start_year, start_month, start_day, 0, 0)
"""
    if is_calib:
        return ps_base + """
// 🔥 MODO CALIBRADOR (Control Group) 🔥
bool signal_buy = bar_index % 50 == 0

var float tp_price = na
var float sl_price = na
bool just_entered = ta.change(strategy.position_size) > 0

if signal_buy and strategy.position_size == 0 and window
    strategy.entry("In", strategy.long, alert_message=wt_enter_long)

if just_entered
    tp_price := math.round(strategy.position_avg_price * 1.002, 5)
    sl_price := math.round(strategy.position_avg_price * 0.998, 5)

// 2. NUEVO MOTOR DE SALIDA MANUAL (Para el Calibrador)
bool hit_tp = high >= tp_price
bool hit_sl = low <= sl_price

if strategy.position_size > 0
    if hit_tp or hit_sl
        strategy.close("In", comment= hit_tp ? "TP_Hit" : "SL_Hit", alert_message=wt_exit_long)
        tp_price := na
        sl_price := na

plotshape(signal_buy, title="COMPRA", style=shape.triangleup, location=location.belowbar, color=color.yellow, size=size.tiny)
"""

    ps_indicators = f"""
hitbox_pct   = {v_hb}
therm_wall   = {v_tw}
adx_trend    = {v_adx}
whale_factor = {v_wf}

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

ap = hlc3
esa = ta.ema(ap, 10)
d_wt = ta.ema(math.abs(ap - esa), 10)
wt1 = ta.ema((ap - esa) / (0.015 * (d_wt == 0 ? 1 : d_wt)), 21)
wt2 = ta.sma(wt1, 4)

basis = ta.sma(close, 20)
stdev20 = ta.stdev(close, 20)
dev = 2.0 * stdev20
bbu = basis + dev
bbl = basis - dev
bb_width = basis == 0 ? 1 : (bbu - bbl) / basis
bb_width_avg = ta.sma(bb_width, 20)
bb_delta = bb_width - nz(bb_width[1], 0)
bb_delta_avg = ta.sma(bb_delta, 10)

kc_u = ta.sma(close, 20) + (atr * 1.5)
kc_l = ta.sma(close, 20) - (atr * 1.5)
squeeze_on = (bbu < kc_u) and (bbl > kc_l)

z_score = stdev20 == 0 ? 0 : (close - basis) / stdev20
rsi_bb_basis = ta.sma(rsi_v, 14)
rsi_bb_dev = ta.stdev(rsi_v, 14) * 2.0

vela_verde = close > open
vela_roja = close < open
rsi_ma = ta.sma(rsi_v, 14)
rsi_cross_up = (rsi_v > rsi_ma) and (nz(rsi_v[1]) <= nz(rsi_ma[1]))
rsi_cross_dn = (rsi_v < rsi_ma) and (nz(rsi_v[1]) >= nz(rsi_ma[1]))
macro_bull = close >= ema200
trinity_safe = macro_bull and not is_falling_knife

[macdLine, signalLine, _] = ta.macd(close, 12, 26, 9)
stoch_k = ta.sma(ta.stoch(close, high, low, 14), 3)
stoch_d = ta.sma(stoch_k, 3)
chop_v = 100 * math.log10(math.sum(ta.tr, 14) / (ta.highest(high, 14) - ta.lowest(low, 14))) / math.log10(14)
gaussian_clean = chop_v < 61.8

local_high = ta.highest(high[1], 30)
local_low = ta.lowest(low[1], 30)
swing_range = local_high - local_low
fib_618_b = local_high - (swing_range * 0.618)
fib_618_s = local_low + (swing_range * 0.618)

pa_engulfing_buy = vela_verde and nz(vela_roja[1]) and close > nz(open[1]) and open < nz(close[1])
pa_engulfing_sell = vela_roja and nz(vela_verde[1]) and close < nz(open[1]) and open > nz(close[1])
pa_pinbar_buy = lower_wick > body_size * 2.5 and upper_wick < body_size
pa_pinbar_sell = upper_wick > body_size * 2.5 and lower_wick < body_size
pa_3_soldiers = vela_verde and nz(vela_verde[1]) and nz(vela_verde[2]) and close > nz(close[1]) and nz(close[1]) > nz(close[2])
pa_3_crows = vela_roja and nz(vela_roja[1]) and nz(vela_roja[2]) and close < nz(close[1]) and nz(close[1]) < nz(close[2])

low_30 = ta.lowest(low[1], 30)
low_100 = ta.lowest(low[1], 100)
low_300 = ta.lowest(low[1], 300)
a_tsup = math.max(nz(low_30, 0), math.max(nz(low_100, 0), nz(low_300, 0)))

high_30 = ta.highest(high[1], 30)
high_100 = ta.highest(high[1], 100)
high_300 = ta.highest(high[1], 300)
a_tres = math.min(nz(high_30, 99999), math.min(nz(high_100, 99999), nz(high_300, 99999)))

a_dsup = close == 0 ? 0 : math.abs(close - a_tsup) / close * 100
a_dres = close == 0 ? 0 : math.abs(close - a_tres) / close * 100
sr_val = atr * 2.0

ceil_w = 0
floor_w = 0
ceil_w += (nz(high_30) > close and nz(high_30) <= close + sr_val) ? 1 : 0
ceil_w += (nz(low_30) > close and nz(low_30) <= close + sr_val) ? 1 : 0
ceil_w += (nz(high_100) > close and nz(high_100) <= close + sr_val) ? 3 : 0
ceil_w += (nz(low_100) > close and nz(low_100) <= close + sr_val) ? 3 : 0
ceil_w += (nz(high_300) > close and nz(high_300) <= close + sr_val) ? 5 : 0
ceil_w += (nz(low_300) > close and nz(low_300) <= close + sr_val) ? 5 : 0

floor_w += (nz(high_30) < close and nz(high_30) >= close - sr_val) ? 1 : 0
floor_w += (nz(low_30) < close and nz(low_30) >= close - sr_val) ? 1 : 0
floor_w += (nz(high_100) < close and nz(high_100) >= close - sr_val) ? 3 : 0
floor_w += (nz(low_100) < close and nz(low_100) >= close - sr_val) ? 3 : 0
floor_w += (nz(high_300) < close and nz(high_300) >= close - sr_val) ? 5 : 0
floor_w += (nz(low_300) < close and nz(low_300) >= close - sr_val) ? 5 : 0

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

tol = atr * 0.5
is_grav_sup = a_dsup < hitbox_pct
is_grav_res = a_dres < hitbox_pct
cross_up_res = (close > a_tres) and (nz(close[1]) <= nz(a_tres[1]))
cross_dn_sup = (close < a_tsup) and (nz(close[1]) >= nz(a_tsup[1]))
cond_lock_buy_bounce = is_grav_sup and (low <= a_tsup + tol) and (close > a_tsup) and vela_verde
cond_lock_buy_break = is_grav_res and cross_up_res and high_vol and vela_verde
cond_lock_sell_reject = is_grav_res and (high >= a_tres - tol) and (close < a_tres) and vela_roja
cond_lock_sell_breakd = is_grav_sup and cross_dn_sup and vela_roja

flash_vol = (rvol > whale_factor * 0.8) and (body_size > atr * 0.3)
whale_buy = flash_vol and vela_verde
whale_sell = flash_vol and vela_roja
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

base_mask_s = retro_peak_sell or rsi_cross_dn or support_sell or div_bear
sell_score = 0.0
sell_score := (base_mask_s and retro_peak_sell) ? 50.0 : (base_mask_s and not retro_peak_sell) ? 30.0 : sell_score
sell_score += is_grav_res ? 25.0 : 0.0
sell_score += whale_memory ? 20.0 : 0.0
sell_score += dump_memory ? 15.0 : 0.0
sell_score += div_bear ? 15.0 : 0.0
sell_score += (rsi_cross_dn and not retro_peak_sell) ? 15.0 : 0.0
sell_score += (z_score > 2.0) ? 15.0 : 0.0

is_magenta = (buy_score >= 70) or retro_peak
is_magenta_sell = (sell_score >= 70) or retro_peak_sell
cond_pink_whale_buy = is_magenta and is_whale_icon

wt_cross_up = (wt1 > wt2) and (nz(wt1[1]) <= nz(wt2[1]))
wt_cross_dn = (wt1 < wt2) and (nz(wt1[1]) >= nz(wt2[1]))
wt_oversold = wt1 < -60
wt_overbought = wt1 > 60

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
    m_cond = "macro_bull" if vault.get('macro') == "Bull Only" else "not macro_bull" if vault.get('macro') == "Bear Only" else "high_vol" if vault.get('macro') == "Organic_Vol" else "squeeze_on" if vault.get('macro') == "Organic_Squeeze" else "trinity_safe" if vault.get('macro') == "Organic_Safe" else "gaussian_clean" if vault.get('macro') == "Organic_Gaussian_Clean" else "true"
    v_cond = "(adx >= adx_trend)" if vault.get('vol') == "Trend" else "(adx < adx_trend)" if vault.get('vol') == "Range" else "pump_memory" if vault.get('vol') == "Organic_Pump" else "dump_memory" if vault.get('vol') == "Organic_Dump" else "gaussian_clean" if vault.get('vol') == "Organic_Gaussian_Clean" else "true"

    b_cond = build_pine_cond(vault.get('b_team', []), vault.get('b_op', '&'))
    s_cond = build_pine_cond(vault.get('s_team', []), vault.get('s_op', '&'))
    
    ps_logic = f"""
float w_rsi = {vault.get('w_rsi',0.0):.4f}
float w_z = {vault.get('w_z',0.0):.4f}
float w_adx = {vault.get('w_adx',0.0):.4f}

float math_score = (rsi_v * w_rsi) + (z_score * w_z) + (adx * w_adx)

bool raw_buy = ({b_cond}) or (math_score > {vault.get('th_buy',99.0):.2f})
bool signal_buy = raw_buy and {m_cond} and {v_cond}

bool signal_sell = ({s_cond}) or (math_score < {vault.get('th_sell',-99.0):.2f})

float atr_tp_mult = {vault.get('atr_tp',2.0):.2f}
float atr_sl_mult = {vault.get('atr_sl',1.0):.2f}
"""

    ps_exec = """
var float locked_atr = na
var float tp_price = na
var float sl_price = na

bool just_entered = ta.change(strategy.position_size) > 0

if signal_buy and strategy.position_size == 0 and window
    strategy.entry("In", strategy.long, alert_message=wt_enter_long)

if just_entered
    locked_atr := atr[1] 
    tp_price := math.round(strategy.position_avg_price + (locked_atr * atr_tp_mult), 5)
    sl_price := math.round(strategy.position_avg_price - (locked_atr * atr_sl_mult), 5)

// 🔥 NUEVO MOTOR DE SALIDA (Control Manual Matemático para Webhooks) 🔥
bool hit_tp = high >= tp_price
bool hit_sl = low <= sl_price
bool dynamic_exit = signal_sell

if strategy.position_size > 0
    if hit_tp or hit_sl or dynamic_exit
        strategy.close("In", comment= hit_tp ? "TP_Hit" : hit_sl ? "SL_Hit" : "Dyn_Exit", alert_message=wt_exit_long)
        locked_atr := na
        tp_price := na
        sl_price := na

plotshape(signal_buy, title="COMPRA", style=shape.triangleup, location=location.belowbar, color=color.aqua, size=size.tiny)
plotshape(signal_sell, title="VENTA", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.tiny)

// Visualización de niveles para auditoría
plot(strategy.position_size > 0 ? tp_price : na, color=color.green, style=plot.style_linebr, linewidth=2, title="Take Profit")
plot(strategy.position_size > 0 ? sl_price : na, color=color.red, style=plot.style_linebr, linewidth=2, title="Stop Loss")
"""
    return ps_base + ps_indicators + ps_logic + ps_exec

# ==========================================
# 🛑 7. BUCLES DE EJECUCIÓN GLOBALES Y PROFUNDOS
# ==========================================
st.session_state['is_calib_mode'] = is_calib_mode

if st.session_state.get('run_global', False):
    time.sleep(0.1) 
    if len(st.session_state['global_queue']) > 0:
        s_id = st.session_state['global_queue'].pop(0)
        ph_holograma.markdown(f"<div style='text-align:center; padding: 20px; background: rgba(0,0,0,0.8); border: 2px solid cyan; border-radius: 10px;'><h2 style='color:cyan;'>⚙️ Forjando ADN: {s_id}...</h2><h4 style='color:lime;'>Quedan {len(st.session_state['global_queue'])} mutantes en incubación.</h4></div>", unsafe_allow_html=True)
        time.sleep(0.1)
        
        v = get_safe_vault(s_id)
        buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
        buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
        
        bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, float(v.get('reinv', 20.0)), float(v.get('ado',4.0)), dias_reales, buy_hold_money, epochs=global_epochs, cur_net=float(v.get('net',-float('inf'))), cur_fit=float(v.get('fit',-float('inf'))), deep_info=None, greed_factor=st.session_state.get(f'greed_{s_id}', 0.8))
        
        st.rerun()
    else:
        st.session_state['run_global'] = False
        ph_holograma.empty()
        st.sidebar.success("✅ ¡Incubación Genética Completada!")
        time.sleep(2)
        if st.session_state.get('ai_algos'):
            st.session_state['selected_mutant'] = st.session_state['ai_algos'][-1]
        st.rerun()

deep_state = st.session_state.get('deep_opt_state', {})
if deep_state and not deep_state.get('paused', False) and deep_state.get('current_epoch', 0) < deep_state.get('target_epochs', 0):
    time.sleep(0.1) 
    
    chunk = 50 
    if deep_state['target_epochs'] - deep_state['current_epoch'] < chunk:
        chunk = deep_state['target_epochs'] - deep_state['current_epoch']
        
    s_id = deep_state['s_id']
    v = get_safe_vault(s_id)
    buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
    buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
    
    deep_info = {'current': deep_state['current_epoch'], 'total': deep_state['target_epochs'], 'start_time': deep_state.get('start_time', time.time())}
    
    bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, float(v.get('reinv', 20.0)), float(v.get('ado',4.0)), dias_reales, buy_hold_money, epochs=chunk, cur_net=float(v.get('net',-float('inf'))), cur_fit=float(v.get('fit',-float('inf'))), deep_info=deep_info, greed_factor=st.session_state.get(f'greed_{s_id}', 0.8))
    
    st.session_state['deep_opt_state']['current_epoch'] += chunk
    
    if st.session_state['deep_opt_state']['current_epoch'] >= deep_state['target_epochs']:
        st.session_state['deep_opt_state']['paused'] = True
        st.session_state['selected_mutant'] = s_id
        ph_holograma.empty()
        st.sidebar.success(f"🌌 ¡FORJA PROFUNDA COMPLETADA PARA {s_id}!")
        time.sleep(2)
        
    st.rerun()

# ==========================================
# 🛑 8. PANEL DE RENDERIZADO VISUAL Y SALÓN DE LA FAMA
# ==========================================
st.title("🛡️ GENESIS LAB - The Omni-Brain")

# 🔥 EL SALÓN DE LA FAMA AHORA ESTÁ SIEMPRE ABIERTO POR DEFECTO 🔥
with st.expander("🏆 SALÓN DE LA FAMA GENÉTICA (Ordenado por Rentabilidad Neta)", expanded=True):
    leaderboard_data = []
    for s in estrategias:
        v = get_safe_vault(s)
        fit = v.get('fit', -float('inf'))
        opt_str = "✅" if fit != -float('inf') else "➖ No Opt"
        net_val = v.get('net', 0.0)
        leaderboard_data.append({"Mutante": s, "Neto_Num": net_val, "Rentabilidad": f"${net_val:,.2f}", "WinRate": f"{v.get('winrate', 0):.1f}%", "Estado": opt_str})
    
    leaderboard_data.sort(key=lambda x: x['Neto_Num'], reverse=True)
    
    for rank, item in enumerate(leaderboard_data):
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"**#{rank+1} | {item['Mutante']}** -> Profit: `{item['Rentabilidad']}` | WR: `{item['WinRate']}` | Estado: {item['Estado']}")
        if col2.button("👉 CARGAR SCRIPT", key=f"btn_load_script_{item['Mutante']}_{rank}"):
            st.session_state['selected_mutant'] = item['Mutante']
            st.rerun()
    st.markdown("---")

st.markdown("<h3 style='text-align: center; color: #00FF00;'>🎡 CARRUSEL DE MUTANTES IA</h3>", unsafe_allow_html=True)
tab_names = list(tab_id_map.keys())

if len(tab_names) > 0:
    default_idx = len(tab_names) - 1
    if 'selected_mutant' in st.session_state:
        target_name = f"🤖 {st.session_state['selected_mutant']}"
        if target_name in tab_names:
            default_idx = tab_names.index(target_name)

    selected_tab_name = st.selectbox("Selecciona un Espécimen:", tab_names, index=default_idx)
    s_id = tab_id_map[selected_tab_name]
    is_opt = st.session_state.get(f'opt_status_{s_id}', False)
    opt_badge = "<span style='color: lime;'>✅ ADN OPTIMIZADO</span>" if is_opt else "<span style='color: gray;'>➖ ADN VIRGEN</span>"
    
    vault = get_safe_vault(s_id) 

    st.markdown(f"### {selected_tab_name} {opt_badge}", unsafe_allow_html=True)

    if is_calib_mode:
        st.warning("⚠️ MODO CALIBRADOR ACTIVO. La IA y los indicadores están apagados. Mostrando trades fijos cada 50 barras con TP/SL exacto de 0.2%. Úsalo para probar CCXT vs TradingView.")
    else:
        with st.expander("🧬 VER ADN DEL MUTANTE Y ARMAS TÁCTICAS", expanded=True):
            st.markdown(f"**🟢 Escuadrón de Compra:** {', '.join(vault.get('b_team', []))} (Táctica: {vault.get('b_op', '&')})")
            st.markdown(f"**🔴 Escuadrón de Venta:** {', '.join(vault.get('s_team', []))} (Táctica: {vault.get('s_op', '&')})")
            st.markdown(f"**🌍 Clima Macro:** `{vault.get('macro', '')}` | **🌪️ Clima Volatilidad:** `{vault.get('vol', '')}`")
            st.markdown(f"**🎛️ Pesos del Perceptrón:** RSI: `{vault.get('w_rsi',0):.2f}` | Z-Score: `{vault.get('w_z',0):.2f}` | ADX: `{vault.get('w_adx',0):.2f}`")
            st.markdown(f"**📏 Gatillos Sensibles:** Buy > `{vault.get('th_buy',0):.2f}` | Sell < `{vault.get('th_sell',0):.2f}`")
            st.markdown(f"**🎯 Camaleón ATR:** TP: `{vault.get('atr_tp',0):.2f}x` | SL: `{vault.get('atr_sl',0):.2f}x`")

    c_ia1, c_ia2, c_ia3 = st.columns([1, 1, 3])
    
    ado_val_ui = float(vault.get('ado', 4.0)) if vault.get('ado') is not None else 4.0
    reinv_val_ui = float(vault.get('reinv', 20.0)) if vault.get('reinv') is not None else 20.0

    ado_ui = c_ia1.slider("🎯 Target ADO (IA Override)", 0.0, 100.0, value=ado_val_ui, key=f"ui_{s_id}_ado_w", step=0.5)
    reinv_ui = c_ia2.slider("💵 Reinversión % (IA Override)", 0.0, 100.0, value=reinv_val_ui, key=f"ui_{s_id}_reinv_w", step=5.0)

    st.session_state[f'champion_{s_id}']['ado'] = ado_ui
    st.session_state[f'champion_{s_id}']['reinv'] = reinv_ui 
    
    ps_buy_pct = reinv_ui
    ps_sell_pct = 100

    c_btn1, c_btn2 = c_ia3.columns(2)
    if c_btn1.button(f"🚀 FORJAR RÁPIDO ({global_epochs*1000})", type="primary", key=f"btn_opt_{s_id}"):
        st.session_state['abort_opt'] = False
        st.session_state['global_queue'] = [s_id]
        st.session_state['run_global'] = True
        st.rerun()

    if c_btn2.button(f"🌌 ACTIVAR FORJA PROFUNDA", type="secondary", key=f"btn_deep_{s_id}"):
        st.session_state['abort_opt'] = False
        st.session_state['deep_opt_state'] = {'s_id': s_id, 'target_epochs': deep_epochs_target, 'current_epoch': 0, 'paused': False, 'start_time': time.time()}
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
            ws = len(exs[exs['Ganancia_$'] > 0])
            wr = (ws / tt) * 100
            gpp = exs[exs['Ganancia_$'] > 0]['Ganancia_$'].sum()
            gll = abs(exs[exs['Ganancia_$'] < 0]['Ganancia_$'].sum())
            pf_val = gpp / gll if gll > 0 else float('inf')

    ado_val = tt / dias_reales if dias_reales > 0 else 0.0
    mdd = abs((((pd.Series(eq_curve) - pd.Series(eq_curve).cummax()) / pd.Series(eq_curve).cummax()) * 100).min())

    col_kpi, col_radar = st.columns([3, 2])
    
    with col_kpi:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Profit", f"${eq_curve[-1]-capital_inicial:,.2f}", f"{ret_pct:.2f}%")
        c2.metric("ALPHA", f"{alpha_pct:.2f}%", delta_color="normal" if alpha_pct > 0 else "inverse")
        c3.metric("Trades", f"{tt}", f"ADO: {ado_val:.2f}")
        c4.metric("Win Rate", f"{wr:.1f}%")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Profit Factor", f"{pf_val:.2f}x")
        c6.metric("Drawdown", f"{mdd:.2f}%", delta_color="inverse")
        c7.metric("Comisiones", f"${total_comms:,.2f}", delta_color="inverse")
        c8.metric("Hold Return", f"{buy_hold_ret:.2f}%")
        
    with col_radar:
        st.plotly_chart(generar_radar(wr, pf_val, ado_val, ret_pct, alpha_pct, ado_val_ui), use_container_width=True)

    # --- EJECUCIÓN DEL TEST DE MONTE CARLO ---
    mc_curves, risk_of_ruin = simular_monte_carlo(t_log, capital_inicial, 500)

    st.markdown("<br>", unsafe_allow_html=True)
    c_mc1, c_mc2 = st.columns([1, 4])

    c_mc1.markdown("### 🎲 Test de Estrés (Monte Carlo)")
    if risk_of_ruin > 10.0:
        c_mc1.error(f"⚠️ RIESGO DE RUINA: {risk_of_ruin:.1f}%")
        c_mc1.caption("La IA es frágil ante malas rachas. Se recomienda re-forjar.")
    elif risk_of_ruin > 0.0:
        c_mc1.warning(f"⚠️ RIESGO DE RUINA: {risk_of_ruin:.1f}%")
        c_mc1.caption("Riesgo moderado. Gestionar tamaño de posición.")
    else:
        c_mc1.success(f"🛡️ RIESGO DE RUINA: {risk_of_ruin:.1f}%")
        c_mc1.caption("Mutante Anti-Frágil. Listo para la guerra.")

    if mc_curves is not None:
        fig_mc = go.Figure()
        for i in range(min(50, len(mc_curves))):
            fig_mc.add_trace(go.Scatter(y=mc_curves[i], mode='lines', line=dict(color='rgba(255, 255, 255, 0.1)', width=1), hoverinfo='skip'))
        real_curve = [capital_inicial] + [capital_inicial + sum([t['Ganancia_$'] for t in t_log if t['Tipo'] in ['TP', 'SL', 'DYN_WIN', 'DYN_LOSS']][:k+1]) for k in range(len([t for t in t_log if t['Tipo'] in ['TP', 'SL', 'DYN_WIN', 'DYN_LOSS']]))]
        fig_mc.add_trace(go.Scatter(y=real_curve, mode='lines', name='Histórico Real', line=dict(color='gold', width=3)))
        fig_mc.update_layout(title='Proyección Estocástica de Equidad (500 Realidades Alternativas)', template='plotly_dark', height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, yaxis_title='Capital ($)')
        c_mc2.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("📝 CÓDIGO DE TRASPLANTE A TRADINGVIEW (PINE SCRIPT)", expanded=False):
        st.info("Traducción Matemática Idéntica. Ejecución Cuántica con Cero Repainting Activa.")
        st.code(generar_pine_script(s_id, vault, ticker.split('/')[0], iv_download, ps_buy_pct, ps_sell_pct, comision_pct, df_strat.index[0], is_calib_mode), language="pine")

    st.markdown("---")
    st.info("🖱️ **TIP GRÁFICO:** Si las velas se ven aplanadas, haz **Doble Clic** dentro del gráfico. Observa la línea naranja punteada que divide el entrenamiento y la validación.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], name="Precio", increasing_line_color='cyan', decreasing_line_color='magenta'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_50'], mode='lines', name='Río Center (EMA 50)', line=dict(color='yellow', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_200'], mode='lines', name='Macro Trend (EMA 200)', line=dict(color='purple', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBU'], mode='lines', name='Squeeze Top (BBU)', line=dict(color='rgba(128,128,128,0.5)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBL'], mode='lines', name='Squeeze Bot (BBL)', line=dict(color='rgba(128,128,128,0.5)', width=1)), row=1, col=1)

    if not dftr.empty:
        ents = dftr[dftr['Tipo'] == 'ENTRY']
        fig.add_trace(go.Scatter(x=ents['Fecha'], y=ents['Precio'], mode='markers', name='COMPRA', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=2, color='white'))), row=1, col=1)
        wins = dftr[dftr['Ganancia_$'] > 0]
        fig.add_trace(go.Scatter(x=wins['Fecha'], y=wins['Precio'], mode='markers', name='WIN', marker=dict(symbol='triangle-down', color='#00FF00', size=14, line=dict(width=2, color='white'))), row=1, col=1)
        loss = dftr[dftr['Ganancia_$'] < 0]
        fig.add_trace(go.Scatter(x=loss['Fecha'], y=loss['Precio'], mode='markers', name='LOSS', marker=dict(symbol='triangle-down', color='#FF0000', size=14, line=dict(width=2, color='white'))), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Total_Portfolio'], mode='lines', name='Equidad', line=dict(color='#00FF00', width=3)), row=2, col=1)
    
    split_idx = int(len(df_strat) * 0.70)
    if split_idx < len(df_strat):
        split_date_str = df_strat.index[split_idx].strftime('%Y-%m-%d %H:%M:%S')
        fig.add_vline(x=split_date_str, line_width=2, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_vline(x=split_date_str, line_width=2, line_dash="dash", line_color="orange", row=2, col=1)
        fig.add_annotation(x=split_date_str, y=1.05, yref="paper", text="⬅️ IS | OOS ➡️", showarrow=False, font=dict(color="orange"), xanchor="center", row=1, col=1)

    y_min_force = df_strat['Low'].min() * 0.98
    y_max_force = df_strat['High'].max() * 1.02
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False, side="right", range=[y_min_force, y_max_force], row=1, col=1)
    fig.update_yaxes(fixedrange=False, side="right", row=2, col=1)
    fig.update_layout(template='plotly_dark', height=800, xaxis_rangeslider_visible=False, dragmode='pan', hovermode='x unified', margin=dict(l=10, r=50, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{s_id}", config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})
