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
import glob
from datetime import datetime, timedelta

# --- MOTOR DE HIPER-VELOCIDAD (NUMBA JIT) ---
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

st.set_page_config(page_title="ROCKET PROTOCOL | Predator Lab", layout="wide", initial_sidebar_state="expanded")
ph_holograma = st.empty()

APP_VERSION = 'V320_PREDATOR_ADO_BONUS'

# ==========================================
# ☢️ PROTOCOLO DE PURGA Y RECUPERACIÓN
# ==========================================
def purga_nuclear():
    st.cache_data.clear()
    st.session_state.clear()
    for f in glob.glob("champ_*.json"):
        try: os.remove(f)
        except: pass
    st.session_state['app_version'] = APP_VERSION
    st.rerun()

if st.session_state.get('app_version') != APP_VERSION:
    purga_nuclear()

if 'ai_algos' not in st.session_state or len(st.session_state['ai_algos']) == 0: 
    loaded_algos = [f.replace("champ_", "").replace(".json", "") for f in glob.glob("champ_*.json")]
    if not loaded_algos: loaded_algos = [f"PREDATOR_{random.randint(100, 999)}"]
    st.session_state['ai_algos'] = list(dict.fromkeys(loaded_algos))

estrategias = st.session_state['ai_algos']
tab_id_map = {f"🤖 {ai_id}": ai_id for ai_id in estrategias}

# ==========================================
# 🧬 DICCIONARIOS GENÉTICOS (FRANCOTIRADOR V320)
# ==========================================
todas_las_armas_b = [
    'Q_Pink_Whale_Buy', 'Q_Nuclear_Buy', 'Q_Climax_Buy', 'Q_Early_Buy',
    'Q_Defcon_Buy', 'Q_Neon_Up', 'Q_Therm_Bounce', 'Q_Therm_Vacuum',
    'Q_Lock_Bounce', 'Q_Lock_Break', 'Q_Rebound_Buy',
    'Q_River_Push_Buy', 'Q_River_Entry_Buy',
    'Q_Div_Bull_Buy', 'Q_WT_Oversold_Buy',
    'PA_Engulfing_Buy', 'PA_Pinbar_Buy'
]
todas_las_armas_s = [
    'Q_Nuclear_Sell', 'Q_Climax_Sell', 'Q_Early_Sell',
    'Q_Defcon_Sell', 'Q_Neon_Dn', 'Q_Therm_Wall_Sell', 'Q_Therm_Panic_Sell',
    'Q_Lock_Reject', 'Q_Lock_Breakd', 'Q_Pullback_Sell',
    'Q_River_Push_Sell', 'Q_River_Entry_Sell',
    'Q_Div_Bear_Sell', 'Q_WT_Overbought_Sell',
    'PA_Engulfing_Sell', 'PA_Pinbar_Sell'
]

pine_map = {
    'Q_Pink_Whale_Buy': 'cond_pink_whale_buy', 'Q_Nuclear_Buy': 'nuclear_buy', 'Q_Climax_Buy': 'climax_buy', 'Q_Early_Buy': 'climax_buy_early',
    'Q_Defcon_Buy': 'cond_defcon_buy', 'Q_Neon_Up': 'neon_up_trig', 'Q_Therm_Bounce': 'cond_therm_buy_bounce', 'Q_Therm_Vacuum': 'cond_therm_buy_vacuum',
    'Q_Lock_Bounce': 'cond_lock_buy_bounce', 'Q_Lock_Break': 'cond_lock_buy_break', 'Q_Rebound_Buy': 'normal_reb_buy',
    'Q_River_Push_Buy': 'river_push_up', 'Q_River_Entry_Buy': 'river_entry_up',
    'Q_Div_Bull_Buy': 'div_bull_trig', 'Q_WT_Oversold_Buy': 'wt_oversold_trig',
    
    'Q_Nuclear_Sell': 'nuclear_sell', 'Q_Climax_Sell': 'climax_sell', 'Q_Early_Sell': 'climax_sell_early',
    'Q_Defcon_Sell': 'cond_defcon_sell', 'Q_Neon_Dn': 'neon_dn_trig', 'Q_Therm_Wall_Sell': 'cond_therm_sell_wall', 'Q_Therm_Panic_Sell': 'cond_therm_sell_panic',
    'Q_Lock_Reject': 'cond_lock_sell_reject', 'Q_Lock_Breakd': 'cond_lock_sell_breakd', 'Q_Pullback_Sell': 'normal_reb_sell',
    'Q_River_Push_Sell': 'river_push_dn', 'Q_River_Entry_Sell': 'river_entry_dn',
    'Q_Div_Bear_Sell': 'div_bear_trig', 'Q_WT_Overbought_Sell': 'wt_overbought_trig',

    'PA_Engulfing_Buy': 'pa_engulfing_buy', 'PA_Engulfing_Sell': 'pa_engulfing_sell', 
    'PA_Pinbar_Buy': 'pa_pinbar_buy', 'PA_Pinbar_Sell': 'pa_pinbar_sell'
}

def get_default_dna():
    return {
        'b_team': random.sample(todas_las_armas_b, 2), 
        's_team': random.sample(todas_las_armas_s, 2), 
        'b_op': '&', 's_op': '|', 'hitbox': 1.5, 'therm_w': 4.0, 
        'adx_th': 25.0, 'whale_f': 2.5, 'ado': 4.0, 'reinv': 20.0, 'fit': -float('inf'), 
        'net': 0.0, 'net_is': 0.0, 'net_oos': 0.0, 'winrate': 0.0, 'pf': 0.0, 'nt': 0, 
        'tp_pct': 25.0, 'sl_pct': 10.0
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
# 🧠 FUNCIONES MATEMÁTICAS C-SPEED
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

@njit(fastmath=True)
def simular_core_rapido(h_arr, l_arr, c_arr, o_arr, f_buy, f_sell, tp_pct_val, sl_pct_val, cap_ini, com_pct, invest_pct, slippage_pct, is_calib):
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
                hit_sl = l_arr[i] <= sl_p; hit_tp = h_arr[i] >= tp_p
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
                    tp_p = round(p_ent * 1.002, 5)
                    sl_p = round(p_ent * 0.998, 5)
                else:
                    tp_p = round(p_ent * (1 + (tp_pct_val / 100.0)), 5)
                    sl_p = round(p_ent * (1 - (sl_pct_val / 100.0)), 5)
                
                en_pos = True; bars_in_trade = 0
                
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    wr = (wins / num_trades) * 100.0 if num_trades > 0 else 0.0
    return (cap_act - cap_ini), pf, num_trades, max_dd, wr

def simular_visual(df_sim, cap_ini, invest_pct, com_pct, slippage_pct=0.0, is_calib=False):
    registro_trades = []; n = len(df_sim); curva = np.full(n, cap_ini, dtype=float)
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
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
                    exec_p = sl_p if c_arr[i] >= o_arr[i] else tp_p; ret = (exec_p - p_ent) / p_ent
                elif hit_sl:
                    exec_p = sl_p if o_arr[i] > sl_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent
                elif hit_tp:
                    exec_p = tp_p if o_arr[i] < tp_p else o_arr[i]; ret = (exec_p - p_ent) / p_ent
                    
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
                    tp_p = np.round(p_ent * 1.002, 5)
                    sl_p = np.round(p_ent * 0.998, 5)
                else:
                    tp_p = np.round(p_ent * (1 + (tp_arr[i] / 100.0)), 5)
                    sl_p = np.round(p_ent * (1 - (sl_arr[i] / 100.0)), 5)
                
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
# 🌍 SIDEBAR UI & DATOS
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🧬 PREDATOR LAB V320</h2>", unsafe_allow_html=True)

with st.sidebar.expander("⚙️ SISTEMA (EMERGENCIAS)", expanded=True):
    if st.button("🛑 ABORTAR RUN GLOBAL", use_container_width=True, key="btn_abort"):
        st.session_state['abort_opt'] = True; st.session_state['global_queue'] = []; st.session_state['run_global'] = False; st.rerun()
    if st.button("☢️ PURGA NUCLEAR (RESET TOTAL)", use_container_width=True, key="btn_purge"): 
        purga_nuclear()

with st.sidebar.expander("🌍 DATOS Y EXCHANGE", expanded=False):
    exchange_sel = st.selectbox("🏦 Exchange", ["coinbase", "kucoin", "kraken", "binance"], index=0)
    ticker = st.text_input("Símbolo Exacto", value="IOTX/USDT")
    utc_offset = st.number_input("🌍 Zona Horaria", value=-5.0, step=0.5)
    intervalos = {"1 Minuto": "1m", "5 Minutos": "5m", "15 Minutos": "15m", "30 Minutos": "30m", "1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d"}
    intervalo_sel = st.selectbox("Temporalidad", list(intervalos.keys()), index=2) 
    iv_download = intervalos[intervalo_sel]
    hoy = datetime.today().date()
    start_date, end_date = st.slider("📅 Scope (Fechas)", min_value=hoy - timedelta(days=1500), max_value=hoy, value=(hoy - timedelta(days=200), hoy), format="YYYY-MM-DD")
    
    if st.button("📥 DESCARGAR MATRIX DE DATOS", use_container_width=True, type="primary"):
        st.session_state['data_params'] = {'ex': exchange_sel, 'sym': ticker, 'start': start_date, 'end': end_date, 'iv': iv_download, 'offset': utc_offset, 'micro': (iv_download in ["1m", "5m", "15m", "30m"])}
        st.rerun()

with st.sidebar.expander("💼 CAPITAL Y COMISIONES", expanded=False):
    capital_inicial = st.number_input("Capital Inicial (USD)", value=1000.0, step=100.0)
    comision_pct = st.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0 
    is_calib_mode = st.checkbox("🛠️ MODO CALIBRACIÓN TV", value=False) 

with st.sidebar.expander("🤖 INTELIGENCIA Y FORJA", expanded=False):
    greed_factor = st.slider("Nivel de Avaricia", 0.0, 1.0, 0.8, 0.1)
    global_epochs = st.slider("Épocas Rápidas (x1000)", 1, 1000, 50)
    deep_epochs_target = st.number_input("Objetivo Épocas Profundas", min_value=10000, max_value=10000000, value=100000, step=10000)
    target_strats = st.multiselect("🎯 Mutantes a Forjar:", estrategias, default=estrategias)
    if st.button(f"🧠 DEEP MINE GLOBAL", type="primary", use_container_width=True, key="btn_global"):
        st.session_state['global_queue'] = target_strats.copy(); st.session_state['abort_opt'] = False; st.session_state['run_global'] = True; st.rerun()
    if st.button("🤖 CREAR NUEVO MUTANTE IA", type="secondary", use_container_width=True, key="btn_mutant"):
        new_id = f"PREDATOR_{int(time.time())}_{random.randint(10, 99)}"
        if new_id not in st.session_state['ai_algos']:
            st.session_state['ai_algos'].append(new_id); get_safe_vault(new_id); st.session_state['global_queue'] = [new_id]; st.session_state['run_global'] = True; st.rerun()

if 'data_params' not in st.session_state:
    st.info("👈 Por favor, configura los datos en el menú lateral y haz clic en **'📥 DESCARGAR MATRIX DE DATOS'** para iniciar el Laboratorio Quant.")
    st.stop()

dp = st.session_state['data_params']

# ==========================================
# 🛑 EXTRACCIÓN Y WARM-UP (V320 MATH)
# ==========================================
@st.cache_data(ttl=3600, show_spinner="📡 Sincronizando Matrix V320 Predator...")
def cargar_matriz(exchange_id, sym, start, end, iv_down, offset, is_micro, version_key):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        warmup_days = 40 if is_micro else 150
        start_ts = int(datetime.combine(start - timedelta(days=warmup_days), datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        
        all_ohlcv, current_ts, error_count = [], start_ts, 0
        req_limit = 1000
        if 'coinbase' in exchange_id.lower(): req_limit = 300

        while current_ts < end_ts:
            try: ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=req_limit); error_count = 0 
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
        
        freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'}
        pd_freq = freq_map.get(iv_down, '15min')
        df = df.resample(pd_freq).asfreq()
        df['Close'] = df['Close'].ffill()
        df['Open'] = df['Open'].fillna(df['Close']); df['High'] = df['High'].fillna(df['Close']); df['Low'] = df['Low'].fillna(df['Close'])
        df['Volume'] = df['Volume'].fillna(0)
            
        a_h, a_l, a_c, a_o = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
        
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean() 
        df['Vol_MA_100'] = df['Volume'].rolling(window=100).mean()
        df['RVol'] = df['Volume'] / np.where(df['Vol_MA_100'] == 0, 1, df['Vol_MA_100'])
        
        tr = np.zeros_like(a_c); tr[0] = a_h[0] - a_l[0]
        for i in range(1, len(a_c)): tr[i] = max(a_h[i] - a_l[i], abs(a_h[i] - a_c[i-1]), abs(a_l[i] - a_c[i-1]))
        df['ATR'] = rma_pine(tr, 14); df['ATR'] = df['ATR'].fillna(df['High']-df['Low'])
        
        delta = np.zeros_like(a_c); delta[1:] = a_c[1:] - a_c[:-1]
        u = np.where(delta > 0, delta, 0.0); d = np.where(delta < 0, -delta, 0.0)
        rs_u = rma_pine(u, 14); rs_d = rma_pine(d, 14); rs = rs_u / np.where(rs_d == 0, 1e-10, rs_d)
        df['RSI'] = np.where(rs_d == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
        df['RSI_MA'] = df['RSI'].rolling(14).mean()
        
        upm = np.zeros_like(a_h); upm[1:] = a_h[1:] - a_h[:-1]
        downm = np.zeros_like(a_l); downm[1:] = a_l[:-1] - a_l[1:]
        plusDM = np.where((upm > downm) & (upm > 0), upm, 0.0); minusDM = np.where((downm > upm) & (downm > 0), downm, 0.0)
        trur = rma_pine(tr, 14); plus = 100 * rma_pine(plusDM, 14) / trur; minus = 100 * rma_pine(minusDM, 14) / trur
        sum_dm = plus + minus; dx = 100 * np.abs(plus - minus) / np.where(sum_dm == 0, 1, sum_dm)
        df['ADX'] = rma_pine(dx, 14)
        
        ap = (df['High'] + df['Low'] + df['Close']) / 3.0
        esa = ap.ewm(span=10, adjust=False).mean(); d_wt = (ap - esa).abs().ewm(span=10, adjust=False).mean()
        df['WT1'] = ((ap - esa) / (0.015 * np.where(d_wt == 0, 1, d_wt))).ewm(span=21, adjust=False).mean()
        df['WT2'] = df['WT1'].rolling(4).mean()
        
        df['Basis'] = df['Close'].rolling(20).mean(); dev = df['Close'].rolling(20).std(ddof=0)
        df['BBU'] = df['Basis'] + (2.0 * dev); df['BBL'] = df['Basis'] - (2.0 * dev)
        df['BB_Width'] = (df['BBU'] - df['BBL']) / np.where(df['Basis'] == 0, 1, df['Basis'])
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
        
        df['PA_Engulfing_Buy'] = (df['Vela_Verde']) & (df['Vela_Roja'].shift(1)) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
        df['PA_Engulfing_Sell'] = (df['Vela_Roja']) & (df['Vela_Verde'].shift(1)) & (df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))
        df['PA_Pinbar_Buy'] = (df['lower_wick'] > df['body_size'] * 2.5) & (df['upper_wick'] < df['body_size'])
        df['PA_Pinbar_Sell'] = (df['upper_wick'] > df['body_size'] * 2.5) & (df['lower_wick'] < df['body_size'])
        df['PA_3_Soldiers'] = (df['Vela_Verde']) & (df['Vela_Verde'].shift(1)) & (df['Vela_Verde'].shift(2)) & (df['Close'] > df['Close'].shift(1)) & (df['Close'].shift(1) > df['Close'].shift(2))
        df['PA_3_Crows'] = (df['Vela_Roja']) & (df['Vela_Roja'].shift(1)) & (df['Vela_Roja'].shift(2)) & (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) < df['Close'].shift(2))

        df['PL100_L'] = df['Low'].shift(1).rolling(100, min_periods=1).min(); df['PH100_L'] = df['High'].shift(1).rolling(100, min_periods=1).max()
        df['PL300_L'] = df['Low'].shift(1).rolling(300, min_periods=1).min(); df['PH300_L'] = df['High'].shift(1).rolling(300, min_periods=1).max()
        df['PL800_L'] = df['Low'].shift(1).rolling(800, min_periods=1).min(); df['PH800_L'] = df['High'].shift(1).rolling(800, min_periods=1).max()
        df['Macro_Bull'] = df['Close'] >= df['EMA_200'] 
        
        target_start = pd.to_datetime(datetime.combine(start, datetime.min.time())) + timedelta(hours=offset)
        df = df[df.index >= target_start]
        gc.collect(); return df, "OK"
    except Exception as e: return pd.DataFrame(), f"❌ ERROR FATAL: {str(e)}"

df_global, status_api = cargar_matriz(dp['ex'], dp['sym'], dp['start'], dp['end'], dp['iv'], dp['offset'], dp['micro'], st.session_state['app_version'])
if df_global.empty: st.error(status_api); st.stop()
dias_reales = max((df_global.index[-1] - df_global.index[0]).days, 1)
st.success(f"📊 Matrix V320 Extraída: **{len(df_global):,} velas** | **{dias_reales} días**")

# ==========================================
# 🧠 CREACIÓN DE MATRICES NUMPY V320 CORE
# ==========================================
a_c, a_o, a_h, a_l = df_global['Close'].values, df_global['Open'].values, df_global['High'].values, df_global['Low'].values
a_rsi, a_rsi_ma, a_adx = df_global['RSI'].values, df_global['RSI_MA'].values, df_global['ADX'].values
a_bbl, a_bbu = df_global['BBL'].values, df_global['BBU'].values
a_wt1, a_wt2 = df_global['WT1'].values, df_global['WT2'].values
a_ema20, a_atr = df_global['EMA_20'].values, df_global['ATR'].values
a_rvol = df_global['RVol'].values
a_vv, a_vr = df_global['Vela_Verde'].values, df_global['Vela_Roja'].values
a_sqz_on = df_global['Squeeze_On'].values
a_bb_delta, a_bb_delta_avg = df_global['BB_Delta'].values, df_global['BB_Delta_Avg'].values
a_zscore, a_rsi_bb_b, a_rsi_bb_d = df_global['Z_Score'].values, df_global['RSI_BB_Basis'].values, df_global['RSI_BB_Dev'].values
a_lw, a_uw, a_bs = df_global['lower_wick'].values, df_global['upper_wick'].values, df_global['body_size'].values
a_mb = df_global['Macro_Bull'].values

a_pa_eng_b, a_pa_eng_s = df_global['PA_Engulfing_Buy'].values, df_global['PA_Engulfing_Sell'].values
a_pa_pin_b, a_pa_pin_s = df_global['PA_Pinbar_Buy'].values, df_global['PA_Pinbar_Sell'].values
a_pa_3sol_b, a_pa_3cro_s = df_global['PA_3_Soldiers'].values, df_global['PA_3_Crows'].values

a_pl100_l, a_ph100_l = df_global['PL100_L'].fillna(0).values, df_global['PH100_L'].fillna(99999).values
a_pl300_l, a_ph300_l = df_global['PL300_L'].fillna(0).values, df_global['PH300_L'].fillna(99999).values
a_pl800_l, a_ph800_l = df_global['PL800_L'].fillna(0).values, df_global['PH800_L'].fillna(99999).values

a_c_s1, a_o_s1, a_l_s1 = npshift(a_c, 1, 0.0), npshift(a_o, 1, 0.0), npshift(a_l, 1, 0.0)
a_l_s5, a_h_s1, a_h_s5 = npshift(a_l, 5, 0.0), npshift(a_h, 1, 0.0), npshift(a_h, 5, 0.0)
a_rsi_s1, a_rsi_s5 = npshift(a_rsi, 1, 50.0), npshift(a_rsi, 5, 50.0)
a_wt1_s1, a_wt2_s1 = npshift(a_wt1, 1, 0.0), npshift(a_wt2, 1, 0.0)

def calcular_señales_numpy(hitbox, therm_w, adx_th, whale_f):
    n_len = len(a_c); s_dict = {}
    
    a_tsup = np.maximum(a_pl100_l, np.maximum(a_pl300_l, a_pl800_l))
    a_tres = np.minimum(a_ph100_l, np.minimum(a_ph300_l, a_ph800_l))
    a_dsup = np.where(a_c == 0, 0, np.abs(a_c - a_tsup) / a_c * 100)
    a_dres = np.where(a_c == 0, 0, np.abs(a_c - a_tres) / a_c * 100)
    sr_val = a_atr * 2.0

    ceil_w = np.where((a_ph100_l > a_c) & (a_ph100_l <= a_c + sr_val), 3, 0) + np.where((a_pl100_l > a_c) & (a_pl100_l <= a_c + sr_val), 3, 0) + np.where((a_ph300_l > a_c) & (a_ph300_l <= a_c + sr_val), 5, 0) + np.where((a_pl300_l > a_c) & (a_pl300_l <= a_c + sr_val), 5, 0) + np.where((a_ph800_l > a_c) & (a_ph800_l <= a_c + sr_val), 8, 0) + np.where((a_pl800_l > a_c) & (a_pl800_l <= a_c + sr_val), 8, 0)
    floor_w = np.where((a_ph100_l < a_c) & (a_ph100_l >= a_c - sr_val), 3, 0) + np.where((a_pl100_l < a_c) & (a_pl100_l >= a_c - sr_val), 3, 0) + np.where((a_ph300_l < a_c) & (a_ph300_l >= a_c - sr_val), 5, 0) + np.where((a_pl300_l < a_c) & (a_pl300_l >= a_c - sr_val), 5, 0) + np.where((a_ph800_l < a_c) & (a_ph800_l >= a_c - sr_val), 8, 0) + np.where((a_pl800_l < a_c) & (a_pl800_l >= a_c - sr_val), 8, 0)

    is_abyss = floor_w == 0
    is_hard_wall = ceil_w >= therm_w
    is_struct_sup = floor_w > 0
    is_struct_res = ceil_w > 0
    
    rsi_cross_up = (a_rsi > a_rsi_ma) & (a_rsi_s1 <= npshift(a_rsi_ma, 1))
    rsi_cross_dn = (a_rsi < a_rsi_ma) & (a_rsi_s1 >= npshift(a_rsi_ma, 1))
    
    rsi_vel = a_rsi - a_rsi_s1
    river_w = a_atr * (1.2 + (np.abs(rsi_vel) / 10.0))
    river_top = a_ema20 + (river_w * 0.5)
    river_bot = a_ema20 - (river_w * 0.5)
    
    neon_up = a_sqz_on & (a_c >= a_bbu * 0.999) & a_vv
    neon_up_trig = neon_up & ~npshift_bool(neon_up, 1) 
    
    neon_dn = a_sqz_on & (a_c <= a_bbl * 1.001) & a_vr
    neon_dn_trig = neon_dn & ~npshift_bool(neon_dn, 1) 
    
    river_push_up = (a_l <= river_top) & (a_c > river_top) & ~neon_dn
    river_push_dn = (a_h >= river_bot) & (a_c < river_bot) & ~neon_up
    river_entry_up = (a_c > river_bot) & (a_c_s1 <= npshift(river_bot, 1))
    river_entry_dn = (a_c < river_top) & (a_c_s1 >= npshift(river_top, 1))
    
    defcon_level = np.full(n_len, 5)
    m4 = neon_up | neon_dn; defcon_level[m4] = 4
    m3 = m4 & (a_bb_delta > 0); defcon_level[m3] = 3
    m2 = m3 & (a_bb_delta > a_bb_delta_avg) & (a_adx > adx_th); defcon_level[m2] = 2
    m1 = m2 & (a_bb_delta > a_bb_delta_avg * 1.5) & (a_adx > adx_th + 5) & (a_rvol > 1.2); defcon_level[m1] = 1

    cond_defcon_buy = (defcon_level <= 2) & neon_up_trig
    cond_defcon_sell = (defcon_level <= 2) & neon_dn_trig
    cond_therm_buy_bounce = (floor_w >= therm_w) & rsi_cross_up & ~is_hard_wall
    cond_therm_sell_wall = (ceil_w >= therm_w) & rsi_cross_dn
    cond_therm_buy_vacuum = (ceil_w <= 3) & neon_up_trig & ~is_abyss
    cond_therm_sell_panic = is_abyss & a_vr

    tol = a_atr * 0.5
    is_grav_sup = (a_c - a_tsup) < (a_atr * 3.0)
    is_grav_res = (a_tres - a_c) < (a_atr * 3.0)
    
    cond_lock_buy_bounce = is_grav_sup & (a_l <= a_tsup + tol) & (a_c > a_tsup) & a_vv
    cond_lock_buy_break = is_grav_res & (a_c > a_tres) & (a_c_s1 <= npshift(a_tres, 1)) & (a_rvol > 1.0) & a_vv
    cond_lock_sell_reject = is_grav_res & (a_h >= a_tres - tol) & (a_c < a_tres) & a_vr
    cond_lock_sell_breakd = is_grav_sup & (a_c < a_tsup) & (a_c_s1 >= npshift(a_tsup, 1)) & a_vr

    flash_vol = (a_rvol > whale_f * 0.8) & (a_bs > a_atr * 0.3)
    whale_buy, whale_sell = flash_vol & a_vv, flash_vol & a_vr
    whale_memory = whale_buy | npshift_bool(whale_buy, 1) | npshift_bool(whale_buy, 2) | whale_sell | npshift_bool(whale_sell, 1) | npshift_bool(whale_sell, 2)
    is_whale_icon = whale_buy & ~npshift_bool(whale_buy, 1)

    pre_pump = ((a_h > a_bbu) | (rsi_vel > 5)) & flash_vol & a_vv
    pump_memory = pre_pump | npshift_bool(pre_pump, 1) | npshift_bool(pre_pump, 2)
    pre_dump = ((a_l < a_bbl) | (rsi_vel < -5)) & flash_vol & a_vr
    dump_memory = pre_dump | npshift_bool(pre_dump, 1) | npshift_bool(pre_dump, 2)

    retro_peak_buy = (a_rsi < 30) & (a_c < a_bbl)
    retro_peak_sell = (a_rsi > 70) & (a_c > a_bbu)
    k_break_up = (a_rsi > (a_rsi_bb_b + a_rsi_bb_d)) & (a_rsi_s1 <= npshift(a_rsi_bb_b + a_rsi_bb_d, 1))
    k_break_dn = (a_rsi < (a_rsi_bb_b - a_rsi_bb_d)) & (a_rsi_s1 >= npshift(a_rsi_bb_b - a_rsi_bb_d, 1))
    
    div_bull = (a_l_s1 < a_l_s5) & (a_rsi_s1 > a_rsi_s5) & (a_rsi < 35)
    div_bear = (a_h_s1 > a_h_s5) & (a_rsi_s1 < a_rsi_s5) & (a_rsi > 65)
    div_bull_trig = div_bull & ~npshift_bool(div_bull, 1)
    div_bear_trig = div_bear & ~npshift_bool(div_bear, 1)

    buy_score = np.zeros(n_len)
    base_mask_b = retro_peak_buy | k_break_up | (is_struct_sup & rsi_cross_up) | div_bull
    buy_score = np.where(base_mask_b & retro_peak_buy, 50.0, np.where(base_mask_b & ~retro_peak_buy, 30.0, buy_score))
    buy_score += np.where(is_struct_sup, 25.0, 0.0); buy_score += np.where(whale_memory, 20.0, 0.0); buy_score += np.where(pump_memory, 15.0, 0.0)
    buy_score += np.where(div_bull, 15.0, 0.0); buy_score += np.where(k_break_up & ~retro_peak_buy, 15.0, 0.0); buy_score += np.where(~a_mb, -15.0, 0.0); buy_score += np.where(a_zscore < -2.0, 15.0, 0.0)
    
    sell_score = np.zeros(n_len)
    base_mask_s = retro_peak_sell | k_break_dn | (is_struct_res & rsi_cross_dn) | div_bear
    sell_score = np.where(base_mask_s & retro_peak_sell, 50.0, np.where(base_mask_s & ~retro_peak_sell, 30.0, sell_score))
    sell_score += np.where(is_struct_res, 25.0, 0.0); sell_score += np.where(whale_memory, 20.0, 0.0); sell_score += np.where(dump_memory, 15.0, 0.0)
    sell_score += np.where(div_bear, 15.0, 0.0); sell_score += np.where(k_break_dn & ~retro_peak_sell, 15.0, 0.0); sell_score += np.where(a_mb, -15.0, 0.0); sell_score += np.where(a_zscore > 2.0, 15.0, 0.0)

    is_magenta_buy = (buy_score >= 70) | retro_peak_buy
    is_magenta_sell = (sell_score >= 70) | retro_peak_sell
    cond_pink_whale_buy = is_magenta_buy & is_whale_icon

    wt_cross_up = (a_wt1 > a_wt2) & (a_wt1_s1 <= a_wt2_s1)
    wt_cross_dn = (a_wt1 < a_wt2) & (a_wt1_s1 >= a_wt2_s1)
    wt_oversold = a_wt1 < -60
    wt_overbought = a_wt1 > 60
    wt_oversold_trig = wt_oversold & wt_cross_up
    wt_overbought_trig = wt_overbought & wt_cross_dn

    dyn_wick_req = np.where(a_adx < 40, 0.4, 0.5)
    final_wick_req = np.where(is_struct_sup | is_struct_res | is_grav_sup | is_grav_res, 0.15, dyn_wick_req)
    final_vol_req  = np.where(is_struct_sup | is_struct_res | is_grav_sup | is_grav_res, 1.2, np.where(a_adx < 40, 1.5, 1.8))
    
    wick_rej_buy = a_lw > (a_bs * final_wick_req)
    wick_rej_sell = a_uw > (a_bs * final_wick_req)
    vol_stop_chk = a_rvol > final_vol_req
    
    climax_buy = is_magenta_buy & (wick_rej_buy | vol_stop_chk)
    climax_sell = is_magenta_sell & (wick_rej_sell | vol_stop_chk)
    
    nuclear_buy = climax_buy & (wt_oversold | wt_cross_up)
    nuclear_sell = climax_sell & (wt_overbought | wt_cross_dn)
    
    s_dict['Q_Pink_Whale_Buy'] = cond_pink_whale_buy
    s_dict['Q_Nuclear_Buy'] = nuclear_buy
    s_dict['Q_Climax_Buy'] = climax_buy
    s_dict['Q_Early_Buy'] = is_magenta_buy
    s_dict['Q_Defcon_Buy'] = cond_defcon_buy
    s_dict['Q_Neon_Up'] = neon_up_trig
    s_dict['Q_Therm_Bounce'] = cond_therm_buy_bounce
    s_dict['Q_Therm_Vacuum'] = cond_therm_buy_vacuum
    s_dict['Q_Lock_Bounce'] = cond_lock_buy_bounce
    s_dict['Q_Lock_Break'] = cond_lock_buy_break
    s_dict['Q_Rebound_Buy'] = rsi_cross_up & ~is_magenta_buy
    s_dict['Q_River_Push_Buy'] = river_push_up
    s_dict['Q_River_Entry_Buy'] = river_entry_up
    s_dict['Q_Div_Bull_Buy'] = div_bull_trig
    s_dict['Q_WT_Oversold_Buy'] = wt_oversold_trig
    
    s_dict['Q_Nuclear_Sell'] = nuclear_sell
    s_dict['Q_Climax_Sell'] = climax_sell
    s_dict['Q_Early_Sell'] = is_magenta_sell
    s_dict['Q_Defcon_Sell'] = cond_defcon_sell
    s_dict['Q_Neon_Dn'] = neon_dn_trig
    s_dict['Q_Therm_Wall_Sell'] = cond_therm_sell_wall
    s_dict['Q_Therm_Panic_Sell'] = cond_therm_sell_panic
    s_dict['Q_Lock_Reject'] = cond_lock_sell_reject
    s_dict['Q_Lock_Breakd'] = cond_lock_sell_breakd
    s_dict['Q_Pullback_Sell'] = rsi_cross_dn & ~is_magenta_sell
    s_dict['Q_River_Push_Sell'] = river_push_dn
    s_dict['Q_River_Entry_Sell'] = river_entry_dn
    s_dict['Q_Div_Bear_Sell'] = div_bear_trig
    s_dict['Q_WT_Overbought_Sell'] = wt_overbought_trig

    s_dict['PA_Engulfing_Buy'] = a_pa_eng_b; s_dict['PA_Engulfing_Sell'] = a_pa_eng_s
    s_dict['PA_Pinbar_Buy'] = a_pa_pin_b; s_dict['PA_Pinbar_Sell'] = a_pa_pin_s
    s_dict['PA_3_Soldiers_Buy'] = a_pa_3sol_b; s_dict['PA_3_Crows_Sell'] = a_pa_3cro_s

    # MODO CALIBRACIÓN (Control Group TV)
    f_calib_buy = np.zeros(n_len, dtype=bool)
    for i in range(0, n_len, 50): f_calib_buy[i] = True
    s_dict['Calibrador'] = f_calib_buy
    return s_dict

# 🔥 EL MOTOR EVOLUTIVO GENÉTICO REAL (ELÁSTICO Y PUNTUADO) 🔥
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
    default_f = np.zeros(n_len, dtype=bool)
    ones_mask = np.ones(n_len, dtype=bool)
    is_calib = st.session_state.get('is_calib_mode', False)

    for c in range(chunks):
        if st.session_state.get('abort_opt', False): break

        for _ in range(chunk_size): 
            # EVOLUCIÓN (75% Mutar al campeón, 25% Explorar azar puro)
            if best_fit_live != -float('inf') and random.random() < 0.75:
                dna_b_team = vault.get('b_team', []).copy()
                dna_s_team = vault.get('s_team', []).copy()
                dna_b_op = vault.get('b_op', '|')
                dna_s_op = vault.get('s_op', '|')
                
                if random.random() < 0.15: dna_b_op = random.choice(['&', '|'])
                if random.random() < 0.15: dna_s_op = random.choice(['&', '|'])

                if random.random() < 0.3 and len(todas_las_armas_b) > 0:
                    if random.random() < 0.5 and len(dna_b_team) > 1:
                        dna_b_team.pop(random.randint(0, len(dna_b_team)-1))
                    else:
                        if len(dna_b_team) < (2 if dna_b_op == '&' else 4): 
                            new_gene = random.choice(todas_las_armas_b)
                            if new_gene not in dna_b_team: dna_b_team.append(new_gene)
                
                if random.random() < 0.3 and len(todas_las_armas_s) > 0:
                    if random.random() < 0.5 and len(dna_s_team) > 1:
                        dna_s_team.pop(random.randint(0, len(dna_s_team)-1))
                    else:
                        if len(dna_s_team) < (2 if dna_s_op == '&' else 4):
                            new_gene = random.choice(todas_las_armas_s)
                            if new_gene not in dna_s_team: dna_s_team.append(new_gene)
                
                r_hitbox = vault.get('hitbox', 1.5)
                r_therm = vault.get('therm_w', 4.0)
                r_adx = vault.get('adx_th', 25.0)
                r_whale = vault.get('whale_f', 2.5)
                if random.random() < 0.15: r_hitbox = random.choice([1.0, 1.5, 2.0, 2.5])
                if random.random() < 0.15: r_adx = random.choice([20.0, 25.0, 30.0])
                
                # TP Y SL AMPLIOS (Redes de seguridad catastrofica, NO estrategia principal)
                r_tp_pct = round(max(5.0, vault.get('tp_pct', 25.0) + random.uniform(-1.0, 1.0)), 2)
                r_sl_pct = round(max(2.0, vault.get('sl_pct', 10.0) + random.uniform(-0.5, 0.5)), 2)
                
            else:
                dna_b_op = random.choice(['&', '|'])
                dna_s_op = random.choice(['&', '|'])
                dna_b_team = random.sample(todas_las_armas_b, random.randint(1, 2 if dna_b_op == '&' else 4))
                dna_s_team = random.sample(todas_las_armas_s, random.randint(1, 2 if dna_s_op == '&' else 4))
                r_hitbox = random.choice([1.0, 1.5, 2.0, 2.5])
                r_therm = random.choice([3.0, 4.0, 5.0, 6.0])
                r_adx = random.choice([20.0, 25.0, 30.0])
                r_whale = random.choice([2.0, 2.5, 3.0])
                r_tp_pct = round(random.uniform(10.0, 30.0), 2) 
                r_sl_pct = round(random.uniform(5.0, 15.0), 2)  
            
            s_dict = calcular_señales_numpy(r_hitbox, r_therm, r_adx, r_whale)
            
            if dna_b_op == '&':
                f_buy_tactical = np.ones(n_len, dtype=bool)
                for r in dna_b_team: f_buy_tactical &= s_dict.get(r, ones_mask)
            else:
                f_buy_tactical = np.zeros(n_len, dtype=bool)
                for r in dna_b_team: f_buy_tactical |= s_dict.get(r, default_f)

            if dna_s_op == '&':
                f_sell_tactical = np.ones(n_len, dtype=bool)
                for r in dna_s_team: f_sell_tactical &= s_dict.get(r, ones_mask)
            else:
                f_sell_tactical = np.zeros(n_len, dtype=bool)
                for r in dna_s_team: f_sell_tactical |= s_dict.get(r, default_f)

            # 🔥 REGLA ANTI-COLISIÓN: Si chocan, prioriza la Venta y anula la Compra (No lo mata) 🔥
            f_buy_tactical = f_buy_tactical & ~f_sell_tactical

            # Predefinir variables por seguridad extrema
            net_is = 0.0; pf_is = 0.0; nt_is = 0; mdd_is = 0.0; wr_is = 0.0; ado_actual = 0.0

            # 🛑 1. SIMULACIÓN IN-SAMPLE
            net_is, pf_is, nt_is, mdd_is, wr_is = simular_core_rapido(
                a_h[:split_idx], a_l[:split_idx], a_c[:split_idx], a_o[:split_idx],
                f_buy_tactical[:split_idx], f_sell_tactical[:split_idx], 
                r_tp_pct, r_sl_pct, float(cap_ini), float(com_pct), float(invest_pct), 0.0, is_calib
            )

            ado_actual = nt_is / max(1, dias_entrenamiento)
            fit_score = -float('inf') 
            
            # 🔥 FITNESS ELÁSTICA (Crecimiento Exponencial) 🔥
            if nt_is >= 3 and net_is > 0: 
                avg_trade_net_pct = (net_is / cap_ini) / nt_is * 100.0
                if avg_trade_net_pct < 0.15: # Flexibilizado
                    fit_score = net_is - 5000.0 
                else:
                    fit_score = net_is
                    if pf_is > 1.0: fit_score *= min(pf_is, 3.0)
                    else: fit_score *= 0.5
                    
                    if mdd_is > 25.0: fit_score -= (mdd_is * 5.0) # Penaliza suavemente, no mata
                    
                    # 🎯 LA FÓRMULA DE CAMPANA DEL ADO (Bonificación o Castigo)
                    target_trades = target_ado * dias_entrenamiento
                    if target_trades > 0:
                        trade_ratio = nt_is / target_trades
                        if 0.5 <= trade_ratio <= 1.5:
                            # Bonificación dorada (+50% si acierta exacto)
                            bonus = 1.0 + (0.5 * (1.0 - abs(1.0 - trade_ratio)))
                            fit_score *= bonus
                        else:
                            # Castigo si se vuelve un bot ametralladora o no opera nada
                            penalty = abs(1.0 - trade_ratio) * 0.2
                            fit_score -= (abs(fit_score) * min(penalty, 0.8)) 

            elif nt_is > 0:
                fit_score = net_is - 1000.0 - (abs(ado_actual - target_ado) * 50.0) # Se penaliza fuertemente pero da una base a mejorar

            # 🛑 2. SIMULACIÓN OUT-OF-SAMPLE
            if fit_score > best_fit_live:
                net_oos, pf_oos, nt_oos, mdd_oos, wr_oos = simular_core_rapido(
                    a_h[split_idx:], a_l[split_idx:], a_c[split_idx:], a_o[split_idx:],
                    f_buy_tactical[split_idx:], f_sell_tactical[split_idx:], 
                    r_tp_pct, r_sl_pct, float(cap_ini), float(com_pct), float(invest_pct), 0.0, is_calib
                )
                
                # Se guarda si OOS es positivo O si es el primer ADN generado (para sacar a la IA del -inf)
                if net_oos > 0 or is_calib or best_fit_live == -float('inf'): 
                    net_tot, pf_tot, nt_tot, mdd_tot, wr_tot = simular_core_rapido(
                        a_h, a_l, a_c, a_o, f_buy_tactical, f_sell_tactical, 
                        r_tp_pct, r_sl_pct, float(cap_ini), float(com_pct), float(invest_pct), 0.0, is_calib
                    )
                    
                    # Si OOS fue malo, reducimos artificialmente el fit_score para que otra mutación lo borre pronto
                    final_fit = fit_score if net_oos > 0 else (fit_score * 0.1)

                    best_fit_live = final_fit
                    vault = {
                        'b_team': dna_b_team, 's_team': dna_s_team, 'b_op': dna_b_op, 's_op': dna_s_op,
                        'hitbox': r_hitbox, 'therm_w': r_therm, 'adx_th': r_adx, 'whale_f': r_whale, 'fit': final_fit, 
                        'net': net_tot, 'net_is': net_is, 'net_oos': net_oos,
                        'winrate': wr_tot, 'pf': pf_tot, 'nt': nt_tot, 'reinv': invest_pct, 'ado': ado_actual, 
                        'tp_pct': r_tp_pct, 'sl_pct': r_sl_pct
                    }
                    save_champion(s_id, vault)
                    st.session_state[f'opt_status_{s_id}'] = True
            
        global_start = deep_info.get('start_time', start_time) if deep_info else start_time
        total_elapsed_sec = time.time() - global_start
        h, rem = divmod(total_elapsed_sec, 3600); m, s = divmod(rem, 60)
        time_str = f"{int(h):02d}h:{int(m):02d}m:{int(s):02d}s"
        
        v = get_safe_vault(s_id)
        current_best_net = v.get('net', 0)
        current_best_nt = v.get('nt', 0)
        current_best_pf = v.get('pf', 0)
        current_best_ado = v.get('ado', 0.0)

        if deep_info:
            current_epoch_val = deep_info['current'] + (c+1)*(chunk_size); macro_pct = int((current_epoch_val / deep_info['total']) * 100)
            title = f"🌌 DEEP FORGE OMNI: {s_id}"; subtitle = f"Épocas: {current_epoch_val:,} / {deep_info['total']:,} ({macro_pct}%)<br>⏱️ Tiempo: {time_str}"; color = "#FF00FF"
        else:
            pct_done = int(((c + 1) / chunks) * 100); combos = (c + 1) * chunk_size
            title = f"OMNI LAB V320: {s_id}"; subtitle = f"Progreso: {pct_done}% | ADN Probados: {combos:,}<br>⏱️ Tiempo Ejecución: {time_str}"; color = "#00FFFF"

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
            <div style="color: cyan; font-size: 1.0rem;">Trades: {current_best_nt} | PF: {current_best_pf:.2f}x</div>
            <div style="color: yellow; font-size: 0.9rem;">ADO Objetivo: {target_ado:.1f} | ADO Actual: {current_best_ado:.2f}</div>
        </div>
        """
        ph_holograma.markdown(html_str, unsafe_allow_html=True)
            
    return get_safe_vault(s_id) if best_fit_live != -float('inf') else None

def run_backtest_eval(s_id, cap_ini, com_pct):
    vault = get_safe_vault(s_id)
    is_calib = st.session_state.get('is_calib_mode', False)
    
    s_dict = calcular_señales_numpy(vault.get('hitbox',1.5), vault.get('therm_w',4.0), vault.get('adx_th',25.0), vault.get('whale_f',2.5))
    n_len = len(a_c)
    
    tp_pct_val = vault.get('tp_pct', 3.0)
    sl_pct_val = vault.get('sl_pct', 1.5)
    f_tp, f_sl = np.full(n_len, tp_pct_val), np.full(n_len, sl_pct_val)
    default_f = np.zeros(n_len, dtype=bool)
    ones_mask = np.ones(n_len, dtype=bool)

    if is_calib:
        f_buy = s_dict['Calibrador']
        f_sell = default_f
    else:
        dna_b_op = vault.get('b_op', '|')
        dna_s_op = vault.get('s_op', '|')
        
        if dna_b_op == '&':
            f_buy = np.ones(n_len, dtype=bool)
            for r in vault.get('b_team', []): f_buy &= s_dict.get(r, ones_mask)
        else:
            f_buy = np.zeros(n_len, dtype=bool)
            for r in vault.get('b_team', []): f_buy |= s_dict.get(r, default_f)

        if dna_s_op == '&':
            f_sell = np.ones(n_len, dtype=bool)
            for r in vault.get('s_team', []): f_sell &= s_dict.get(r, ones_mask)
        else:
            f_sell = np.zeros(n_len, dtype=bool)
            for r in vault.get('s_team', []): f_sell |= s_dict.get(r, default_f)
            
        f_buy = f_buy & ~f_sell # Anti-colisión final para el gráfico visual

    df_strat = df_global.copy()
    df_strat['Signal_Buy'], df_strat['Signal_Sell'], df_strat['Active_TP'], df_strat['Active_SL'] = f_buy, f_sell, f_tp, f_sl
    eq_curve, divs, cap_act, t_log, en_pos, total_comms = simular_visual(df_strat, cap_ini, float(vault.get('reinv', 20.0)), com_pct, 0.0, is_calib)
    return df_strat, eq_curve, t_log, total_comms

def build_pine_cond(team, operator):
    if not team: return "false"
    joiner = " and " if operator == '&' else " or "
    return joiner.join([pine_map.get(x, 'false') for x in team])

def generar_pine_script(s_id, vault, sym, tf, buy_pct, sell_pct, com_pct, start_date_obj, is_calib=False):
    v_hb = vault.get('hitbox', 1.5); v_tw = vault.get('therm_w', 4.0); v_adx = vault.get('adx_th', 25.0); v_wf = vault.get('whale_f', 2.5)
    v_tp = vault.get('tp_pct', 3.0); v_sl = vault.get('sl_pct', 1.5)
    
    b_cond = build_pine_cond(vault.get('b_team', []), vault.get('b_op', '|'))
    s_cond = build_pine_cond(vault.get('s_team', []), vault.get('s_op', '|'))
    
    json_buy = f'{{"passphrase": "ASTRONAUTA", "action": "buy", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {buy_pct}, "order_type": "limit", "limit_price": {{{{close}}}}, "slippage_pct": 1.0, "side": "🟢 COMPRA LIMIT"}}'
    json_sell = f'{{"passphrase": "ASTRONAUTA", "action": "sell", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": {sell_pct}, "order_type": "market", "side": "🔴 VENTA MARKET"}}'

    ps_base = f"""//@version=5
strategy("{s_id} PREDATOR [{sym} {tf}]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value={buy_pct}, commission_type=strategy.commission.percent, commission_value={com_pct*100}, process_orders_on_close=true)

wt_enter_long = input.text_area(defval='{json_buy}', title="🟢 Webhook de Compra (Limit)")
wt_exit_long  = input.text_area(defval='{json_sell}', title="🔴 Webhook de Venta (Market)")

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

if strategy.position_size > 0
    strategy.exit("TP/SL", "In", limit=tp_price, stop=sl_price, alert_profit=wt_exit_long, alert_loss=wt_exit_long)

if strategy.position_size == 0
    tp_price := na
    sl_price := na

plotshape(signal_buy, title="COMPRA", style=shape.triangleup, location=location.belowbar, color=color.yellow, size=size.tiny)
"""

    ps_indicators = f"""
hitbox_pct   = {v_hb}
therm_wall   = {v_tw}
adx_trend    = {v_adx}
whale_factor = {v_wf}
tp_pct_val   = {v_tp}
sl_pct_val   = {v_sl}

vol_ma_100 = ta.sma(volume, 100)
rsi_v = ta.rsi(close, 14)
atr_val = ta.atr(14)
rvol = volume / (vol_ma_100 == 0 ? 1 : vol_ma_100)

ema20 = ta.ema(close, 20)
basis = ta.sma(close, 20)
stdev20 = ta.stdev(close, 20)
dev = 2.0 * stdev20
bbu = basis + dev
bbl = basis - dev
bb_width = basis == 0 ? 1 : (bbu - bbl) / basis
bb_delta = bb_width - nz(bb_width[1], 0)
bb_delta_avg = ta.sma(bb_delta, 10)

kc_u = ta.sma(close, 20) + (atr_val * 1.5)
kc_l = ta.sma(close, 20) - (atr_val * 1.5)
squeeze_on = (bbu < kc_u) and (bbl > kc_l)
z_score = stdev20 == 0 ? 0 : (close - basis) / stdev20

vela_verde = close > open
vela_roja = close < open
rsi_ma = ta.sma(rsi_v, 14)
rsi_cross_up = (rsi_v > rsi_ma) and (nz(rsi_v[1]) <= nz(rsi_ma[1]))
rsi_cross_dn = (rsi_v < rsi_ma) and (nz(rsi_v[1]) >= nz(rsi_ma[1]))

rsi_vel = rsi_v - nz(rsi_v[1])
river_width = atr_val * (1.2 + (math.abs(rsi_vel) / 10.0))
river_top = ema20 + (river_width * 0.5)
river_bot = ema20 - (river_width * 0.5)

low_100 = ta.lowest(low[1], 100)
low_300 = ta.lowest(low[1], 300)
low_800 = ta.lowest(low[1], 800)
a_tsup = math.max(nz(low_100, 0), math.max(nz(low_300, 0), nz(low_800, 0)))

high_100 = ta.highest(high[1], 100)
high_300 = ta.highest(high[1], 300)
high_800 = ta.highest(high[1], 800)
a_tres = math.min(nz(high_100, 99999), math.min(nz(high_300, 99999), nz(high_800, 99999)))

sr_val = atr_val * 2.0
ceil_w = 0
floor_w = 0
ceil_w += (nz(high_100) > close and nz(high_100) <= close + sr_val) ? 3 : 0
ceil_w += (nz(low_100) > close and nz(low_100) <= close + sr_val) ? 3 : 0
ceil_w += (nz(high_300) > close and nz(high_300) <= close + sr_val) ? 5 : 0
ceil_w += (nz(low_300) > close and nz(low_300) <= close + sr_val) ? 5 : 0
ceil_w += (nz(high_800) > close and nz(high_800) <= close + sr_val) ? 8 : 0
ceil_w += (nz(low_800) > close and nz(low_800) <= close + sr_val) ? 8 : 0

floor_w += (nz(high_100) < close and nz(high_100) >= close - sr_val) ? 3 : 0
floor_w += (nz(low_100) < close and nz(low_100) >= close - sr_val) ? 3 : 0
floor_w += (nz(high_300) < close and nz(high_300) >= close - sr_val) ? 5 : 0
floor_w += (nz(low_300) < close and nz(low_300) >= close - sr_val) ? 5 : 0
floor_w += (nz(high_800) < close and nz(high_800) >= close - sr_val) ? 8 : 0
floor_w += (nz(low_800) < close and nz(low_800) >= close - sr_val) ? 8 : 0

is_abyss = floor_w == 0
is_hard_wall = ceil_w >= therm_wall
is_struct_sup = floor_w > 0
is_struct_res = ceil_w > 0

neon_up = squeeze_on and (close >= bbu * 0.999) and vela_verde
neon_up_trig = neon_up and not nz(neon_up[1])
neon_dn = squeeze_on and (close <= bbl * 1.001) and vela_roja
neon_dn_trig = neon_dn and not nz(neon_dn[1])

river_push_up = (low <= river_top) and (close > river_top) and not neon_dn
river_push_dn = (high >= river_bot) and (close < river_bot) and not neon_up
river_entry_up = (close > river_bot) and (nz(close[1]) <= nz(river_bot[1]))
river_entry_dn = (close < river_top) and (nz(close[1]) >= nz(river_top[1]))

[di_plus, di_minus, adx] = ta.dmi(14, 14)
defcon_level = 5
if neon_up or neon_dn
    defcon_level := 4
    if bb_delta > 0
        defcon_level := 3
        if bb_delta > bb_delta_avg and adx > adx_trend
            defcon_level := 2
            if bb_delta > (bb_delta_avg * 1.5) and adx > (adx_trend + 5) and rvol > 1.2
                defcon_level := 1

cond_defcon_buy = defcon_level <= 2 and neon_up_trig
cond_defcon_sell = defcon_level <= 2 and neon_dn_trig
cond_therm_buy_bounce = (floor_w >= therm_wall) and rsi_cross_up and not is_hard_wall
cond_therm_sell_wall = (ceil_w >= therm_wall) and rsi_cross_dn
cond_therm_buy_vacuum = (ceil_w <= 3) and neon_up_trig and not is_abyss
cond_therm_sell_panic = is_abyss and vela_roja

tol = atr_val * 0.5
is_grav_sup = (close - a_tsup) < (atr_val * 3.0)
is_grav_res = (a_tres - close) < (atr_val * 3.0)
cond_lock_buy_bounce = is_grav_sup and (low <= a_tsup + tol) and (close > a_tsup) and vela_verde
cond_lock_buy_break = is_grav_res and (close > a_tres) and (nz(close[1]) <= nz(a_tres[1])) and (rvol > 1.0) and vela_verde
cond_lock_sell_reject = is_grav_res and (high >= a_tres - tol) and (close < a_tres) and vela_roja
cond_lock_sell_breakd = is_grav_sup and (close < a_tsup) and (nz(close[1]) >= nz(a_tsup[1])) and vela_roja

body_size = math.abs(close - open)
upper_wick = high - math.max(open, close)
lower_wick = math.min(open, close) - low

flash_vol = (rvol > whale_factor * 0.8) and (body_size > atr_val * 0.3)
whale_buy = flash_vol and vela_verde
whale_sell = flash_vol and vela_roja
whale_memory = whale_buy or nz(whale_buy[1]) or nz(whale_buy[2]) or whale_sell or nz(whale_sell[1]) or nz(whale_sell[2])
is_whale_icon = whale_buy and not nz(whale_buy[1])

pre_pump = (high > bbu or rsi_vel > 5) and flash_vol and vela_verde
pump_memory = pre_pump or nz(pre_pump[1]) or nz(pre_pump[2])
pre_dump = (low < bbl or rsi_vel < -5) and flash_vol and vela_roja
dump_memory = pre_dump or nz(pre_dump[1]) or nz(pre_dump[2])

rsi_bb_basis = ta.sma(rsi_v, 14)
rsi_bb_dev = ta.stdev(rsi_v, 14) * 2.0
retro_peak_buy = (rsi_v < 30) and (close < bbl)
retro_peak_sell = (rsi_v > 70) and (close > bbu)
k_break_up = (rsi_v > (rsi_bb_basis + rsi_bb_dev)) and (nz(rsi_v[1]) <= (nz(rsi_bb_basis[1]) + nz(rsi_bb_dev[1])))
k_break_dn = (rsi_v < (rsi_bb_basis - rsi_bb_dev)) and (nz(rsi_v[1]) >= (nz(rsi_bb_basis[1]) - nz(rsi_bb_dev[1])))

div_bull = nz(low[1]) < nz(low[5]) and nz(rsi_v[1]) > nz(rsi_v[5]) and (rsi_v < 35)
div_bear = nz(high[1]) > nz(high[5]) and nz(rsi_v[1]) < nz(rsi_v[5]) and (rsi_v > 65)
div_bull_trig = div_bull and not nz(div_bull[1])
div_bear_trig = div_bear and not nz(div_bear[1])

macro_bull = close >= ta.ema(close, 200)

base_mask = retro_peak_buy or k_break_up or (is_struct_sup and rsi_cross_up) or div_bull
buy_score = 0.0
buy_score := (base_mask and retro_peak_buy) ? 50.0 : (base_mask and not retro_peak_buy) ? 30.0 : buy_score
buy_score += is_struct_sup ? 25.0 : 0.0
buy_score += whale_memory ? 20.0 : 0.0
buy_score += pump_memory ? 15.0 : 0.0
buy_score += div_bull ? 15.0 : 0.0
buy_score += (k_break_up and not retro_peak_buy) ? 15.0 : 0.0
buy_score += not macro_bull ? -15.0 : 0.0
buy_score += (z_score < -2.0) ? 15.0 : 0.0

base_mask_s = retro_peak_sell or k_break_dn or (is_struct_res and rsi_cross_dn) or div_bear
sell_score = 0.0
sell_score := (base_mask_s and retro_peak_sell) ? 50.0 : (base_mask_s and not retro_peak_sell) ? 30.0 : sell_score
sell_score += is_struct_res ? 25.0 : 0.0
sell_score += whale_memory ? 20.0 : 0.0
sell_score += dump_memory ? 15.0 : 0.0
sell_score += div_bear ? 15.0 : 0.0
sell_score += (k_break_dn and not retro_peak_sell) ? 15.0 : 0.0
sell_score += macro_bull ? -15.0 : 0.0
sell_score += (z_score > 2.0) ? 15.0 : 0.0

is_magenta_buy = (buy_score >= 70) or retro_peak_buy
is_magenta_sell = (sell_score >= 70) or retro_peak_sell
cond_pink_whale_buy = is_magenta_buy and is_whale_icon

ap = hlc3
esa = ta.ema(ap, 10)
d_wt = ta.ema(math.abs(ap - esa), 10)
wt1 = ta.ema((ap - esa) / (0.015 * (d_wt == 0 ? 1 : d_wt)), 21)
wt2 = ta.sma(wt1, 4)
wt_cross_up = (wt1 > wt2) and (nz(wt1[1]) <= nz(wt2[1]))
wt_cross_dn = (wt1 < wt2) and (nz(wt1[1]) >= nz(wt2[1]))
wt_oversold = wt1 < -60
wt_overbought = wt1 > 60
wt_oversold_trig = wt_oversold and wt_cross_up
wt_overbought_trig = wt_overbought and wt_cross_dn

dyn_wick_req = adx < 40 ? 0.4 : 0.5 
matrix_active = is_struct_sup or is_struct_res or is_grav_sup or is_grav_res
final_wick_req = matrix_active ? 0.15 : dyn_wick_req 
final_vol_req  = matrix_active ? 1.2 : (adx < 40 ? 1.5 : 1.8)  

wick_rej_buy = lower_wick > (body_size * final_wick_req)
wick_rej_sell = upper_wick > (body_size * final_wick_req)
vol_stop_chk = rvol > final_vol_req

climax_buy = is_magenta_buy and (wick_rej_buy or vol_stop_chk)
climax_sell = is_magenta_sell and (wick_rej_sell or vol_stop_chk)

nuclear_buy = climax_buy and (wt_oversold or wt_cross_up)
nuclear_sell = climax_sell and (wt_overbought or wt_cross_dn)
climax_buy_early = is_magenta_buy
climax_sell_early = is_magenta_sell
normal_reb_buy = rsi_cross_up and not is_magenta_buy
normal_reb_sell = rsi_cross_dn and not is_magenta_sell

pa_engulfing_buy = vela_verde and nz(vela_roja[1]) and close > nz(open[1]) and open < nz(close[1])
pa_engulfing_sell = vela_roja and nz(vela_verde[1]) and close < nz(open[1]) and open > nz(close[1])
pa_pinbar_buy = lower_wick > body_size * 2.5 and upper_wick < body_size
pa_pinbar_sell = upper_wick > body_size * 2.5 and lower_wick < body_size
pa_3_soldiers = vela_verde and nz(vela_verde[1]) and nz(vela_verde[2]) and close > nz(close[1]) and nz(close[1]) > nz(close[2])
pa_3_crows = vela_roja and nz(vela_roja[1]) and nz(vela_roja[2]) and close < nz(close[1]) and nz(close[1]) < nz(close[2])
"""

    ps_logic = f"""
bool raw_buy = ({b_cond})
bool raw_sell = ({s_cond})

// Filtro anti-colisión matemático: La venta anula la compra si ocurren simultáneamente
bool signal_buy = raw_buy and not raw_sell
bool signal_sell = raw_sell
"""

    ps_exec = """
var float locked_entry = na
var float locked_tp = na
var float locked_sl = na
bool just_entered = ta.change(strategy.position_size) > 0

if signal_buy and strategy.position_size == 0 and window
    strategy.entry("In", strategy.long, alert_message=wt_enter_long)

if just_entered
    locked_entry := strategy.position_avg_price
    locked_tp := locked_entry * (1 + (tp_pct_val / 100))
    locked_sl := locked_entry * (1 - (sl_pct_val / 100))

if strategy.position_size > 0
    bool hit_tp = high >= locked_tp or close >= locked_tp
    bool hit_sl = low <= locked_sl or close <= locked_sl
    
    if signal_sell or hit_tp or hit_sl
        string exit_msg = hit_tp ? "TP_Hit" : hit_sl ? "SL_Hit" : "Dyn_Exit"
        strategy.close("In", comment=exit_msg, alert_message=wt_exit_long)
        locked_entry := na
        locked_tp := na
        locked_sl := na

if strategy.position_size == 0
    locked_entry := na
    locked_tp := na
    locked_sl := na

plotshape(signal_buy, title="COMPRA", style=shape.triangleup, location=location.belowbar, color=color.aqua, size=size.small)
plotshape(signal_sell, title="VENTA", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small)
plot(strategy.position_size > 0 ? locked_tp : na, color=color.green, style=plot.style_linebr, linewidth=2)
plot(strategy.position_size > 0 ? locked_sl : na, color=color.red, style=plot.style_linebr, linewidth=2)
"""
    return ps_base + ps_indicators + ps_logic + ps_exec

# ==========================================
# 🛑 EJECUCIÓN GLOBAL
# ==========================================
if st.session_state.get('run_global', False):
    time.sleep(0.1) 
    if len(st.session_state['global_queue']) > 0:
        s_id = st.session_state['global_queue'].pop(0)
        ph_holograma.markdown(f"<div style='text-align:center; padding: 20px; background: rgba(0,0,0,0.8); border: 2px solid cyan; border-radius: 10px;'><h2 style='color:cyan;'>⚙️ Forjando Evolución Estricta: {s_id}...</h2><h4 style='color:lime;'>Quedan {len(st.session_state['global_queue'])} mutantes en incubación.</h4></div>", unsafe_allow_html=True)
        time.sleep(0.1)
        
        v = get_safe_vault(s_id)
        buy_hold_ret = ((df_global['Close'].iloc[-1] - df_global['Open'].iloc[0]) / df_global['Open'].iloc[0]) * 100
        buy_hold_money = capital_inicial * (buy_hold_ret / 100.0)
        
        bp = optimizar_ia_tracker(s_id, capital_inicial, comision_pct, float(v.get('reinv', 20.0)), float(v.get('ado',4.0)), dias_reales, buy_hold_money, epochs=global_epochs, cur_net=float(v.get('net',-float('inf'))), cur_fit=float(v.get('fit',-float('inf'))), deep_info=None, greed_factor=st.session_state.get(f'greed_{s_id}', 0.8))
        
        st.rerun()
    else:
        st.session_state['run_global'] = False
        ph_holograma.empty()
        st.sidebar.success("✅ ¡Evolución Estricta Completada!")
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
        st.sidebar.success(f"🌌 ¡FORJA PROFUNDA PREDATOR COMPLETADA PARA {s_id}!")
        time.sleep(2)
        
    st.rerun()

# ==========================================
# 🛑 UI Y RENDERIZADO
# ==========================================
st.title("🛡️ PREDATOR LAB (V320)")

with st.expander("🏆 SALÓN DE LA FAMA GENÉTICA", expanded=True):
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
    opt_badge = "<span style='color: lime;'>✅ ADN PREDATOR OPTIMIZADO</span>" if is_opt else "<span style='color: gray;'>➖ ADN VIRGEN</span>"
    
    vault = get_safe_vault(s_id) 

    st.markdown(f"### {selected_tab_name} {opt_badge}", unsafe_allow_html=True)

    with st.expander("🧬 VER ADN DEL MUTANTE Y ARMAS TÁCTICAS", expanded=True):
        st.markdown(f"**🟢 Escuadrón de Compra (V320 Core):** {', '.join(vault.get('b_team', []))} (Táctica: {vault.get('b_op', '&')})")
        st.markdown(f"**🔴 Escuadrón de Venta (V320 Core):** {', '.join(vault.get('s_team', []))} (Táctica: {vault.get('s_op', '&')})")
        st.markdown(f"**📡 Densidad Radial:** Hitbox: `{vault.get('hitbox',0):.2f}%` | Muro Térmico: `{vault.get('therm_w',0)}`")
        st.markdown(f"**🐋 Anomalías:** Fuerza ADX: `{vault.get('adx_th',0):.1f}` | Factor Ballena: `{vault.get('whale_f',0):.1f}x`")
        st.markdown(f"**🎯 Redes de Seguridad Macro:** TP: `{vault.get('tp_pct',0):.2f}%` | SL: `{vault.get('sl_pct',0):.2f}%`")

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

    mc_curves, risk_of_ruin = simular_monte_carlo(t_log, capital_inicial, 500)
    st.markdown("<br>", unsafe_allow_html=True)
    c_mc1, c_mc2 = st.columns([1, 4])

    c_mc1.markdown("### 🎲 Test de Estrés")
    if risk_of_ruin > 10.0: c_mc1.error(f"⚠️ RIESGO: {risk_of_ruin:.1f}%")
    elif risk_of_ruin > 0.0: c_mc1.warning(f"⚠️ RIESGO: {risk_of_ruin:.1f}%")
    else: c_mc1.success(f"🛡️ RIESGO: {risk_of_ruin:.1f}%")

    if mc_curves is not None:
        fig_mc = go.Figure()
        for i in range(min(50, len(mc_curves))):
            fig_mc.add_trace(go.Scatter(y=mc_curves[i], mode='lines', line=dict(color='rgba(255, 255, 255, 0.1)', width=1), hoverinfo='skip'))
        real_curve = [capital_inicial] + [capital_inicial + sum([t['Ganancia_$'] for t in t_log if t['Tipo'] in ['TP', 'SL', 'DYN_WIN', 'DYN_LOSS']][:k+1]) for k in range(len([t for t in t_log if t['Tipo'] in ['TP', 'SL', 'DYN_WIN', 'DYN_LOSS']]))]
        fig_mc.add_trace(go.Scatter(y=real_curve, mode='lines', name='Histórico', line=dict(color='gold', width=3)))
        fig_mc.update_layout(template='plotly_dark', height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        c_mc2.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("📝 CÓDIGO PINE SCRIPT PREDATOR (LISTO PARA TRADINGVIEW)", expanded=False):
        st.info("Traducción Matemática Exacta. Pégalo en TradingView.")
        st.code(generar_pine_script(s_id, vault, ticker.split('/')[0], iv_download, ps_buy_pct, ps_sell_pct, comision_pct, df_strat.index[0], is_calib_mode), language="pine")

    st.markdown("---")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], name="Precio", increasing_line_color='cyan', decreasing_line_color='magenta'), row=1, col=1)
    
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

    fig.update_layout(template='plotly_dark', height=800, xaxis_rangeslider_visible=False, dragmode='pan', hovermode='x unified', margin=dict(l=10, r=50, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
