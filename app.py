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
from datetime import datetime, timedelta

st.set_page_config(page_title="ROCKET PROTOCOL | Apex Quant", layout="wide", initial_sidebar_state="expanded")

# --- MEMORIA IA INSTITUCIONAL ---
buy_rules = ['Pink_Whale_Buy', 'Lock_Bounce', 'Lock_Break', 'Defcon_Buy', 'Neon_Up', 'Therm_Bounce', 'Therm_Vacuum', 'Nuclear_Buy', 'Early_Buy', 'Rebound_Buy']
sell_rules = ['Defcon_Sell', 'Neon_Dn', 'Therm_Wall_Sell', 'Therm_Panic_Sell', 'Lock_Reject', 'Lock_Breakd', 'Nuclear_Sell', 'Early_Sell']

# INICIALIZADOR DE LLAVES FÍSICAS
if 'winning_dna' not in st.session_state: st.session_state['winning_dna'] = ""
if 'gen_bull_tp' not in st.session_state: st.session_state['gen_bull_tp'] = 5.0
if 'gen_bull_sl' not in st.session_state: st.session_state['gen_bull_sl'] = 2.0
if 'gen_bear_tp' not in st.session_state: st.session_state['gen_bear_tp'] = 3.0
if 'gen_bear_sl' not in st.session_state: st.session_state['gen_bear_sl'] = 1.5

for r in buy_rules:
    if f'gen_bull_b_{r}' not in st.session_state: st.session_state[f'gen_bull_b_{r}'] = False
    if f'gen_bear_b_{r}' not in st.session_state: st.session_state[f'gen_bear_b_{r}'] = False
for r in sell_rules:
    if f'gen_bull_s_{r}' not in st.session_state: st.session_state[f'gen_bull_s_{r}'] = False
    if f'gen_bear_s_{r}' not in st.session_state: st.session_state[f'gen_bear_s_{r}'] = False

# Fuerza G Inicial de Gatillos (Para no arrancar en blanco)
if not any([st.session_state[f'gen_bull_b_{r}'] for r in buy_rules]): 
    st.session_state['gen_bull_b_Nuclear_Buy'] = True
    st.session_state['gen_bear_b_Pink_Whale_Buy'] = True
if not any([st.session_state[f'gen_bull_s_{r}'] for r in sell_rules]): 
    st.session_state['gen_bull_s_Nuclear_Sell'] = True
    st.session_state['gen_bear_s_Therm_Panic_Sell'] = True

for s in ["TRINITY", "JUGGERNAUT", "DEFCON"]:
    if f'sld_tp_{s}' not in st.session_state: st.session_state[f'sld_tp_{s}'] = 3.0
    if f'sld_sl_{s}' not in st.session_state: st.session_state[f'sld_sl_{s}'] = 1.5
    if f'sld_wh_{s}' not in st.session_state: st.session_state[f'sld_wh_{s}'] = 2.5
    if f'sld_rd_{s}' not in st.session_state: st.session_state[f'sld_rd_{s}'] = 1.5
    if f'sld_reinv_{s}' not in st.session_state: st.session_state[f'sld_reinv_{s}'] = 50.0

# --- 1. PANEL LATERAL ---
css_spinner = """
<style>
.loader-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; pointer-events: none; background: transparent; }
.rocket { font-size: 10rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 35px rgba(0, 255, 255, 1)); }
@keyframes spin { 0% { transform: scale(1) rotate(0deg); } 50% { transform: scale(1.2) rotate(180deg); } 100% { transform: scale(1) rotate(360deg); } }
</style>
<div class="loader-container"><div class="rocket">🌌</div></div>
"""
ph_holograma = st.empty()

st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 APEX QUANT LAB</h2>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Purgar Memoria & Sincronizar", use_container_width=True): 
    st.cache_data.clear()
    gc.collect()

st.sidebar.markdown("---")
exchange_sel = st.sidebar.selectbox("🏦 Exchange", ["coinbase", "binance", "kraken", "kucoin"], index=0)
ticker = st.sidebar.text_input("Símbolo Exacto", value="BTC/USD")
utc_offset = st.sidebar.number_input("🌍 Zona Horaria", value=-5.0, step=0.5)

intervalos = {"1 Minuto": "1min", "5 Minutos": "5min", "15 Minutos": "15min", "30 Minutos": "30min", "1 Hora": "1h", "4 Horas": "4h", "1 Día": "1d", "1 Semana": "1W"}
iv_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=4) 
iv_resample = intervalos[iv_sel]
iv_download = "1m" if "min" in iv_resample else "1h" if "h" in iv_resample else "1d"

hoy = datetime.today().date()
limite_dias = 7 if iv_download == "1m" else 180 if iv_download == "1h" else 1800
start_date, end_date = st.sidebar.slider("📅 Time Frame Global", min_value=hoy - timedelta(days=limite_dias), max_value=hoy, value=(hoy - timedelta(days=min(30, limite_dias)), hoy), format="YYYY-MM-DD")
dias_analizados = max((end_date - start_date).days, 1)

capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=13364.0, step=1000.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0

# --- 2. EXTRACCIÓN MAESTRA ---
@st.cache_data(ttl=120)
def cargar_matriz(exchange_id, sym, start, end, iv_down, iv_res, offset):
    try:
        ex_class = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp() * 1000)
        end_ts = int((datetime.combine(end, datetime.min.time()) + timedelta(days=1)).timestamp() * 1000)
        
        all_ohlcv, current_ts = [], start_ts
        while current_ts < end_ts:
            ohlcv = ex_class.fetch_ohlcv(sym, iv_down, since=current_ts, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
            if len(all_ohlcv) > 50000: break
            
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index + timedelta(hours=offset)
        df = df[~df.index.duplicated(keep='first')]
        
        if iv_down != iv_res: 
            df = df.resample(iv_res).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        if len(df) > 50:
            df['EMA_200'] = ta.ema(df['Close'], length=200).fillna(df['Close'])
            df['Vol_MA_100'] = ta.sma(df['Volume'], length=100).fillna(df['Volume'])
            df['RVol'] = df['Volume'] / df['Vol_MA_100'].replace(0, 1)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(df['High'] - df['Low']).replace(0, 0.001)
            df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50.0)
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0].fillna(0.0)

            df['KC_Upper'] = df['EMA_200'] + (df['ATR'] * 1.5)
            df['KC_Lower'] = df['EMA_200'] - (df['ATR'] * 1.5)
            bb = ta.bbands(df['Close'], length=20, std=2.0)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
                df.rename(columns={bb.columns[0]: 'BBL', bb.columns[2]: 'BBU'}, inplace=True)
            else: df['BBU'], df['BBL'] = df['Close'], df['Close']
            df['BBU'], df['BBL'] = df['BBU'].fillna(df['Close']), df['BBL'].fillna(df['Close'])

            df['Squeeze_On'] = (df['BBU'] < df['KC_Upper']) & (df['BBL'] > df['KC_Lower'])
            df['BB_Delta'] = (df['BBU'] - df['BBL']).diff().fillna(0)
            df['BB_Delta_Avg'] = df['BB_Delta'].rolling(10).mean().fillna(0)
            df['Vela_Verde'] = df['Close'] > df['Open']
            df['Vela_Roja'] = df['Close'] < df['Open']
            df['Cuerpo_Vela'] = abs(df['Close'] - df['Open'])
            
            df['PL30'] = df['Low'].rolling(30).min().fillna(df['Low'])
            df['PH30'] = df['High'].rolling(30).max().fillna(df['High'])
            df['PL100'] = df['Low'].rolling(100).min().fillna(df['Low'])
            df['PH100'] = df['High'].rolling(100).max().fillna(df['High'])
            df['PL300'] = df['Low'].rolling(300).min().fillna(df['Low'])
            df['PH300'] = df['High'].rolling(300).max().fillna(df['High'])
            
            basis_sigma = df['Close'].rolling(20).mean()
            dev_sigma = df['Close'].rolling(20).std().replace(0, 1)
            df['Z_Score'] = (df['Close'] - basis_sigma) / dev_sigma
            rsi_ma = df['RSI'].rolling(14).mean()
            df['RSI_Cross_Up'] = (df['RSI'] > rsi_ma) & (df['RSI'].shift(1) <= rsi_ma.shift(1))
            df['RSI_Cross_Dn'] = (df['RSI'] < rsi_ma) & (df['RSI'].shift(1) >= rsi_ma.shift(1))
            df['Retro_Peak'] = (df['RSI'] < 30) & (df['Close'] < df['BBL'])
            
            ap = (df['High'] + df['Low'] + df['Close']) / 3
            esa = ap.ewm(span=10).mean()
            d_wt = abs(ap - esa).ewm(span=10).mean()
            ci = (ap - esa) / (0.015 * d_wt.replace(0, 1))
            wt1 = ci.ewm(span=21).mean()
            wt2 = wt1.rolling(4).mean()
            df['WT_Cross_Up'] = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
            df['WT_Cross_Dn'] = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
            df['WT_Oversold'] = wt1 < -60
            df['WT_Overbought'] = wt1 > 60
            
            df['Macro_Bull'] = df['Close'] >= df['EMA_200']
            gc.collect()

        return df
    except Exception as e: 
        return pd.DataFrame()

df_global = cargar_matriz(exchange_sel, ticker, start_date, end_date, iv_download, iv_resample, utc_offset)
if df_global.empty:
    st.error(f"🚨 ERROR API: No hay datos para {ticker} en {exchange_sel.upper()}.")

# --- 3. MOTOR PRE-CÁLCULO TOPOLÓGICO ---
def inyectar_adn(df_sim, r_sens=1.5, w_factor=2.5):
    df_sim['Whale_Cond'] = df_sim['Cuerpo_Vela'] > (df_sim['ATR'] * 0.3)
    df_sim['Flash_Vol'] = (df_sim['RVol'] > (w_factor * 0.8)) & df_sim['Whale_Cond']
    
    df_sim['Target_Lock_Sup'] = df_sim[['PL30', 'PL100', 'PL300']].max(axis=1)
    df_sim['Target_Lock_Res'] = df_sim[['PH30', 'PH100', 'PH300']].min(axis=1)
    tol = df_sim['ATR'] * 0.5
    
    df_sim['Lock_Bounce'] = (df_sim['Low'] <= (df_sim['Target_Lock_Sup'] + tol)) & (df_sim['Close'] > df_sim['Target_Lock_Sup']) & df_sim['Vela_Verde']
    df_sim['Lock_Break'] = (df_sim['Close'] > df_sim['Target_Lock_Res']) & (df_sim['Open'] <= df_sim['Target_Lock_Res']) & df_sim['Flash_Vol'] & df_sim['Vela_Verde']
    df_sim['Lock_Reject'] = (df_sim['High'] >= (df_sim['Target_Lock_Res'] - tol)) & (df_sim['Close'] < df_sim['Target_Lock_Res']) & df_sim['Vela_Roja']
    df_sim['Lock_Breakd'] = (df_sim['Close'] < df_sim['Target_Lock_Sup']) & (df_sim['Open'] >= df_sim['Target_Lock_Sup']) & df_sim['Vela_Roja']
    
    dist_sup = (abs(df_sim['Close'] - df_sim['PL30']) / df_sim['Close']) * 100
    dist_res = (abs(df_sim['Close'] - df_sim['PH30']) / df_sim['Close']) * 100
    df_sim['Radar_Activo'] = (dist_sup <= r_sens) | (dist_res <= r_sens)

    buy_score = np.zeros(len(df_sim))
    buy_score = np.where(df_sim['Retro_Peak'] | df_sim['RSI_Cross_Up'], 30, buy_score)
    buy_score = np.where(df_sim['Retro_Peak'], 50, buy_score)
    buy_score = np.where((buy_score > 0) & df_sim['Radar_Activo'], buy_score + 25, buy_score)
    buy_score = np.where((buy_score > 0) & (df_sim['Z_Score'] < -2.0), buy_score + 15, buy_score)
    
    is_magenta = (buy_score >= 70) | df_sim['Retro_Peak']
    is_whale_icon = df_sim['Flash_Vol'] & df_sim['Vela_Verde'] & (~df_sim['Flash_Vol'].shift(1).fillna(False))
    df_sim['Pink_Whale_Buy'] = is_magenta & is_whale_icon
    
    df_sim['Neon_Up'] = df_sim['Squeeze_On'] & (df_sim['Close'] >= df_sim['BBU'] * 0.999) & df_sim['Vela_Verde']
    df_sim['Neon_Dn'] = df_sim['Squeeze_On'] & (df_sim['Close'] <= df_sim['BBL'] * 1.001) & df_sim['Vela_Roja']
    df_sim['Defcon_Buy'] = df_sim['Neon_Up'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    df_sim['Defcon_Sell'] = df_sim['Neon_Dn'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    
    scan_range = df_sim['ATR'] * 2.0
    ceil_w = np.zeros(len(df_sim))
    floor_w = np.zeros(len(df_sim))
    for p_col, w in [('PL30', 1), ('PH30', 1), ('PL100', 3), ('PH100', 3), ('PL300', 5), ('PH300', 5)]:
        p_val = df_sim[p_col].values
        c_val = df_sim['Close'].values
        ceil_w += np.where((p_val > c_val) & (p_val <= c_val + scan_range), w, 0)
        floor_w += np.where((p_val < c_val) & (p_val >= c_val - scan_range), w, 0)

    df_sim['Therm_Bounce'] = (floor_w >= 4) & df_sim['RSI_Cross_Up'] & ~(ceil_w >= 4)
    df_sim['Therm_Vacuum'] = (ceil_w <= 3) & df_sim['Neon_Up'] & ~(floor_w == 0)
    df_sim['Therm_Wall_Sell'] = (ceil_w >= 4) & df_sim['RSI_Cross_Dn']
    df_sim['Therm_Panic_Sell'] = (floor_w == 0) & df_sim['Vela_Roja']
    df_sim['Cielo_Libre'] = dist_res > (r_sens * 2) 

    df_sim['Nuclear_Buy'] = is_magenta & (df_sim['WT_Oversold'] | df_sim['WT_Cross_Up'])
    df_sim['Early_Buy'] = is_magenta
    df_sim['Nuclear_Sell'] = (df_sim['RSI'] > 70) & (df_sim['WT_Overbought'] | df_sim['WT_Cross_Dn'])
    df_sim['Early_Sell'] = (df_sim['RSI'] > 70) & df_sim['Vela_Roja']
    df_sim['Rebound_Buy'] = df_sim['RSI_Cross_Up'] & ~is_magenta
    return df_sim

# --- NÚCLEO FÍSICO C++ CÁLCULO DE INTERÉS COMPUESTO ---
def simular_crecimiento_exponencial(high_arr, low_arr, close_arr, open_arr, sig_buy_arr, sig_sell_arr, tp_arr, sl_arr, cap_ini, com_pct):
    n = len(high_arr)
    cap_activo = cap_ini
    en_pos = False
    precio_ent = 0.0
    tp_active = 0.0
    sl_active = 0.0
    
    g_profit, g_loss, num_trades, max_dd, peak = 0.0, 0.0, 0, cap_ini, 0.0
    
    for i in range(n):
        if en_pos:
            tp_price = precio_ent * (1 + tp_active/100)
            sl_price = precio_ent * (1 - sl_active/100)
            
            if high_arr[i] >= tp_price:
                bruta = cap_activo * (tp_active/100)
                costo = (cap_activo + bruta + cap_activo) * com_pct
                neta = bruta - costo
                cap_activo += neta
                if neta > 0: g_profit += neta
                else: g_loss += abs(neta)
                num_trades += 1
                en_pos = False
            elif low_arr[i] <= sl_price:
                bruta = cap_activo * (sl_active/100)
                costo = (cap_activo - bruta + cap_activo) * com_pct
                neta = -(bruta + costo)
                cap_activo += neta
                g_loss += abs(neta)
                num_trades += 1
                en_pos = False
            elif sig_sell_arr[i]:
                ret = (close_arr[i] - precio_ent) / precio_ent
                bruta = cap_activo * ret
                costo = (cap_activo + bruta + cap_activo) * com_pct
                neta = bruta - costo
                cap_activo += neta
                if neta > 0: g_profit += neta
                else: g_loss += abs(neta)
                num_trades += 1
                en_pos = False
                
            if cap_activo > peak: peak = cap_activo
            dd = (peak - cap_activo) / peak * 100
            if dd > max_dd: max_dd = dd
            if cap_activo <= 0: break
                
        if not en_pos and sig_buy_arr[i] and i+1 < n:
            precio_ent = open_arr[i+1]
            tp_active, sl_active = tp_arr[i], sl_arr[i]
            en_pos = True
            
    pf = g_profit / g_loss if g_loss > 0 else (1.0 if g_profit > 0 else 0.0)
    return cap_activo - cap_ini, pf, num_trades, max_dd

# NÚCLEO VISUAL PARA DIBUJAR
def simular_visual(df_sim, cap_ini, reinvest, com_pct):
    registro_trades = []
    n = len(df_sim)
    curva = np.full(n, cap_ini, dtype=float)
    
    h_arr, l_arr, c_arr, o_arr = df_sim['High'].values, df_sim['Low'].values, df_sim['Close'].values, df_sim['Open'].values
    buy_arr, sell_arr = df_sim['Signal_Buy'].values, df_sim['Signal_Sell'].values
    tp_arr, sl_arr = df_sim['Active_TP'].values, df_sim['Active_SL'].values
    f_arr = df_sim.index
    
    en_pos, p_ent, cap_act, divs, tp_act, sl_act = False, 0.0, cap_ini, 0.0, 0.0, 0.0
    
    for i in range(n):
        cierra = False
        if en_pos:
            tp_p = p_ent * (1 + tp_act/100)
            sl_p = p_ent * (1 - sl_act/100)
            if h_arr[i] >= tp_p:
                bruta = cap_act * (tp_act/100)
                costo = (cap_act + bruta + cap_act) * com_pct
                neta = bruta - costo
                reinv_amt = neta * (reinvest/100) if neta > 0 else neta
                divs += (neta - reinv_amt) if neta > 0 else 0
                cap_act += reinv_amt
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'TP', 'Precio': tp_p, 'Ganancia_$': neta})
                en_pos, cierra = False, True
            elif l_arr[i] <= sl_p:
                bruta = cap_act * (sl_act/100)
                costo = (cap_act - bruta + cap_act) * com_pct
                neta = -(bruta + costo)
                cap_act += neta
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'SL', 'Precio': sl_p, 'Ganancia_$': neta})
                en_pos, cierra = False, True
            elif sell_arr[i]:
                ret = (c_arr[i] - p_ent) / p_ent
                bruta = cap_act * ret
                costo = (cap_act + bruta + cap_act) * com_pct
                neta = bruta - costo
                reinv_amt = neta * (reinvest/100) if neta > 0 else neta
                divs += (neta - reinv_amt) if neta > 0 else 0
                cap_act += reinv_amt
                if cap_act <= 0: cap_act = 0
                registro_trades.append({'Fecha': f_arr[i], 'Tipo': 'DYN_WIN' if neta>0 else 'DYN_LOSS', 'Precio': c_arr[i], 'Ganancia_$': neta})
                en_pos, cierra = False, True

        if not en_pos and not cierra and buy_arr[i] and i+1 < n and cap_act > 0:
            p_ent = o_arr[i+1]
            tp_act, sl_act = tp_arr[i], sl_arr[i]
            en_pos = True
            registro_trades.append({'Fecha': f_arr[i+1], 'Tipo': 'ENTRY', 'Precio': p_ent, 'Ganancia_$': 0})

        if en_pos and cap_act > 0:
            ret_flot = (c_arr[i] - p_ent) / p_ent
            curva[i] = cap_act + (cap_act * ret_flot) + divs
        else:
            curva[i] = cap_act + divs
            
    return curva.tolist(), divs, cap_act, registro_trades, en_pos

# --- 4. TERMINAL RENDER ---
st.title("🛡️ The Apex Quant Terminal")
tab_tri, tab_jug, tab_def, tab_gen = st.tabs(["💠 TRINITY V357", "⚔️ JUGGERNAUT V356", "🚀 DEFCON V329", "🌌 GÉNESIS V320 (QUANTUM)"])

def renderizar_estrategia(strat_name, tab_obj, df_base):
    with tab_obj:
        if df_base.empty:
            return

        s_id = strat_name.split()[0]
        
        # 🔥 PATRÓN PRE-RENDER (EL SECRETO DEL HACKEO DE LA UI) 🔥
        # Si la IA dictó una orden, sobrescribimos la memoria ANTES de pintar la interfaz
        if st.session_state.get(f'update_pending_{s_id}', False):
            bp = st.session_state[f'pending_bp_{s_id}']
            if s_id == "GENESIS":
                for r in buy_rules:
                    st.session_state[f'gen_bull_b_{r}'] = (r in bp['b_bull'])
                    st.session_state[f'gen_bear_b_{r}'] = (r in bp['b_bear'])
                for r in sell_rules:
                    st.session_state[f'gen_bull_s_{r}'] = (r in bp['s_bull'])
                    st.session_state[f'gen_bear_s_{r}'] = (r in bp['s_bear'])
                st.session_state['gen_bull_tp'] = float(round(bp['tp_bull'], 1))
                st.session_state['gen_bull_sl'] = float(round(bp['sl_bull'], 1))
                st.session_state['gen_bear_tp'] = float(round(bp['tp_bear'], 1))
                st.session_state['gen_bear_sl'] = float(round(bp['sl_bear'], 1))
            else:
                st.session_state[f'sld_tp_{s_id}'] = float(round(bp['tp'], 1))
                st.session_state[f'sld_sl_{s_id}'] = float(round(bp['sl'], 1))
                if s_id == "TRINITY": st.session_state[f'sld_reinv_{s_id}'] = float(bp['reinv'])
                if s_id != "DEFCON":
                    st.session_state[f'sld_wh_{s_id}'] = float(bp['wh'])
                    st.session_state[f'sld_rd_{s_id}'] = float(bp['rd'])
            st.session_state[f'update_pending_{s_id}'] = False

        # --- MÓDULO GÉNESIS ---
        if s_id == "GENESIS":
            st.markdown("### 🌌 Singularidad Genética (Panel HFT)")
            st.info("Al presionar el botón de Recocido Cuántico, la IA probará miles de mutaciones. Si encuentra un crecimiento exponencial real, reescribirá estos controles automáticamente.")
            
            c_bull, c_bear = st.columns(2)
            c_bull.markdown("""
            <div style='background-color:rgba(0, 255, 0, 0.1); padding:10px; border-radius:10px; border-left: 5px solid lime;'>
            <h4 style='color:lime; margin-top:0;'>🟢 PROTOCOLO ALCISTA (EMA 200+)</h4>
            </div>""", unsafe_allow_html=True)
            
            for r in buy_rules: c_bull.checkbox(f"COMPRA: {r.replace('_', ' ')}", key=f"gen_bull_b_{r}")
            for r in sell_rules: c_bull.checkbox(f"CIERRE: {r.replace('_', ' ')}", key=f"gen_bull_s_{r}")
            c_bull.slider("🎯 TP Alcista (%)", 0.5, 20.0, step=0.5, key="gen_bull_tp")
            c_bull.slider("🛑 SL Alcista (%)", 0.5, 15.0, step=0.5, key="gen_bull_sl")

            c_bear.markdown("""
            <div style='background-color:rgba(255, 0, 0, 0.1); padding:10px; border-radius:10px; border-left: 5px solid red; margin-top:20px;'>
            <h4 style='color:red; margin-top:0;'>🔴 PROTOCOLO BAJISTA (EMA 200-)</h4>
            </div>""", unsafe_allow_html=True)
            
            for r in buy_rules: c_bear.checkbox(f"COMPRA: {r.replace('_', ' ')}", key=f"gen_bear_b_{r}")
            for r in sell_rules: c_bear.checkbox(f"CIERRE: {r.replace('_', ' ')}", key=f"gen_bear_s_{r}")
            c_bear.slider("🎯 TP Bajista (%)", 0.5, 20.0, step=0.5, key="gen_bear_tp")
            c_bear.slider("🛑 SL Bajista (%)", 0.5, 15.0, step=0.5, key="gen_bear_sl")

            st.markdown("---")
            if st.button("🌌 INICIAR RECOCIDO CUÁNTICO (Destruir al Mercado)", type="primary"):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                
                df_p = inyectar_adn(df_base.copy(), 1.5, 2.5)
                h_a, l_a, c_a, o_a = df_p['High'].values, df_p['Low'].values, df_p['Close'].values, df_p['Open'].values
                is_bull = df_p['Macro_Bull'].values
                
                b_mat = {r: df_p[r].values for r in buy_rules}
                s_mat = {r: df_p[r].values for r in sell_rules}
                
                best_fit = -float('inf')
                bp = None
                
                # 3000 UNIVERSOS. IA AGRESIVA, SIN MIEDO.
                for i in range(3000): 
                    # El algoritmo elige hasta 6 combinaciones (DEPREDADOR ACTIVO)
                    b_bull = random.sample(buy_rules, random.randint(1, 6))
                    b_bear = random.sample(buy_rules, random.randint(1, 6))
                    s_bull = random.sample(sell_rules, random.randint(1, 6))
                    s_bear = random.sample(sell_rules, random.randint(1, 6))
                    
                    b_cond_bull, b_cond_bear = np.zeros(len(df_p), dtype=bool), np.zeros(len(df_p), dtype=bool)
                    s_cond_bull, s_cond_bear = np.zeros(len(df_p), dtype=bool), np.zeros(len(df_p), dtype=bool)
                    
                    for r in b_bull: b_cond_bull |= b_mat[r]
                    for r in b_bear: b_cond_bear |= b_mat[r]
                    for r in s_bull: s_cond_bull |= s_mat[r]
                    for r in s_bear: s_cond_bear |= s_mat[r]
                    
                    f_buy = np.where(is_bull, b_cond_bull, b_cond_bear)
                    f_sell = np.where(is_bull, s_cond_bull, s_cond_bear)
                    
                    tp_bull, sl_bull = random.uniform(2.0, 15.0), random.uniform(1.0, 5.0)
                    tp_bear, sl_bear = random.uniform(2.0, 8.0), random.uniform(1.0, 3.0)
                    
                    f_tp = np.where(is_bull, tp_bull, tp_bear)
                    f_sl = np.where(is_bull, sl_bull, sl_bear)
                    
                    net, pf, nt, mdd = simular_crecimiento_exponencial(h_a, l_a, c_a, o_a, f_buy, f_sell, f_tp, f_sl, capital_inicial, comision_pct)
                    
                    # LA ECUACIÓN DEL DEPREDADOR (Si hace menos de 3 trades, es basura. Recompensa el Net Profit masivo)
                    if nt >= 3 and net > 0:
                        fit = (net * pf * np.sqrt(nt)) / (mdd + 1.0)
                        if fit > best_fit:
                            best_fit = fit
                            bp = {
                                'b_bull': b_bull, 'b_bear': b_bear, 's_bull': s_bull, 's_bear': s_bear,
                                'tp_bull': tp_bull, 'sl_bull': sl_bull, 'tp_bear': tp_bear, 'sl_bear': sl_bear,
                                'net': net, 'pf': pf, 'nt': nt
                            }
                
                ph_holograma.empty()
                if bp: 
                    # ORDEN DE HACKEO DE LA UI (PATRÓN SOMBRA)
                    st.session_state[f'update_pending_{s_id}'] = True
                    st.session_state[f'pending_bp_{s_id}'] = bp
                    
                    dna_str = f"🌌 THE QUANTUM SINGULARITY\nProfit {bp['pf']:.2f}x | Net +${bp['net']:,.2f} | Trades: {bp['nt']}\n\n🟢 BULL BUY: {bp['b_bull']}\n🟢 BULL SELL: {bp['s_bull']}\n🎯 BULL TP/SL: {round(bp['tp_bull'],1)}% / {round(bp['sl_bull'],1)}%\n\n🔴 BEAR BUY: {bp['b_bear']}\n🔴 BEAR SELL: {bp['s_bear']}\n🎯 BEAR TP/SL: {round(bp['tp_bear'],1)}% / {round(bp['sl_bear'],1)}%"
                    st.session_state['winning_dna'] = dna_str
                    st.rerun() # REINICIA EL SERVIDOR Y FUERZA LOS BOTONES
                else:
                    st.error("❌ 3000 universos. El mercado está totalmente plano o en colapso por las comisiones. No hay crecimiento exponencial posible aquí. Cambie de moneda.")

            if st.session_state.get('winning_dna') != "":
                st.success("¡ADN Extraído Exitosamente! Cópiame este bloque para el código PineScript:")
                st.code(st.session_state['winning_dna'], language="text")

            # RECONSTRUIR LÓGICA BASADA EN LA UI FÍSICA PARA DIBUJARLA
            df_strat = inyectar_adn(df_base.copy(), 1.5, 2.5)
            bull_b_cond, bear_b_cond = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            bull_s_cond, bear_s_cond = np.zeros(len(df_strat), dtype=bool), np.zeros(len(df_strat), dtype=bool)
            
            for r in buy_rules:
                if st.session_state.get(f'gen_bull_b_{r}'): bull_b_cond |= df_strat[r].values
                if st.session_state.get(f'gen_bear_b_{r}'): bear_b_cond |= df_strat[r].values
            for r in sell_rules:
                if st.session_state.get(f'gen_bull_s_{r}'): bull_s_cond |= df_strat[r].values
                if st.session_state.get(f'gen_bear_s_{r}'): bear_s_cond |= df_strat[r].values
                
            df_strat['Signal_Buy'] = np.where(df_strat['Macro_Bull'], bull_b_cond, bear_b_cond)
            df_strat['Signal_Sell'] = np.where(df_strat['Macro_Bull'], bull_s_cond, bear_s_cond)
            df_strat['Active_TP'] = np.where(df_strat['Macro_Bull'], st.session_state.get('gen_bull_tp', 5.0), st.session_state.get('gen_bear_tp', 3.0))
            df_strat['Active_SL'] = np.where(df_strat['Macro_Bull'], st.session_state.get('gen_bull_sl', 2.0), st.session_state.get('gen_bear_sl', 1.5))
            
            # En Génesis, asumimos el 100% de reinversión para mostrar el potencial cuántico
            eq_curve, divs, cap_act, t_log, pos_ab = simular_visual(df_strat, capital_inicial, 100.0, comision_pct)

        # --- BLOQUES NORMALES (TRINITY/JUGG/DEFCON) ---
        else:
            with st.form(f"form_{s_id}"):
                c1, c2, c3, c4 = st.columns(4)
                st.session_state[f'sld_tp_{s_id}'] = c1.slider(f"🎯 TP Base (%)", 0.5, 15.0, value=float(st.session_state.get(f'sld_tp_{s_id}', 3.0)), step=0.1)
                st.session_state[f'sld_sl_{s_id}'] = c2.slider(f"🛑 SL (%)", 0.5, 10.0, value=float(st.session_state.get(f'sld_sl_{s_id}', 1.5)), step=0.1)
                
                mac_sh, atr_sh, d_buy, d_sell = True, True, True, True
                if s_id == "TRINITY":
                    st.session_state[f'sld_reinv_{s_id}'] = c3.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state.get(f'sld_reinv_{s_id}', 50.0)), step=5.0)
                    st.session_state[f'sld_wh_{s_id}'] = c4.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state.get(f'sld_wh_{s_id}', 2.5)), step=0.1)
                elif s_id == "JUGGERNAUT":
                    st.session_state[f'sld_wh_{s_id}'] = c3.slider("🐋 Factor", 1.0, 5.0, value=float(st.session_state.get(f'sld_wh_{s_id}', 2.5)), step=0.1)
                    mac_sh = st.checkbox("Bloqueo Macro (EMA)", value=True)
                else:
                    d_buy = st.checkbox("Squeeze Up", value=True)
                    
                if st.form_submit_button("⚡ Aplicar"): st.rerun()

            c_ia1, c_ia2 = st.columns([1, 3])
            
            if c_ia2.button(f"🚀 Ejecutar IA Cuántica ({s_id})", use_container_width=True):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                best_fit = -999999
                bp = {}
                
                df_precalc = inyectar_adn(df_base.copy(), 1.5, st.session_state.get(f'sld_wh_{s_id}', 2.5))
                if s_id == "TRINITY":
                    b_cond = df_precalc['Pink_Whale_Buy'] | df_precalc['Lock_Bounce'] | df_precalc['Defcon_Buy']
                    s_cond = df_precalc['Defcon_Sell'] | df_precalc['Therm_Wall_Sell']
                elif s_id == "JUGGERNAUT":
                    b_cond = df_precalc['Pink_Whale_Buy'] | (df_precalc['Lock_Bounce'] & df_precalc['Macro_Bull'])
                    s_cond = df_precalc['Defcon_Sell'] | df_precalc['Therm_Wall_Sell']
                else:
                    b_cond, s_cond = df_precalc['Defcon_Buy'], df_precalc['Defcon_Sell']

                h_arr, l_arr, c_arr, o_arr = df_precalc['High'].values, df_precalc['Low'].values, df_precalc['Close'].values, df_precalc['Open'].values
                b_c, s_c = b_cond.values, s_cond.values
                
                for _ in range(120): 
                    rtp = round(random.uniform(1.2, 8.0), 1)
                    rsl = round(random.uniform(0.5, 3.5), 1)
                    rrv = round(random.uniform(20, 100), -1) if s_id == "TRINITY" else 0.0
                    rwh = round(random.uniform(1.5, 3.5), 1) if s_id != "DEFCON" else 2.5
                    rrd = round(random.uniform(0.5, 3.0), 1) if s_id != "DEFCON" else 1.5
                    
                    t_arr, sl_arr = np.full(len(df_precalc), rtp), np.full(len(df_precalc), rsl)
                    net, pf, nt, mdd = simular_crecimiento_exponencial(h_arr, l_arr, c_arr, o_arr, b_c, s_c, t_arr, sl_arr, capital_inicial, comision_pct)
                    
                    if nt > 2 and net > 0:
                        fit = (net * pf) / (mdd + 1.0)
                        if fit > best_fit:
                            best_fit, bp = fit, {'tp':rtp, 'sl':rsl, 'reinv':rrv, 'wh':rwh, 'rd':rrd}
                
                ph_holograma.empty()
                if bp:
                    st.session_state[f'update_pending_{s_id}'] = True
                    st.session_state[f'pending_bp_{s_id}'] = bp
                    st.rerun()
                else: st.error("IA: Mercado demasiado hostil para esta estrategia.")
            
            df_strat = inyectar_adn(df_base.copy(), st.session_state.get(f'sld_rd_{s_id}', 1.5), st.session_state.get(f'sld_wh_{s_id}', 2.5))
            if s_id == "TRINITY":
                df_strat['Signal_Buy'] = df_strat['Pink_Whale_Buy'] | df_strat['Lock_Bounce'] | df_strat['Defcon_Buy']
                df_strat['Signal_Sell'] = df_strat['Defcon_Sell'] | df_strat['Therm_Wall_Sell']
                df_strat['Active_TP'], df_strat['Active_SL'] = st.session_state[f'sld_tp_{s_id}'], st.session_state[f'sld_sl_{s_id}']
            elif s_id == "JUGGERNAUT":
                df_strat['Signal_Buy'] = df_strat['Pink_Whale_Buy'] | (df_strat['Lock_Bounce'] & df_strat['Macro_Bull'])
                df_strat['Signal_Sell'] = df_strat['Defcon_Sell'] | df_strat['Therm_Wall_Sell']
                df_strat['Active_TP'], df_strat['Active_SL'] = st.session_state[f'sld_tp_{s_id}'], st.session_state[f'sld_sl_{s_id}']
            else:
                df_strat['Signal_Buy'], df_strat['Signal_Sell'] = df_strat['Defcon_Buy'], df_strat['Defcon_Sell']
                df_strat['Active_TP'], df_strat['Active_SL'] = st.session_state[f'sld_tp_{s_id}'], st.session_state[f'sld_sl_{s_id}']
                
            eq_curve, divs, cap_act, t_log, pos_ab = simular_visual(df_strat, capital_inicial, st.session_state.get(f'sld_reinv_{s_id}', 0.0), comision_pct)

        # --- SECCIÓN COMÚN (MÉTRICAS Y GRÁFICO) ---
        df_strat['Total_Portfolio'] = eq_curve
        ret_pct = ((eq_curve[-1] - capital_inicial) / capital_inicial) * 100

        dftr = pd.DataFrame(t_log)
        tt, wr, pf_val, ado_act = 0, 0.0, 0.0, 0.0
        if not dftr.empty:
            exs = dftr[dftr['Tipo'].isin(['TP', 'SL', 'DYN_WIN', 'DYN_LOSS'])]
            tt = len(exs)
            ado_act = tt / dias_analizados if dias_analizados > 0 else 0
            if tt > 0:
                ws = len(exs[exs['Tipo'].isin(['TP', 'DYN_WIN'])])
                wr = (ws / tt) * 100
                gpp = exs[exs['Ganancia_$'] > 0]['Ganancia_$'].sum()
                gll = abs(exs[exs['Ganancia_$'] < 0]['Ganancia_$'].sum())
                pf_val = gpp / gll if gll > 0 else float('inf')
        
        mdd = abs((((pd.Series(eq_curve) - pd.Series(eq_curve).cummax()) / pd.Series(eq_curve).cummax()) * 100).min())

        st.markdown(f"### 📊 Auditoría: {s_id}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Portafolio Neto", f"${eq_curve[-1]:,.2f}", f"{ret_pct:.2f}%")
        c2.metric("Flujo Neto", f"${divs:,.2f}")
        c3.metric("Win Rate", f"{wr:.1f}%")
        c4.metric("Profit Factor", f"{pf_val:.2f}x")
        c5.metric("Max Drawdown", f"{mdd:.2f}%", delta_color="inverse")
        c6.metric("ADO ⚡", f"{ado_act:.2f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], name="Precio"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_200'], mode='lines', name='EMA 200', line=dict(color='orange', width=2)), row=1, col=1)

        if not dftr.empty:
            ents = dftr[dftr['Tipo'] == 'ENTRY']
            fig.add_trace(go.Scatter(x=ents['Fecha'], y=ents['Precio'] * 0.96, mode='markers', marker=dict(symbol='triangle-up', color='cyan', size=14)), row=1, col=1)
            wins = dftr[dftr['Tipo'].isin(['TP', 'DYN_WIN'])]
            fig.add_trace(go.Scatter(x=wins['Fecha'], y=wins['Precio'] * 1.04, mode='markers', marker=dict(symbol='triangle-down', color='#00FF00', size=14)), row=1, col=1)
            loss = dftr[dftr['Tipo'].isin(['SL', 'DYN_LOSS'])]
            fig.add_trace(go.Scatter(x=loss['Fecha'], y=loss['Precio'] * 1.04, mode='markers', marker=dict(symbol='triangle-down', color='#FF0000', size=14)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Total_Portfolio'], mode='lines', name='Equidad', line=dict(color='#00FF00', width=3)), row=2, col=1)

        fig.update_yaxes(side="right")
        fig.update_layout(template='plotly_dark', height=750, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), hovermode="closest", dragmode="pan")
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True}, key=f"chart_{s_id}")

renderizar_estrategia("TRINITY V357", tab_tri, df_global)
renderizar_estrategia("JUGGERNAUT V356", tab_jug, df_global)
renderizar_estrategia("DEFCON V329", tab_def, df_global)
renderizar_estrategia("GENESIS V320", tab_gen, df_global)
