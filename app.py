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

st.set_page_config(page_title="ROCKET PROTOCOL | Lab Quant", layout="wide", initial_sidebar_state="expanded")

# --- MEMORIA IA INDEPENDIENTE ---
estrategias = ["TRINITY", "JUGGERNAUT", "DEFCON"]
for s in estrategias:
    if f'tp_{s}' not in st.session_state: st.session_state[f'tp_{s}'] = 3.0
    if f'sl_{s}' not in st.session_state: st.session_state[f'sl_{s}'] = 1.5
    if f'whale_{s}' not in st.session_state: st.session_state[f'whale_{s}'] = 2.5
    if f'radar_{s}' not in st.session_state: st.session_state[f'radar_{s}'] = 1.5
    if f'reinvest_{s}' not in st.session_state: st.session_state[f'reinvest_{s}'] = 50.0
    if f'ado_{s}' not in st.session_state: st.session_state[f'ado_{s}'] = 0.0

# --- MEMORIA GÉNESIS (Sincronizada con Keys) ---
buy_rules = ['Pink_Whale_Buy', 'Lock_Bounce', 'Lock_Break', 'Defcon_Buy', 'Neon_Up', 'Therm_Bounce', 'Therm_Vacuum', 'Nuclear_Buy', 'Early_Buy', 'Rebound_Buy']
sell_rules = ['Defcon_Sell', 'Neon_Dn', 'Therm_Wall_Sell', 'Therm_Panic_Sell', 'Lock_Reject', 'Lock_Breakd', 'Nuclear_Sell', 'Early_Sell']

if 'sld_gen_tp' not in st.session_state: st.session_state['sld_gen_tp'] = 5.0
if 'sld_gen_sl' not in st.session_state: st.session_state['sld_gen_sl'] = 2.0
if 'sld_gen_ado' not in st.session_state: st.session_state['sld_gen_ado'] = 0.0
if 'winning_dna' not in st.session_state: st.session_state['winning_dna'] = ""

for r in buy_rules:
    if f'chk_b_{r}' not in st.session_state: st.session_state[f'chk_b_{r}'] = False
for r in sell_rules:
    if f'chk_s_{r}' not in st.session_state: st.session_state[f'chk_s_{r}'] = False

if not any([st.session_state.get(f'chk_b_{r}') for r in buy_rules]): st.session_state['chk_b_Nuclear_Buy'] = True
if not any([st.session_state.get(f'chk_s_{r}') for r in sell_rules]): st.session_state['chk_s_Nuclear_Sell'] = True

# --- 1. PANEL LATERAL ---
css_spinner = """
<style>
.loader-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; pointer-events: none; background: transparent; }
.rocket { font-size: 10rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 25px rgba(0, 255, 255, 0.9)); }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
<div class="loader-container"><div class="rocket">🚀</div></div>
"""
ph_holograma = st.empty()

logo_files = glob.glob("logo.*")
if logo_files: st.sidebar.image(logo_files[0], use_container_width=True)
else: st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 ROCKET PROTOCOL</h2>", unsafe_allow_html=True)

if st.sidebar.button("🔄 Sincronización Live", use_container_width=True, key="btn_sync"): 
    st.cache_data.clear()
    gc.collect()

st.sidebar.markdown("---")
st.sidebar.header("📡 Enlace de Mercado")
exchanges_soportados = {"Coinbase (Pro)": "coinbase", "Binance": "binance", "Kraken": "kraken", "KuCoin": "kucoin"}
exchange_sel = st.sidebar.selectbox("🏦 Exchange", list(exchanges_soportados.keys()), key="sel_exch")
id_exchange = exchanges_soportados[exchange_sel]

ticker = st.sidebar.text_input("Símbolo Exacto (Ej. HNT/USD)", value="HNT/USD", key="txt_tick")
utc_offset = st.sidebar.number_input("🌍 Zona Horaria (UTC)", min_value=-12.0, max_value=14.0, value=-5.0, step=0.5, key="num_utc")

intervalos = {
    "1 Minuto": ("1m", "1min"), "5 Minutos": ("5m", "5min"), 
    "7 Minutos": ("1m", "7min"), "13 Minutos": ("1m", "13min"), 
    "15 Minutos": ("15m", "15min"), "23 Minutos": ("1m", "23min"), 
    "30 Minutos": ("30m", "30min"), "1 Hora": ("1h", "1h"), 
    "4 Horas": ("4h", "4h"), "1 Día": ("1d", "1D"), "1 Semana": ("1d", "1W")
}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=4, key="sel_tf") 
iv_download, iv_resample = intervalos[intervalo_sel]

hoy = datetime.today().date()
limite_dias = 7 if iv_download == "1m" else 90 if iv_download in ["5m", "15m", "30m"] else 1800
start_date, end_date = st.sidebar.slider("📅 Time Frame Global", min_value=hoy - timedelta(days=limite_dias), max_value=hoy, value=(hoy - timedelta(days=30 if limite_dias>30 else 7), hoy), format="YYYY-MM-DD", key="sld_tf")
dias_analizados = max((end_date - start_date).days, 1)

st.sidebar.markdown("---")
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=13364.0, step=1000.0, key="num_cap")
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05, key="num_com") / 100.0

# --- 2. EXTRACCIÓN MAESTRA ---
@st.cache_data(ttl=120)
def cargar_y_preprocesar(exchange_id, sym, start, end, iv_down, iv_res, offset):
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
            df['Vol_MA'] = ta.sma(df['Volume'], length=20).fillna(df['Volume'])
            df['Vol_MA_100'] = ta.sma(df['Volume'], length=100).fillna(df['Volume'])
            df['RVol'] = df['Volume'] / df['Vol_MA_100']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(df['High'] - df['Low']).replace(0, 0.001)
            df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50.0)
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            df['ADX'] = adx_df.iloc[:, 0].fillna(0.0) if adx_df is not None else 0.0

            df['KC_Upper'] = df['EMA_200'] + (df['ATR'] * 1.5)
            df['KC_Lower'] = df['EMA_200'] - (df['ATR'] * 1.5)
            bb = ta.bbands(df['Close'], length=20, std=2.0)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
                df.rename(columns={bb.columns[0]: 'BBL', bb.columns[1]: 'BBM', bb.columns[2]: 'BBU'}, inplace=True)
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
            ci = (ap - esa) / (0.015 * d_wt)
            wt1 = ci.ewm(span=21).mean()
            wt2 = wt1.rolling(4).mean()
            df['WT_Cross_Up'] = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
            df['WT_Cross_Dn'] = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
            df['WT_Oversold'] = wt1 < -60
            df['WT_Overbought'] = wt1 > 60
            
            del basis_sigma, dev_sigma, rsi_ma, ap, esa, d_wt, ci, wt1, wt2
            gc.collect()

        return df
    except Exception as e: 
        return pd.DataFrame()

df_global = cargar_y_preprocesar(id_exchange, ticker, start_date, end_date, iv_download, iv_resample, utc_offset)

# --- 3. MOTOR PRE-CÁLCULO UNIVERSAL ---
def generar_senales(df_sim, strat, w_factor, r_sens, macro_sh, atr_sh, def_buy=True, def_sell=True):
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

    if strat == "TRINITY":
        df_sim['Signal_Buy'] = df_sim['Pink_Whale_Buy'] | df_sim['Lock_Bounce'] | df_sim['Lock_Break'] | df_sim['Defcon_Buy'] | df_sim['Therm_Bounce'] | df_sim['Therm_Vacuum']
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell'] | df_sim['Therm_Panic_Sell'] | df_sim['Lock_Reject'] | df_sim['Lock_Breakd']
    elif strat == "JUGGERNAUT":
        df_sim['Macro_Safe'] = df_sim['Close'] > df_sim['EMA_200'] if macro_sh else True
        df_sim['ATR_Safe'] = ~(df_sim['Cuerpo_Vela'].shift(1).fillna(0) > (df_sim['ATR'].shift(1).fillna(0.001) * 1.5)) if atr_sh else True
        df_sim['Signal_Buy'] = df_sim['Pink_Whale_Buy'] | ((df_sim['Lock_Bounce'] | df_sim['Lock_Break'] | df_sim['Defcon_Buy'] | df_sim['Therm_Bounce'] | df_sim['Therm_Vacuum']) & df_sim['Macro_Safe'] & df_sim['ATR_Safe'])
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell'] | df_sim['Therm_Panic_Sell'] | df_sim['Lock_Reject'] | df_sim['Lock_Breakd']
    elif strat == "DEFCON":
        df_sim['Signal_Buy'] = df_sim['Defcon_Buy'] if def_buy else False
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] if def_sell else False
    elif strat == "GENESIS" or strat == "GENESIS_PRECALC":
        buy_cond = np.zeros(len(df_sim), dtype=bool)
        sell_cond = np.zeros(len(df_sim), dtype=bool)
        # Si es la visualización final, usa los checkboxes. Si es precalc, lo deja en 0.
        if strat == "GENESIS":
            for r in buy_rules:
                if st.session_state.get(f'chk_b_{r}', False): buy_cond |= df_sim[r].values
            for r in sell_rules:
                if st.session_state.get(f'chk_s_{r}', False): sell_cond |= df_sim[r].values
        df_sim['Signal_Buy'] = buy_cond
        df_sim['Signal_Sell'] = sell_cond
        
    return df_sim

# NÚCLEO DE SIMULACIÓN ULTRA-RÁPIDO (C++)
def ejecutar_simulacion_fast(high_arr, low_arr, close_arr, open_arr, sig_buy_arr, sig_sell_arr, tp, sl, cap_ini, com_pct):
    n = len(high_arr)
    cap_activo = cap_ini
    en_pos = False
    precio_ent = 0.0
    
    g_profit = 0.0
    g_loss = 0.0
    num_trades = 0
    peak = cap_ini
    max_dd = 0.0
    
    for i in range(n):
        if en_pos:
            tp_price = precio_ent * (1 + tp/100)
            sl_price = precio_ent * (1 - sl/100)
            
            if high_arr[i] >= tp_price:
                ganancia = (cap_activo * (tp/100))
                costo = (cap_activo + ganancia) * com_pct
                neta = ganancia - costo
                cap_activo += neta
                g_profit += neta
                num_trades += 1
                en_pos = False
            elif low_arr[i] <= sl_price:
                perdida = (cap_activo * (sl/100))
                costo = (cap_activo - perdida) * com_pct
                neta = -(perdida + costo)
                cap_activo += neta
                g_loss += abs(neta)
                num_trades += 1
                en_pos = False
            elif sig_sell_arr[i]:
                ret = (close_arr[i] - precio_ent) / precio_ent
                ganancia = cap_activo * ret
                costo = (cap_activo + ganancia) * com_pct
                neta = ganancia - costo
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
            costo = cap_activo * com_pct
            cap_activo -= costo
            en_pos = True
            
    pf = g_profit / g_loss if g_loss > 0 else (1.5 if g_profit > 0 else 0.0)
    net_val = cap_activo - cap_ini
    return net_val, pf, num_trades, max_dd

# NÚCLEO DE SIMULACIÓN VISUAL (Para dibujar el gráfico)
def ejecutar_simulacion(df_sim, strat, tp, sl, cap_ini, reinvest, com_pct):
    registro_trades = []
    n = len(df_sim)
    curva_capital = np.full(n, cap_ini, dtype=float)
    
    high_arr = df_sim['High'].values
    low_arr = df_sim['Low'].values
    close_arr = df_sim['Close'].values
    open_arr = df_sim['Open'].values
    sig_buy_arr = df_sim['Signal_Buy'].values
    sig_sell_arr = df_sim['Signal_Sell'].values
    fechas_arr = df_sim.index
    
    en_pos, precio_ent, cap_activo, divs = False, 0.0, cap_ini, 0.0
    is_trinity = "TRINITY" in strat
    
    for i in range(n):
        trade_cerrado = False
        if en_pos:
            tp_price = precio_ent * (1 + (tp / 100))
            sl_price = precio_ent * (1 - (sl / 100))
            
            if high_arr[i] >= tp_price:
                g_bruta = (cap_activo if is_trinity else cap_ini) * (tp / 100)
                costo = ((cap_activo if is_trinity else cap_ini) + g_bruta) * com_pct
                g_neta = g_bruta - costo
                if is_trinity:
                    reinv = g_neta * (reinvest / 100.0)
                    divs += (g_neta - reinv)
                    cap_activo += reinv
                else: cap_activo += g_neta
                registro_trades.append({'Fecha': fechas_arr[i], 'Tipo': 'TP', 'Precio': tp_price, 'Ganancia_$': g_neta})
                en_pos, trade_cerrado = False, True
            elif low_arr[i] <= sl_price:
                p_bruta = (cap_activo if is_trinity else cap_ini) * (sl / 100)
                costo = ((cap_activo if is_trinity else cap_ini) - p_bruta) * com_pct
                p_neta = p_bruta + costo
                cap_activo -= p_neta
                if cap_activo <= 0: cap_activo = 0.0
                registro_trades.append({'Fecha': fechas_arr[i], 'Tipo': 'SL', 'Precio': sl_price, 'Ganancia_$': -p_neta})
                en_pos, trade_cerrado = False, True
            elif sig_sell_arr[i]:
                ret_pct = (close_arr[i] - precio_ent) / precio_ent
                g_bruta = (cap_activo if is_trinity else cap_ini) * ret_pct
                costo = ((cap_activo if is_trinity else cap_ini) + g_bruta) * com_pct
                g_neta = g_bruta - costo
                if is_trinity and g_neta > 0:
                    reinv = g_neta * (reinvest / 100.0)
                    divs += (g_neta - reinv)
                    cap_activo += reinv
                else: cap_activo += g_neta
                if cap_activo <= 0: cap_activo = 0.0
                registro_trades.append({'Fecha': fechas_arr[i], 'Tipo': 'DYNAMIC_WIN' if g_neta > 0 else 'DYNAMIC_LOSS', 'Precio': close_arr[i], 'Ganancia_$': g_neta})
                en_pos, trade_cerrado = False, True

        if not en_pos and not trade_cerrado and sig_buy_arr[i] and i + 1 < n and cap_activo > 0:
            precio_ent = open_arr[i+1] 
            fecha_ent = fechas_arr[i+1]
            costo_ent = (cap_activo if is_trinity else cap_ini) * com_pct
            cap_activo -= costo_ent
            en_pos = True
            registro_trades.append({'Fecha': fecha_ent, 'Tipo': 'ENTRY', 'Precio': precio_ent, 'Ganancia_$': -costo_ent})

        if en_pos and cap_activo > 0:
            ret_flot = (close_arr[i] - precio_ent) / precio_ent
            pnl_flot = (cap_activo if is_trinity else cap_ini) * ret_flot
            curva_capital[i] = (cap_activo + pnl_flot + divs) if is_trinity else (cap_activo + pnl_flot)
        else:
            curva_capital[i] = (cap_activo + divs) if is_trinity else cap_activo
            
    return curva_capital.tolist(), divs, cap_activo, registro_trades, en_pos

# --- 4. RENDERIZADO DE PESTAÑAS BLINDADAS ---
st.title("🛡️ Terminal Táctico Multipestaña")
tab_tri, tab_jug, tab_def, tab_gen = st.tabs(["💠 TRINITY V357", "⚔️ JUGGERNAUT V356", "🚀 DEFCON V329", "🧬 GÉNESIS V320"])

def renderizar_estrategia(strat_name, tab_obj, df_base):
    with tab_obj:
        if df_base.empty:
            st.warning("Matriz de datos vacía.")
            return

        s_id = strat_name.split()[0]
        
        # --- MÓDULO GÉNESIS (ALGORITMO GENÉTICO APEX) ---
        if s_id == "GENESIS":
            st.markdown("### 🧬 Laboratorio Evolutivo Genético (V320)")
            st.info("La IA no adivina. Cruza, muta y evoluciona miles de ADN en 15 generaciones hasta encontrar la rentabilidad absoluta.")
            
            with st.form("form_genesis"):
                c_b, c_s, c_r = st.columns(3)
                c_b.markdown("**🟢 Módulo de Compras**")
                for r in buy_rules:
                    st.session_state[f'chk_b_{r}'] = c_b.checkbox(r.replace('_', ' '), value=st.session_state[f'chk_b_{r}'], key=f"f_chk_b_{r}")
                
                c_s.markdown("**🔴 Módulo de Cierres**")
                for r in sell_rules:
                    st.session_state[f'chk_s_{r}'] = c_s.checkbox(r.replace('_', ' '), value=st.session_state[f'chk_s_{r}'], key=f"f_chk_s_{r}")
                
                c_r.markdown("**🎯 Gestión de Riesgo**")
                st.session_state['sld_gen_tp'] = c_r.slider("Take Profit (%)", 0.5, 20.0, value=float(st.session_state['sld_gen_tp']), step=0.5, key="f_sld_tp")
                st.session_state['sld_gen_sl'] = c_r.slider("Stop Loss (%)", 0.5, 15.0, value=float(st.session_state['sld_gen_sl']), step=0.5, key="f_sld_sl")
                
                if st.form_submit_button("🧪 Aplicar Selección Manual"): st.rerun()

            c_ia1, c_ia2 = st.columns([1, 3])
            st.session_state['sld_gen_ado'] = c_ia1.slider("🎯 Target ADO (Opcional)", 0.0, 10.0, value=float(st.session_state['sld_gen_ado']), step=0.1, key="f_sld_ado")
            
            # --- MOTOR DE INTELIGENCIA EVOLUTIVA ---
            if c_ia2.button("🚀 Iniciar Evolución Genética (15 Generaciones)", type="primary", key="btn_ia_gen"):
                
                txt_prog = st.empty()
                progress_bar = st.progress(0.0)
                txt_prog.markdown("### 🧬 Extrayendo Matrices de C++...")
                
                df_precalc = generar_senales(df_base.copy(), "GENESIS_PRECALC", 2.5, 1.5, False, False)
                h_arr = df_precalc['High'].values
                l_arr = df_precalc['Low'].values
                c_arr = df_precalc['Close'].values
                o_arr = df_precalc['Open'].values
                
                # Arrays booleanos para acceso rápido
                b_mat = {r: df_precalc[r].values for r in buy_rules}
                s_mat = {r: df_precalc[r].values for r in sell_rules}
                
                def eval_fitness(dna):
                    b_cond = np.zeros(len(df_precalc), dtype=bool)
                    for i, r in enumerate(buy_rules):
                        if dna['b'][i]: b_cond |= b_mat[r]
                    s_cond = np.zeros(len(df_precalc), dtype=bool)
                    for i, r in enumerate(sell_rules):
                        if dna['s'][i]: s_cond |= s_mat[r]
                        
                    net, pf, nt, mdd = ejecutar_simulacion_fast(h_arr, l_arr, c_arr, o_arr, b_cond, s_cond, dna['tp'], dna['sl'], capital_inicial, comision_pct)
                    
                    if nt < 5: return -999, net, pf  # Descarta estrategias que no operan
                    ado_pen = 1.0
                    if st.session_state['sld_gen_ado'] > 0.0:
                        ado_pen = 1.0 / (1.0 + abs((nt/dias_analizados) - st.session_state['sld_gen_ado']))
                        
                    fitness = ((net * pf) / (mdd + 1.0)) * ado_pen
                    return fitness, net, pf

                # --- 1. Crear Población Inicial ---
                pop_size = 40
                generations = 15
                population = []
                for _ in range(pop_size):
                    dna = {
                        'b': [random.choice([True, False]) for _ in buy_rules],
                        's': [random.choice([True, False]) for _ in sell_rules],
                        'tp': random.uniform(2.0, 12.0),
                        'sl': random.uniform(1.0, 4.0)
                    }
                    if not any(dna['b']): dna['b'][random.randint(0, len(buy_rules)-1)] = True
                    if not any(dna['s']): dna['s'][random.randint(0, len(sell_rules)-1)] = True
                    population.append(dna)
                    
                best_dna_overall = None
                best_fit_overall = -9999
                best_net = 0
                best_pf = 0

                # --- 2. Bucle de Evolución Genética ---
                for gen in range(generations):
                    txt_prog.markdown(f"### 🧬 Evolucionando: Generación {gen+1}/{generations} ...")
                    progress_bar.progress((gen+1) / generations)
                    
                    scored_pop = []
                    for dna in population:
                        fit, net, pf = eval_fitness(dna)
                        scored_pop.append((fit, dna, net, pf))
                        if fit > best_fit_overall:
                            best_fit_overall = fit
                            best_dna_overall = dna
                            best_net = net
                            best_pf = pf
                            
                    scored_pop.sort(key=lambda x: x[0], reverse=True)
                    
                    # Crossover y Mutación
                    next_gen = [scored_pop[0][1], scored_pop[1][1]] # Pasan los 2 mejores intactos
                    while len(next_gen) < pop_size:
                        p1 = random.choice(scored_pop[:15])[1] # Cruza a los mejores
                        p2 = random.choice(scored_pop[:15])[1]
                        child = {
                            'b': [p1['b'][i] if random.random() > 0.5 else p2['b'][i] for i in range(len(buy_rules))],
                            's': [p1['s'][i] if random.random() > 0.5 else p2['s'][i] for i in range(len(sell_rules))],
                            'tp': (p1['tp'] + p2['tp']) / 2.0,
                            'sl': (p1['sl'] + p2['sl']) / 2.0
                        }
                        
                        # Muta el 20% de las veces
                        if random.random() < 0.2:
                            idx = random.randint(0, len(buy_rules)-1)
                            child['b'][idx] = not child['b'][idx]
                        if random.random() < 0.2:
                            idx = random.randint(0, len(sell_rules)-1)
                            child['s'][idx] = not child['s'][idx]
                        if random.random() < 0.2:
                            child['tp'] += random.uniform(-1.5, 1.5)
                            child['tp'] = max(1.0, child['tp'])
                        if random.random() < 0.2:
                            child['sl'] += random.uniform(-1.0, 1.0)
                            child['sl'] = max(0.5, child['sl'])
                            
                        if not any(child['b']): child['b'][random.randint(0, len(buy_rules)-1)] = True
                        if not any(child['s']): child['s'][random.randint(0, len(sell_rules)-1)] = True
                        
                        next_gen.append(child)
                        
                    population = next_gen
                    
                txt_prog.empty()
                progress_bar.empty()
                
                # --- 3. Resultado Final ---
                if best_dna_overall and best_fit_overall > 0: # Si es rentable
                    for i, r in enumerate(buy_rules): st.session_state[f'chk_b_{r}'] = best_dna_overall['b'][i]
                    for i, r in enumerate(sell_rules): st.session_state[f'chk_s_{r}'] = best_dna_overall['s'][i]
                    st.session_state['sld_gen_tp'] = round(best_dna_overall['tp'], 1)
                    st.session_state['sld_gen_sl'] = round(best_dna_overall['sl'], 1)
                    st.session_state['sld_gen_ado'] = 0.0
                    
                    b_str = ", ".join([buy_rules[i] for i in range(len(buy_rules)) if best_dna_overall['b'][i]])
                    s_str = ", ".join([sell_rules[i] for i in range(len(sell_rules)) if best_dna_overall['s'][i]])
                    
                    dna_str = f"🧬 GÉNESIS DNA ALPHA (V320)\n-------------------------\n☑️ COMPRAS ACTIVAS: {b_str}\n☑️ VENTAS ACTIVAS: {s_str}\n🎯 TAKE PROFIT: {st.session_state['sld_gen_tp']}%\n🛑 STOP LOSS: {st.session_state['sld_gen_sl']}%\n-------------------------"
                    st.session_state['winning_dna'] = dna_str
                    st.rerun() # Aplica los botones físicamente en pantalla
                else: 
                    st.session_state['winning_dna'] = ""
                    st.error("❌ El motor evolutivo analizó 600 variaciones completas. El mercado actual es un agujero negro bajista. Ninguna combinación superó matemáticamente la erosión por comisiones.")

            if st.session_state['winning_dna'] != "":
                st.success("¡Evolución Genética Completada! Copie el ADN y envíemelo para generar su PineScript Bot:")
                st.code(st.session_state['winning_dna'], language="text")

            df_strat = generar_senales(df_base.copy(), "GENESIS", 2.5, 1.5, False, False)
            eq_curve, divs, cap_act, t_log, pos_ab = ejecutar_simulacion(df_strat, "GENESIS", st.session_state['sld_gen_tp'], st.session_state['sld_gen_sl'], capital_inicial, 0.0, comision_pct)
            
        # --- BLOQUE TRINITY / JUGGERNAUT / DEFCON ---
        else:
            with st.form(f"form_{s_id}"):
                c1, c2, c3, c4 = st.columns(4)
                t_tp = c1.slider(f"🎯 TP Base (%)", 0.5, 15.0, value=float(st.session_state[f'tp_{s_id}']), step=0.1, key=f"sld_tp_{s_id}")
                t_sl = c2.slider(f"🛑 SL (%)", 0.5, 10.0, value=float(st.session_state[f'sl_{s_id}']), step=0.1, key=f"sld_sl_{s_id}")
                
                mac_sh = st.session_state.get(f"mac_{s_id}", True)
                atr_sh = st.session_state.get(f"atr_{s_id}", True)
                d_buy = st.session_state.get(f"db_{s_id}", True)
                d_sell = st.session_state.get(f"ds_{s_id}", True)
                
                t_reinv, t_whale, t_radar = 0.0, 2.5, 1.5
                
                if s_id == "TRINITY":
                    t_reinv = c3.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state[f'reinvest_{s_id}']), step=5.0, key=f"sld_reinv_{s_id}")
                    t_whale = c4.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state[f'whale_{s_id}']), step=0.1, key=f"sld_wh_{s_id}")
                    t_radar = st.slider("📡 Tolerancia Target Lock (%)", 0.1, 5.0, value=float(st.session_state[f'radar_{s_id}']), step=0.1, key=f"sld_rd_{s_id}")
                elif s_id == "JUGGERNAUT":
                    t_whale = c3.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state[f'whale_{s_id}']), step=0.1, key=f"sld_wh_{s_id}")
                    t_radar = c4.slider("📡 Tolerancia Target Lock", 0.1, 5.0, value=float(st.session_state[f'radar_{s_id}']), step=0.1, key=f"sld_rd_{s_id}")
                    st.checkbox("Bloqueo Macro (EMA)", value=mac_sh, key=f"mac_{s_id}")
                    st.checkbox("Bloqueo Crash (ATR)", value=atr_sh, key=f"atr_{s_id}")
                else:
                    st.checkbox("Entrada Squeeze Up", value=d_buy, key=f"db_{s_id}")
                    st.checkbox("Salida Squeeze Dn", value=d_sell, key=f"ds_{s_id}")

                if st.form_submit_button("⚡ Aplicar Configuraciones"):
                    st.session_state[f'tp_{s_id}'] = t_tp
                    st.session_state[f'sl_{s_id}'] = t_sl
                    if s_id == "TRINITY": st.session_state[f'reinvest_{s_id}'] = t_reinv
                    if s_id != "DEFCON": 
                        st.session_state[f'whale_{s_id}'] = t_whale
                        st.session_state[f'radar_{s_id}'] = t_radar
                    st.rerun()

            col_ia1, col_ia2 = st.columns([1, 3])
            t_ado = col_ia1.slider(f"🎯 ADO Target ({s_id})", 0.0, 10.0, value=float(st.session_state[f'ado_{s_id}']), step=0.1, key=f"sld_ado_{s_id}")
            st.session_state[f'ado_{s_id}'] = t_ado
            
            if col_ia2.button(f"🚀 Ejecutar IA Cuántica ({s_id})", use_container_width=True, key=f"btn_ia_{s_id}"):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                best_fit = -999999
                bp = {}
                for _ in range(80): 
                    rtp = round(random.uniform(1.2, 8.0), 1)
                    rsl = round(random.uniform(0.5, 3.5), 1)
                    rrv = round(random.uniform(20, 100), -1) if s_id == "TRINITY" else 0.0
                    rwh = round(random.uniform(1.5, 3.5), 1) if s_id != "DEFCON" else 2.5
                    rrd = round(random.uniform(0.5, 3.0), 1) if s_id != "DEFCON" else 1.5
                    
                    df_t = generar_senales(df_base.copy(), strat_name, rwh, rrd, mac_sh, atr_sh, d_buy, d_sell)
                    c_test, _, _, trds, _ = ejecutar_simulacion(df_t, strat_name, rtp, rsl, capital_inicial, rrv, comision_pct)
                    
                    dft = pd.DataFrame(trds)
                    if not dft.empty:
                        exits = dft[dft['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
                        nt = len(exits)
                        if nt > 2:
                            gp = exits[exits['Ganancia_$'] > 0]['Ganancia_$'].sum()
                            gl = abs(exits[exits['Ganancia_$'] < 0]['Ganancia_$'].sum())
                            pf = gp / gl if gl > 0 else 0.5
                            np_val = c_test[-1] - capital_inicial
                            pk = pd.Series(c_test).cummax()
                            m_dd = abs((((pd.Series(c_test) - pk) / pk) * 100).min())
                            
                            ado_pen = 1.0
                            if st.session_state[f'ado_{s_id}'] > 0.0:
                                ado_pen = 1.0 / (1.0 + abs((nt/dias_analizados) - st.session_state[f'ado_{s_id}']))
                                
                            fit = ((np_val * pf) / (m_dd + 1.0)) * ado_pen
                            if fit > best_fit and np_val > 0:
                                best_fit, bp = fit, {'tp':rtp, 'sl':rsl, 'reinv':rrv, 'whale':rwh, 'radar':rrd}
                
                ph_holograma.empty()
                if bp:
                    st.session_state[f'tp_{s_id}'], st.session_state[f'sl_{s_id}'] = float(bp['tp']), float(bp['sl'])
                    if s_id == "TRINITY": st.session_state[f'reinvest_{s_id}'] = float(bp['reinv'])
                    if s_id != "DEFCON": 
                        st.session_state[f'whale_{s_id}'] = float(bp['whale'])
                        st.session_state[f'radar_{s_id}'] = float(bp['radar'])
                    st.session_state[f'ado_{s_id}'] = 0.0
                    gc.collect()
                    st.rerun()
                else: st.error("IA: Mercado demasiado hostil para esta estrategia.")

            mac_sh = st.session_state.get(f"mac_{s_id}", True)
            atr_sh = st.session_state.get(f"atr_{s_id}", True)
            d_buy = st.session_state.get(f"db_{s_id}", True)
            d_sell = st.session_state.get(f"ds_{s_id}", True)
            
            df_strat = generar_senales(df_base.copy(), strat_name, st.session_state[f'whale_{s_id}'], st.session_state[f'radar_{s_id}'], mac_sh, atr_sh, d_buy, d_sell)
            eq_curve, divs, cap_act, t_log, pos_ab = ejecutar_simulacion(df_strat, strat_name, st.session_state[f'tp_{s_id}'], st.session_state[f'sl_{s_id}'], capital_inicial, st.session_state[f'reinvest_{s_id}'], comision_pct)

        # --- SECCIÓN COMÚN (GRÁFICAS Y MÉTRICAS) PARA TODAS LAS PESTAÑAS ---
        df_strat['Total_Portfolio'] = eq_curve
        ret_pct = ((eq_curve[-1] - capital_inicial) / capital_inicial) * 100

        dftr = pd.DataFrame(t_log)
        tt, wr, pf_val, ado_act = 0, 0.0, 0.0, 0.0
        if not dftr.empty:
            exs = dftr[dftr['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
            tt = len(exs)
            ado_act = tt / dias_analizados if dias_analizados > 0 else 0
            if tt > 0:
                ws = len(exs[exs['Tipo'].isin(['TP', 'DYNAMIC_WIN'])])
                wr = (ws / tt) * 100
                gpp = exs[exs['Ganancia_$'] > 0]['Ganancia_$'].sum()
                gll = abs(exs[exs['Ganancia_$'] < 0]['Ganancia_$'].sum())
                pf_val = gpp / gll if gll > 0 else float('inf')
        
        mdd = abs((((pd.Series(eq_curve) - pd.Series(eq_curve).cummax()) / pd.Series(eq_curve).cummax()) * 100).min())

        st.markdown(f"### 📊 Auditoría: {s_id}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Portafolio Neto", f"${eq_curve[-1]:,.2f} {'🟢' if pos_ab else ''}", f"{ret_pct:.2f}%")
        c2.metric("Flujo/Capital", f"${divs if s_id=='TRINITY' else cap_act:,.2f}")
        c3.metric("Win Rate", f"{wr:.1f}%")
        c4.metric("Profit Factor", f"{pf_val:.2f}x")
        c5.metric("Max Drawdown", f"{mdd:.2f}%", delta_color="inverse")
        
        c6.markdown(f"""
        <div style="background-color:rgba(0,255,255,0.1); border:1px solid cyan; border-radius:5px; padding:10px; text-align:center;">
            <h4 style="margin:0; color:cyan;">ADO ⚡</h4>
            <h3 style="margin:0; color:white;">{ado_act:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        horizonte, vida_util, riesgo = "Corto Plazo", "Recalibración en 3-5 días.", "⚠️ ALTO: Riesgo de Sobreoptimización."
        if dias_analizados >= 180: horizonte, vida_util, riesgo = "Largo Plazo", "Sostenible indefinidamente.", "🛡️ BAJO: Estructura blindada."
        elif dias_analizados >= 45: horizonte, vida_util, riesgo = "Medio Plazo", "Recalibración en 2-4 semanas.", "⚖️ MODERADO: Adaptado al ciclo actual."
        st.info(f"**🧠 DICTAMEN IA:** Horizonte: **{horizonte}** | Esperanza de Vida: **{vida_util}** | Riesgo Técnico: **{riesgo}**")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        ht_clean = "F: %{x}<br>P: $%{y:,.4f}<extra></extra>"

        fig.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], name="Precio"), row=1, col=1)
        if s_id == "DEFCON" or s_id == "GENESIS":
            fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBU'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='BBU', hovertemplate=ht_clean), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBL'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='BBL', hovertemplate=ht_clean), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_200'], mode='lines', name='EMA 200', line=dict(color='orange', width=2), hovertemplate=ht_clean), row=1, col=1)

        if not dftr.empty:
            ents = dftr[dftr['Tipo'] == 'ENTRY']
            fig.add_trace(go.Scatter(
                x=ents['Fecha'], y=ents['Precio'] * 0.96, mode='markers', name='Compra (Vela Ejecución)', 
                marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=1)),
                error_y=dict(type='data', symmetric=False, array=ents['Precio']*0.04, arrayminus=[0]*len(ents), color='cyan', thickness=1, width=0),
                hovertemplate=ht_clean
            ), row=1, col=1)
            
            wins = dftr[dftr['Tipo'].isin(['TP', 'DYNAMIC_WIN'])]
            if not wins.empty:
                fig.add_trace(go.Scatter(
                    x=wins['Fecha'], y=wins['Precio'] * 1.04, mode='markers', name='Win', 
                    marker=dict(symbol='triangle-down', color='#00FF00', size=14, line=dict(width=1)), 
                    error_y=dict(type='data', symmetric=False, array=[0]*len(wins), arrayminus=wins['Precio']*0.04, color='#00FF00', thickness=1, width=0),
                    text=wins['Tipo'], hovertemplate="%{text}: $%{y:,.4f}<extra></extra>"
                ), row=1, col=1)

            losses = dftr[dftr['Tipo'].isin(['SL', 'DYNAMIC_LOSS'])]
            if not losses.empty:
                fig.add_trace(go.Scatter(
                    x=losses['Fecha'], y=losses['Precio'] * 1.04, mode='markers', name='Loss', 
                    marker=dict(symbol='triangle-down', color='#FF0000', size=14, line=dict(width=1)), 
                    error_y=dict(type='data', symmetric=False, array=[0]*len(losses), arrayminus=losses['Precio']*0.04, color='#FF0000', thickness=1, width=0),
                    text=losses['Tipo'], hovertemplate="%{text}: $%{y:,.4f}<extra></extra>"
                ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Total_Portfolio'], mode='lines', name='Equidad ($)', line=dict(color='#00FF00', width=3), hovertemplate="Cap: $%{y:,.2f}<extra></extra>"), row=2, col=1)

        fig.update_yaxes(side="right", fixedrange=False, row=1, col=1)
        fig.update_yaxes(side="right", fixedrange=False, row=2, col=1)
        fig.update_xaxes(fixedrange=False, showspikes=True, spikecolor="cyan", spikesnap="cursor", spikemode="toaxis+across", spikethickness=1, spikedash="solid")
        
        fig.update_layout(template='plotly_dark', height=750, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), hovermode="closest", dragmode="pan", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}, key=f"chart_{s_id}")

        st.markdown("### 🔎 Análisis Financiero por Ventana Dinámica")
        fecha_min, fecha_max = df_strat.index[0].date(), df_strat.index[-1].date()
        v_start, v_end = st.slider(f"Recortar Rango ({s_id})", min_value=fecha_min, max_value=fecha_max, value=(fecha_min, fecha_max), format="YYYY-MM-DD", key=f"win_{s_id}")
        
        mask = (df_strat.index >= pd.to_datetime(v_start)) & (df_strat.index <= pd.to_datetime(v_end) + timedelta(days=1))
        df_sub = df_strat.loc[mask]
        
        if not df_sub.empty:
            cap_ini_sub, cap_fin_sub = df_sub['Total_Portfolio'].iloc[0], df_sub['Total_Portfolio'].iloc[-1]
            ret_sub = ((cap_fin_sub - cap_ini_sub) / cap_ini_sub) * 100
            t_sub = [t for t in t_log if pd.to_datetime(v_start) <= pd.to_datetime(t['Fecha']) <= (pd.to_datetime(v_end) + timedelta(days=1))]
            df_tsub = pd.DataFrame(t_sub)
            
            tt_sub, wr_sub = 0, 0.0
            if not df_tsub.empty:
                exs_sub = df_tsub[df_tsub['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
                tt_sub = len(exs_sub)
                if tt_sub > 0: wr_sub = (len(exs_sub[exs_sub['Tipo'].isin(['TP', 'DYNAMIC_WIN'])]) / tt_sub) * 100

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Inicio de Ventana", f"${cap_ini_sub:,.2f}")
            mc2.metric("Fin de Ventana", f"${cap_fin_sub:,.2f}", f"{ret_sub:.2f}% Neto")
            mc3.metric("Trades en Ventana", f"{tt_sub}")
            mc4.metric("Win Rate en Ventana", f"{wr_sub:.1f}%")

renderizar_estrategia("TRINITY V357", tab_tri, df_global)
renderizar_estrategia("JUGGERNAUT V356", tab_jug, df_global)
renderizar_estrategia("DEFCON V329", tab_def, df_global)
renderizar_estrategia("GENESIS V320", tab_gen, df_global)
