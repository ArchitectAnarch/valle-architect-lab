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

# --- MEMORIA IA INDEPENDIENTE Y GENÉTICA ---
estrategias = ["TRINITY", "JUGGERNAUT", "DEFCON", "FORJA"]
for s in estrategias:
    if f'tp_{s}' not in st.session_state: st.session_state[f'tp_{s}'] = 3.0
    if f'sl_{s}' not in st.session_state: st.session_state[f'sl_{s}'] = 1.5
    if f'whale_{s}' not in st.session_state: st.session_state[f'whale_{s}'] = 2.5
    if f'radar_{s}' not in st.session_state: st.session_state[f'radar_{s}'] = 1.5
    if f'reinvest_{s}' not in st.session_state: st.session_state[f'reinvest_{s}'] = 50.0
    if f'ado_{s}' not in st.session_state: st.session_state[f'ado_{s}'] = 0.0

# Memoria Genética (Checkboxes de la Forja)
toggles_forja = ['n_b', 'c_b', 'm_b', 'd_b', 'w_b', 'n_s', 'c_s', 'm_s', 'd_s', 't_s']
for t in toggles_forja:
    if f'fj_{t}' not in st.session_state: st.session_state[f'fj_{t}'] = True

if 'forja_dna_result' not in st.session_state: st.session_state['forja_dna_result'] = ""

css_spinner = """
<style>
.loader-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; pointer-events: none; background: transparent; }
.rocket { font-size: 10rem; animation: spin 1s linear infinite; filter: drop-shadow(0 0 25px rgba(0, 255, 255, 0.9)); }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
<div class="loader-container"><div class="rocket">🚀</div></div>
"""
ph_holograma = st.empty()

# --- 1. PANEL LATERAL ---
logo_files = glob.glob("logo.*")
if logo_files: st.sidebar.image(logo_files[0], use_container_width=True)
else: st.sidebar.markdown("<h2 style='text-align: center; color: cyan;'>🚀 ROCKET PROTOCOL</h2>", unsafe_allow_html=True)

if st.sidebar.button("🔄 Sincronización Live", use_container_width=True): 
    st.cache_data.clear()
    gc.collect()

st.sidebar.markdown("---")
st.sidebar.header("📡 Enlace de Mercado")
exchanges_soportados = {"Coinbase (Pro)": "coinbase", "Binance": "binance", "Kraken": "kraken", "KuCoin": "kucoin"}
exchange_sel = st.sidebar.selectbox("🏦 Exchange", list(exchanges_soportados.keys()))
id_exchange = exchanges_soportados[exchange_sel]

ticker = st.sidebar.text_input("Símbolo Exacto (Ej. HNT/USD)", value="HNT/USD")
utc_offset = st.sidebar.number_input("🌍 Zona Horaria (UTC)", min_value=-12.0, max_value=14.0, value=-5.0, step=0.5)

intervalos = {
    "1 Minuto": ("1m", "1T"), "5 Minutos": ("5m", "5T"), 
    "7 Minutos": ("1m", "7T"), "13 Minutos": ("1m", "13T"), 
    "15 Minutos": ("15m", "15T"), "23 Minutos": ("1m", "23T"), 
    "30 Minutos": ("30m", "30T"), "1 Hora": ("1h", "1H"), 
    "2 Horas": ("1h", "2H"), "4 Horas": ("4h", "4H"), "1 Día": ("1d", "1D"), "1 Semana": ("1d", "1W")
}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=4) 
iv_download, iv_resample = intervalos[intervalo_sel]

hoy = datetime.today().date()
limite_dias = 7 if iv_download == "1m" else 90 if iv_download in ["5m", "15m", "30m"] else 1800
start_date, end_date = st.sidebar.slider("📅 Time Frame Global", min_value=hoy - timedelta(days=limite_dias), max_value=hoy, value=(hoy - timedelta(days=30 if limite_dias>30 else 7), hoy), format="YYYY-MM-DD")
dias_analizados = max((end_date - start_date).days, 1)

st.sidebar.markdown("---")
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=13364.0, step=1000.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0

# --- 2. EXTRACCIÓN Y PRE-CÁLCULO EXPANDIDO (V320) ---
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
        
        if len(df) > 300:
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
            
            df['Target_Lock_Sup'] = df[['PL30', 'PL100', 'PL300']].max(axis=1)
            df['Target_Lock_Res'] = df[['PH30', 'PH100', 'PH300']].min(axis=1)

            basis_sigma = df['Close'].rolling(20).mean()
            dev_sigma = df['Close'].rolling(20).std().replace(0, 1)
            df['Z_Score'] = (df['Close'] - basis_sigma) / dev_sigma
            df['RSI_Velocity'] = df['RSI'].diff().fillna(0)
            
            rsi_ma = df['RSI'].rolling(14).mean()
            df['RSI_Cross_Up'] = (df['RSI'] > rsi_ma) & (df['RSI'].shift(1) <= rsi_ma.shift(1))
            df['RSI_Cross_Dn'] = (df['RSI'] < rsi_ma) & (df['RSI'].shift(1) >= rsi_ma.shift(1))
            df['Retro_Peak'] = (df['RSI'] < 30) & (df['Close'] < df['BBL'])
            
            # --- AGREGADOS V320 PARA LA FORJA ---
            # Wavetrend
            ap = (df['High'] + df['Low'] + df['Close']) / 3
            esa = ta.ema(ap, length=10).fillna(ap)
            d_wt = ta.ema(abs(ap - esa), length=10).fillna(0.001).replace(0, 0.001)
            ci = (ap - esa) / (0.015 * d_wt)
            df['WT1'] = ta.ema(ci, length=21).fillna(0)
            df['WT2'] = ta.sma(df['WT1'], length=4).fillna(0)
            df['WT_Cross_Up'] = (df['WT1'] > df['WT2']) & (df['WT1'].shift(1) <= df['WT2'].shift(1))
            df['WT_Cross_Dn'] = (df['WT1'] < df['WT2']) & (df['WT1'].shift(1) >= df['WT2'].shift(1))
            df['WT_Oversold'] = df['WT1'] < -60
            df['WT_Overbought'] = df['WT1'] > 60

            # Wicks y Divergencias
            df['Upper_Wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['Lower_Wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
            df['Div_Bull'] = (df['Low'].shift(1) < df['Low'].shift(5)) & (df['RSI'].shift(1) > df['RSI'].shift(5)) & (df['RSI'] < 35)
            df['Div_Bear'] = (df['High'].shift(1) > df['High'].shift(5)) & (df['RSI'].shift(1) < df['RSI'].shift(5)) & (df['RSI'] > 65)

            del basis_sigma, dev_sigma, rsi_ma, ap, esa, d_wt, ci
            gc.collect()

        return df
    except Exception as e: 
        return pd.DataFrame()

ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
df_global = cargar_y_preprocesar(id_exchange, ticker, start_date, end_date, iv_download, iv_resample, utc_offset)
ph_holograma.empty() 

# --- 3. MOTOR CUÁNTICO MULTI-ESTRATEGIA ---
def generar_senales(df_sim, strat, w_factor, r_sens, macro_sh, atr_sh, def_buy, def_sell, toggles=None):
    df_sim['Whale_Cond'] = df_sim['Cuerpo_Vela'] > (df_sim['ATR'] * 0.3)
    df_sim['Flash_Vol'] = (df_sim['RVol'] > (w_factor * 0.8)) & df_sim['Whale_Cond']
    
    dist_sup = (abs(df_sim['Close'] - df_sim['Pivot_Low_30']) / df_sim['Close']) * 100
    dist_res = (abs(df_sim['Close'] - df_sim['Pivot_High_30']) / df_sim['Close']) * 100
    df_sim['Radar_Activo'] = (dist_sup <= r_sens) | (dist_res <= r_sens)
    
    buy_score = np.zeros(len(df_sim))
    buy_score = np.where(df_sim['Retro_Peak'] | df_sim['RSI_Cross_Up'], 30, buy_score)
    buy_score = np.where(df_sim['Retro_Peak'], 50, buy_score)
    buy_score = np.where((buy_score > 0) & df_sim['Radar_Activo'], buy_score + 25, buy_score)
    buy_score = np.where((buy_score > 0) & (df_sim['Z_Score'] < -2.0), buy_score + 15, buy_score)
    is_magenta_buy = (buy_score >= 70) | df_sim['Retro_Peak']
    
    sell_score = np.zeros(len(df_sim))
    retro_sell = (df_sim['RSI'] > 70) & (df_sim['Close'] > df_sim['BBU'])
    sell_score = np.where(retro_sell | df_sim['RSI_Cross_Dn'], 30, sell_score)
    sell_score = np.where(retro_sell, 50, sell_score)
    sell_score = np.where((sell_score > 0) & df_sim['Radar_Activo'], sell_score + 25, sell_score)
    sell_score = np.where((sell_score > 0) & (df_sim['Z_Score'] > 2.0), sell_score + 15, sell_score)
    is_magenta_sell = (sell_score >= 70) | retro_sell

    is_whale_icon = df_sim['Flash_Vol'] & df_sim['Vela_Verde'] & (~df_sim['Flash_Vol'].shift(1).fillna(False))
    
    df_sim['Neon_Up'] = df_sim['Squeeze_On'] & (df_sim['Close'] >= df_sim['BBU'] * 0.999) & df_sim['Vela_Verde']
    df_sim['Neon_Dn'] = df_sim['Squeeze_On'] & (df_sim['Close'] <= df_sim['BBL'] * 1.001) & df_sim['Vela_Roja']
    df_sim['Defcon_Buy'] = df_sim['Neon_Up'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    df_sim['Defcon_Sell'] = df_sim['Neon_Dn'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    df_sim['Therm_Wall_Sell'] = (df_sim['RSI'] > 70) & (df_sim['Close'] > df_sim['BBU']) & df_sim['Vela_Roja']
    
    if strat == "FORJA":
        # Cálculos de Fricción (Wicks) de V320
        dyn_w_req = np.where(df_sim['ADX'] < 40, 0.4, 0.5)
        dyn_v_req = np.where(df_sim['ADX'] < 40, 1.5, 1.8)
        f_w_req = np.where(df_sim['Radar_Activo'], 0.15, dyn_w_req)
        f_v_req = np.where(df_sim['Radar_Activo'], 1.2, dyn_v_req)
        
        wick_rej_buy = df_sim['Lower_Wick'] > (df_sim['Cuerpo_Vela'] * f_w_req)
        wick_rej_sell = df_sim['Upper_Wick'] > (df_sim['Cuerpo_Vela'] * f_w_req)
        vol_stop = df_sim['RVol'] > f_v_req
        
        climax_b = is_magenta_buy & (wick_rej_buy | vol_stop)
        climax_s = is_magenta_sell & (wick_rej_sell | vol_stop)
        
        nuclear_b = climax_b & (df_sim['WT_Oversold'] | df_sim['WT_Cross_Up'])
        nuclear_s = climax_s & (df_sim['WT_Overbought'] | df_sim['WT_Cross_Dn'])
        
        # Ensamble del ADN Genético
        sig_buy = np.zeros(len(df_sim), dtype=bool)
        if toggles['n_b']: sig_buy |= nuclear_b
        if toggles['c_b']: sig_buy |= climax_b
        if toggles['m_b']: sig_buy |= is_magenta_buy
        if toggles['d_b']: sig_buy |= df_sim['Defcon_Buy']
        if toggles['w_b']: sig_buy |= is_whale_icon
        
        sig_sell = np.zeros(len(df_sim), dtype=bool)
        if toggles['n_s']: sig_sell |= nuclear_s
        if toggles['c_s']: sig_sell |= climax_s
        if toggles['m_s']: sig_sell |= is_magenta_sell
        if toggles['d_s']: sig_sell |= df_sim['Defcon_Sell']
        if toggles['t_s']: sig_sell |= df_sim['Therm_Wall_Sell']
        
        df_sim['Signal_Buy'] = sig_buy
        df_sim['Signal_Sell'] = sig_sell

    elif "TRINITY" in strat:
        tol = df_sim['ATR'] * 0.5
        lock_bounce = df_sim['Radar_Activo'] & (df_sim['Low'] <= (df_sim['Target_Lock_Sup'] + tol)) & (df_sim['Close'] > df_sim['Target_Lock_Sup']) & df_sim['Vela_Verde']
        lock_break = df_sim['Radar_Activo'] & (df_sim['Close'] > df_sim['Target_Lock_Res']) & (df_sim['Open'] <= df_sim['Target_Lock_Res']) & df_sim['Flash_Vol'] & df_sim['Vela_Verde']
        lock_rej = df_sim['Radar_Activo'] & (df_sim['High'] >= (df_sim['Target_Lock_Res'] - tol)) & (df_sim['Close'] < df_sim['Target_Lock_Res']) & df_sim['Vela_Roja']
        lock_brk_dn = df_sim['Radar_Activo'] & (df_sim['Close'] < df_sim['Target_Lock_Sup']) & (df_sim['Open'] >= df_sim['Target_Lock_Sup']) & df_sim['Vela_Roja']

        df_sim['Signal_Buy'] = (is_magenta_buy & is_whale_icon) | lock_bounce | lock_break | df_sim['Defcon_Buy']
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell'] | lock_rej | lock_brk_dn

    elif "JUGGERNAUT" in strat:
        df_sim['Macro_Safe'] = df_sim['Close'] > df_sim['EMA_200'] if macro_sh else True
        df_sim['ATR_Safe'] = ~(df_sim['Cuerpo_Vela'].shift(1).fillna(0) > (df_sim['ATR'].shift(1).fillna(0.001) * 1.5)) if atr_sh else True
        df_sim['Signal_Buy'] = (is_magenta_buy & is_whale_icon) | ((df_sim['Radar_Activo'] | df_sim['Defcon_Buy']) & df_sim['Vela_Verde'] & df_sim['Macro_Safe'] & df_sim['ATR_Safe'])
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell']

    elif "DEFCON" in strat:
        df_sim['Signal_Buy'] = df_sim['Defcon_Buy'] if def_buy else False
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] if def_sell else False
        
    return df_sim

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
                registro_trades.append({'Fecha': fechas_arr[i], 'Tipo': 'DYNAMIC_WIN' if g_neta > 0 else 'DYNAMIC_LOSS', 'Precio': close_arr[i], 'Ganancia_$': g_neta})
                en_pos, trade_cerrado = False, True

        if not en_pos and not trade_cerrado and sig_buy_arr[i] and i + 1 < n:
            precio_ent = open_arr[i+1] 
            fecha_ent = fechas_arr[i+1]
            costo_ent = (cap_activo if is_trinity else cap_ini) * com_pct
            cap_activo -= costo_ent
            en_pos = True
            registro_trades.append({'Fecha': fecha_ent, 'Tipo': 'ENTRY', 'Precio': precio_ent, 'Ganancia_$': -costo_ent})

        if en_pos:
            ret_flot = (close_arr[i] - precio_ent) / precio_ent
            pnl_flot = (cap_activo if is_trinity else cap_ini) * ret_flot
            curva_capital[i] = (cap_activo + pnl_flot + divs) if is_trinity else (cap_activo + pnl_flot)
        else:
            curva_capital[i] = (cap_activo + divs) if is_trinity else cap_activo
            
    return curva_capital.tolist(), divs, cap_activo, registro_trades, en_pos

# --- 4. RENDERIZADO GENERAL ---
st.title("🛡️ Terminal Táctico Multipestaña")
tab_tri, tab_jug, tab_def, tab_forja = st.tabs(["💠 TRINITY V357", "⚔️ JUGGERNAUT V356", "🚀 DEFCON V329", "🧬 FORJA V320 (Creador)"])

def render_grafico(df_strat, s_id, t_log, eq_curve):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    ht_clean = "F: %{x}<br>P: $%{y:,.4f}<extra></extra>"
    fig.add_trace(go.Candlestick(x=df_strat.index, open=df_strat['Open'], high=df_strat['High'], low=df_strat['Low'], close=df_strat['Close'], name="Precio"), row=1, col=1)
    
    if 'BBU' in df_strat.columns:
        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_200'], mode='lines', name='EMA 200', line=dict(color='orange', width=2), hovertemplate=ht_clean), row=1, col=1)

    dftr = pd.DataFrame(t_log)
    if not dftr.empty:
        ents = dftr[dftr['Tipo'] == 'ENTRY']
        fig.add_trace(go.Scatter(
            x=ents['Fecha'], y=ents['Precio'] * 0.96, mode='markers', name='Compra', 
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

    fig.add_trace(go.Scatter(x=df_strat.index, y=eq_curve, mode='lines', name='Equidad ($)', line=dict(color='#00FF00', width=3), hovertemplate="Cap: $%{y:,.2f}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(side="right", fixedrange=False, row=1, col=1)
    fig.update_yaxes(side="right", fixedrange=False, row=2, col=1)
    fig.update_xaxes(fixedrange=False, showspikes=True, spikecolor="cyan", spikesnap="cursor", spikemode="toaxis+across", spikethickness=1, spikedash="solid")
    fig.update_layout(template='plotly_dark', height=750, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), hovermode="closest", dragmode="pan", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

# --- PESTAÑA 4: FORJA GENÉTICA ---
with tab_forja:
    if df_global.empty:
        st.warning("Datos vacíos.")
    else:
        s_id = "FORJA"
        st.markdown("### 🧬 Constructora de ADN Algorítmico (V320)")
        
        with st.form("form_forja"):
            c1, c2, c3, c4 = st.columns(4)
            t_tp = c1.slider("🎯 TP Base (%)", 0.5, 15.0, value=float(st.session_state[f'tp_{s_id}']), step=0.1)
            t_sl = c2.slider("🛑 SL (%)", 0.5, 10.0, value=float(st.session_state[f'sl_{s_id}']), step=0.1)
            t_whale = c3.slider("🐋 F. Ballena", 1.0, 5.0, value=float(st.session_state[f'whale_{s_id}']), step=0.1)
            t_radar = c4.slider("📡 Radar Sens.", 0.1, 5.0, value=float(st.session_state[f'radar_{s_id}']), step=0.1)
            
            st.markdown("**Activar Motores de Compra 🟢**")
            cb1, cb2, cb3, cb4, cb5 = st.columns(5)
            f_nb = cb1.checkbox("Nuclear Buy", value=st.session_state['fj_n_b'])
            f_cb = cb2.checkbox("Clímax Buy", value=st.session_state['fj_c_b'])
            f_mb = cb3.checkbox("Magenta Score (>70)", value=st.session_state['fj_m_b'])
            f_db = cb4.checkbox("Defcon Squeeze Up", value=st.session_state['fj_d_b'])
            f_wb = cb5.checkbox("Whale Icon", value=st.session_state['fj_w_b'])
            
            st.markdown("**Activar Motores de Venta 🔴**")
            cs1, cs2, cs3, cs4, cs5 = st.columns(5)
            f_ns = cs1.checkbox("Nuclear Sell", value=st.session_state['fj_n_s'])
            f_cs = cs2.checkbox("Clímax Sell", value=st.session_state['fj_c_s'])
            f_ms = cs3.checkbox("Magenta Score (>70)", value=st.session_state['fj_m_s'])
            f_ds = cs4
