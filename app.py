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

# --- MEMORIA GÉNESIS ---
if 'gen_tp' not in st.session_state: st.session_state.gen_tp = 3.0
if 'gen_sl' not in st.session_state: st.session_state.gen_sl = 1.5
if 'gen_ado' not in st.session_state: st.session_state.gen_ado = 0.0
if 'winning_dna' not in st.session_state: st.session_state.winning_dna = ""

buy_rules = ['Pink_Whale_Buy', 'Lock_Bounce', 'Lock_Break', 'Defcon_Buy', 'Neon_Up', 'Therm_Bounce', 'Therm_Vacuum', 'Nuclear_Buy', 'Early_Buy', 'Rebound_Buy']
sell_rules = ['Defcon_Sell', 'Neon_Dn', 'Therm_Wall_Sell', 'Therm_Panic_Sell', 'Lock_Reject', 'Lock_Breakd', 'Nuclear_Sell', 'Early_Sell']

for r in buy_rules:
    if f'gen_b_{r}' not in st.session_state: st.session_state[f'gen_b_{r}'] = False
for r in sell_rules:
    if f'gen_s_{r}' not in st.session_state: st.session_state[f'gen_s_{r}'] = False

if not any([st.session_state[f'gen_b_{r}'] for r in buy_rules]): st.session_state['gen_b_Nuclear_Buy'] = True
if not any([st.session_state[f'gen_s_{r}'] for r in sell_rules]): st.session_state['gen_s_Nuclear_Sell'] = True

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
    "2 Horas": ("1h", "2h"), "4 Horas": ("4h", "4h"), "1 Día": ("1d", "1D"), "1 Semana": ("1d", "1W")
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

ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
df_global = cargar_y_preprocesar(id_exchange, ticker, start_date, end_date, iv_download, iv_resample, utc_offset)
ph_holograma.empty() 

# --- 3. MOTOR PRE-CÁLCULO UNIVERSAL ---
def generar_senales(df_sim, strat, w_factor, r_sens, macro_sh, atr_sh):
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

    # Mapeo según la estrategia seleccionada
    if strat == "TRINITY":
        df_sim['Signal_Buy'] = df_sim['Pink_Whale_Buy'] | df_sim['Lock_Bounce'] | df_sim['Lock_Break'] | df_sim['Defcon_Buy'] | df_sim['Therm_Bounce'] | df_sim['Therm_Vacuum']
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell'] | df_sim['Therm_Panic_Sell'] | df_sim['Lock_Reject'] | df_sim['Lock_Breakd']
    elif strat == "JUGGERNAUT":
        df_sim['Macro_Safe'] = df_sim['Close'] > df_sim['EMA_200'] if macro_sh else True
        df_sim['ATR_Safe'] = ~(df_sim['Cuerpo_Vela'].shift(1).fillna(0) > (df_sim['ATR'].shift(1).fillna(0.001) * 1.5)) if atr_sh else True
        df_sim['Signal_Buy'] = df_sim['Pink_Whale_Buy'] | ((df_sim['Lock_Bounce'] | df_sim['Lock_Break'] | df_sim['Defcon_Buy'] | df_sim['Therm_Bounce'] | df_sim['Therm_Vacuum']) & df_sim['Macro_Safe'] & df_sim['ATR_Safe'])
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell'] | df_sim['Therm_Panic_Sell'] | df_sim['Lock_Reject'] | df_sim['Lock_Breakd']
    elif strat == "GENESIS":
        buy_cond = np.zeros(len(df_sim), dtype=bool)
        for r in buy_rules:
            if st.session_state.get(f'gen_b_{r}', False): buy_cond |= df_sim[r].values
        
        sell_cond = np.zeros(len(df_sim), dtype=bool)
        for r in sell_rules:
            if st.session_state.get(f'gen_s_{r}', False): sell_cond |= df_sim[r].values
            
        df_sim['Signal_Buy'] = buy_cond
        df_sim['Signal_Sell'] = sell_cond
        
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
    cielo_arr = df_sim['Cielo_Libre'].values if 'Cielo_Libre' in df_sim.columns else np.zeros(n, dtype=bool)
    whale_arr = df_sim['Pink_Whale_Buy'].values if 'Pink_Whale_Buy' in df_sim.columns else np.zeros(n, dtype=bool)
    fechas_arr = df_sim.index
    
    en_pos, precio_ent, cap_activo, divs = False, 0.0, cap_ini, 0.0
    is_trinity = "TRINITY" in strat
    tp_dinamico_activo = False 
    
    for i in range(n):
        trade_cerrado = False
        if en_pos:
            tp_efectivo = tp * 1.5 if tp_dinamico_activo else tp
            tp_price = precio_ent * (1 + (tp_efectivo / 100))
            sl_price = precio_ent * (1 - (sl / 100))
            
            if high_arr[i] >= tp_price:
                g_bruta = (cap_activo if is_trinity else cap_ini) * (tp_efectivo / 100)
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
            tp_dinamico_activo = whale_arr[i] or cielo_arr[i] 
            registro_trades.append({'Fecha': fecha_ent, 'Tipo': 'ENTRY', 'Precio': precio_ent, 'Ganancia_$': -costo_ent})

        if en_pos:
            ret_flot = (close_arr[i] - precio_ent) / precio_ent
            pnl_flot = (cap_activo if is_trinity else cap_ini) * ret_flot
            curva_capital[i] = (cap_activo + pnl_flot + divs) if is_trinity else (cap_activo + pnl_flot)
        else:
            curva_capital[i] = (cap_activo + divs) if is_trinity else cap_activo
            
    return curva_capital.tolist(), divs, cap_activo, registro_trades, en_pos

# --- 4. RENDERIZADO DE PESTAÑAS ---
st.title("🛡️ Terminal Táctico Multipestaña")
tab_tri, tab_jug, tab_def, tab_gen = st.tabs(["💠 TRINITY V357", "⚔️ JUGGERNAUT V356", "🚀 DEFCON V329", "🧬 GÉNESIS V320"])

def renderizar_estrategia(strat_name, tab_obj, df_base):
    with tab_obj:
        if df_base.empty:
            st.warning("Matriz de datos vacía.")
            return

        s_id = strat_name.split()[0]
        
        # --- MÓDULO GÉNESIS (DEEP SCAN) ---
        if s_id == "GENESIS":
            st.markdown("### 🧬 Laboratorio Genético Deep Scan (V320)")
            st.info("La IA vectorizará matrices probando de 1 a 4 lógicas simultáneas en compras y ventas hasta encontrar la combinación perfecta.")
            
            with st.form("form_genesis"):
                c_b, c_s, c_r = st.columns(3)
                c_b.markdown("**🟢 Módulo de Compras**")
                for r in buy_rules:
                    st.session_state[f'gen_b_{r}'] = c_b.checkbox(r.replace('_', ' '), value=st.session_state[f'gen_b_{r}'], key=f"chk_b_{r}")
                
                c_s.markdown("**🔴 Módulo de Cierres**")
                for r in sell_rules:
                    st.session_state[f'gen_s_{r}'] = c_s.checkbox(r.replace('_', ' '), value=st.session_state[f'gen_s_{r}'], key=f"chk_s_{r}")
                
                c_r.markdown("**🎯 Gestión de Riesgo**")
                st.session_state.gen_tp = c_r.slider("Take Profit (%)", 0.5, 20.0, value=float(st.session_state.gen_tp), step=0.1, key="sld_gen_tp")
                st.session_state.gen_sl = c_r.slider("Stop Loss (%)", 0.5, 15.0, value=float(st.session_state.gen_sl), step=0.1, key="sld_gen_sl")
                
                if st.form_submit_button("🧪 Aplicar Selección Manual"): st.rerun()

            c_ia1, c_ia2 = st.columns([1, 3])
            st.session_state.gen_ado = c_ia1.slider("🎯 Target ADO", 0.0, 10.0, value=float(st.session_state.gen_ado), step=0.1, key="sld_gen_ado")
            
            # --- MOTOR DE FUERZA BRUTA VECTORIZADA ---
            if c_ia2.button("🚀 Iniciar Deep Scan (500 Mutaciones)", type="primary", key="btn_ia_gen"):
                ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
                
                # Precalculamos todo UNA VEZ para que la IA vuele
                df_precalc = generar_senales(df_base.copy(), "GENESIS_PRECALC", 2.5, 1.5, False, False)
                
                best_fit = -999999
                bp = {}
                
                # 500 ciclos profundos
                for _ in range(500): 
                    # Elige entre 1 y 4 lógicas de compra al mismo tiempo
                    b_sample = random.sample(buy_rules, random.randint(1, 4))
                    # Elige entre 1 y 4 lógicas de venta
                    s_sample = random.sample(sell_rules, random.randint(1, 4))
                    
                    buy_cond = np.zeros(len(df_precalc), dtype=bool)
                    for r in b_sample: buy_cond |= df_precalc[r].values
                        
                    sell_cond = np.zeros(len(df_precalc), dtype=bool)
                    for r in s_sample: sell_cond |= df_precalc[r].values
                    
                    df_test = df_precalc.copy()
                    df_test['Signal_Buy'] = buy_cond
                    df_test['Signal_Sell'] = sell_cond
                    
                    tp_test = round(random.uniform(2.0, 15.0), 1)
                    sl_test = round(random.uniform(1.0, 5.0), 1)
                    
                    c_test, _, _, trds, _ = ejecutar_simulacion(df_test, "GENESIS", tp_test, sl_test, capital_inicial, 0, comision_pct)
                    
                    dft = pd.DataFrame(trds)
                    if not dft.empty:
                        exits = dft[dft['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
                        nt = len(exits)
                        if nt > 5:
                            gp = exits[exits['Ganancia_$'] > 0]['Ganancia_$'].sum()
                            gl = abs(exits[exits['Ganancia_$'] < 0]['Ganancia_$'].sum())
                            pf = gp / gl if gl > 0 else 0.5
                            np_val = c_test[-1] - capital_inicial
                            pk = pd.Series(c_test).cummax()
                            m_dd = abs((((pd.Series(c_test) - pk) / pk) * 100).min())
                            
                            ado_pen = 1.0
                            if st.session_state.gen_ado > 0.0:
                                ado_pen = 1.0 / (1.0 + abs((nt/dias_analizados) - st.session_state.gen_ado))
                                
                            fit = ((np_val * pf) / (m_dd + 1.0)) * ado_pen
                            if fit > best_fit and np_val > 0:
                                best_fit = fit
                                bp = {'b':b_sample, 's':s_sample, 'tp':tp_test, 'sl':sl_test}
                
                ph_holograma.empty()
                if bp:
                    # Enciende los botones en el panel
                    for k in buy_rules: st.session_state[f'gen_b_{k}'] = (k in bp['b'])
                    for k in sell_rules: st.session_state[f'gen_s_{k}'] = (k in bp['s'])
                    st.session_state.gen_tp = float(bp['tp'])
                    st.session_state.gen_sl = float(bp['sl'])
                    st.session_state.gen_ado = 0.0
                    
                    b_str = ", ".join(bp['b'])
                    s_str = ", ".join(bp['s'])
                    
                    # Genera el Bloque de Código para Copiar
                    dna_str = f"🧬 FÓRMULA GANADORA V320:\n\nCOMPRAS = [ {b_str} ]\nVENTAS  = [ {s_str} ]\nTP = {bp['tp']}%\nSL = {bp['sl']}%"
                    st.session_state.winning_dna = dna_str
                    st.rerun()
                else: 
                    st.error("No se encontraron sinergias rentables en 500 universos.")

            if st.session_state.winning_dna != "":
                st.success("¡Evolución Genética Completada! Copie el ADN y envíemelo para generar su PineScript:")
                st.code(st.session_state.winning_dna, language="text")

            df_strat = generar_senales(df_base.copy(), "GENESIS", 2.5, 1.5, False, False)
            eq_curve, divs, cap_act, t_log, pos_ab = ejecutar_simulacion(df_strat, "GENESIS", st.session_state.gen_tp, st.session_state.gen_sl, capital_inicial, 0.0, comision_pct)
            
        # --- BLOQUE TRINITY / JUGGERNAUT / DEFCON ---
        else:
            with st.form(f"form_{s_id}"):
                c1, c2, c3, c4 = st.columns(4)
                t_tp = c1.slider(f"🎯 TP Base (%)", 0.5, 15.0, value=float(st.session_state[f'tp_{s_id}']), step=0.1, key=f"sld_tp_{s_id}")
                t_sl = c2.slider(f"🛑 SL (%)", 0.5, 10.0, value=float(st.session_state[f'sl_{s_id}']), step=0.1, key=f"sld_sl_{s_id}")
                
                mac_sh = st.session_state.get(f"mac_{s_id}", True)
                atr_sh = st.session_state.get(f"atr_{s_id}", True)
                
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
                    
                    df_t = generar_senales(df_base.copy(), strat_name, rwh, rrd, mac_sh, atr_sh)
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
            df_strat = generar_senales(df_base.copy(), strat_name, st.session_state[f'whale_{s_id}'], st.session_state[f'radar_{s_id}'], mac_sh, atr_sh)
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

        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Total_Portfolio'], mode='lines', name='Equidad ($)', line=dict(color='#00FF00', width=3), hovertemplate="Cap: $%{y:,.2f}<extra></extra>"), row=2, col=1)

        fig.update_yaxes(side="right", fixedrange=False, row=1, col=1)
        fig.update_yaxes(side="right", fixedrange=False, row=2, col=1)
        fig.update_xaxes(fixedrange=False, showspikes=True, spikecolor="cyan", spikesnap="cursor", spikemode="toaxis+across", spikethickness=1, spikedash="solid")
        
        fig.update_layout(template='plotly_dark', height=750, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), hovermode="closest", dragmode="pan", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

renderizar_estrategia("TRINITY V357", tab_tri, df_global)
renderizar_estrategia("JUGGERNAUT V356", tab_jug, df_global)
renderizar_estrategia("DEFCON V329", tab_def, df_global)
renderizar_estrategia("GENESIS V320", tab_gen, df_global)
