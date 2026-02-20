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

# --- HOLOGRAMA COHETE CSS ---
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

if st.sidebar.button("🔄 Sincronización Live", use_container_width=True): st.cache_data.clear()
st.sidebar.markdown("---")

st.sidebar.header("📡 Enlace de Mercado")
exchanges_soportados = {"Coinbase (Pro)": "coinbase", "Binance": "binance", "Kraken": "kraken", "KuCoin": "kucoin"}
exchange_sel = st.sidebar.selectbox("🏦 Exchange", list(exchanges_soportados.keys()))
id_exchange = exchanges_soportados[exchange_sel]

ticker = st.sidebar.text_input("Símbolo Exacto (Ej. HNT/USD)", value="HNT/USD")

intervalos = {
    "1 Minuto": ("1m", "1T"), "5 Minutos": ("5m", "5T"), 
    "7 Minutos": ("1m", "7T"), "13 Minutos": ("1m", "13T"), 
    "15 Minutos": ("15m", "15T"), "23 Minutos": ("1m", "23T"), 
    "30 Minutos": ("30m", "30T"), "1 Hora": ("1h", "1H"), 
    "2 Horas": ("1h", "2H"), "4 Horas": ("4h", "4H"), "1 Día": ("1d", "1D")
}
intervalo_sel = st.sidebar.selectbox("Temporalidad", list(intervalos.keys()), index=4) 
iv_download, iv_resample = intervalos[intervalo_sel]

hoy = datetime.today().date()
limite_dias = 30 if iv_download == "1m" else 730 if iv_download in ["5m", "15m", "30m"] else 1800
start_date, end_date = st.sidebar.slider("📅 Time Frame Global", min_value=hoy - timedelta(days=limite_dias), max_value=hoy, value=(hoy - timedelta(days=30), hoy), format="YYYY-MM-DD")
dias_analizados = max((end_date - start_date).days, 1)

st.sidebar.markdown("---")
capital_inicial = st.sidebar.number_input("Capital Inicial (USD)", value=13364.0, step=1000.0)
comision_pct = st.sidebar.number_input("Comisión (%)", value=0.25, step=0.05) / 100.0

# --- 2. EXTRACCIÓN Y PRE-CÁLCULO ---
@st.cache_data(ttl=60)
def cargar_y_preprocesar(exchange_id, sym, start, end, iv_down, iv_res):
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
            if len(all_ohlcv) > 200000: break
            
        if not all_ohlcv: return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        if iv_down != iv_res: df = df.resample(iv_res).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        if len(df) > 5:
            df['EMA_200'] = ta.ema(df['Close'], length=200).fillna(df['Close'])
            df['Vol_MA'] = ta.sma(df['Volume'], length=20).fillna(df['Volume'])
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
            df['Vela_Verde'], df['Vela_Roja'] = df['Close'] > df['Open'], df['Close'] < df['Open']
            df['Cuerpo_Vela'] = abs(df['Close'] - df['Open'])
            df['Pivot_Low_30'] = df['Low'].rolling(window=30, center=False).min().fillna(df['Low'])
            df['Pivot_High_30'] = df['High'].rolling(window=30, center=False).max().fillna(df['High'])
            
        return df
    except Exception: return pd.DataFrame()

ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
df_global = cargar_y_preprocesar(id_exchange, ticker, start_date, end_date, iv_download, iv_resample)
ph_holograma.empty() 

# --- 3. MOTOR ACELERADO POR NUMPY (V18.0) ---
def generar_senales(df_sim, strat, w_factor, r_sens, macro_sh, atr_sh, def_buy, def_sell):
    df_sim['Whale_Cond'] = df_sim['Cuerpo_Vela'] > (df_sim['ATR'] * 0.3)
    df_sim['Vol_Anormal'] = (df_sim['Volume'] > (df_sim['Vol_MA'] * w_factor)) & df_sim['Whale_Cond']
    df_sim['Radar_Activo'] = ((abs(df_sim['Close'] - df_sim['Pivot_Low_30']) / df_sim['Close']) * 100 <= r_sens) | ((abs(df_sim['Close'] - df_sim['Pivot_High_30']) / df_sim['Close']) * 100 <= r_sens)
    
    df_sim['Neon_Up'] = df_sim['Squeeze_On'] & (df_sim['Close'] >= df_sim['BBU'] * 0.999) & df_sim['Vela_Verde']
    df_sim['Neon_Dn'] = df_sim['Squeeze_On'] & (df_sim['Close'] <= df_sim['BBL'] * 1.001) & df_sim['Vela_Roja']
    df_sim['Defcon_Buy'] = df_sim['Neon_Up'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    df_sim['Defcon_Sell'] = df_sim['Neon_Dn'] & (df_sim['BB_Delta'] > df_sim['BB_Delta_Avg']) & (df_sim['ADX'] > 20)
    df_sim['Therm_Wall_Sell'] = (df_sim['RSI'] > 70) & (df_sim['Close'] > df_sim['BBU']) & df_sim['Vela_Roja']

    if "TRINITY" in strat:
        df_sim['Signal_Buy'] = (df_sim['Vol_Anormal'] & df_sim['Vela_Verde']) | ((df_sim['Radar_Activo'] | df_sim['Defcon_Buy']) & df_sim['Vela_Verde'])
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell']
    elif "JUGGERNAUT" in strat:
        df_sim['Macro_Safe'] = df_sim['Close'] > df_sim['EMA_200'] if macro_sh else True
        df_sim['ATR_Safe'] = ~(df_sim['Cuerpo_Vela'].shift(1).fillna(0) > (df_sim['ATR'].shift(1).fillna(0.001) * 1.5)) if atr_sh else True
        df_sim['Signal_Buy'] = (df_sim['Vol_Anormal'] & df_sim['Vela_Verde']) | ((df_sim['Radar_Activo'] | df_sim['Defcon_Buy']) & df_sim['Vela_Verde'] & df_sim['Macro_Safe'] & df_sim['ATR_Safe'])
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] | df_sim['Therm_Wall_Sell']
    elif "DEFCON" in strat:
        df_sim['Signal_Buy'] = df_sim['Defcon_Buy'] if def_buy else False
        df_sim['Signal_Sell'] = df_sim['Defcon_Sell'] if def_sell else False
    return df_sim

def ejecutar_simulacion(df_sim, strat, tp, sl, cap_ini, reinvest, com_pct):
    registro_trades = []
    n = len(df_sim)
    curva_capital = np.full(n, cap_ini, dtype=float)
    
    # ⚡ EXTRACCIÓN A MATRICES DE C (NUMPY) PARA VELOCIDAD EXTREMA ⚡
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

# --- 4. RENDERIZADO DE PESTAÑAS ---
st.title("🛡️ Terminal Táctico Multipestaña")
tab_tri, tab_jug, tab_def = st.tabs(["💠 TRINITY V357", "⚔️ JUGGERNAUT V356", "🚀 DEFCON V329"])

def renderizar_estrategia(strat_name, tab_obj, df_base):
    with tab_obj:
        if df_base.empty:
            st.warning("Matriz de datos vacía.")
            return

        s_id = strat_name.split()[0]
        
        with st.form(f"form_{s_id}"):
            c1, c2, c3, c4 = st.columns(4)
            t_tp = c1.slider(f"🎯 TP (%)", 0.5, 15.0, value=float(st.session_state[f'tp_{s_id}']), step=0.1)
            t_sl = c2.slider(f"🛑 SL (%)", 0.5, 10.0, value=float(st.session_state[f'sl_{s_id}']), step=0.1)
            
            t_reinv, t_whale, t_radar, mac_sh, atr_sh, d_buy, d_sell = 0.0, 2.5, 1.5, True, True, True, True
            
            if s_id == "TRINITY":
                t_reinv = c3.slider("💵 Reinversión (%)", 0.0, 100.0, value=float(st.session_state[f'reinvest_{s_id}']), step=5.0)
                t_whale = c4.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state[f'whale_{s_id}']), step=0.1)
                t_radar = st.slider("📡 Hitbox Radar (%)", 0.1, 5.0, value=float(st.session_state[f'radar_{s_id}']), step=0.1)
            elif s_id == "JUGGERNAUT":
                t_whale = c3.slider("🐋 Factor Ballena", 1.0, 5.0, value=float(st.session_state[f'whale_{s_id}']), step=0.1)
                t_radar = c4.slider("📡 Hitbox Radar", 0.1, 5.0, value=float(st.session_state[f'radar_{s_id}']), step=0.1)
                mac_sh = st.checkbox("Bloqueo Macro (EMA)", value=True, key=f"mac_{s_id}")
                atr_sh = st.checkbox("Bloqueo Crash (ATR)", value=True, key=f"atr_{s_id}")
            else:
                d_buy = c3.checkbox("Entrada Squeeze Up", value=True, key=f"db_{s_id}")
                d_sell = c4.checkbox("Salida Squeeze Dn", value=True, key=f"ds_{s_id}")

            if st.form_submit_button("⚡ Aplicar Configuraciones"):
                st.session_state[f'tp_{s_id}'], st.session_state[f'sl_{s_id}'] = t_tp, t_sl
                if s_id == "TRINITY": st.session_state[f'reinvest_{s_id}'] = t_reinv
                if s_id != "DEFCON": st.session_state[f'whale_{s_id}'], st.session_state[f'radar_{s_id}'] = t_whale, t_radar
                st.rerun()

        col_ia1, col_ia2 = st.columns([1, 3])
        t_ado = col_ia1.slider(f"🎯 ADO Target ({s_id})", 0.0, 10.0, value=float(st.session_state[f'ado_{s_id}']), step=0.1)
        st.session_state[f'ado_{s_id}'] = t_ado
        
        if col_ia2.button(f"🚀 Ejecutar IA Cuántica ({s_id})", use_container_width=True):
            ph_holograma.markdown(css_spinner, unsafe_allow_html=True)
            best_fit = -999999
            bp = {}
            for _ in range(150):
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
                            sim_ado = nt / dias_analizados
                            ado_pen = 1.0 / (1.0 + abs(sim_ado - st.session_state[f'ado_{s_id}']))
                            
                        fit = ((np_val * pf) / (m_dd + 1.0)) * ado_pen
                        if fit > best_fit and np_val > 0:
                            best_fit, bp = fit, {'tp':rtp, 'sl':rsl, 'reinv':rrv, 'whale':rwh, 'radar':rrd}
            
            ph_holograma.empty()
            if bp:
                st.session_state[f'tp_{s_id}'], st.session_state[f'sl_{s_id}'] = float(bp['tp']), float(bp['sl'])
                if s_id == "TRINITY": st.session_state[f'reinvest_{s_id}'] = float(bp['reinv'])
                if s_id != "DEFCON": st.session_state[f'whale_{s_id}'], st.session_state[f'radar_{s_id}'] = float(bp['whale']), float(bp['radar'])
                st.session_state[f'ado_{s_id}'] = 0.0
                st.rerun()
            else: st.error("IA: Mercado demasiado hostil para esta estrategia bajo estos parámetros.")

        df_strat = generar_senales(df_base.copy(), strat_name, st.session_state[f'whale_{s_id}'], st.session_state[f'radar_{s_id}'], mac_sh, atr_sh, d_buy, d_sell)
        eq_curve, divs, cap_act, t_log, pos_ab = ejecutar_simulacion(df_strat, strat_name, st.session_state[f'tp_{s_id}'], st.session_state[f'sl_{s_id}'], capital_inicial, st.session_state[f'reinvest_{s_id}'], comision_pct)
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
        if s_id == "DEFCON":
            fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBU'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='BBU', hovertemplate=ht_clean), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['BBL'], mode='lines', line=dict(color='rgba(0,255,255,0.3)', width=1), name='BBL', hovertemplate=ht_clean), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['EMA_200'], mode='lines', name='EMA 200', line=dict(color='orange', width=2), hovertemplate=ht_clean), row=1, col=1)

        if not dftr.empty:
            ents = dftr[dftr['Tipo'] == 'ENTRY']
            fig.add_trace(go.Scatter(x=ents['Fecha'], y=ents['Precio'] * 0.98, mode='markers', name='Compra', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=1))), row=1, col=1)
            sals = dftr[dftr['Tipo'].isin(['TP', 'SL', 'DYNAMIC_WIN', 'DYNAMIC_LOSS'])]
            cs = ['#00FF00' if t in ['TP', 'DYNAMIC_WIN'] else '#FF0000' for t in sals['Tipo']]
            fig.add_trace(go.Scatter(x=sals['Fecha'], y=sals['Precio'] * 1.02, mode='markers', name='Venta', marker=dict(symbol='triangle-down', color=cs, size=14, line=dict(width=1)), text=sals['Tipo'], hovertemplate="%{text}: $%{y:,.4f}<extra></extra>"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_strat.index, y=df_strat['Total_Portfolio'], mode='lines', name='Equidad ($)', line=dict(color='#00FF00', width=3), hovertemplate="Cap: $%{y:,.2f}<extra></extra>"), row=2, col=1)

        fig.update_yaxes(side="right", fixedrange=False, row=1, col=1)
        fig.update_yaxes(side="right", fixedrange=False, row=2, col=1)
        fig.update_xaxes(fixedrange=False, showspikes=True, spikecolor="cyan", spikesnap="cursor", spikemode="toaxis+across", spikethickness=1, spikedash="solid")
        fig.update_yaxes(showspikes=True, spikecolor="cyan", spikesnap="cursor", spikemode="toaxis+across", spikethickness=1, spikedash="solid")

        fig.update_layout(template='plotly_dark', height=750, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20), hovermode="closest", dragmode="pan", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']})

        # MÓDULO DE VENTANA DINÁMICA
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
