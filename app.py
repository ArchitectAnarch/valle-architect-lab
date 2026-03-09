import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import itertools

st.set_page_config(page_title="Forja Genética V355", page_icon="🧬", layout="wide")
st.title("🧬 FORJA GENÉTICA V355: Optimizador de Algoritmos")
st.markdown("Descarga data real, simula miles de escenarios en segundos y forja el código Pine Script perfecto para tu token.")

# --- 1. CONFIGURACIÓN DEL ENTORNO ---
col1, col2, col3, col4 = st.columns(4)
symbol = col1.text_input("Token (ej. IOTX/USDT)", "IOTX/USDT")
timeframe = col2.selectbox("Temporalidad", ["1m", "5m", "15m", "1h", "4h"], index=1)
limit_bars = col3.number_input("Velas Históricas", 1000, 5000, 2000)
exchange_id = col4.selectbox("Exchange", ["binance", "coinbase", "kraken"], index=0)

st.sidebar.header("⚙️ RANGOS DE MUTACIÓN")
tp_min, tp_max = st.sidebar.slider("Rango Take Profit (%)", 0.5, 10.0, (1.0, 5.0), 0.5)
sl_min, sl_max = st.sidebar.slider("Rango Stop Loss (%)", 0.5, 5.0, (1.0, 3.0), 0.5)
whale_min, whale_max = st.sidebar.slider("Rango Ballena (xVol)", 1.5, 5.0, (2.0, 4.0), 0.5)

if st.button("🔥 FORJAR ALGORITMO (INICIAR BACKTEST MASIVO)", use_container_width=True, type="primary"):
    
    with st.spinner(f"📡 Descargando datos de {symbol} en {timeframe}..."):
        try:
            # Descarga de Data
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class()
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit_bars)
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            # --- 2. CÁLCULO DE MOTORES MATEMÁTICOS (PANDAS TA) ---
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['vol_ma_100'] = ta.sma(df['volume'], length=100)
            df['rvol'] = df['volume'] / df['vol_ma_100'].replace(0, 1)
            
            # Keltner & Bollinger (Para Squeeze / DEFCON)
            bb = ta.bbands(df['close'], length=20, std=2.0)
            kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
            
            if bb is not None and kc is not None:
                df = pd.concat([df, bb, kc], axis=1)
                
                # Nombres dinámicos de pandas_ta
                bb_u, bb_m, bb_l = bb.columns[2], bb.columns[1], bb.columns[0]
                kc_u, kc_m, kc_l = kc.columns[2], kc.columns[1], kc.columns[0]
                
                # Condición Squeeze & Defcon
                df['squeeze'] = (df[bb_u] < df[kc_u]) & (df[bb_l] > df[kc_l])
                df['neon_up'] = df['squeeze'] & (df['close'] >= df[bb_u] * 0.999) & (df['close'] > df['open'])
            else:
                df['neon_up'] = False
            
            st.success(f"✅ Data procesada: {len(df)} velas calculadas.")
            
        except Exception as e:
            st.error(f"Error descargando datos: {e}")
            st.stop()

    with st.spinner("🧬 Forjando genomas... Calculando miles de combinaciones..."):
        # --- 3. MOTOR DE OPTIMIZACIÓN (GRID SEARCH) ---
        tp_steps = np.arange(tp_min, tp_max + 0.1, 0.5)
        sl_steps = np.arange(sl_min, sl_max + 0.1, 0.5)
        whale_steps = np.arange(whale_min, whale_max + 0.1, 0.5)
        
        best_pnl = -99999
        best_params = {}
        best_trades = 0
        
        # Iteración de combinaciones
        for tp, sl, w_th in itertools.product(tp_steps, sl_steps, whale_steps):
            
            # Condición de Compra Vectorizada
            flash_vol = (df['rvol'] > (w_th * 0.8)) & (abs(df['close'] - df['open']) > (df['atr'] * 0.3))
            whale_buy = flash_vol & (df['close'] > df['open'])
            
            # Simulación Vectorizada Rápida de Backtest
            buy_signals = df.index[whale_buy | df['neon_up']].tolist()
            
            pnl = 0.0
            trades = 0
            in_position = False
            entry_price = 0.0
            
            for i in range(1, len(df)):
                if not in_position and i in buy_signals:
                    in_position = True
                    entry_price = df['close'].iloc[i]
                    trades += 1
                
                elif in_position:
                    high_p = df['high'].iloc[i]
                    low_p = df['low'].iloc[i]
                    
                    target_tp = entry_price * (1 + (tp / 100))
                    target_sl = entry_price * (1 - (sl / 100))
                    
                    if high_p >= target_tp:
                        pnl += tp
                        in_position = False
                    elif low_p <= target_sl:
                        pnl -= sl
                        in_position = False
            
            # Registrar el mejor mutante
            if pnl > best_pnl and trades > 0:
                best_pnl = pnl
                best_trades = trades
                best_params = {'tp': round(tp, 2), 'sl': round(sl, 2), 'whale': round(w_th, 2)}

    # --- 4. RESULTADOS DE LA FORJA ---
    if best_trades > 0:
        st.markdown("---")
        st.subheader("🏆 ADN GANADOR ENCONTRADO")
        colA, colB, colC = st.columns(3)
        colA.metric("Beneficio Neto Estimado", f"{round(best_pnl, 2)}%")
        colB.metric("Trades Simulados", best_trades)
        colC.metric("Ratio PnL/Trades", f"{round(best_pnl/best_trades, 2)}%")
        
        st.markdown(f"> **Mejores Parámetros:** Take Profit **{best_params['tp']}%** | Stop Loss **{best_params['sl']}%** | Ballena: **{best_params['whale']}x**")
        
        # --- 5. GENERACIÓN DEL PINE SCRIPT EXACTO ---
        st.markdown("### 📋 Código Pine Script V355 Optimizado")
        st.markdown("Copia este código y pégalo en TradingView. Los Webhooks ya están configurados con el Modelo Asimétrico (Limit/Market) y los parámetros son exactamente los ganadores.")
        
        pine_code = f"""//@version=5
// 🛡️ V355: VALLE ARCHITECT [FORJA {symbol.replace('/','')} {timeframe}]
strategy("VALLE ARCHITECT [V355 OPTIMIZADO]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, max_lines_count=500, max_boxes_count=500, max_bars_back=5000, max_labels_count=500, commission_type=strategy.commission.percent, commission_value=0.25, process_orders_on_close=true)

// ==========================================
// 1. CONFIGURACIÓN OPTIMIZADA POR PYTHON
// ==========================================
grp_main = "⚙️ CONTROL DE NÚCLEOS"
bot_enable = input.bool(true, "Activar TRINITY ENGINE", group=grp_main)
bot_pink_whale = input.bool(true, "Comprar: Vela Rosa + Icono Ballena", group=grp_main)

grp_risk = "🛡️ GESTIÓN DE RIESGO"
use_risk_mgt    = input.bool(true, "Activar Take Profit / Stop Loss", group=grp_risk)
tp_pct          = input.float({best_params['tp']}, "🎯 Take Profit (%)", step=0.1, group=grp_risk)
sl_pct          = input.float({best_params['sl']}, "🛑 Stop Loss (%)", step=0.1, group=grp_risk)
whale_threshold = input.float({best_params['whale']}, "Factor Ballena (xVol)", step=0.1, group=grp_risk)
hitbox_pct      = input.float(1.5, "Sensibilidad de Radares (%)", step=0.1, group=grp_risk)

grp_adv = "⚙️ WEBHOOKS ASIMÉTRICOS"
msg_buy  = input.text_area('{{"passphrase": "ASTRONAUTA", "action": "buy", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": 100, "order_type": "limit", "limit_price": {{{{close}}}}, "slippage_pct": 1.0, "side": "🟢 COMPRA"}}', "Webhook Compra", group=grp_adv)
msg_sell = input.text_area('{{"passphrase": "ASTRONAUTA", "action": "sell", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": 100, "order_type": "market", "side": "🔴 VENTA"}}', "Webhook Venta", group=grp_adv)

// --- LÓGICA TRINITY BASE ---
var color C_MAG = #FF00FF
float atr_val = ta.atr(14)
float vol_ma_long = ta.sma(volume, 100)
float rvol = volume / (vol_ma_long == 0 ? 1 : vol_ma_long)
[bb_mid, bb_top, bb_bot] = ta.bb(close, 20, 2.0)
[kc_m, kc_u, kc_l] = ta.kc(close, 20, 1.5)

bool squeeze_on = (bb_top < kc_u) and (bb_bot > kc_l)
bool neon_break_up = squeeze_on and (close >= bb_top * 0.999) and (close > open)

bool flash_vol = rvol > (whale_threshold * 0.8) and math.abs(close-open) > (atr_val * 0.3)
bool whale_buy = flash_vol and close > open
bool is_whale_icon = whale_buy and not nz(whale_buy[1])

bool cond_buy = is_whale_icon or neon_break_up

var float locked_entry = na
var float locked_tp = na
var float locked_sl = na
bool just_entered = strategy.position_size > 0 and strategy.position_size[1] == 0

if bot_enable
    if cond_buy and strategy.position_size == 0
        strategy.entry("TRINITY_LONG", strategy.long, alert_message=msg_buy)

    if just_entered and use_risk_mgt
        locked_entry := strategy.position_avg_price
        locked_tp := locked_entry * (1 + (tp_pct / 100))
        locked_sl := locked_entry * (1 - (sl_pct / 100))

    if strategy.position_size > 0 
        bool hit_tp = high >= locked_tp or close >= locked_tp
        bool hit_sl = low <= locked_sl or close <= locked_sl
        
        if hit_tp or hit_sl
            strategy.close("TRINITY_LONG", comment=hit_tp ? "TP" : "SL", alert_message=msg_sell)
            locked_entry := na, locked_tp := na, locked_sl := na

barcolor(cond_buy ? C_MAG : na)
plot(strategy.position_size > 0 ? locked_tp : na, color=color.green, style=plot.style_linebr, linewidth=2)
plot(strategy.position_size > 0 ? locked_sl : na, color=color.red, style=plot.style_linebr, linewidth=2)
"""
        st.code(pine_code, language="pine")
    else:
        st.warning("No se encontró ninguna configuración rentable en este rango. ¡Intenta ajustar los rangos de mutación o cambia de temporalidad!")
