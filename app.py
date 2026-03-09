import streamlit as st
import random

st.set_page_config(page_title="Forja Genética V355", page_icon="🧬", layout="wide")

st.title("🧬 Laboratorio Genético: VALLE ARCHITECT V355")
st.markdown("Generador de mutantes para Backtesting Institucional (Velas Japonesas + Ejecución Asimétrica Limit/Market).")

# --- 1. CONFIGURACIÓN DEL MUTANTE ---
st.sidebar.header("⚙️ Parámetros de Mutación")

# RANGOS DE MUTACIÓN (Puedes ajustarlos según tu apetito de riesgo)
tp_range = st.sidebar.slider("Rango de Take Profit (%)", 0.5, 10.0, (1.0, 5.0), 0.1)
sl_range = st.sidebar.slider("Rango de Stop Loss (%)", 0.5, 5.0, (1.0, 3.0), 0.1)
hitbox_range = st.sidebar.slider("Sensibilidad de Radares (%)", 0.5, 5.0, (1.0, 2.5), 0.1)
whale_range = st.sidebar.slider("Multiplicador de Ballena (xVol)", 1.5, 5.0, (2.0, 3.5), 0.1)

if st.button("⚡ GENERAR NUEVO MUTANTE", use_container_width=True, type="primary"):
    
    # --- 2. GENERACIÓN DEL ADN (Valores Aleatorios) ---
    mutant_tp = round(random.uniform(tp_range[0], tp_range[1]), 1)
    mutant_sl = round(random.uniform(sl_range[0], sl_range[1]), 1)
    mutant_hitbox = round(random.uniform(hitbox_range[0], hitbox_range[1]), 1)
    mutant_whale = round(random.uniform(whale_range[0], whale_range[1]), 1)
    
    # Mutación de módulos (Activa/Desactiva aleatoriamente para ver qué combinación funciona mejor)
    m_whale = str(random.choice(['true', 'false']))
    m_defcon_buy = str(random.choice(['true', 'false']))
    m_therm_bounce = str(random.choice(['true', 'false']))
    m_therm_vacuum = str(random.choice(['true', 'false']))
    m_lock_bounce = str(random.choice(['true', 'false']))
    m_lock_break = str(random.choice(['true', 'false']))
    
    version_id = f"V355-M{random.randint(1000, 9999)}"
    
    st.success(f"✅ Mutante {version_id} generado con éxito. Cópialo y pégalo en TradingView.")
    
    st.markdown(f"""
    **🧬 ADN del Mutante:**
    * **Take Profit:** {mutant_tp}% | **Stop Loss:** {mutant_sl}%
    * **Radar Hitbox:** {mutant_hitbox}% | **Ballena xVol:** {mutant_whale}x
    """)

    # --- 3. PLANTILLA PINE SCRIPT V355 (LA MÁQUINA DE LA VERDAD) ---
    pine_code = f"""//@version=5
// 🛡️ {version_id}: VALLE ARCHITECT [TRINITY + PINK WHALE] - REALITY EDITION
// Creado por la Forja Genética de Python.
strategy("VALLE ARCHITECT [{version_id}]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, max_lines_count=500, max_boxes_count=500, max_bars_back=5000, max_labels_count=500, commission_type=strategy.commission.percent, commission_value=0.25, process_orders_on_close=true)

// ==========================================
// 1. CONFIGURACIÓN E INPUTS (MUTADOS POR IA)
// ==========================================
grp_main = "⚙️ CONTROL DE NÚCLEOS"
bot_enable = input.bool(true, "Activar TRINITY ENGINE", group=grp_main)

grp_whale = "🐋 CAPA EXTRA: PINK WHALE DEEPS"
bot_pink_whale = input.bool({m_whale}, "Comprar: Vela Rosa + Icono Ballena", group=grp_whale)

grp_defcon = "🚀 NÚCLEO 1: DEFCON (V329)"
bot_defcon_buy  = input.bool({m_defcon_buy}, "Comprar: DEFCON 1/2 (Expansión Alcista)", group=grp_defcon)
bot_defcon_sell = input.bool(true, "Vender: DEFCON 1/2 (Expansión Bajista)", group=grp_defcon)

grp_thermal = "🌡️ NÚCLEO 2: TERMÓMETRO (V331)"
bot_therm_bounce = input.bool({m_therm_bounce}, "Comprar: Rebote en Suelo Fuerte", group=grp_thermal)
bot_therm_vacuum = input.bool({m_therm_vacuum}, "Comprar: Breakout en Cielo Libre", group=grp_thermal)
bot_therm_wall   = input.bool(true, "Vender: Choque en Techo Fuerte", group=grp_thermal)
bot_therm_panic  = input.bool(true, "Vender: Caída al Abismo", group=grp_thermal)

grp_lock = "🎯 NÚCLEO 3: TARGET LOCK (V332)"
bot_lock_bounce = input.bool({m_lock_bounce}, "Comprar: Rebote en Lock", group=grp_lock)
bot_lock_break  = input.bool({m_lock_break}, "Comprar: Ruptura de Lock", group=grp_lock)
bot_lock_reject = input.bool(true, "Vender: Rechazo en Lock", group=grp_lock)
bot_lock_breakd = input.bool(true, "Vender: Ruptura de Lock", group=grp_lock)

grp_risk = "🛡️ GESTIÓN DE RIESGO UNIFICADA"
use_risk_mgt    = input.bool(true, "Activar Take Profit / Stop Loss", group=grp_risk)
tp_pct          = input.float({mutant_tp}, "🎯 Take Profit (%)", step=0.1, minval=0.5, group=grp_risk)
sl_pct          = input.float({mutant_sl}, "🛑 Stop Loss (%)", step=0.1, minval=0.1, group=grp_risk)

grp_adv = "⚙️ CALIBRACIÓN AVANZADA"
btc_ticker      = input.symbol("COINBASE:BTCUSD", "Ticker Bitcoin Ref", group=grp_adv)
hitbox_pct      = input.float({mutant_hitbox}, "Sensibilidad de Radares (%)", step=0.1, group=grp_adv)
whale_threshold = input.float({mutant_whale}, "Factor Ballena (xVol)", minval=1.5, step=0.1, group=grp_adv)
matrix_show     = input.bool(true, "Dibujar Malla Tensorial", group=grp_adv)

// 🔥 EJECUCIÓN ASIMÉTRICA: Compra Limit (Slippage 1%) | Venta Market (Pánico/Seguridad) 🔥
msg_buy  = input.text_area('{{"passphrase": "ASTRONAUTA", "action": "buy", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": 100, "order_type": "limit", "limit_price": {{{{close}}}}, "slippage_pct": 1.0, "side": "🟢 COMPRA"}}', "Webhook Compra", group=grp_adv)
msg_sell = input.text_area('{{"passphrase": "ASTRONAUTA", "action": "sell", "ticker": "{{{{syminfo.basecurrency}}}}/{{{{syminfo.currency}}}}", "reinvest_pct": 100, "order_type": "market", "side": "🔴 VENTA"}}', "Webhook Venta", group=grp_adv)

// ==========================================
// 2. MOTORES Y EJECUCIÓN DE PRECIO REAL
// ==========================================
var color C_UP = #00FF00, var color C_DN = #FF0000, var color C_MAG = #FF00FF, var color C_ULTRA = #00FFFF, var color C_GOLD = #FFD700 
var line[] matrix_lines = array.new_line(0)
var int[] matrix_weights = array.new_int(0)

float atr_val = ta.atr(14)
float vol_ma = ta.sma(volume, 20)
bool high_vol = volume > vol_ma
float vol_ma_long = ta.sma(volume, 100)
float rvol = volume / (vol_ma_long == 0 ? 1 : vol_ma_long)

float rsi_v = ta.rsi(close, 14)       
float rsi_ma = ta.sma(rsi_v, 14) 
bool rsi_cross_up = ta.crossover(rsi_v, rsi_ma)
bool rsi_cross_down = ta.crossunder(rsi_v, rsi_ma)

[di_p, di_m, adx_val] = ta.dmi(14, 14)
[bb_mid, bb_top, bb_bot] = ta.bb(close, 20, 2.0)
[kc_m, kc_u, kc_l] = ta.kc(close, 20, 1.5)

project_diagonal(x1, y1, x2, y2, col, wid, sty, weight) =>
    if (bar_index - x1) < 15000 
        line l = line.new(x1, y1, x2, y2, extend=extend.right, color=col, width=wid, style=sty)
        array.push(matrix_lines, l)
        array.push(matrix_weights, weight)
        if array.size(matrix_lines) > 350
            line.delete(array.shift(matrix_lines))
            array.shift(matrix_weights)

float pl_1 = ta.pivotlow(low, 30, 3), ph_1 = ta.pivothigh(high, 30, 3)
if not na(pl_1) and matrix_show
    project_diagonal(bar_index[3], pl_1, bar_index, pl_1, color.new(color.teal, 30), 1, line.style_solid, 1)
float pl_2 = ta.pivotlow(low, 100, 5)
if not na(pl_2) and matrix_show
    project_diagonal(bar_index[5], pl_2, bar_index, pl_2, color.new(color.yellow, 40), 1, line.style_dotted, 3)

float scan_range = atr_val * 2.0
float ceil_weight = 0.0, float floor_weight = 0.0
bool is_gravity = false 
float target_lock = na 
int max_grav = 0 

if array.size(matrix_lines) > 0 
    for i = 0 to array.size(matrix_lines) - 1
        line l_chk = array.get(matrix_lines, i)
        int w_chk = array.get(matrix_weights, i)
        float p_chk = line.get_price(l_chk, bar_index)
        
        if p_chk > close and p_chk <= (close + scan_range)
            ceil_weight += w_chk
        if p_chk < close and p_chk >= (close - scan_range)
            floor_weight += w_chk
            
        if w_chk >= 3 and math.abs(close - p_chk) < (close * (hitbox_pct/100.0))
            if w_chk > max_grav
                is_gravity := true, target_lock := p_chk, max_grav := w_chk

// NÚCLEO 1: DEFCON
bool squeeze_on = (bb_top < kc_u) and (bb_bot > kc_l)
bool neon_break_up = squeeze_on and (close >= bb_top * 0.999) and (close > open)
bool neon_break_dn = squeeze_on and (close <= bb_bot * 1.001) and (close < open)
float bb_delta = (bb_top - bb_bot) - nz((bb_top[1] - bb_bot[1]), 0)
float bb_delta_avg = ta.sma(bb_delta, 10)

int defcon_level = 5 
if neon_break_up or neon_break_dn
    defcon_level := 4
    if bb_delta > 0
        defcon_level := 3
    if bb_delta > bb_delta_avg and adx_val > 20
        defcon_level := 2
    if bb_delta > (bb_delta_avg * 1.5) and adx_val > 25 and rvol > 1.2
        defcon_level := 1

bool cond_defcon_buy  = (defcon_level <= 2) and neon_break_up
bool cond_defcon_sell = (defcon_level <= 2) and neon_break_dn

// NÚCLEO 2: TERMÓMETRO
bool is_abyss = floor_weight == 0 
bool is_hard_wall = ceil_weight >= 4 
bool cond_therm_buy_bounce = (floor_weight >= 4) and rsi_cross_up and not is_hard_wall
bool cond_therm_buy_vacuum = (ceil_weight <= 3) and neon_break_up and not is_abyss
bool cond_therm_sell_wall  = (ceil_weight >= 4) and rsi_cross_down
bool cond_therm_sell_panic = is_abyss and (close < open)

// NÚCLEO 3: TARGET LOCK
float tol = atr_val * 0.5
bool cond_lock_buy_bounce = is_gravity and (low <= target_lock + tol) and (close > target_lock) and (close > open)
bool cond_lock_buy_break  = is_gravity and ta.crossover(close, target_lock) and high_vol and (close > open)
bool cond_lock_sell_reject = is_gravity and (high >= target_lock - tol) and (close < target_lock) and (close < open)
bool cond_lock_sell_breakd = is_gravity and ta.crossunder(close, target_lock) and (close < open)

// CAPA EXTRA: PINK WHALE
float basis_sigma = ta.sma(close, 20), float dev_sigma = ta.stdev(close, 20)
float z_score = (close - basis_sigma) / (dev_sigma == 0 ? 1 : dev_sigma)
bool flash_vol = rvol > (whale_threshold * 0.8) and math.abs(close-open) > (atr_val * 0.3)
bool whale_buy = flash_vol and close > open

bool retro_peak_buy = (rsi_v < 30 and close < bb_bot)
float buy_score = retro_peak_buy ? 50.0 : 30.0
if is_gravity 
    buy_score += 25.0 
if z_score < -2.0 
    buy_score += 15.0

bool is_magenta = buy_score >= 70 or retro_peak_buy
bool cond_pink_whale_buy = is_magenta and whale_buy and not nz(whale_buy[1])

// EJECUCIÓN (Verdad Cruda)
var float locked_entry = na, var float locked_tp = na, var float locked_sl = na
bool just_entered = strategy.position_size > 0 and strategy.position_size[1] == 0

if bot_enable
    bool do_buy = false, bool do_sell = false
    string buy_reason = "", string sell_reason = ""
    
    if bot_pink_whale and cond_pink_whale_buy
        do_buy := true, buy_reason := "PINK_WHALE"
    if bot_defcon_buy and cond_defcon_buy
        do_buy := true, buy_reason := "DEFCON_UP"
    if bot_therm_bounce and cond_therm_buy_bounce
        do_buy := true, buy_reason := "THERM_BOUNCE"
    if bot_therm_vacuum and cond_therm_buy_vacuum
        do_buy := true, buy_reason := "THERM_VACUUM"
    if bot_lock_bounce and cond_lock_buy_bounce
        do_buy := true, buy_reason := "LOCK_BOUNCE"
    if bot_lock_break and cond_lock_buy_break
        do_buy := true, buy_reason := "LOCK_BREAK"
        
    if bot_defcon_sell and cond_defcon_sell
        do_sell := true, sell_reason := "DEFCON_DN"
    if bot_therm_wall and cond_therm_sell_wall
        do_sell := true, sell_reason := "THERM_WALL"
    if bot_therm_panic and cond_therm_sell_panic
        do_sell := true, sell_reason := "THERM_PANIC"
    if bot_lock_reject and cond_lock_sell_reject
        do_sell := true, sell_reason := "LOCK_REJECT"
    if bot_lock_breakd and cond_lock_sell_breakd
        do_sell := true, sell_reason := "LOCK_BREAKD"

    if do_buy and strategy.position_size == 0
        strategy.entry("TRINITY_LONG", strategy.long, alert_message=msg_buy)

    if just_entered and use_risk_mgt
        locked_entry := strategy.position_avg_price
        float dyn_tp = (buy_reason == "PINK_WHALE" or buy_reason == "THERM_VACUUM") and nz(ceil_weight[1]) <= 3 ? tp_pct * 1.5 : tp_pct
        locked_tp := locked_entry * (1 + (dyn_tp / 100))
        locked_sl := locked_entry * (1 - (sl_pct / 100))

    if strategy.position_size > 0 
        bool hit_tp = use_risk_mgt and (high >= locked_tp)
        bool hit_sl = use_risk_mgt and (low <= locked_sl)
        
        if do_sell or hit_tp or hit_sl
            string exit_msg = hit_tp ? "TP_Hit" : hit_sl ? "SL_Hit" : sell_reason
            strategy.close("TRINITY_LONG", comment=exit_msg, alert_message=msg_sell)
            locked_entry := na, locked_tp := na, locked_sl := na

    if strategy.position_size == 0
        locked_entry := na, locked_tp := na, locked_sl := na

// PINTURA
barcolor(is_magenta ? C_MAG : na)
plot(strategy.position_size > 0 ? locked_tp : na, color=color.green, style=plot.style_linebr, linewidth=2)
plot(strategy.position_size > 0 ? locked_sl : na, color=color.red, style=plot.style_linebr, linewidth=2)
"""
    
    st.code(pine_code, language="pine")
