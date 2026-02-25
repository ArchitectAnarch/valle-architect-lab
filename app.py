//@version=5
// 🛡️ V320: VALLE ARCHITECT [HYBRID]
// BASE: V319 (DEFCON Protocol)

strategy("VALLE ARCHITECT [DEFCON COMMAND HUD]", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.25, max_lines_count=500, max_boxes_count=500, max_bars_back=5000, max_labels_count=500)

// ==========================================
// 0. INTEGRACIÓN WUNDERTRADING & FECHA
// ==========================================
grp_wt = "🔌 WUNDERTRADING WEBHOOKS"
wt_enter_long = input.text_area(defval='{"action": "buy"}', title="🟢 WT: Mensaje Enter Long", group=grp_wt)
wt_exit_long  = input.text_area(defval='{"action": "sell"}', title="🔴 WT: Mensaje Exit Long", group=grp_wt)

grp_time = "📅 FILTRO DE FECHA"
start_year = input.int(2025, "Año de Inicio", group=grp_time)
start_month = input.int(1, "Mes de Inicio", group=grp_time)
start_day = input.int(1, "Día de Inicio", group=grp_time)
window = time >= timestamp(syminfo.timezone, start_year, start_month, start_day, 0, 0)

grp_trade = "💼 GESTIÓN DE OPERACIONES"
tp_pct = input.float(20.0, "Take Profit (%)", step=0.5, group=grp_trade)
sl_pct = input.float(5.0, "Stop Loss (%)", step=0.5, group=grp_trade)

// ==========================================
// 1. CONFIGURACIÓN E INPUTS
// ==========================================
grp_brain = "🧠 INTELIGENCIA ROCKET PROTOCOL"
learning_show   = input.bool(true, "Panel de Estado (HUD)", group=grp_brain)
popup_show      = input.bool(true, "Activar Pop-Up Clímax", group=grp_brain)
popup_duration  = input.int(10, "Duración Pop-up (Velas)", minval=1, group=grp_brain)
utc_offset      = input.float(-5.0, "Zona Horaria (UTC)", step=0.5, group=grp_brain)

grp_visual = "🌊 RÍO CUÁNTICO & SEÑALES"
river_show      = input.bool(true, "Ver Río Predictivo (Quantum)", group=grp_visual)
hud_show        = input.bool(true, "Ver Termómetro Fluid Dynamics", group=grp_visual)
gravity_show    = input.bool(true, "Ver Gravity Halos (Target Lock)", group=grp_visual)
pingpong_show   = input.bool(true, "🏓 Ver Sigma Vector (Predicción Física)", group=grp_visual)
tactical_show   = input.bool(true, "🎯 Ver Etiquetas Tácticas", group=grp_visual)

grp_struct = "🕸️ MALLA & ESTRUCTURA (MATRIX)"
matrix_show     = input.bool(true, "Ver Malla Diagonal (Deep Memory)", group=grp_struct)

grp_adv = "⚙️ CALIBRACIÓN AVANZADA"
btc_ticker      = input.symbol("COINBASE:BTCUSD", "Ticker Bitcoin Ref", group=grp_adv)
hitbox_pct      = input.float(1.5, "Sensibilidad de Contacto (%)", step=0.1, group=grp_adv) 
whale_threshold = input.float(2.5, "Factor Ballena (xVol)", minval=1.5, step=0.1, group=grp_adv)
hud_zoom        = input.float(2.5, "Zoom Termómetro", minval=1.0, maxval=10.0, step=0.5, group=grp_adv)

grp_col = "🎨 PALETA DE COLORES"
color c_up_in   = input.color(#00FF00, "Compra (Neón)", group=grp_col)
color c_dn_in   = input.color(#FF0000, "Venta (Neón)", group=grp_col)
color c_mag_in  = input.color(#FF00FF, "Nuclear (Rosa Vivo)", group=grp_col)
color c_gold_in = input.color(#FFD700, "Ballena (Oro Sólido)", group=grp_col)
color c_squeeze = input.color(#708090, "Squeeze Base", group=grp_col)
color c_diag_up = input.color(#00FF00, "Diagonal UP", group=grp_col) 
color c_diag_dn = input.color(#FF00FF, "Diagonal DOWN", group=grp_col)
color c_ultra   = input.color(#00FFFF, "Ultra (Turquesa)", group=grp_col)
color c_master  = input.color(#E0E0E0, "Maestra (Plata)", group=grp_col)

// ==========================================
// 2. ARRAYS Y VARIABLES GLOBALES
// ==========================================
var color C_UP      = c_up_in
var color C_DN      = c_dn_in
var color C_MAG     = c_mag_in
var color C_ULTRA   = c_ultra
var color C_MASTER  = c_master
var color C_MAG_PALE = color.new(c_mag_in, 30) 
var color C_GOLD    = c_gold_in 
var color C_ORANGE  = #FFA500   
var color C_NEU     = #808080  
var color C_TRANS   = #00000000 
var color C_TXT     = #FFFFFF
var string SZ_TINY  = size.tiny

var line[] matrix_lines = array.new_line(0)
var int[] matrix_weights = array.new_int(0)
var line[] proj_lines = array.new_line(0)
var linefill[] proj_fills = array.new_linefill(0)
var box[] hud_boxes = array.new_box(0)

// Estado Global
var bool is_structure_support = false
var bool is_gravity_zone = false 
var int friction_weight = 0       
var float nearest_wall_global = na 
var float gravity_target_price = na 
var int gravity_target_mass = 0
var table master_ui = table.new(position.middle_left, 2, 9, bgcolor=color.new(#000000, 30), border_width=1)
var int last_climax_time = 0 

// Variables para DEFCON System Visual
var string last_defcon_status = "DEFCON 5"
var color last_defcon_color = color.blue
var float last_defcon_price = na
var int last_defcon_time = 0
var string last_defcon_dir = ""

// ==========================================
// 3. FUNCIÓN DE TEJIDO (DEEP MEMORY 15K)
// ==========================================
project_diagonal(x1, y1, x2, y2, col, wid, sty, weight) =>
    if (bar_index - x1) < 15000 
        line l = line.new(x1, y1, x2, y2, extend=extend.right, color=col, width=wid, style=sty)
        array.push(matrix_lines, l)
        array.push(matrix_weights, weight)
        if array.size(matrix_lines) > 350
            line.delete(array.shift(matrix_lines))
            array.shift(matrix_weights)

// ==========================================
// 4. MOTORES MATEMÁTICOS & FÍSICA
// ==========================================
float atr_val = ta.atr(14)
float btc_close = request.security(btc_ticker, timeframe.period, close)
float btc_open = request.security(btc_ticker, timeframe.period, open)
bool btc_bull = btc_close > btc_open

// FÍSICA GLOBAL
float basis_sigma = ta.sma(close, 20)
float dev_sigma = ta.stdev(close, 20)
float z_score = (close - basis_sigma) / (dev_sigma == 0 ? 1 : dev_sigma)
float pp_slope = ta.linreg(close, 5, 0) - ta.linreg(close, 5, 1)

// RSI
float rsi_v = ta.rsi(close, 14)       
float rsi_ma = ta.sma(rsi_v, 14) 
float rsi_velocity = rsi_v - nz(rsi_v[1])
bool rsi_cross_up = ta.crossover(rsi_v, rsi_ma)
bool rsi_cross_down = ta.crossunder(rsi_v, rsi_ma)

// ADX
[di_p, di_m, adx_val] = ta.dmi(14, 14)

// WHALE HOLOGRAPH
float halo_high = ta.ema(high, 3)
float halo_low = ta.ema(low, 3)
bool show_halo_sell = rsi_v > 70
bool show_halo_buy = rsi_v < 30

// WAVETREND
float ap = hlc3 
float esa = ta.ema(ap, 10)
float d_wt = ta.ema(math.abs(ap - esa), 10)
float ci = (ap - esa) / (0.015 * (d_wt == 0 ? 1 : d_wt))
float tci = ta.ema(ci, 21)
float wt1 = tci
float wt2 = ta.sma(wt1, 4)
bool wt_cross_up = ta.crossover(wt1, wt2)
bool wt_cross_dn = ta.crossunder(wt1, wt2)
bool wt_oversold = wt1 < -60 
bool wt_overbought = wt1 > 60 

// BALLENAS & VOLUMEN
float vol_ma = ta.sma(volume, 100)
float rvol = volume / (vol_ma == 0 ? 1 : vol_ma)
bool flash_vol = rvol > (whale_threshold * 0.8) and math.abs(close-open) > (atr_val * 0.3)
bool whale_buy = flash_vol and close > open
bool whale_sell = flash_vol and close < open
bool whale_memory = whale_buy or nz(whale_buy[1]) or nz(whale_buy[2]) or whale_sell or nz(whale_sell[1]) or nz(whale_sell[2])

// BANDAS & DIVERGENCIAS
float rsi_bb_basis = ta.sma(rsi_v, 14)
float rsi_bb_dev = ta.stdev(rsi_v, 14) * 2.0
float rsi_bb_upper = rsi_bb_basis + rsi_bb_dev
float rsi_bb_lower = rsi_bb_basis - rsi_bb_dev
bool k_break_up = ta.crossover(rsi_v, rsi_bb_upper) 
bool k_break_dn = ta.crossunder(rsi_v, rsi_bb_lower) 

// 🔥 CORRECCIÓN DEL ERROR 'rsi': SE DEBE USAR rsi_v 🔥
bool div_bull = nz(low[1]) < nz(low[5]) and nz(rsi_v[1]) > nz(rsi_v[5]) and rsi_v < 35
bool div_bear = nz(high[1]) > nz(high[5]) and nz(rsi_v[1]) < nz(rsi_v[5]) and rsi_v > 65

// PUMP/DUMP
[bb_mid, bb_top, bb_bot] = ta.bb(close, 20, 2.0)
bool pre_pump = (high > bb_top or rsi_velocity > 5) and flash_vol and close > open
bool pre_dump = (low < bb_bot or rsi_velocity < -5) and flash_vol and close < open
bool pump_memory = pre_pump or nz(pre_pump[1]) or nz(pre_pump[2])
bool dump_memory = pre_dump or nz(pre_dump[1]) or nz(pre_dump[2])

// ==========================================
// 5. EJECUCIÓN DE LA MALLA
// ==========================================
var int p1_idx = na, var float p1_val = na, var int h1_idx = na, var float h1_val = na
var int p2_idx = na, var float p2_val = na, var int h2_idx = na, var float h2_val = na
var int p3_idx = na, var float p3_val = na, var int h3_idx = na, var float h3_val = na
var int p4_idx = na, var float p4_val = na, var int h4_idx = na, var float h4_val = na

if matrix_show
    // Micro
    float pl_1 = ta.pivotlow(low, 30, 3), ph_1 = ta.pivothigh(high, 30, 3)
    if not na(pl_1)
        if not na(p1_val)
            project_diagonal(p1_idx, p1_val, bar_index[3], pl_1, color.new(color.teal, 20), 1, line.style_solid, 1)
        p1_idx := bar_index[3], p1_val := pl_1
    if not na(ph_1)
        if not na(h1_val)
            project_diagonal(h1_idx, h1_val, bar_index[3], ph_1, color.new(color.teal, 20), 1, line.style_solid, 1)
        h1_idx := bar_index[3], h1_val := ph_1
    
    // Macro
    float pl_2 = ta.pivotlow(low, 100, 5), ph_2 = ta.pivothigh(high, 100, 5)
    if not na(pl_2)
        if not na(p2_val)
            project_diagonal(p2_idx, p2_val, bar_index[5], pl_2, color.new(color.yellow, 30), 1, line.style_dotted, 3)
        p2_idx := bar_index[5], p2_val := pl_2
    if not na(ph_2)
        if not na(h2_val)
            project_diagonal(h2_idx, h2_val, bar_index[5], ph_2, color.new(color.yellow, 30), 1, line.style_dotted, 3)
        h2_idx := bar_index[5], h2_val := ph_2
    
    // MAESTRO
    float pl_3 = ta.pivotlow(low, 300, 5), ph_3 = ta.pivothigh(high, 300, 5)
    if not na(pl_3)
        if not na(p3_val)
            project_diagonal(p3_idx, p3_val, bar_index[5], pl_3, color.new(C_MASTER, 20), 1, line.style_dashed, 5)
        p3_idx := bar_index[5], p3_val := pl_3
    if not na(ph_3)
        if not na(h3_val)
            project_diagonal(h3_idx, h3_val, bar_index[5], ph_3, color.new(C_MASTER, 20), 1, line.style_dashed, 5)
        h3_idx := bar_index[5], h3_val := ph_3
    
    // ULTRA
    float pl_4 = ta.pivotlow(low, 800, 10), ph_4 = ta.pivothigh(high, 800, 10)
    if not na(pl_4)
        if not na(p4_val)
            project_diagonal(p4_idx, p4_val, bar_index[10], pl_4, color.new(C_ULTRA, 0), 2, line.style_solid, 8)
        p4_idx := bar_index[10], p4_val := pl_4
    if not na(ph_4)
        if not na(h4_val)
            project_diagonal(h4_idx, h4_val, bar_index[10], ph_4, color.new(C_ULTRA, 0), 2, line.style_solid, 8)

// ==========================================
// 6. LÓGICA APEX HYBRID & MOTOR VISUAL
// ==========================================

bool final_signal_buy = wt_cross_up and window
bool final_signal_sell = wt_cross_dn

plotchar(final_signal_buy, title="COMPRA", char="🚀", location=location.belowbar, color=color.aqua, size=size.tiny)
plotchar(final_signal_sell, title="VENTA", char="🛑", location=location.abovebar, color=color.red, size=size.tiny)

if final_signal_buy and strategy.position_size == 0
    strategy.entry("Hybrid_Long", strategy.long, alert_message=wt_enter_long)

if signal_sell and strategy.position_size > 0
    strategy.close("Hybrid_Long", comment="Dyn_Exit", alert_message=wt_exit_long)

if strategy.position_size > 0 
    entry_price = strategy.opentrades.entry_price(strategy.opentrades - 1)
    target_price = entry_price * (1 + (tp_pct / 100))
    stop_price = entry_price * (1 - (sl_pct / 100))
    strategy.exit("TP/SL", "Hybrid_Long", limit=target_price, stop=stop_price, alert_message=wt_exit_long)
