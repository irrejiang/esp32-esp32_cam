from machine import Pin, ADC, PWM
from time import ticks_ms, ticks_diff, sleep

# ======================== 1. 硬件配置 ========================
SENSOR_PINS = [27, 33, 32, 35, 34]
sensors = [ADC(Pin(p)) for p in SENSOR_PINS]
for adc in sensors:
    adc.atten(ADC.ATTN_11DB)

MOTOR_L_DIR = Pin(15, Pin.OUT, value=0)
MOTOR_L_PWM = PWM(Pin(13), freq=1000, duty=1023)
MOTOR_R_DIR = Pin(14, Pin.OUT, value=0)
MOTOR_R_PWM = PWM(Pin(25), freq=1000, duty=1023)

# ======================== 2. 参数配置 ========================
CONFIG = {
    'CONTRAST_MIN': 80,
    'THR_EMA_ALPHA': 0.3,
    'KP': 30.0,
    'KI': 0.5,
    'KD': 1.0,
    'PID_OUT_LIMIT': 20,
    'INT_LIMIT': 40,
    'INT_DEADZONE': 1.5,
    'BASE_SPEED': 50,
    'CURVE_FACTOR': 0.7,
    'BASE_MIN': 0,
    'SPEED_SMOOTH_ALPHA': 0.3,
    'SAMPLE_FILTER_N': 3,
    'SAMPLE_PERIOD_MS': 10,
    'DEBUG_PRINT_INTERVAL': 20,
    'STEER_INVERT': True,
}

SENSOR_WEIGHT = [-2, -1, 0, 1, 2]

# ======================== 3. 状态变量 ========================
last_error = 0.0
integral = 0.0
last_deriv = 0.0
last_valid_error = 0.0
smooth_left = 0
smooth_right = 0
debug_cnt = 0
lost_count = 0
search_mode = False

# ======================== 4. 电机驱动（反逻辑PWM） ========================
def set_motor_raw(left_raw: int, right_raw: int):
    if left_raw > 0:
        duty = 1023 - int(1023 * left_raw / 100)
        MOTOR_L_PWM.duty(duty)
    else:
        MOTOR_L_PWM.duty(1023)
    
    if right_raw > 0:
        duty = 1023 - int(1023 * right_raw / 100)
        MOTOR_R_PWM.duty(duty)
    else:
        MOTOR_R_PWM.duty(1023)
    
    MOTOR_L_DIR.on()
    MOTOR_R_DIR.on()

def set_motor_smooth(left_cmd: int, right_cmd: int):
    global smooth_left, smooth_right
    alpha = CONFIG['SPEED_SMOOTH_ALPHA']
    smooth_left = int(alpha * left_cmd + (1 - alpha) * smooth_left)
    smooth_right = int(alpha * right_cmd + (1 - alpha) * smooth_right)
    set_motor_raw(smooth_left, smooth_right)

def car_stop():
    global smooth_left, smooth_right
    smooth_left = smooth_right = 0
    set_motor_raw(0, 0)

# ======================== 5. 传感器读取 ========================
def read_sensor_filtered() -> list:
    n = CONFIG['SAMPLE_FILTER_N']
    acc = [0] * 5
    for _ in range(n):
        for i, adc in enumerate(sensors):
            acc[i] += adc.read()
    return [v // n for v in acc]

# ======================== 6. 开机自动校准 ========================
def auto_calibrate():
    print("Calibrating sensors...")
    samples = 50
    calib_min = [4095] * 5
    calib_max = [0] * 5
    for _ in range(samples):
        raw = read_sensor_filtered()
        for i in range(5):
            if raw[i] < calib_min[i]:
                calib_min[i] = raw[i]
            if raw[i] > calib_max[i]:
                calib_max[i] = raw[i]
        sleep(0.01)
    thresholds = [(calib_max[i] + calib_min[i]) // 2 + 10 for i in range(5)]
    black_ema = calib_max[:]
    white_ema = calib_min[:]
    print(f"Calibration done. Thresholds: {thresholds}")
    return thresholds, black_ema, white_ema

adc_thresholds, black_ema, white_ema = auto_calibrate()

# ======================== 7. 自适应阈值更新（带限幅） ========================
def update_adaptive_thresholds(raw: list):
    global black_ema, white_ema, adc_thresholds
    alpha = CONFIG['THR_EMA_ALPHA']
    max_change = 30
    for i in range(5):
        val = raw[i]
        if val > adc_thresholds[i]:
            new_black = int(alpha * val + (1 - alpha) * black_ema[i])
            if abs(new_black - black_ema[i]) > max_change:
                new_black = black_ema[i] + max_change if new_black > black_ema[i] else black_ema[i] - max_change
            black_ema[i] = new_black
        else:
            new_white = int(alpha * val + (1 - alpha) * white_ema[i])
            if abs(new_white - white_ema[i]) > max_change:
                new_white = white_ema[i] + max_change if new_white > white_ema[i] else white_ema[i] - max_change
            white_ema[i] = new_white
        adc_thresholds[i] = (black_ema[i] + white_ema[i]) // 2

# ======================== 8. 偏差计算（使用动态阈值） ========================
def calc_error(raw: list) -> tuple:
    global last_valid_error, lost_count
    min_val = min(raw)
    max_val = max(raw)
    
    if max_val - min_val < CONFIG['CONTRAST_MIN']:
        lost_count += 1
        return last_valid_error, False
    
    # ====== 改动：用当前帧的动态阈值替代固定阈值 ======
    dynamic_thr = (max_val + min_val) // 2
    binary = [1 if raw[i] > dynamic_thr else 0 for i in range(5)]
    
    sum_bin = sum(binary)
    
    if sum_bin == 0 or sum_bin == 5:
        lost_count += 1
        return last_valid_error, False
    
    lost_count = 0
    weighted_sum = 0.0
    for i, b in enumerate(binary):
        weighted_sum += SENSOR_WEIGHT[i] * b
    error = weighted_sum / sum_bin
    last_valid_error = error
    global last_binary
    last_binary = binary
    return error, True

last_binary = [0] * 5

# ======================== 9. PID控制器 ========================
def pid_calc(error: float, dt: float) -> float:
    global last_error, integral, last_deriv
    p_out = CONFIG['KP'] * error
    if abs(error) < CONFIG['INT_DEADZONE']:
        integral += error * dt
    else:
        integral *= 0.9
    integral = max(-CONFIG['INT_LIMIT'], min(CONFIG['INT_LIMIT'], integral))
    i_out = CONFIG['KI'] * integral
    
    if abs(error - last_error) < 0.001:
        deriv = last_deriv
    else:
        raw_deriv = (error - last_error) / dt if dt > 0 else 0
        deriv = 0.4 * raw_deriv + 0.6 * last_deriv
    last_deriv = deriv
    d_out = CONFIG['KD'] * deriv
    
    steer = p_out + i_out + d_out
    steer = max(-CONFIG['PID_OUT_LIMIT'], min(CONFIG['PID_OUT_LIMIT'], steer))
    last_error = error
    return steer

# ======================== 10. 速度计算 ========================
def calc_wheel_speed(steer: float) -> tuple:
    if CONFIG.get('STEER_INVERT', False):
        steer = -steer
    base = CONFIG['BASE_SPEED'] - abs(steer) * CONFIG['CURVE_FACTOR']
    if base < CONFIG['BASE_MIN']:
        base = CONFIG['BASE_MIN']
    left = base + steer
    right = base - steer
    left = max(0, min(100, left))
    right = max(0, min(100, right))
    if 0 < left < 3:
        left = 3
    if 0 < right < 3:
        right = 3
    return int(left), int(right)

# ======================== 11. 主循环 ========================
print("=== ESP32 Line Follower ===")
car_stop()
sleep(1)

last_tick = ticks_ms()
left_cmd = 0
right_cmd = 0

while True:
    now = ticks_ms()
    if ticks_diff(now, last_tick) < CONFIG['SAMPLE_PERIOD_MS']:
        continue
    dt = ticks_diff(now, last_tick) / 1000.0
    last_tick = now
    
    raw = read_sensor_filtered()
    
    err, online = calc_error(raw)
    
    # 只有成功检测到线时，才更新自适应阈值（保留，但阈值已不再用于二值化）
    if online:
        update_adaptive_thresholds(raw)
    
    if online:
        steer = pid_calc(err, dt)
        left_cmd, right_cmd = calc_wheel_speed(steer)
        set_motor_smooth(left_cmd, right_cmd)
        search_mode = False
    else:
        if not search_mode:
            car_stop()
            search_mode = True
        
        left_search = 5
        right_search = 25
        set_motor_smooth(left_search, right_search)
        
        integral = 0.0
        last_deriv = 0.0
    
    debug_cnt += 1
    if debug_cnt >= CONFIG['DEBUG_PRINT_INTERVAL']:
        debug_cnt = 0
        status = "SEARCH" if search_mode else "TRACK"
        print(f"RAW:{raw} | Bin:{last_binary} | Err:{err:.2f} | "
              f"L:{left_cmd:3d} R:{right_cmd:3d} | {status}")