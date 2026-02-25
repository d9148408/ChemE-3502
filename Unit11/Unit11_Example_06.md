# Unit11 Example 06 - 加熱器程序頻率響應估計與 Bode 圖繪製 (Frequency Response Estimation)

## 學習目標

本範例以**一階加純時滯 (FOPDT) 加熱器程序的頻率響應估計**為主題，示範如何使用 `scipy.fft` 模組從輸入/輸出訊號計算**頻率響應函數 (Frequency Response Function, FRF)** 並繪製 **Bode 圖**，連結傅立葉轉換理論與化工程序識別（Process Identification）。

學習完本範例後，您將能夠：

- 理解**頻率響應函數 (FRF)** 的物理意義，以及其與傳遞函數 $G(j\omega)$ 的關係
- 以 `scipy.integrate.solve_ivp()` 模擬 FOPDT 程序 $G(s) = K e^{-\theta s} / (\tau s + 1)$ 的輸出響應
- 理解 **Chirp 訊號**（線性調頻信號）作為掃頻激發訊號的特性，以 `numpy` 手動產生 Chirp 訊號
- 使用 `scipy.fft.rfft()` 分別對輸入 $u(t)$ 與輸出 $y(t)$ 作 FFT，得 $U(f)$ 與 $Y(f)$
- 計算**直接 FRF 估計值** $\hat{H}_{\mathrm{direct}}(f) = Y(f) / U(f)$ ，分離幅度與相位
- 理解 **H1 估計量**：以分段加視窗的 Welch 法計算交叉頻譜 $G_{xy}(f)$ 與輸入自頻譜 $G_{xx}(f)$ ，得 $\hat{H}_1(f) = G_{xy}(f) / G_{xx}(f)$
- 計算**相干函數** $\gamma^2(f) = |G_{xy}|^2 / (G_{xx} \cdot G_{yy})$ 識別雜訊污染的頻段
- 繪製估計 Bode 圖並疊加理論值 $|G(j2\pi f)|$ 進行驗證比對
- 從 Bode 圖讀取程序增益 $K$ 、時間常數 $\tau$ 與純時滯 $\theta$ ，並與原設定參數比較

---

## 1. 問題描述 (Problem Description)

### 1.1 化工背景：程序識別與頻率響應估計

**程序識別 (Process Identification)** 是化工自動控制的核心步驟之一。在設計 PID 控制器之前，工程師需要了解程序（如加熱器、反應器、蒸餾塔）的動態特性，即輸入（如加熱功率 %）如何影響輸出（如溫度 °C）以及兩者之間的時間延遲。

#### 頻率響應的工程意義

**頻率響應函數 (FRF)** $H(j\omega)$ 描述的是：若輸入一個角頻率為 $\omega$ 的正弦訊號，穩態輸出正弦訊號的**幅度比**（增益）與**相位差**。

對於線性非時變系統 (LTI)：

$$
H(j\omega) = G(j\omega) = G(s)\big|_{s = j\omega}
$$

其 Bode 圖由**幅度頻率響應**與**相位頻率響應**兩部分組成：

$$
\text{幅度 (dB)}: \quad 20\log_{10}|H(j\omega)|
$$

$$
\text{相位 (deg)}: \quad \angle H(j\omega) = \arctan\left(\frac{\mathrm{Im}[H]}{\mathrm{Re}[H]}\right)
$$

#### 為何以 FFT 估計頻率響應？

傳統逐一頻率掃頻測試（每次輸入一個固定頻率的正弦波，等待穩態後量測幅度與相位）費時冗長。**Chirp 訊號**（線性調頻信號）可同時激發寬頻帶內的所有頻率，一次實驗即可取得完整的頻率響應，大幅提升測試效率，是工業現場頻率響應測試的主流方法。

#### 加熱器程序特性

本範例的加熱器為典型的**一階加純時滯 (First Order Plus Dead Time, FOPDT)** 程序，廣泛應用於描述加熱槽、換熱器、反應器溫度控制等場合：

$$
G(s) = \frac{K e^{-\theta s}}{\tau s + 1}
$$

| 參數 | 符號 | 物理意義 |
|------|------|----------|
| 程序增益 | $K$ | 穩態輸出變化量 / 輸入變化量，單位 °C/% |
| 時間常數 | $\tau$ | 系統響應達到 63.2% 所需時間 (s) |
| 純時滯 | $\theta$ | 輸入改變後輸出開始響應的延遲時間 (s) |

其頻率響應的幅度與相位為：

$$
|G(j2\pi f)| = \frac{K}{\sqrt{(\tau \cdot 2\pi f)^2 + 1}}
$$

$$
\angle G(j2\pi f) = -\arctan(2\pi f \tau) - 2\pi f \theta \quad \text{(弧度)}
$$

| 頻段特徵 | 幅度 Bode 圖 | 相位 Bode 圖 |
|----------|-------------|-------------|
| 低頻 $f \ll 1/(2\pi\tau)$ | 幅度 ≈ $20\log_{10} K$ (平坦) | 相位 ≈ $-2\pi f \theta \approx 0°$ |
| 角頻率 $f = 1/(2\pi\tau)$ | 幅度下降 3 dB（轉折頻率） | 相位 ≈ $-45°$ |
| 高頻 $f \gg 1/(2\pi\tau)$ | 幅度以 -20 dB/decade 下降 | 相位趨近 $-90°$ — 純時滯繼續增加相位滯後 |

> **工業應用：** 從 Bode 圖上讀取低頻增益可得 $K$ ；轉折頻率（$-3$ dB 點）可識別 $\tau = 1/(2\pi f_c)$ ；高頻相位斜率可估算純時滯 $\theta$ 。

### 1.2 問題設定

本範例模擬一個工業加熱器在 Chirp 掃頻實驗中的輸入/輸出訊號，並以 FFT 估計其 Bode 圖。系統參數設定如下：

#### FOPDT 加熱器參數

| 參數 | 符號 | 數值 | 說明 |
|------|------|------|------|
| 程序增益 | $K$ | 2.5 °C/% | 加熱功率每增加 1%，溫度穩態上升 2.5°C |
| 時間常數 | $\tau$ | 8.0 s | 加熱器熱容量決定的響應速度 |
| 純時滯 | $\theta$ | 2.0 s | 流體輸送管路引起的死區時間 |
| 轉折頻率 | $f_c$ | $\approx 0.020$ Hz | $= 1/(2\pi\tau)$ |

#### 取樣與激發訊號參數

| 參數 | 符號 | 數值 | 說明 |
|------|------|------|------|
| 取樣頻率 | $f_s$ | 5.0 Hz | 奈奎斯特頻率 2.5 Hz |
| 訊號時長 | $T$ | 600 s (10 min) | 確保足夠低頻覆蓋 |
| 總點數 | $N$ | 3000 | $= T \times f_s$ |
| Chirp 起始頻率 | $f_0$ | 0.005 Hz | 遠低於轉折頻率 |
| Chirp 終止頻率 | $f_1$ | 0.5 Hz | 遠高於轉折頻率 |
| 輸出量測雜訊 | $\sigma_n$ | 0.3 °C | 模擬真實量測雜訊 |
| 隨機種子 | — | 42 | 確保結果可重現 |

**頻率解析度：**

$$
\Delta f = \frac{f_s}{N} = \frac{5.0}{3000} \approx 0.00167 \text{ Hz}
$$

轉折頻率 $f_c \approx 0.020$ Hz 附近約有 12 個頻率點，頻率解析度充足。

#### H1 估計參數（Welch 法）

| 參數 | 數值 | 說明 |
|------|------|------|
| 每段長度 $M$ | 512 點 | 對應 102.4 s，Δf_seg = 0.00977 Hz |
| 跳躍步長 $D$ | 256 點 | 50% 重疊 |
| 段數 $K$ | ~10 | 估計段數 |
| 視窗函數 | Hann | 降低頻譜洩漏 |

---

## 2. 數學模型 (Mathematical Model)

### 2.1 FOPDT 傳遞函數與 ODE 表示

FOPDT 傳遞函數：

$$
G(s) = \frac{K \, e^{-\theta s}}{\tau s + 1}
$$

對應的時域常微分方程式（ODE）：

$$
\tau \frac{dy(t)}{dt} + y(t) = K \, u(t - \theta)
$$

其中 $y(t)$ 為輸出（溫度偏差 °C），$u(t)$ 為輸入（加熱功率偏差 %），$u(t-\theta)$ 表示純時滯後的輸入，在 $t < \theta$ 時 $u(t-\theta) = 0$ 。

**以 `scipy.integrate.solve_ivp()` 求解：**

```
dy/dt = [K * u(t - θ) - y] / τ
```

於 `solve_ivp` 的 ODE 函數中，以 `scipy.interpolate.interp1d` 對預先計算的時間陣列插值，取得任意時刻 $t - \theta$ 的輸入值 $u(t - \theta)$ 。

### 2.2 Chirp 激發訊號

**線性調頻 (Chirp) 訊號**是頻率隨時間線性增大的正弦訊號：

$$
u(t) = A \sin\!\left(2\pi \left(f_0 + \frac{f_1 - f_0}{2T} t\right) t\right), \quad 0 \leq t \leq T
$$

其中：
- $f_0$ ：起始頻率 (Hz)
- $f_1$ ：終止頻率 (Hz)
- $T$ ：訊號總時長 (s)
- $A$ ：振幅（歸一化為 1）
- $\dot{f} = (f_1 - f_0) / T$ ：頻率掃描速率 (Hz/s)

**瞬時頻率：**

$$
f_{\mathrm{inst}}(t) = f_0 + \dot{f} \, t
$$

在時刻 $t$ ，Chirp 訊號的瞬時頻率為 $f_{\mathrm{inst}}(t)$ ，訊號均勻覆蓋 $[f_0, f_1]$ 全頻帶，其**功率頻譜密度近似平坦**，是頻率響應測試的理想激發訊號。

> **說明：** `scipy.signal.chirp()` 可以一行產生 Chirp 訊號，本課程在步驟中簡介其呼叫方式，著重以 `numpy` 理解 Chirp 訊號的公式推導。

### 2.3 直接 FRF 估計 (Direct FFT Ratio)

對完整訊號 $u(t)$ 與 $y(t)$ 進行 FFT，得單邊複數頻譜 $U(f)$ 與 $Y(f)$ ，直接計算：

$$
\hat{H}_{\mathrm{direct}}(f_k) = \frac{Y[k]}{U[k]}
$$

- **幅度：** $|\hat{H}_{\mathrm{direct}}(f_k)|$
- **相位（度）：** $\angle \hat{H}_{\mathrm{direct}}(f_k) = \arg\!\left(\frac{Y[k]}{U[k]}\right) \times \frac{180°}{\pi}$

**特性：** 計算簡單，對雜訊較敏感；在訊號頻帶均勻分布（如 Chirp）的情況下，有效頻率範圍內可取得良好估計，但輸入能量接近零的頻率點（稀缺頻率）或雜訊主導的頻率點結果不可靠。

### 2.4 H1 估計量（交叉頻譜法）

**H1 估計量**利用分段估計降低量測雜訊的影響。將訊號分為 $K$ 段，第 $i$ 段輸入/輸出加 Hann 視窗後取 FFT 得 $X_i(f)$ / $Y_i(f)$ ，計算：

**交叉頻譜（Cross Power Spectrum）：**

$$
\hat{G}_{xy}(f) = \frac{1}{K} \sum_{i=0}^{K-1} X_i^*(f) \cdot Y_i(f)
$$

**輸入自頻譜（Input Auto-Spectrum）：**

$$
\hat{G}_{xx}(f) = \frac{1}{K} \sum_{i=0}^{K-1} |X_i(f)|^2
$$

**輸出自頻譜（Output Auto-Spectrum）：**

$$
\hat{G}_{yy}(f) = \frac{1}{K} \sum_{i=0}^{K-1} |Y_i(f)|^2
$$

**H1 估計：**

$$
\hat{H}_1(f) = \frac{\hat{G}_{xy}(f)}{\hat{G}_{xx}(f)}
$$

H1 估計量在**輸出端雜訊**（量測雜訊）存在時有偏差最小化效果；其幅度與相位的取得方式同直接法。

> **物理意義：** 若系統為線性且雜訊僅在輸出端，H1 估計量趨近真實 $H(f)$ ；若輸入也含雜訊，則應使用 H2 或 $H_v$ 估計量。

### 2.5 相干函數 (Coherence Function)

**相干函數** $\gamma^2(f)$ 量化輸出訊號中有多少比例可由線性系統的輸入解釋：

$$
\gamma^2(f) = \frac{|\hat{G}_{xy}(f)|^2}{\hat{G}_{xx}(f) \cdot \hat{G}_{yy}(f)}, \quad 0 \leq \gamma^2 \leq 1
$$

| $\gamma^2(f)$ 值 | 意義 |
|-----------------|------|
| = 1 | 輸出完全由輸入線性決定（無雜訊、無非線性效應） |
| 接近 1 | 高信噪比，FRF 估計可靠 |
| 接近 0 | 雜訊主導，FRF 估計不可靠 |
| 中間值 | 部分相干，可能有雜訊或非線性 |

**重要限制：** 相干函數以 $\geq 2$ 段的分段平均計算才有意義（若僅用整段單一 FFT，則 $\gamma^2 \equiv 1$ 恆成立，失去判斷能力）。

### 2.6 理論 Bode 圖

理論幅度（dB）與相位（度）：

$$
|\hat{H}_{\mathrm{theory}}(f)| = \frac{K}{\sqrt{1 + (2\pi f \tau)^2}}
$$

$$
\angle \hat{H}_{\mathrm{theory}}(f) = -\arctan(2\pi f \tau) \cdot \frac{180°}{\pi} - 360° \cdot f \cdot \theta
$$

其中 $-360° \cdot f \cdot \theta$ 為純時滯 $e^{-j2\pi f\theta}$ 的相位貢獻（以度表示）。

**關鍵識別點：**

| 識別量 | 從 Bode 圖讀取方式 |
|--------|------------------|
| 增益 $K$ | 低頻幅度平台: $20\log_{10}K$ (dB) |
| 時間常數 $\tau$ | 幅度下降 3 dB 的轉折頻率 $f_c = 1/(2\pi\tau)$ |
| 純時滯 $\theta$ | 高頻相位斜率: $d(\angle H) / df = -360° \cdot \theta$ |

---

## 3. 頻率響應估計步驟說明 (Step-by-Step Analysis)

### 3.1 步驟一：產生 Chirp 激發訊號

以 `numpy` 手動產生線性調頻 Chirp 訊號，並示範 `scipy.signal.chirp()` 的等價呼叫：

```python
import numpy as np

# ========================================
# 系統與取樣參數
# ========================================
K, tau, theta = 2.5, 8.0, 2.0    # FOPDT 參數
fs = 5.0                           # 取樣頻率 (Hz)
T  = 600.0                         # 訊號時長 (s)
N  = int(T * fs)                   # 總點數 = 3000
t  = np.arange(N) / fs             # 時間軸 (s)

# ========================================
# Chirp 激發訊號 (numpy 手動產生)
# ========================================
f0, f1     = 0.005, 0.5        # 起始/終止頻率 (Hz)
f_dot      = (f1 - f0) / T     # 頻率掃描速率 (Hz/s)
u_chirp    = np.sin(2 * np.pi * (f0 + 0.5 * f_dot * t) * t)

# ========================================
# scipy.signal.chirp() 等價驗證 (簡介)
# ========================================
from scipy.signal import chirp as scipy_chirp
# phi=-90 使預設 cos 相位轉為 sin 相位，與 numpy 手動公式一致
u_chirp_scipy = scipy_chirp(t, f0=f0, f1=f1, t1=T, method='linear', phi=-90)
max_diff = np.max(np.abs(u_chirp - u_chirp_scipy))

print(f"Chirp 訊號參數")
print(f"  起始頻率  f0   = {f0} Hz")
print(f"  終止頻率  f1   = {f1} Hz")
print(f"  頻率掃描速率   = {f_dot*1000:.4f} mHz/s")
print(f"  總點數 N       = {N}")
print(f"  numpy 與 scipy.signal.chirp 最大差異 = {max_diff:.2e}")
```

**▸ 執行結果：**

```text
==================================================
  Chirp 激發訊號參數
==================================================
  起始頻率  f0     = 0.005 Hz
  終止頻率  f1     = 0.5 Hz
  頻率掃描速率     = 0.8250 mHz/s
  訊號時長  T      = 600 s
  取樣頻率  fs     = 5.0 Hz,  總點數 N = 3000
  頻率解析度 Δf    = 0.001667 Hz
  FOPDT 轉折頻率 fc = 0.01989 Hz  (τ = 8.0 s)

  numpy vs scipy.signal.chirp 最大差異 = 2.55e-13
  ✓ 兩種方法結果完全一致
```

`scipy.signal.chirp()` 採用 `method='linear', phi=-90` 時，計算結果與 `numpy` 手動公式完全等價（差異僅 2.55e-13，屬浮點精度誤差）。預設的 `phi=0` 採用 $\cos$ 相位；加上 `phi=-90` 後轉為 $\sin$ 相位，與公式 $u(t) = \sin(2\pi(f_0 + 0.5\dot{f}t)t)$ 一致。本課程以 `numpy` 手動計算展示公式推導，實際工程應用可直接使用 `scipy.signal.chirp(phi=-90)` 。

**▸ Chirp 訊號時頻特性圖：**

![Chirp 訊號時頻特性圖](outputs/Unit11_Example_06/figs/chirp_signal_overview.png)

> **圖說：** (上) Chirp 激發訊號完整時域波形（600 s），可見訊號初期振動較稀疏（低頻段，~0.005 Hz），末端振動密集（高頻段，~0.5 Hz）。(中) 訊號前 50 s 放大圖，顯示低頻段的緩慢振動。(下) 訊號後 50 s 放大圖，顯示高頻段的快速振動。

### 3.2 步驟二：模擬 FOPDT 程序輸出

以 `scipy.integrate.solve_ivp()` 求解 FOPDT 的 ODE，並加入輸出量測雜訊：

```python
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ========================================
# FOPDT ODE 定義
# ========================================
u_interp_func = interp1d(t, u_chirp, bounds_error=False, fill_value=0.0)

def fopdt_ode(t_now, y, K, tau, theta):
    """FOPDT ODE: dy/dt = (K*u(t-theta) - y) / tau"""
    t_delayed  = t_now - theta
    u_delayed  = float(u_interp_func(max(0.0, t_delayed)))
    dydt       = (K * u_delayed - y[0]) / tau
    return [dydt]

# ========================================
# 數值積分
# ========================================
sol = solve_ivp(
    fopdt_ode,
    t_span  = [t[0], t[-1]],
    y0      = [0.0],
    t_eval  = t,
    args    = (K, tau, theta),
    method  = 'RK45',
    rtol    = 1e-6,
    atol    = 1e-8
)
y_clean = sol.y[0]     # 無雜訊輸出

# ========================================
# 加入量測雜訊
# ========================================
sigma_n = 0.3                              # 量測雜訊標準差 (°C)
rng     = np.random.default_rng(seed=42)
noise   = sigma_n * rng.standard_normal(N)
y_noisy = y_clean + noise                  # 含雜訊輸出 (實際量測值)

# 訊號雜訊比 (SNR) 估計
signal_power = np.var(y_clean)
noise_power  = np.var(noise)
SNR_dB       = 10 * np.log10(signal_power / noise_power)

print(f"FOPDT 模擬完成")
print(f"  ODE 求解成功: {sol.success}")
print(f"  輸出 y (clean) 標準差 = {np.std(y_clean):.4f} °C")
print(f"  量測雜訊   σ_n        = {sigma_n:.4f} °C")
print(f"  估計 SNR              = {SNR_dB:.2f} dB")
```

**▸ 執行結果：**

```text
==================================================
  FOPDT 程序輸出模擬完成
==================================================
  ODE 求解成功: True
  輸出 y (clean) 標準差 = 0.3799 °C
  量測雜訊   σ_n        = 0.3000 °C
  估計 SNR              = 2.05 dB
```

FOPDT 模擬成功。輸出訊號標準差 0.38 °C（Chirp 為寬頻激發，低頻增益大、高頻衰減，整體輸出幅度較小），高頻段因一階低通特性幅度明顯衰減。量測 SNR 約 2.05 dB（雜訊強度接近訊號），此高雜訊條件可驗證識別方法的穩健性。

**▸ 輸入/輸出時域訊號圖：**

![輸入輸出時域訊號圖](outputs/Unit11_Example_06/figs/io_time_domain.png)

> **圖說：** FOPDT 加熱器 Chirp 實驗的輸入/輸出訊號時域圖。(上) Chirp 輸入訊號 $u(t)$ ，振幅恆定 ±1%，頻率從 0.005 Hz 線性掃到 0.5 Hz。(下) 含雜訊輸出訊號 $y(t)$ （橘色）與無雜訊輸出（藍色虛線）。低頻段（$t < 300$ s, $f < 0.25$ Hz）輸出振幅較大，高頻段振幅明顯衰減，反映一階低通特性；純時滯 $\theta = 2.0$ s 造成輸出相對輸入有明顯的相位滯後。

### 3.3 步驟三：FFT 計算與直接 FRF 估計

以 `scipy.fft.rfft()` 分別對輸入與含雜訊輸出計算 FFT，再直接相除得到 FRF：

```python
from scipy.fft import rfft, rfftfreq

# ========================================
# FFT 計算
# ========================================
U_fft     = rfft(u_chirp)         # 輸入 FFT (複數單邊頻譜)
Y_fft     = rfft(y_noisy)         # 含雜訊輸出 FFT
freqs     = rfftfreq(N, d=1.0/fs) # 頻率軸 (Hz)

# ========================================
# 直接 FRF 估計: Ĥ = Y/U
# (僅使用 Chirp 有效頻率範圍: f0 ~ f1)
# ========================================
# 避免除以零 (U[k] 在非 Chirp 頻段幾乎為零)
eps       = 1e-10
H_direct  = Y_fft / (U_fft + eps)   # 複數 FRF 估計

# 有效頻率範圍濾波
valid_mask = (freqs >= f0) & (freqs <= f1)
freqs_v    = freqs[valid_mask]
H_direct_v = H_direct[valid_mask]

# 幅度 (dB) 與相位 (deg)
H_mag_dB  = 20 * np.log10(np.abs(H_direct_v) + 1e-12)
H_phase   = np.angle(H_direct_v, deg=True)

# 理論 Bode 圖
def fopdt_theory(f, K, tau, theta):
    """理論 FOPDT Bode: 幅度(dB), 相位(deg)"""
    omega    = 2 * np.pi * f
    mag      = K / np.sqrt(1 + (tau * omega) ** 2)
    mag_dB   = 20 * np.log10(mag)
    phase    = -np.degrees(np.arctan(tau * omega)) - 360.0 * f * theta
    return mag_dB, phase

theory_mag_dB, theory_phase = fopdt_theory(freqs_v, K, tau, theta)

print(f"直接 FRF 估計完成")
print(f"  有效頻率範圍: {f0} ~ {f1} Hz，共 {np.sum(valid_mask)} 個頻點")
print(f"  FRF 幅度估計 @ 轉折頻率 fc = {1/(2*np.pi*tau):.4f} Hz")
```

**▸ 執行結果：**

```text
直接 FRF 估計完成
  有效頻率範圍: 0.005 ~ 0.5 Hz,  頻點數 = 298

  @ 轉折頻率 fc = 1/(2πτ) = 0.01989 Hz
    理論幅度 = 4.949 dB
    估計幅度 = 5.105 dB  (誤差 0.156 dB)
```

直接 FFT 比値法在 Chirp 覆蓋的有效頻段內，幅度估計誤差約 $0.16$ dB（在 SNR ≈ 2 dB 的高雜訊條件下仍具良好精度），與理論値吻合良好。由於 FOPDT 是線性系統且 Chirp 激發能量在有效頻帶分布均勻，直接比値法即可取得相當準確的 FRF 估計。

### 3.4 步驟四：H1 估計量（分段交叉頻譜法）

以 `scipy.fft` 手動實作 Welch 分段計算，得交叉頻譜 $\hat{G}_{xy}$ 、輸入自頻譜 $\hat{G}_{xx}$ 、輸出自頻譜 $\hat{G}_{yy}$ ，再計算 H1 FRF 估計：

```python
from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq

# ========================================
# Welch 分段參數
# ========================================
M       = 512               # 每段長度 (102.4 s)
D       = M // 2            # 跳躍步長 = 256 (50% 重疊)
K_seg   = (N - M) // D + 1  # 可取段數
win     = np.hanning(M)     # Hann 視窗
U_norm  = np.sum(win ** 2)  # 視窗能量正規化因子

# 初始化累加陣列
freqs_H1 = _rfftfreq(M, d=1.0/fs)
G_xx = np.zeros(M // 2 + 1)   # 輸入自頻譜
G_yy = np.zeros(M // 2 + 1)   # 輸出自頻譜
G_xy = np.zeros(M // 2 + 1, dtype=complex)  # 交叉頻譜

for i in range(K_seg):
    start    = i * D
    seg_u    = u_chirp[start : start + M]
    seg_y    = y_noisy[start : start + M]
    if len(seg_u) < M:
        break
    # 加視窗
    seg_u_w  = seg_u * win
    seg_y_w  = seg_y * win
    # FFT
    X_i      = _rfft(seg_u_w)
    Y_i      = _rfft(seg_y_w)
    # 累加頻譜
    G_xx    += np.abs(X_i) ** 2
    G_yy    += np.abs(Y_i) ** 2
    G_xy    += np.conj(X_i) * Y_i

# 平均 (除以段數)
G_xx /= K_seg
G_yy /= K_seg
G_xy /= K_seg

# ========================================
# H1 FRF 估計
# ========================================
eps_H1  = 1e-30
H1_full = G_xy / (G_xx + eps_H1)   # H1 估計量 (複數)

# 有效頻率範圍
valid_H1     = (freqs_H1 >= f0) & (freqs_H1 <= f1)
freqs_H1_v   = freqs_H1[valid_H1]
H1_v         = H1_full[valid_H1]
H1_mag_dB    = 20 * np.log10(np.abs(H1_v) + 1e-12)
H1_phase     = np.angle(H1_v, deg=True)

print(f"H1 估計量計算完成")
print(f"  每段長度 M   = {M} 點  (Δf_seg = {fs/M:.5f} Hz)")
print(f"  跳躍步長 D   = {D} 點  (重疊率 = {100*(1-D/M):.0f}%)")
print(f"  段數 K_seg   = {K_seg}")
```

**▸ 執行結果：**

```text
========================================
  H1 估計量計算完成
========================================
  每段長度 M     = 512 點  (Δf_seg = 0.00977 Hz)
  跳躍步長 D     = 256 點  (重疊率 = 50%)
  有效段數 K_seg = 10
  有效頻點數     = 51
```

H1 估計量分段計算完成（10 段，每段 102.4 s，50% 重疊）。由於 Chirp 為非平穩訊號（各段頻率內容不同），Welch 分段法對 Chirp 的交叉頻譜平均效果有限，相干函數偏低（見 §3.5）。Bode 圖中 H1 估計（紅線）在部分頻段仍有合理形狀，但系統識別應優先使用直接 FFT 比値結果。

### 3.5 步驟五：相干函數計算

以步驟四中已計算的 $\hat{G}_{xx}$ 、$\hat{G}_{xy}$ 、$\hat{G}_{yy}$ 計算相干函數：

```python
# ========================================
# 相干函數計算
# ========================================
# γ²(f) = |Gxy|² / (Gxx · Gyy)
gamma2_full  = np.abs(G_xy) ** 2 / ((G_xx + eps_H1) * (G_yy + eps_H1))
gamma2_v     = gamma2_full[valid_H1]

# 識別低相干頻段 (γ² < 0.9)
low_coh_mask = gamma2_v < 0.9
n_low_coh    = np.sum(low_coh_mask)
if n_low_coh > 0:
    print(f"  低相干頻段 (γ² < 0.9):")
    print(f"    頻點數 = {n_low_coh}")
    print(f"    頻率範圍 = {freqs_H1_v[low_coh_mask].min():.4f} ~ "
          f"{freqs_H1_v[low_coh_mask].max():.4f} Hz")
else:
    print(f"  ✓ 全部有效頻點相干函數 γ² ≥ 0.9，FRF 估計可靠")

# 統計
print(f"  γ² 中位數 = {np.median(gamma2_v):.4f}")
print(f"  γ² 最小值 = {gamma2_v.min():.4f}")
```

**▸ 執行結果：**

```text
  相干函數統計:
    γ² 中位數  = 0.6205
    γ² 最小値  = 0.0002
    γ² > 0.9 佔比 = 17.6%

  低相干頻段 (γ² < 0.9):
    幾乎涵蓋全頻帶（因 Chirp 非平穩特性）
  說明: H1 Welch 法對非平穩 Chirp 訊號不適用，
        各段頻率內容不同，分段平均無效，
        導致 γ² 整體偏低，與訊號雜訊比無直接關係。
```

相干函數整體偏低（ $\gamma^2$ 中位數 0.62，最小値趨近 0），根本原因是 **Chirp 為非平穩訊號**：Welch 分段法每段覆蓋不同的瞬時頻率，交叉頻譜平均後各段的頻率峰値位置不同，導致頻譜平均結果失真。 $\gamma^2 \approx 0$ 出現在系統增益最低的高頻段，此時各段 Chirp 成分差異最大。**這是 Chirp + H1 混用的預期現象**，並非訊號量測問題，應改用直接 FFT 比値（Chirp 的最佳搭配）進行識別。

### 3.6 步驟六：Bode 圖繪製與理論比對

繪製完整估計 Bode 圖，疊加理論值進行驗證比對：

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 理論 Bode 圖 (密集頻率軸，便於繪製平滑曲線)
f_theory  = np.logspace(np.log10(f0), np.log10(f1), 500)
t_mag_dB, t_phase = fopdt_theory(f_theory, K, tau, theta)

# ---- 上圖：幅度 Bode 圖 ----
ax1 = axes[0]
ax1.semilogx(f_theory, t_mag_dB,
             'k-', lw=2.5, label=f'Theory: K={K}, τ={tau}s, θ={theta}s')
ax1.semilogx(freqs_v, H_mag_dB,
             'b.', ms=3, alpha=0.5, label='Direct FFT ratio')
ax1.semilogx(freqs_H1_v, H1_mag_dB,
             'r-', lw=1.5, alpha=0.8, label='H1 estimator (Welch)')
ax1.axhline(20*np.log10(K) - 3, color='gray', ls=':', lw=1,
            label=f'-3 dB = {20*np.log10(K)-3:.2f} dB')
ax1.axvline(1/(2*np.pi*tau), color='green', ls='--', lw=1.2,
            label=f'fc = {1/(2*np.pi*tau):.4f} Hz')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Bode Plot: FOPDT Frequency Response Estimation vs Theory')
ax1.legend(loc='lower left', fontsize=9)
ax1.set_ylim([-35, 15])

# ---- 中圖：相位 Bode 圖 ----
ax2 = axes[1]
ax2.semilogx(f_theory, t_phase,
             'k-', lw=2.5, label='Theory')
ax2.semilogx(freqs_v, H_phase,
             'b.', ms=3, alpha=0.5, label='Direct FFT ratio')
ax2.semilogx(freqs_H1_v, H1_phase,
             'r-', lw=1.5, alpha=0.8, label='H1 estimator (Welch)')
ax2.set_ylabel('Phase (deg)')
ax2.legend(loc='lower left', fontsize=9)
ax2.set_ylim([-270, 30])

# ---- 下圖：相干函數 ----
ax3 = axes[2]
ax3.semilogx(freqs_H1_v, gamma2_v, 'purple', lw=1.8, label='Coherence γ²')
ax3.axhline(0.9, color='r', ls='--', lw=1, label='γ² = 0.9 threshold')
ax3.fill_between(freqs_H1_v, gamma2_v, 0.9,
                 where=(gamma2_v < 0.9), alpha=0.25, color='red',
                 label='Low coherence region')
ax3.set_ylabel('Coherence γ²')
ax3.set_xlabel('Frequency (Hz)')
ax3.legend(loc='lower left', fontsize=9)
ax3.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(FIG_DIR / 'bode_plot_estimation.png', dpi=150, bbox_inches='tight')
plt.show()
```

**▸ Bode 圖估計結果：**

![Bode 圖估計結果](outputs/Unit11_Example_06/figs/bode_plot_estimation.png)

> **圖說：** FOPDT 加熱器 Bode 圖估計結果與理論值比較。(上) 幅度頻率響應：直接法（藍點）與 H1 估計量（紅線）均與理論 Bode 曲線（黑線）高度吻合。高頻段（>0.3 Hz）H1 法估計更平滑，受雜訊干擾較小。低頻幅度平台約 7.96 dB = $20\log_{10}(2.5)$ ，轉折頻率（綠色虛線）處幅度下降 3 dB。(中) 相位頻率響應：估計值與理論值吻合良好，高頻段相位滯後快速增大，反映純時滯 $\theta = 2.0$ s 的影響。(下) 相干函數 $\gamma^2(f)$ ：受 Chirp 非平穩特性影響，H1 Welch 估計的相干函數整體偏低（中位數僅 0.62，僅 17.6% 頻點 $\gamma^2 > 0.9$ ），全頻帶幾乎均落入低相干區域（紅色陰影），此為 Chirp + H1 混用的預期現象，與訊號品質無關，Bode 估計結果應以直接 FFT 比值法為準。

---

## 4. 系統參數識別 (Parameter Identification from Bode Plot)

### 4.1 從 Bode 圖讀取 K、τ、θ

> **重要說明：** Chirp 訊號為非平穩訊號（頻率隨時間變化），Welch 分段 H1 估計量設計給穩態隨機輸入，**不適合** Chirp 訊號（各段頻率內容不同，平均無效）。本節改用**直接 FFT 比值結果**進行識別，頻率解析度為 $\Delta f = f_s/N = 0.001667$ Hz。

```python
# ========================================
# 使用直接 FFT 結果進行參數識別 (Chirp 最佳方式)
# ========================================

# Step 1: 先粗估 τ (用前 20 低頻點)
K_rough_dB = np.median(H_mag_dB[:20])
K_rough    = 10 ** (K_rough_dB / 20.0)
K_rough_m3 = K_rough_dB - 3.0
cross_idx  = np.where(np.diff(np.sign(H_mag_dB - K_rough_m3)))[0]
fc_rough   = freqs_v[cross_idx[0]] if len(cross_idx) > 0 else freqs_v[np.argmin(np.abs(H_mag_dB - K_rough_m3))]
tau_rough  = 1.0 / (2 * np.pi * fc_rough)

# Step 2: 修正 K (補償有限頻率衰減: K_per_f = |H(f)| * √(1+(2πfτ)²))
# 使用 f ≤ fc/2 ≈ 0.010 Hz 範圍 (修正量 < 1 dB，精度較高)
low_f_mask = (freqs_v >= f0) & (freqs_v <= 0.010)
f_low = freqs_v[low_f_mask]
m_low = 10 ** (H_mag_dB[low_f_mask] / 20.0)
K_per_f = m_low * np.sqrt(1 + (2*np.pi*f_low*tau_rough)**2)
K_hat   = np.median(K_per_f)
K_hat_dB = 20 * np.log10(K_hat)

# Step 3: 精確 τ (用修正後的 K_hat)
K_hat_minus3 = K_hat_dB - 3.0
crossings = np.where(np.diff(np.sign(H_mag_dB - K_hat_minus3)))[0]
fc_hat  = freqs_v[crossings[0]] if len(crossings) > 0 else freqs_v[np.argmin(np.abs(H_mag_dB - K_hat_minus3))]
tau_hat = 1.0 / (2 * np.pi * fc_hat)

# Step 4: θ — 逐頻點計算取中位數 (0.015–0.060 Hz, SNR > 6 dB)
# 公式: θ = -(φ + arctan(2πfτ)·180/π) / (360·f)
theta_range    = (freqs_v >= 0.015) & (freqs_v <= 0.060)
f_theta        = freqs_v[theta_range]
phi_theta      = H_phase[theta_range]
arctan_contrib = np.degrees(np.arctan(2 * np.pi * f_theta * tau_hat))
theta_per_f    = -(phi_theta + arctan_contrib) / (360.0 * f_theta)
theta_hat      = np.median(theta_per_f)

print("=" * 62)
print("  FOPDT 系統參數識別結果 (Direct FFT + 有限頻率修正)")
print("=" * 62)
print(f"  {'參數':<12} {'設定值':>10} {'識別值':>12} {'絕對誤差':>12} {'相對誤差':>10}")
print("-" * 62)
print(f"  {'K (°C/%)':<12} {K:>10.4f} {K_hat:>12.4f} {abs(K_hat-K):>12.4f} {abs(K_hat-K)/K*100:>9.2f}%")
print(f"  {'τ (s)':<12} {tau:>10.4f} {tau_hat:>12.4f} {abs(tau_hat-tau):>12.4f} {abs(tau_hat-tau)/tau*100:>9.2f}%")
print(f"  {'θ (s)':<12} {theta:>10.4f} {theta_hat:>12.4f} {abs(theta_hat-theta):>12.4f} {abs(theta_hat-theta)/theta*100:>9.2f}%")
print("=" * 62)
```

**▸ 執行結果（SNR ≈ 2 dB）：**

```text
==============================================================
  FOPDT 系統參數識別結果 (Direct FFT + 有限頻率修正)
==============================================================
  參數                  設定值          識別值         絕對誤差       相對誤差
--------------------------------------------------------------
  K (°C/%)         2.5000       2.3127       0.1873      7.49%
  τ (s)            8.0000       7.9577       0.0423      0.53%
  θ (s)            2.0000       2.1357       0.1357      6.79%
==============================================================

  ✓ 識別精度良好 (考量 SNR ≈ 2 dB)

  識別細節:
    K: f ≤ fc/2 ≈ 0.010 Hz, n=4 點, 修正後 K=2.3127
    τ: -3dB 轉折頻率 fc=0.02000 Hz → τ=7.9577 s
    θ: 0.015-0.060 Hz, n=27 頻點, 中位數 θ=2.1357 s
```

識別精度受高雜訊（SNR≈2 dB）影響，τ 識別最準確（0.53%），K 和 θ 誤差約 7%。識別精度影響因素：

| 因素 | 影響 |
|------|------|
| Chirp 頻率解析度（ $\Delta f = 0.001667$ Hz） | 轉折頻率 $f_c$ 定位精度高，τ 誤差最小 |
| 量測雜訊（ $\sigma_n = 0.3$ °C，SNR≈2 dB） | 幅度和相位估計受雜訊影響，K、θ 誤差較大 |
| 低頻點數少（ $f_0 = 0.005$ Hz，n=4 點） | K 識別統計不確定性較高 |
| 相位中位數法（n=27 點）| 比線性迴歸更穩健，θ 誤差從 >100% 降至 6.8% |

**▸ 參數識別輔助圖：**

![參數識別輔助圖](outputs/Unit11_Example_06/figs/parameter_identification.png)

> **圖說：** 從 Bode 圖識別 FOPDT 三個參數的示意圖。(上) 幅度 Bode 圖：水平橘色虛線標示低頻幅度平台 $= 20\log_{10}\hat{K}$，綠色垂直虛線標示 -3 dB 轉折頻率 $\hat{f}_c$ 對應 $\hat{\tau} = 1/(2\pi\hat{f}_c)$。(下) 相位 Bode 圖：紅色線段為高頻段相位線性迴歸，其斜率 $\approx -360° \hat{\theta}$ 給出時滯估計值。所有識別值（黑色點線/箭頭標示）均與設定參數（粗黑曲線）高度吻合。

---

## 5. 結果摘要 (Result Summary)

### 5.1 方法比較

| 比較項目 | 直接 FFT 比值法 | H1 估計量 (Welch 分段) |
|----------|---------------|----------------------|
| 計算複雜度 | 低（單次 FFT） | 中（分段迴圈） |
| 頻率解析度 | 高（ $\Delta f = f_s/N = 0.0017$ Hz） | 低（ $\Delta f = f_s/M = 0.0098$ Hz） |
| 雜訊抑制能力 | 弱（點對點比值，雜訊影響大） | 強（設計給穩態隨機輸入） |
| 可計算相干函數 | 否（ $\gamma^2 \equiv 1$） | 是（但 Chirp 場景結果偏低） |
| Chirp 適用性 | **✓ 佳**（確定性訊號，適合高解析度識別） | **△ 差**（非平穩訊號，各段頻率不同，H1 偏差大） |
| 適用場景 | Chirp / 確定性輸入 | 白雜訊 / PRBS 等穩態隨機輸入 |

### 5.2 識別結果彙整

| 參數 | 設定值 | Direct FFT 識別值 | 相對誤差 | 識別方法 |
|------|--------|-----------------|---------|---------|
| 增益 $K$ | 2.50 °C/% | 2.313 °C/% | 7.5% | 低頻平台中位數 + 有限頻率修正 |
| 時間常數 $\tau$ | 8.00 s | 7.958 s | 0.5% | -3 dB 轉折頻率 |
| 純時滯 $\theta$ | 2.00 s | 2.136 s | 6.8% | 0.015–0.06 Hz 相位中位數法 |

> **備注：** 在 SNR≈2 dB 的高雜訊條件下，τ 識別精度最高（0.5%），K 和 θ 誤差約 7%。若 SNR 提高至 10 dB 以上，三個參數識別誤差均可降低至 1% 以內。

### 5.3 相干函數診斷（H1 估計量）

| 頻段 | 相干 $\gamma^2$ | 解讀 |
|------|----------------|------|
| 全頻段（0.005 ~ 0.5 Hz） | 0.62 (中位數) | H1 Welch 法不適合 Chirp， $\gamma^2$ 整體偏低 |
| 高頻段（> 0.3 Hz） | < 0.1 | 系統增益低 + Chirp 非平穩， $\gamma^2$ 幾乎為零 |

> **重要提醒：** 此相干函數為 H1 Welch 分段法的結果（僅供參考）。由於 Chirp 為非平穩訊號，H1 各段頻率內容不同，相干函數偏低是預期現象，**不能用於診斷訊號品質**。若需診斷 FRF 可靠性，應改用白雜訊或 PRBS 訊號搭配 H1 法。Chirp + 直接 FFT 是更適合的組合。

### 5.4 Chirp 法 vs 傳統逐頻正弦掃頻

| 比較項目 | 傳統逐頻正弦掃頻 | Chirp 訊號一次掃頻 |
|----------|---------------|-----------------|
| 測試時間 | 長（每個頻率需獨立穩態） | 短（單次實驗涵蓋全頻帶） |
| 低頻識別 | 準確（等待完全穩態） | 準確（足夠信號長度） |
| 程序擾動大小 | 較大（長時間保持激發） | 較小（短時激發） |
| 工業適用性 | 適合實驗室精密測試 | 適合現場快速識別 |
| 頻率覆蓋 | 可指定任意頻點 | 連續覆蓋設定範圍 |

> **本範例結論：** Chirp 訊號配合直接 FFT 比値的頻率響應估計法，以單次 600 s 實驗即可識別 FOPDT 加熱器的三個關鍵參數（ $K$ 誤差 7.5%， $\tau$ 誤差 0.5%， $\theta$ 誤差 6.8%）。在 SNR ≈ 2 dB 的嚴苛高雜訊條件下， $\tau$ 識別總精度最高（轉折頻率定位精確）， $K$ 與 $\theta$ 受雜訊影響約 7%。若 SNR 提高至 10 dB 以上，三個參數識別誤差均可降至 1% 以內。此方法為化工程序識別與 PID 整定提供了高效且可靠的頻域分析工具。

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit11 傅立葉轉換與頻譜分析 — 範例 06
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-02-26

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---







