# Unit11 傅立葉轉換與頻譜分析 (Fourier Transform & Spectral Analysis)

## 📚 單元簡介

化工製程中的感測器訊號——溫度振盪、壓力脈動、流量波動——往往是多個頻率成分的疊加。**傅立葉轉換（Fourier Transform）**是將時域訊號轉換到頻域的核心數學工具，讓我們得以識別訊號的週期性成分、分析設備異常特徵頻率、建立液泛預警指標，乃至估計程序的動態模型（Bode 圖）。

本單元以 **`scipy.fft`** 為核心工具，系統性介紹從連續傅立葉轉換（CFT）到離散傅立葉轉換（DFT）、快速傅立葉演算法（FFT）的理論體系，以及視窗函數、功率頻譜密度（PSD）、短時傅立葉轉換（STFT）、自相關函數等進階頻譜分析技術。透過六個化工實際案例，從感測器訊號主頻識別、批次反應器週期性分析、CSTR 振盪偵測，到蒸餾塔液泛預警、泵浦葉片通過頻率識別，以及加熱器頻率響應估計（Bode 圖），完整呈現 FFT 頻譜分析在化工製程監控與診斷中的應用。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握傅立葉轉換的理論基礎**：理解 CFT、DFT、FFT 的關係，以及取樣定理（Nyquist 定理）、頻率解析度、混疊現象等核心概念
2. **熟練使用 `scipy.fft` 工具**：使用 `rfft()`、`rfftfreq()`、`fftshift()`、`next_fast_len()` 等函式計算單邊幅度頻譜並正確歸一化
3. **應用視窗函數與前處理技術**：去均值、去趨勢、套用 Hann/Hamming/Blackman 視窗，理解頻譜洩漏的成因與改善策略
4. **計算與解讀功率頻譜密度（PSD）**：以週期圖法（Periodogram）與 Welch 法（手動實作多段平均）估計 PSD，理解兩者的方差與解析度取捨
5. **進行化工製程頻譜診斷**：識別製程訊號的特徵頻率（設備振動、週期性干擾、系統振盪），建立頻域量化指標用於異常偵測與狀態監控

---

## 📖 單元內容架構

### 1️⃣ Unit11_Fourier_Transform — 傅立葉轉換理論與頻譜分析工具

**檔案**：
- 講義：[Unit11_Fourier_Transform.md](Unit11_Fourier_Transform.md)
- 程式範例：[Unit11_Fourier_Transform.ipynb](Unit11_Fourier_Transform.ipynb)

**內容重點**：

#### 傅立葉轉換理論基礎

| 類型 | 定義 | 說明 |
|------|------|------|
| **連續傅立葉轉換（CFT）** | $\mathcal{F}\{f(t)\} = \int_{-\infty}^{\infty} f(t) e^{-j2\pi ft} dt$ | 時域 → 頻域 |
| **反傅立葉轉換** | $f(t) = \int_{-\infty}^{\infty} F(\nu) e^{j2\pi \nu t} d\nu$ | 頻域 → 時域 |
| **離散傅立葉轉換（DFT）** | $X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$ | 離散訊號的頻域表示 |
| **FFT（快速演算）** | Cooley-Tukey 蝶形運算 | $O(N^2) \to O(N \log N)$ |

**取樣定理（Nyquist-Shannon Theorem）**：
- 取樣頻率 $f_s \geq 2f_{max}$，奈奎斯特頻率 $f_N = f_s / 2$
- 頻率解析度：$\Delta f = f_s / N$
- 混疊（Aliasing）：取樣頻率不足時的頻率折疊效應

#### `scipy.fft` 核心函式總覽

| 函式 | 功能 | 說明 |
|------|------|------|
| `fft(x)` | 雙邊複數頻譜 | 含正負頻率，複數輸出 |
| `ifft(X)` | 反 FFT | 從頻域還原時域 |
| `rfft(x)` | **單邊頻譜（推薦）** | 實數訊號專用，節省計算量 |
| `irfft(X)` | 實數反 FFT | 從單邊頻譜還原 |
| `rfftfreq(n, d)` | 實數 FFT 頻率軸 | 自動建立 $[0, f_N]$ 頻率陣列 |
| `fftfreq(n, d)` | 完整頻率軸 | 含負頻率（雙邊用） |
| `fftshift(X)` | 零頻移至中央 | 雙邊頻譜視覺化 |
| `next_fast_len(n)` | FFT 最佳點數 | 零填充至 2 的冪次或高合成數 |
| `fft2(x)` | 二維 FFT | 影像或空間場頻譜分析 |

**幅度歸一化**：
- 單邊幅度：$A = |X[k]| \times \dfrac{2}{N}$（DC 與 Nyquist 點除外）
- 雙邊幅度：$A = |X[k]| \times \dfrac{1}{N}$

#### 視窗函數與頻譜洩漏

| 視窗函數 | 主葉寬度 | 旁葉抑制 | 適用場景 |
|---------|---------|---------|---------|
| **Rectangular（矩形）** | 最窄 | 最差（-13 dB） | 訊號恰好為整數週期 |
| **Hann** | 中 | 好（-31 dB） | 一般用途，最推薦 |
| **Hamming** | 中 | 較好（-43 dB） | 需要更高旁葉抑制 |
| **Blackman** | 寬 | 最好（-58 dB） | 弱訊號分析 |

- **頻譜洩漏成因**：有限時間截斷等效於乘上矩形視窗，頻域產生 sinc 函數卷積
- **幅度修正因子（ACF）**：套用視窗後需補償幅度衰減：$ACF = N / \sum w[n]$

#### 功率頻譜密度（PSD）

- **PSD 定義**：$S(f) = \lim_{T\to\infty} \frac{1}{T}|X_T(f)|^2$，描述訊號能量在頻率上的分布
- **週期圖法（Periodogram）**：直接由 `rfft()` 計算，頻譜方差大（不平滑）
- **Welch 方法原理**：分段 → 加視窗 → 各段 FFT → 平均，降低頻譜方差（以頻率解析度為代價）
- **補充說明**：`scipy.signal.welch()` 可直接計算 Welch PSD，本課程著重以 `scipy.fft` 理解底層原理

#### 短時傅立葉轉換（STFT）

- **STFT 定義**：對訊號加滑動視窗後逐段 FFT，得到時頻二維表示
- **手動實作**：迴圈切段 + Hann 視窗 + `rfft()`（理解底層原理）
- **頻譜圖（Spectrogram）**：以 `matplotlib.pyplot.pcolormesh()` 顯示頻率-時間-能量熱圖
- **時間 vs 頻率解析度取捨**：視窗長度 ↑ → 頻率解析度 ↑，時間解析度 ↓
- **補充說明**：`scipy.signal.stft()` / `scipy.signal.spectrogram()` 可一行完成，本課程範例示範呼叫方式

#### 自相關函數與 Wiener-Khinchin 定理

$$
R(\tau) = \int_{-\infty}^{\infty} x(t) x(t+\tau) dt = \mathcal{F}^{-1}\{|X(f)|^2\}
$$

- 以 `fft()` + `ifft()` 計算循環自相關（快速且精確）
- **Wiener-Khinchin 定理**：$S(f) = \mathcal{F}\{R(\tau)\}$（自相關函數的頻譜等於 PSD）
- 應用：從自相關函數估計訊號週期，交叉驗證 FFT 峰值頻率

#### 化工頻譜分析前處理最佳實踐
- **去趨勢（Detrending）**：`numpy.polyfit()` 擬合並去除線性趨勢（消除低頻假象）
- **去均值（Demean）**：去除 DC 分量，避免零頻峰掩蓋有用頻率資訊
- **零填充（Zero Padding）**：`next_fast_len()` 補零提升頻率軸密度（插值效果，不提升解析度）
- **峰值識別**：`numpy` 手動尋找頻譜局部最大值，或簡介 `scipy.signal.find_peaks()` 的用途

---

## 🧪 化工案例演練

### 📡 Example 01 — 製程感測器訊號之頻譜分析與主頻識別

**檔案**：[Unit11_Example_01.md](Unit11_Example_01.md) | [Unit11_Example_01.ipynb](Unit11_Example_01.ipynb)

**問題概述**：化工廠溫度/壓力感測器量測訊號，含兩個已知頻率成分（設備振動、週期性干擾）、DC 偏差與隨機雜訊。以 FFT 識別主要頻率成分並驗證幅度歸一化的正確性。

**合成測試訊號**：

$$
x(t) = \mu + A_1 \sin(2\pi f_1 t) + A_2 \sin(2\pi f_2 t) + \sigma\epsilon(t)
$$

**化工重點**：
- `scipy.fft.rfft()` + `rfftfreq()` 計算單邊頻譜，歸一化因子 $2/N$ 驗證峰值幅度
- 去均值（demean）前處理，觀察 DC 分量對頻譜的影響
- 手動套用 Hann 視窗（numpy 陣列乘法），比較有/無視窗的頻譜洩漏改善效果
- 計算各頻率成分能量佔總訊號能量的百分比
- 繪製：時域波形圖、雙邊複數頻譜圖、單邊幅度頻譜圖（含/不含視窗比較）

---

### 🌡️ Example 02 — 批次反應器溫度訊號之週期性分析與 PSD 計算

**檔案**：[Unit11_Example_02.md](Unit11_Example_02.md) | [Unit11_Example_02.ipynb](Unit11_Example_02.ipynb)

**問題概述**：批次反應器攪拌槳轉動引發週期性溫度脈動，同時含低頻溫度上升趨勢（製程升溫）與高頻量測雜訊。以頻譜分析解析溫度訊號的各頻率成分，計算功率頻譜密度（PSD）。

**合成訊號**：線性升溫趨勢 + 攪拌主頻 $f_1$ + 二次諧波 $2f_1$ + 白雜訊

**化工重點**：
- **去趨勢前處理**：`numpy.polyfit()` 擬合並去除線性趨勢，比較去趨勢前後頻譜差異
- 使用 `rfft()` 計算週期圖（Periodogram），推導 PSD 估計值（除以 $f_s \cdot N$）
- 比較 Rectangular、Hann、Hamming 三種視窗的幅度解析度與旁葉抑制差異
- `next_fast_len()` 零填充，觀察頻率軸密度的提升效果
- 以 `fft()` + `ifft()` 計算自相關函數，用 Wiener-Khinchin 定理驗證 PSD
- 繪製：原始/去趨勢訊號對比、不同視窗 PSD 比較、自相關函數圖

---

### 🔄 Example 03 — CSTR 反應器濃度振盪之頻譜分析（Limit Cycle Detection）

**檔案**：[Unit11_Example_03.md](Unit11_Example_03.md) | [Unit11_Example_03.ipynb](Unit11_Example_03.ipynb)

**問題概述**：非恆溫 CSTR 在特定操作條件下出現週期性濃度振盪（Limit Cycle），透過 FFT 頻譜分析確認振盪週期，比較不同冷卻溫度下的系統動態特性（穩態收斂 → 週期振盪 → 擬混沌）。

**數學模型**：使用 `scipy.integrate.solve_ivp()` 模擬 CSTR 動態（延伸自 Unit09_Example_01），分析穩態後的時間序列頻譜

**化工重點**：
- 識別主要振盪頻率 $f_0$ 與諧波（$2f_0$、$3f_0$）
- 比較三種操作條件的頻譜特徵：
  - **穩態收斂**：DC 分量主導，無顯著峰值
  - **週期振盪（Limit Cycle）**：清晰的 $f_0$ 及諧波峰值
  - **擬似混沌**：寬帶連續頻譜（無明顯峰值）
- 自相關函數估計振盪週期，與 FFT 峰值頻率交叉驗證
- 繪製：相平面圖（Phase Portrait）、時域曲線、單邊幅度頻譜、自相關函數
- **與 Unit09 的連結**：直接使用 Unit09 的 CSTR ODE 模型，展示 ODE 模擬 + FFT 分析的整合工作流程

---

### 🏭 Example 04 — 蒸餾塔差壓訊號之時頻分析與液泛預警

**檔案**：[Unit11_Example_04.md](Unit11_Example_04.md) | [Unit11_Example_04.ipynb](Unit11_Example_04.ipynb)

**問題概述**：蒸餾塔差壓訊號在接近液泛（Flooding）時出現特定低頻振盪增強。以手動實作的滑動視窗 FFT（STFT）即時監測頻率成分的時間演變，建立液泛早期預警指標 $I_{flood}(t)$。

**合成訊號**（三個操作階段）：
1. **正常操作**：隨機白雜訊，無低頻振盪
2. **液泛初期**：低頻振盪逐漸增強，幅度緩慢上升
3. **完全液泛**：低頻成分主導，幅度急遽增大

**化工重點**：
- 以 `scipy.fft.rfft()` 手動實作滑動視窗 FFT（迴圈切段 + Hann 視窗），理解 STFT 底層原理
- 繪製**時頻頻譜圖（Spectrogram）**：`matplotlib.pyplot.pcolormesh()` 顯示頻率-時間-能量熱圖
- 提取低頻頻段（0.01–0.1 Hz）的頻譜能量，定義液泛指標 $I_{flood}(t)$
- 比較時域統計指標（滾動標準差）vs 頻域 $I_{flood}$ 的早期預警靈敏度
- **補充說明**：`scipy.signal.stft()` 示範（一行完成），與手動結果比對驗證等價性

---

### ⚙️ Example 05 — 管道壓力脈動之 PSD 分析與泵浦葉片通過頻率識別

**檔案**：[Unit11_Example_05.md](Unit11_Example_05.md) | [Unit11_Example_05.ipynb](Unit11_Example_05.ipynb)

**問題概述**：離心泵浦壓力脈動訊號含葉片通過頻率（BPF）及其諧波，以 PSD 分析識別特徵頻率，評估泵浦運行狀態。手動實作 Welch 法（多段平均），體驗降低頻譜方差的效果。

**葉片通過頻率（BPF）**：

$$
f_{BPF} = \frac{n_{rpm}}{60} \times Z_{blade}
$$

**化工重點**：
- **方法一（Periodogram）**：直接 `rfft()` 計算，頻譜方差大，不平滑
- **方法二（手動 Welch 法）**：迴圈切段 + Hann 視窗 + FFT 後取平均，體驗降噪效果
- 比較兩種方法的頻譜平滑程度，說明頻率解析度與方差之間的取捨
- `next_fast_len()` 零填充精確定位 BPF 峰值
- 識別 $f_{BPF}$、$2f_{BPF}$、$3f_{BPF}$ 諧波，計算各諧波能量分布
- **補充說明**：`scipy.signal.welch()` 示範（一行完成），與手動結果比對驗證
- 繪製：時域訊號、Periodogram vs Welch PSD 比較、諧波能量分布條形圖

---

### 📊 Example 06 — 加熱器程序頻率響應估計與 Bode 圖繪製

**檔案**：[Unit11_Example_06.md](Unit11_Example_06.md) | [Unit11_Example_06.ipynb](Unit11_Example_06.ipynb)

**問題概述**：對一階加純時滯（FOPDT）加熱器程序施加正弦掃頻（Chirp）激發訊號，以輸入/輸出訊號的 FFT 估計程序傳遞函數的幅度與相位，繪製 Bode 圖並識別系統參數 $K$、$\tau$、$\theta$。

**程序模型（FOPDT）**：

$$
G(s) = \frac{K e^{-\theta s}}{\tau s + 1}
$$

**頻率響應函數（FRF）估計**：

$$
\hat{H}(f) = \frac{Y(f)}{U(f)}, \quad |G(j2\pi f)| \text{ (dB)} = 20\log_{10}|\hat{H}(f)|
$$

**化工重點**：
- `scipy.integrate.solve_ivp()` 模擬 FOPDT 程序輸出（與 Unit09 的整合）
- Chirp 激發訊號：以 `numpy` 手動產生（簡介 `scipy.signal.chirp()` 可自動產生）
- 分別對輸入 $u(t)$ 與輸出 $y(t)$ 作 `rfft()`，計算 FRF 估計 $\hat{H}(f)$
- 計算**相干函數** $\gamma^2(f) = |G_{xy}|^2 / (G_{xx} \cdot G_{yy})$：識別雜訊污染頻段
- 繪製估計 Bode 圖（幅度 + 相位），疊加理論值驗證
- 從 Bode 圖讀取 $K$（低頻增益）、$\tau$（-3dB 截止頻率）、$\theta$（相位延遲）
- **與程序控制的連結**：以 FFT 實驗識別法建立實際程序的數學模型

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit11_Fourier_Transform.md](Unit11_Fourier_Transform.md) | 📄 教學講義 | CFT/DFT/FFT 理論、`scipy.fft` 工具、視窗函數、PSD、STFT、自相關 |
| [Unit11_Fourier_Transform.ipynb](Unit11_Fourier_Transform.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit11_Example_01.md](Unit11_Example_01.md) | 📄 案例講義 | 製程感測器頻譜分析（基礎 FFT + 視窗比較） |
| [Unit11_Example_01.ipynb](Unit11_Example_01.ipynb) | 💻 程式演練 | 感測器頻譜分析實作 |
| [Unit11_Example_02.md](Unit11_Example_02.md) | 📄 案例講義 | 批次反應器溫度頻譜（去趨勢 + PSD + 視窗比較） |
| [Unit11_Example_02.ipynb](Unit11_Example_02.ipynb) | 💻 程式演練 | 批次反應器頻譜實作 |
| [Unit11_Example_03.md](Unit11_Example_03.md) | 📄 案例講義 | CSTR 濃度振盪分析（Limit Cycle 偵測，CSTR ODE 整合） |
| [Unit11_Example_03.ipynb](Unit11_Example_03.ipynb) | 💻 程式演練 | CSTR 振盪分析實作 |
| [Unit11_Example_04.md](Unit11_Example_04.md) | 📄 案例講義 | 蒸餾塔液泛預警（手動 STFT + 時頻頻譜圖） |
| [Unit11_Example_04.ipynb](Unit11_Example_04.ipynb) | 💻 程式演練 | 液泛預警實作 |
| [Unit11_Example_05.md](Unit11_Example_05.md) | 📄 案例講義 | 泵浦 BPF 分析（Periodogram vs 手動 Welch PSD） |
| [Unit11_Example_05.ipynb](Unit11_Example_05.ipynb) | 💻 程式演練 | 泵浦頻率分析實作 |
| [Unit11_Example_06.md](Unit11_Example_06.md) | 📄 案例講義 | 加熱器頻率響應估計（Bode 圖，程序識別） |
| [Unit11_Example_06.ipynb](Unit11_Example_06.ipynb) | 💻 程式演練 | 頻率響應估計實作 |
| [Unit11_Homework.ipynb](Unit11_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit10 偏微分方程式之求解](../Unit10/README.md)**：`py-pde`、Method of Lines、熱傳/質傳/動量傳遞的 PDE 模擬

### ➡️ 下一單元
- **[Unit12 程序最適化](../Unit12/README.md)**：`scipy.optimize.minimize()`、有限制條件最適化、全域最適化演算法

---

## 📈 本單元在課程中的定位

```
Unit10 (py-pde + scipy) — PDE 求解
      ↓
   Unit11 ← 你在這裡
 ┌──────────────────────────────────────────────────────────────┐
 │  傅立葉轉換與頻譜分析 (scipy.fft)                              │
 │  基礎工具：rfft / rfftfreq / fftshift / next_fast_len        │
 │  頻譜技術：視窗函數 / PSD（Periodogram + Welch）/ STFT        │
 │  進階分析：自相關函數 / 相干函數 / 頻率響應估計（Bode 圖）     │
 │  前處理：去均值 / 去趨勢 / 零填充 / 峰值識別                  │
 └──────────────────────────────────────────────────────────────┘
      ↓
 Unit12 (scipy.optimize) — 程序最適化
      ↓
 Unit13 (scipy.optimize) — 參數估計
      ↓
 ...（後續各應用單元）
```

**與化工問題的對應**：
```
訊號主頻識別（設備振動、週期干擾）  → FFT 基礎 (Unit11 Example 01)  ← 本單元
製程趨勢去除 + PSD 分析              → 去趨勢 + Welch (Unit11 Example 02)  ← 本單元
反應器振盪偵測（Limit Cycle）        → FFT + 自相關 (Unit11 Example 03)  ← 本單元
製程異常預警（液泛、設備故障）       → STFT 時頻分析 (Unit11 Example 04)  ← 本單元
旋轉設備診斷（泵浦 BPF）             → PSD + Welch (Unit11 Example 05)  ← 本單元
程序數學模型識別（Bode 圖）          → FRF 估計 (Unit11 Example 06)  ← 本單元
```

**Unit11 的重要橫向聯繫**：
- **與 Unit09 的連結**：Example 03（CSTR 振盪）使用 `solve_ivp()` 模擬時間序列後再進行 FFT 分析；Example 06（頻率響應）亦使用 `solve_ivp()` 模擬程序輸出
- **與 Unit15 的連結**：Unit11 介紹 FFT 頻域分析基礎；Unit15（`scipy.signal`）將深入信號處理的濾波器設計與系統分析

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit11_Fourier_Transform.md`（約 60–90 分鐘）建立 FFT 理論框架
   - Step 2：執行 `Unit11_Fourier_Transform.ipynb` 熟悉 `rfft()`、`rfftfreq()`、視窗函數的基本操作
   - Step 3：依序完成 Example 01–06（由簡到難：基礎頻譜 → PSD → 時頻分析 → 頻率響應）

2. **重點關注**：
   - **幅度歸一化**：$2/N$（單邊）是初學者最容易出錯的地方，Example 01 完整示範驗證流程
   - **去趨勢的重要性**：有線性趨勢的訊號不去趨勢直接 FFT，零頻附近會出現假性寬帶洩漏（Example 02）
   - **Example 04（STFT）**：手動實作與 `scipy.signal.stft()` 對比，理解底層後再使用工具更有把握
   - **Example 06（Bode 圖）**：本單元與程序控制理論的橋接，展示 FFT 作為系統識別工具的完整流程

3. **`scipy.fft` vs `scipy.signal` 的定位**：

   | 功能 | 本課程工具 | 說明 |
   |------|---------|------|
   | 基礎 FFT 計算 | `scipy.fft`（主要） | 理解底層原理 |
   | Welch PSD | 手動實作 + `scipy.signal.welch()`（示範） | 了解介面即可 |
   | STFT / Spectrogram | 手動實作 + `scipy.signal.stft()`（示範） | 了解介面即可 |
   | 視窗函數 | 手動建立 + `scipy.signal.get_window()`（簡介） | 了解介面即可 |
   | 濾波器設計 | Unit15 深入介紹 | 本單元不涉及 |

4. **常見錯誤提醒**：
   - `rfft()` 返回的陣列長度為 `n//2 + 1`，不是 `n`，頻率軸需用 `rfftfreq()` 而非 `fftfreq()`
   - 套用視窗後若不補幅度修正因子（ACF），頻譜幅度會偏低，誤判訊號強度
   - Welch 分段後每段需**重疊**（通常 50%）才能有效降低方差，不重疊效果有限

5. **參考外部資源**：
   - [SciPy fft 官方文件](https://docs.scipy.org/doc/scipy/reference/fft.html)
   - [SciPy signal 官方文件](https://docs.scipy.org/doc/scipy/reference/signal.html)（濾波器設計，單元 15 深入學習）
   - [The Scientist and Engineer's Guide to DSP](http://www.dspguide.com/)（免費數位訊號處理教材）

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit11 傅立葉轉換與頻譜分析
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
