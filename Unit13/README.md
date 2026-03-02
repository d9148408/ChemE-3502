# Unit13 參數估計 (Parameter Estimation)

## 📚 單元簡介

在化工程序設計與操作中，**「參數估計（Parameter Estimation）」**是建構實用數學模式的核心技術。無論是反應動力學速率常數、相平衡活性係數交互參數、傳熱係數、或程序轉移函數參數，都必須透過將數學模式與實驗數據進行擬合，才能確認模式中的未知參數值，使模式完備且能真實反映系統行為。

本單元以 **`scipy.optimize`** 的參數估計工具為核心，從理論推導到 Python 實作，完整涵蓋線性最小平方法（`scipy.linalg.lstsq()`）、非線性曲線擬合（`scipy.optimize.curve_fit()`）與有界非線性最小平方法（`scipy.optimize.least_squares()`）。透過七個化工與工程實際案例，從線性模式、可線性化非線性模式、固體觸媒速率式，到活性碳吸附等溫線、程序動態響應轉移函數，再到結合 ODE 求解的醱酵程序動力學參數估計，完整呈現參數估計技術在化工模式建構中的廣泛應用。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **建立參數估計問題的數學框架**：正確定義誤差平方和目標函數 $J = \sum_{i=1}^{n} e_i^2$，理解最小平方法的核心概念
2. **推導並應用線性最小平方法解析解**：以設計矩陣 $\mathbf{X}$ 表示線性模式，推導正規方程組，求得 $\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$
3. **熟練使用 `scipy.linalg.lstsq()`**：求解線性與可線性化非線性模式的參數估計
4. **識別並處理可線性化的非線性模式**：透過取對數、變數代換等方法將非線性模式轉化為線性問題
5. **熟練使用 `scipy.optimize.curve_fit()`**：求解非線性參數估計問題並取得協方差矩陣 `pcov`
6. **熟練使用 `scipy.optimize.least_squares()`**：進行含物理上下限約束的有界非線性參數估計
7. **由協方差矩陣推算置信區間**：從 `pcov` 計算各參數的 95% 置信區間，評估模式可辨識性
8. **整合 ODE 求解器與參數估計**：以 `solve_ivp()` 作為 `curve_fit()` 的內層計算工具，處理動態系統參數估計

---

## 📖 單元內容架構

### 1️⃣ Unit13_Parameter_Estimation — 參數估計理論與工具總覽

**檔案**：
- 講義：[Unit13_Parameter_Estimation.md](Unit13_Parameter_Estimation.md)
- 程式範例：[Unit13_Parameter_Estimation.ipynb](Unit13_Parameter_Estimation.ipynb)

**內容重點**：

#### 參數估計問題分類

| 問題類型 | 模式特性 | Python 求解工具 |
|---------|---------|----------------|
| **線性參數估計** | 模式輸出為參數的線性組合 $\mathbf{Y}^M = \mathbf{X}\boldsymbol{\theta}$ | `scipy.linalg.lstsq()` |
| **可線性化非線性模式** | 透過取對數或變數代換化為線性問題 | 轉換後用 `lstsq()` |
| **非線性參數估計** | 模式輸出與參數呈非線性關係 | `scipy.optimize.curve_fit()` |
| **有界非線性估計** | 非線性模式且參數有物理上下限 | `scipy.optimize.least_squares()` |

**線性最小平方法**（核心公式）：

$$
\boldsymbol{\theta} = \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{Y}
$$

**非線性最小平方問題**：

$$
\min_{\mathbf{p}} J = \sum_{i=1}^{n} \left[y_i - f(x_i, \mathbf{p})\right]^2
$$

#### 三大參數估計工具比較

| 函式 | 適用情境 | 優點 | 限制 |
|------|---------|------|------|
| `scipy.linalg.lstsq(X, Y)` | 線性參數模式 | 解析解，快速穩定，有秩診斷 | 僅限線性模式 |
| `scipy.optimize.curve_fit(f, x, y)` | 非線性模式，需置信區間 | 自動返回協方差矩陣 `pcov` | 無法直接設定嚴格上下限 |
| `scipy.optimize.least_squares(res, p0, bounds=)` | 非線性模式，需上下限約束 | 嚴格支援上下限，多種求解演算法 | 不直接返回協方差矩陣 |

#### 方法選擇決策流程

```
問題類型判斷
│
├── 線性參數模式（模式輸出為參數之線性組合）
│   └── 使用 scipy.linalg.lstsq()
│
├── 可線性化之非線性模式（可透過變數代換化為線性）
│   └── 線性化後使用 scipy.linalg.lstsq()
│       （注意：最小化之為變換空間的誤差，非原始空間）
│
└── 非線性模式（無法線性化）
    │
    ├── 無上下限限制，需置信區間
    │   └── 使用 scipy.optimize.curve_fit()
    │
    ├── 有上下限限制，需置信區間
    │   ├── 先用 scipy.optimize.least_squares(bounds=) 求解
    │   └── 再用 curve_fit(bounds=) 獲取協方差矩陣
    │
    └── 有上下限限制，不需置信區間
        └── 使用 scipy.optimize.least_squares(bounds=)
```

#### 95% 置信區間計算

```python
popt, pcov = curve_fit(model_func, xdata, ydata, p0=p0_init)
perr = np.sqrt(np.diag(pcov))   # 各參數標準差
ci95 = 1.96 * perr              # 95% CI 半寬度（大樣本近似）
```

---

## 🧪 化工案例演練

### 📊 Example 01 — 線性模式參數估計

**檔案**：[Unit13_Example_01.md](Unit13_Example_01.md) | [Unit13_Example_01.ipynb](Unit13_Example_01.ipynb)

**問題概述**：給定 4 組實驗數據 $(x_i, y_i)$，估計線性模式 $y = a + be^{3x} + ce^{-3x}$ 的三個未知參數；同場加映可線性化非線性模式 $y = axe^{-bx}$ 的參數估計（取對數線性化），並比較兩模式的擬合品質。

**數學模型**：線性設計矩陣法 + `scipy.linalg.lstsq()`

$$
\begin{bmatrix} y_1 \\ y_2 \\ y_3 \\ y_4 \end{bmatrix} = \begin{bmatrix} 1 & e^{3x_1} & e^{-3x_1} \\ 1 & e^{3x_2} & e^{-3x_2} \\ 1 & e^{3x_3} & e^{-3x_3} \\ 1 & e^{3x_4} & e^{-3x_4} \end{bmatrix} \begin{bmatrix} a \\ b \\ c \end{bmatrix}
$$

**化工重點**：
- 建構設計矩陣 $\mathbf{A}$，使用 `scipy.linalg.lstsq()` 求解線性最小平方問題
- 可線性化非線性模式：$\ln(y/x) = \ln a - bx$（取對數後線性化）
- 比較兩模式目標函數 $J$（模式一 $J = 0.0178$ vs 模式二 $J = 0.3071$），說明模式選擇方法
- 繪製實驗數據與兩模式預測值的比較圖

---

### 📐 Example 02 — 非線性模式參數估計與置信區間

**檔案**：[Unit13_Example_02.md](Unit13_Example_02.md) | [Unit13_Example_02.ipynb](Unit13_Example_02.ipynb)

**問題概述**：給定 10 組實驗數據，估計非線性模式 $y = \alpha x^2 + \beta \sin(x) + \gamma x^3$ 的三個參數，並由協方差矩陣計算各參數的 95% 置信區間以評估模式可辨識性。

**數學模型**：非線性最小平方法（`curve_fit`）

$$
\min_{\alpha, \beta, \gamma} J = \sum_{i=1}^{n} \left[y_i - \alpha x_i^2 - \beta \sin(x_i) - \gamma x_i^3\right]^2
$$

**化工重點**：
- `scipy.optimize.curve_fit()` 求解非線性參數，獲取協方差矩陣 `pcov`
- 由 `perr = np.sqrt(np.diag(pcov))` 計算標準差，推算 95% 置信區間
- 置信區間寬窄的物理意義：$\gamma$ 置信區間極窄（精確），$\beta$ 置信區間跨越零點（統計上不顯著）
- 繪製模式預測結果與實驗數據的吻合度圖

---

### 🌊 Example 03 — 二氧化硫溶解度模式參數估計

**檔案**：[Unit13_Example_03.md](Unit13_Example_03.md) | [Unit13_Example_03.ipynb](Unit13_Example_03.ipynb)

**問題概述**：在 15°C 下，以 8 組 $\mathrm{SO_2}$ 在水溶液中的溶解度數據，估計溶解度與分壓的關係模式 $x = ap + b\sqrt{p}$ 的兩個參數 $a, b$（改編自 ch7 範例 7-2-1）。

**數學模型**：線性設計矩陣法（`lstsq`）

$$
x = ap + b\sqrt{p} \quad \Rightarrow \quad \mathbf{A} = [p, \sqrt{p}], \quad \boldsymbol{\theta} = [a, b]^T
$$

**化工重點**：
- 數據單位換算：$p_{\mathrm{SO_2}}$ 由 mmHg 轉換為 atm，濃度由 g-SO2/100g-H2O 轉換為莫耳分率 $x$
- 建構設計矩陣 $\mathbf{A} = [p, \sqrt{p}]$，使用 `lstsq()` 求解兩參數
- 繪製分壓 vs 莫耳分率之實驗數據與模式擬合曲線

---

### ⚗️ Example 04 — 固體觸媒 Hougen-Watson 型速率式線性化

**檔案**：[Unit13_Example_04.md](Unit13_Example_04.md) | [Unit13_Example_04.ipynb](Unit13_Example_04.ipynb)

**問題概述**：苯氫化合成環己烷的初期反應速率遵循 Hougen-Watson 速率式，透過模式線性化手法將非線性速率參數 $k, K_H, K_B$ 轉化為線性問題，再由線性解反推原始參數（改編自 ch7 範例 7-2-2）。

**數學模型**：速率式線性化 → 線性最小平方

$$
r_0 = \frac{kK_H^3 K_B P_H^3 P_B}{(1+K_H P_H+K_B P_B)^4} \xrightarrow{\text{取 } 1/4 \text{ 次方}} R = a + bP_H + cP_B
$$

**化工重點**：
- 模式線性化：兩側取倒數後再取 $1/4$ 次方，得線性模式 $R = a + bP_H + cP_B$
- 建構設計矩陣並以 `lstsq()` 求解中間參數 $a, b, c$
- 由 $a, b, c$ 反推原始速率常數：$K_H = b/a$，$K_B = c/a$，$k = a^{-4}K_H^{-3}K_B^{-1}$
- 驗證模式：列印各量測點的 $r_0$ 實驗值、模式預測值與誤差

---

### 🔬 Example 05 — 活性碳吸附等溫模式（有界非線性估計與置信區間）

**檔案**：[Unit13_Example_05.md](Unit13_Example_05.md) | [Unit13_Example_05.ipynb](Unit13_Example_05.ipynb)

**問題概述**：活性碳對溶質的吸附量遵循廣義 Langmuir-Freundlich 等溫模式，給定 16 組實驗數據，在參數的物理約束範圍內估計三個參數，並計算置信區間（改編自 ch7 範例 7-2-3）。

**數學模型**：有界非線性最小平方（`least_squares` + `curve_fit`）

$$
Q = \frac{bC}{1 + aC^\beta}, \quad 0 \leq a \leq 10, \quad 100 \leq b \leq 200, \quad 0 \leq \beta \leq 1
$$

**化工重點**：
- `scipy.optimize.least_squares(bounds=)` 在物理上下限約束下求最優參數
- 再以 `scipy.optimize.curve_fit(bounds=)` 獲取協方差矩陣，計算 95% 置信區間
- 結果：$a = 3.847$，$b = 150.51$，$\beta = 0.790$（均在物理合理範圍內）
- 說明設定物理上下限的重要性（避免優化器收斂至不具物理意義的解）
- 繪製吸附等溫線實驗數據與模式擬合結果比較圖（半對數座標）

---

### 🌡️ Example 06 — 加熱程序動態響應之轉移函數參數估計

**檔案**：[Unit13_Example_06.md](Unit13_Example_06.md) | [Unit13_Example_06.ipynb](Unit13_Example_06.ipynb)

**問題概述**：加熱程序的階梯響應數據遵循二階加純時間延遲（Second-Order plus Time-Delay, SOPDT）模式，使用 `curve_fit()` 估計轉移函數的四個參數（改編自 ch7 範例 7-2-4）。

**數學模型**：非線性最小平方（`curve_fit`）

$$
G(s) = \frac{K_p e^{-t_d s}}{(\tau_1 s+1)(\tau_2 s+1)}
$$

對階梯輸入進行反 Laplace 轉換，得時域模式輸出 $T(t)$，以此作為 `curve_fit` 的模式函數。

**化工重點**：
- 推導 SOPDT 模式對階梯輸入的時域響應解析式
- 使用 `scipy.optimize.curve_fit()` 估計 $K_p$、$\tau_1$、$\tau_2$、$t_d$ 四個參數
- 計算各參數的 95% 置信區間，評估參數可辨識性
- 繪製實驗響應曲線與模式預測值的比較圖，驗證擬合品質

---

### 🧫 Example 07 — 結合 ODE 求解之醱酵程序動力學參數估計

**檔案**：[Unit13_Example_07.md](Unit13_Example_07.md) | [Unit13_Example_07.ipynb](Unit13_Example_07.ipynb)

**問題概述**：盤尼西林醱酵程序動力學由 ODE 系統描述，模式輸出無解析式，必須在每次函數評估時以 `solve_ivp()` 數值積分取得預測值，再以外層 `curve_fit()` 進行參數搜尋（改編自 ch7 範例 7-2-5）。

**數學模型**：ODE 系統參數估計（`solve_ivp()` 嵌套在 `curve_fit()` 中）

$$
\frac{dy_1}{dt} = k_1 y_1\left(1-\frac{y_1}{k_2}\right), \quad \frac{dy_2}{dt} = k_3 y_1 - k_4 y_2
$$

估計速率常數 $k_1, k_2, k_3, k_4$（$y_1$：細胞濃度，$y_2$：盤尼西林濃度）。

**化工重點**：
- **關鍵技術**：模式函數內呼叫 `scipy.integrate.solve_ivp()` 求解 ODE，取時間點插值作為預測值
- `scipy.optimize.curve_fit()` 外層參數搜尋：估計 $k_1 \sim k_4$ 並計算 95% 置信區間
- **與 Unit09 的整合**：ODE 求解器作為參數估計的內層計算工具（跨單元工具整合的重要範例）
- 分別繪製細胞濃度 $y_1$ 與盤尼西林濃度 $y_2$ 的實驗值與模式預測值比較圖

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit13_Parameter_Estimation.md](Unit13_Parameter_Estimation.md) | 📄 教學講義 | 參數估計理論、Python 工具總覽、方法選擇決策流程 |
| [Unit13_Parameter_Estimation.ipynb](Unit13_Parameter_Estimation.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit13_Example_01.md](Unit13_Example_01.md) | 📄 案例講義 | 線性模式參數估計（設計矩陣法 vs 線性化法比較） |
| [Unit13_Example_01.ipynb](Unit13_Example_01.ipynb) | 💻 程式演練 | 線性模式估計實作 |
| [Unit13_Example_02.md](Unit13_Example_02.md) | 📄 案例講義 | 非線性模式估計與 95% 置信區間（`curve_fit`） |
| [Unit13_Example_02.ipynb](Unit13_Example_02.ipynb) | 💻 程式演練 | 非線性估計與置信區間實作 |
| [Unit13_Example_03.md](Unit13_Example_03.md) | 📄 案例講義 | SO₂ 溶解度線性模式參數估計（化工案例一） |
| [Unit13_Example_03.ipynb](Unit13_Example_03.ipynb) | 💻 程式演練 | SO₂ 溶解度模式實作 |
| [Unit13_Example_04.md](Unit13_Example_04.md) | 📄 案例講義 | 固體觸媒 Hougen-Watson 速率式線性化（化工案例二） |
| [Unit13_Example_04.ipynb](Unit13_Example_04.ipynb) | 💻 程式演練 | 速率式線性化參數估計實作 |
| [Unit13_Example_05.md](Unit13_Example_05.md) | 📄 案例講義 | 活性碳吸附等溫線有界非線性估計與置信區間（化工案例三） |
| [Unit13_Example_05.ipynb](Unit13_Example_05.ipynb) | 💻 程式演練 | 有界非線性估計實作 |
| [Unit13_Example_06.md](Unit13_Example_06.md) | 📄 案例講義 | 加熱程序轉移函數參數估計（化工案例四） |
| [Unit13_Example_06.ipynb](Unit13_Example_06.ipynb) | 💻 程式演練 | 程序動態響應估計實作 |
| [Unit13_Example_07.md](Unit13_Example_07.md) | 📄 案例講義 | 醱酵程序 ODE+curve_fit 動力學參數估計（化工案例五） |
| [Unit13_Example_07.ipynb](Unit13_Example_07.ipynb) | 💻 程式演練 | ODE 動力學參數估計實作 |
| [Unit13_Homework.ipynb](Unit13_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit12 程序最適化](../Unit12/README.md)**：`scipy.optimize`、`minimize()`、`linprog()`、全域最適化（差分進化、雙退火）

### ➡️ 下一單元
- **[Unit14 統計分析](../Unit14/README.md)**：`scipy.stats`、機率分布、假設檢定、ANOVA、相關分析

---

## 📈 本單元在課程中的定位

```
Unit12 (scipy.optimize) — 程序最適化
      ↓
   Unit13 ← 你在這裡
 ┌──────────────────────────────────────────────────────────────┐
 │  參數估計 (scipy.optimize + scipy.linalg)                     │
 │  線性估計：scipy.linalg.lstsq()                               │
 │  可線性化非線性：取對數/變數代換 + lstsq()                      │
 │  非線性估計：scipy.optimize.curve_fit()（含置信區間）           │
 │  有界非線性估計：scipy.optimize.least_squares(bounds=)         │
 │  進階應用：solve_ivp() + curve_fit() 嵌套（ODE 系統參數估計）   │
 └──────────────────────────────────────────────────────────────┘
      ↓
 Unit14 (scipy.stats) — 統計分析
      ↓
 Unit15 (scipy.signal) — 信號模擬與處理
      ↓
 ...（後續各應用單元）
```

**與化工問題的對應**：
```
線性物性關係（SO₂ 溶解度、多項式擬合）         → 線性 lstsq() (Unit13 Ex01, Ex03)
觸媒速率式（Langmuir-Hinshelwood 型）           → 線性化 + lstsq() (Unit13 Ex04)
非線性吸附等溫線（Freundlich, Langmuir 修正型）  → 有界 least_squares() (Unit13 Ex05)
程序動態模式識別（FOPDT, SOPDT 轉移函數）       → 非線性 curve_fit() (Unit13 Ex06)
生化動力學（Monod, 盤尼西林醱酵 ODE 系統）      → ODE + curve_fit() 嵌套 (Unit13 Ex07)
```

**Unit13 的重要橫向聯繫**：
- **與 Unit06 的連結**：Unit13 線性最小平方法 `lstsq()` 是 Unit06 線性方程組求解的延伸，從精確解擴展為過確定系統的最佳近似解
- **與 Unit09 的連結**：Example 07（醱酵動力學）直接整合 `solve_ivp()` 作為 `curve_fit()` 的內層計算工具，兩個單元的工具在此匯聚
- **與 Unit12 的連結**：Unit13 的參數估計本質上是最小平方最適化的特例，`curve_fit()` 和 `least_squares()` 是 `scipy.optimize` 的專門化版本
- **與 Unit14 的連結**：Unit14 的 `scipy.stats.linregress()` 著重統計推論（t 檢定、p 值）；Unit13 的 `lstsq()` 著重矩陣求解，兩者在線性回歸上互補

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit13_Parameter_Estimation.md`（約 60 分鐘）建立線性/非線性最小平方法的理論框架
   - Step 2：執行 `Unit13_Parameter_Estimation.ipynb` 熟悉三大工具的基本介面
   - Step 3：依序完成 Example 01–07（由簡到難：線性 → 線性化 → 非線性 → 有界非線性 → ODE 嵌套）

2. **重點關注**：
   - **Example 01（模式比較）**：理解 $J$ 值（誤差平方和）作為模式選擇準則的意義
   - **Example 02（置信區間）**：`pcov` 對角線元素開方 = 標準差，乘 1.96 得 95% CI 半寬；置信區間跨越零點表示參數統計上不顯著
   - **Example 05（有界估計）**：物理上下限設定的重要性，以及 `least_squares()` → `curve_fit()` 兩步驟取得協方差矩陣的工程流程
   - **Example 07（ODE 嵌套）**：模式函數內呼叫 `solve_ivp()` 是處理動態系統參數估計的關鍵技術，也是本課程最進階的應用案例

3. **工具選擇速查**：

   | 情境 | 推薦工具 |
   |------|---------| 
   | 模式輸出為參數線性組合 | `scipy.linalg.lstsq(X, Y)` |
   | 可取對數/變數代換之非線性模式 | 轉換後用 `lstsq()` |
   | 一般非線性模式，需置信區間 | `scipy.optimize.curve_fit(f, x, y, p0=)` |
   | 非線性模式，有物理上下限 | `scipy.optimize.least_squares(res, p0, bounds=)` |
   | ODE 系統之動力學參數 | `solve_ivp()` 嵌套在 `curve_fit()` 中 |

4. **常見錯誤提醒**：
   - `curve_fit()` 預設初始猜測值為全 1，非線性問題對初始值敏感，務必透過物理分析或圖形法給定良好初始猜測值
   - `least_squares()` 返回的 `result.cost = J/2`（而非 $J$），計算目標函數值時須乘以 2
   - 由 `pcov` 計算置信區間基於大樣本正態分布假設，樣本數不足或模式高度非線性時不夠準確
   - 線性化後最小平方法最小化的是**變換空間**的誤差，而非原始空間，可能使原始空間擬合品質較差

5. **參考外部資源**：
   - [scipy.optimize.curve_fit 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
   - [scipy.optimize.least_squares 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
   - [scipy.linalg.lstsq 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit13 參數估計
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
