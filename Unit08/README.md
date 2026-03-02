# Unit08 插值、微分與積分之運算 (Interpolation, Differentiation & Integration)

## 📚 單元簡介

化工工程中的許多問題涉及**離散實驗數據的處理**：黏度、擴散係數等物性數據隨溫度或濃度的變化往往只有少數量測點，需要透過**插值法（Interpolation）**估計中間值；由批次實驗的濃度時間數據推算反應速率，需要**數值微分（Numerical Differentiation）**；而填充塔高度計算、平均滯留時間分析、反應熱計算等，則需要**數值積分（Numerical Integration）**。

本單元以 **`scipy.interpolate`** 與 **`scipy.integrate`** 為核心工具，系統性介紹從一維到二維的插值方法比較、差分近似法的理論與誤差分析，以及多種數值積分方案的選用策略。透過六個化工實際案例，涵蓋物性插值、反應速率推斷、流體化床分析、停留時間分布（RTD）分析與吸收塔塔高計算，完整呈現這三項數值工具在化工數據處理中的應用。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握一維與二維插值方法**：比較 nearest、linear、cubic 插值與三次樣條（CubicSpline）的精確度與平滑度，選用適合場景的插值方案
2. **正確使用數值微分工具**：理解前向、後向、中間差分的原理與截斷誤差，使用 `numpy.gradient()` 處理非等間距實驗數據
3. **熟練進行數值積分**：依據問題性質（數值數據 vs 解析函數、一維 vs 多維）選用 `trapezoid()`、`simpson()`、`quad()`、`dblquad()` 等工具
4. **建立化工數據處理流程**：從物性數據插值、批次反應速率分析到 RTD 分析，掌握完整的「實驗數據 → 數值運算 → 化工解釋」流程
5. **驗證數值結果的正確性**：殘差分析、與留存測試點比較（插值驗證）、質量守恆/能量守恆檢查（積分驗證）

---

## 📖 單元內容架構

### 1️⃣ Unit08_Interpolation_Differentiation_Integration — 理論與工具總覽

**檔案**：
- 講義：[Unit08_Interpolation_Differentiation_Integration.md](Unit08_Interpolation_Differentiation_Integration.md)
- 程式範例：[Unit08_Interpolation_Differentiation_Integration.ipynb](Unit08_Interpolation_Differentiation_Integration.ipynb)

**內容重點**：

#### 插值法基礎（Interpolation）

| 方法 | SciPy 函式 | 特性 |
|------|-----------|------|
| **最近鄰插值** | `interp1d(kind='nearest')` | 階梯狀，不連續 |
| **線性插值** | `interp1d(kind='linear')` | 連續但不可微，折點明顯 |
| **三次多項式插值** | `interp1d(kind='cubic')` | 平滑，可能有龍格（Runge）現象 |
| **三次樣條插值** | `CubicSpline()` | 最平滑，二階連續可微，最推薦 |
| **二維規則網格插值** | `RegularGridInterpolator()` | 雙變數規則格點數據 |
| **二維散點插值** | `griddata()` | 任意散佈數據點 |
| **二維樣條插值** | `RectBivariateSpline()` | 規則格點平滑曲面擬合 |

- **外插問題**：超出數據範圍的估計風險，`fill_value` 與 `bounds_error` 參數設定
- **逆向插值**：給定目標函數值，反推對應自變數

#### 數值微分（Numerical Differentiation）

| 方法 | 公式 | 誤差階 | 適用場景 |
|------|------|--------|---------|
| 前向差分 | $y'_i = (y_{i+1} - y_i)/\Delta x$ | $O(\Delta x)$ | 邊界起點 |
| 後向差分 | $y'_i = (y_i - y_{i-1})/\Delta x$ | $O(\Delta x)$ | 邊界終點 |
| 中間差分 | $y'_i = (y_{i+1} - y_{i-1})/(2\Delta x)$ | $O(\Delta x^2)$ | 內部點，最準確 |

- `numpy.diff()`：相鄰差分值計算
- `numpy.gradient()`：自動處理邊界點的中間差分，**非等間距數據適用**
- `scipy.misc.derivative()`：解析函數在指定點的導數

#### 數值積分（Numerical Integration）

| 函式 | 適用場景 |
|------|---------|
| `scipy.integrate.trapezoid(y, x)` | 數值數據（非等間距），梯形法 |
| `scipy.integrate.simpson(y, x)` | 數值數據（等間距），Simpson 法（更精確） |
| `scipy.integrate.quad(f, a, b)` | 解析函數定積分，廣義積分（無窮限） |
| `scipy.integrate.dblquad(f, ...)` | 二維重積分 |
| `scipy.integrate.tplquad(f, ...)` | 三維重積分 |
| `scipy.integrate.nquad(f, ...)` | N 維積分（動態積分限） |

---

## 🧪 化工案例演練

### 🌡️ Example 01 — 化工物性數據之一維插值

**檔案**：[Unit08_Example_01.md](Unit08_Example_01.md) | [Unit08_Example_01.ipynb](Unit08_Example_01.ipynb)

**問題概述**：已知液體黏度在若干溫度點的實驗量測值（離散數據），使用不同插值方法估計任意溫度下的黏度值，並進行逆向插值——給定目標黏度值，反推對應溫度。

**數學模型**：一維插值問題

$$
\mu = f(T), \quad T \in [T_{min}, T_{max}]
$$

**化工重點**：
- 四種插值方法（nearest、linear、cubic、CubicSpline）的精確度與平滑度比較
- 使用留存測試點（hold-out test points）驗證插值精確度：計算 RMSE 與最大誤差
- **逆向插值**：將插值函數反轉，由目標黏度值反推溫度（物性設計的實際需求）
- 繪製不同插值方法的比較圖，直觀展示各方法的優缺點

---

### 🔬 Example 02 — 擴散係數之二維插值

**檔案**：[Unit08_Example_02.md](Unit08_Example_02.md) | [Unit08_Example_02.ipynb](Unit08_Example_02.ipynb)

**問題概述**：NO₂ 在 MDEA（甲基二乙醇胺）溶液中的擴散係數同時受**溫度**與**濃度**影響，已有規則網格量測數據，建立二維插值模型以估計任意溫度與濃度條件下的擴散係數。

**數學模型**：二維插值問題

$$
D_{NO_2} = f(T, C_{MDEA}), \quad \text{規則格點數據} (T_i, C_j) \to D_{ij}
$$

**化工重點**：
- `RegularGridInterpolator()` 與 `RectBivariateSpline()` 的比較（精度 vs 平滑度）
- 二維等高線圖（Contour Plot）與三維曲面圖（Surface Plot）的化工意義解讀
- 規則網格 vs 散點數據的插值方法選擇策略
- 估計設計條件（特定操作溫度與溶劑濃度）下的擴散係數用於傳質計算

---

### ⚗️ Example 03 — 批次反應器之反應速率推斷

**檔案**：[Unit08_Example_03.md](Unit08_Example_03.md) | [Unit08_Example_03.ipynb](Unit08_Example_03.ipynb)

**問題概述**：批次反應 A + B → R，等莫耳初濃度下的二次反應動力學實驗。由量測的產物濃度 $C_R(t)$ 時間序列數據，使用數值微分推斷各時刻的反應速率，並利用最小平方法估計反應階數 $N$ 與速率常數 $k$。

**數學模型**：由數值微分推斷反應速率（$-r_A = dC_R/dt$）

$$
\ln(-r_A) = \ln(k) + N \ln(C_A) \quad \Rightarrow \quad \text{線性回歸推斷 } N \text{ 與 } k
$$

**化工重點**：
- 使用 `numpy.gradient()` 計算非等間距時間序列的數值微分（注意邊界點處理）
- 對數線性化（log-linearization）與 `numpy.polyfit()` 或 `scipy.stats.linregress()` 估計動力學參數
- 取整數反應階數後重新精算速率常數 $k$
- 繪製實驗速率與動力學模式預測值的比較圖（驗證模型適合度）

---

### 🌪️ Example 04 — 氣固流體化床壓力梯度與固體粒子體積分率

**檔案**：[Unit08_Example_04.md](Unit08_Example_04.md) | [Unit08_Example_04.ipynb](Unit08_Example_04.ipynb)

**問題概述**：已知氣固流體化床沿軸向位置的壓力量測值，使用數值微分計算各位置的壓力梯度 $dP/dz$，再由動量平衡方程式推算固體粒子體積分率 $\varepsilon_s$ 的軸向分布。

**數學模型**：由壓力梯度計算固體體積分率（動量平衡）

$$
\varepsilon_s(z) = -\frac{1}{\rho_s g} \frac{dP}{dz}, \quad \frac{dP}{dz} \approx \frac{\Delta P}{\Delta z} \text{（數值微分）}
$$

**化工重點**：
- 前向差分、後向差分、中間差分三種方法的結果比較（邊界點處理差異）
- `numpy.gradient()` 的自動邊界處理機制
- 計算固體粒子**平均體積分率**：$\bar{\varepsilon}_s = \frac{1}{H}\int_0^H \varepsilon_s(z) dz$（積分再次應用）
- 繪製軸向壓力分布與固體體積分率分布圖，解讀流體化床的物理行為

---

### 🧪 Example 05 — 追蹤劑響應之流動特性分析（RTD Analysis）

**檔案**：[Unit08_Example_05.md](Unit08_Example_05.md) | [Unit08_Example_05.ipynb](Unit08_Example_05.ipynb)

**問題概述**：脈衝追蹤劑（Pulse Tracer）實驗，量測反應器出口追蹤劑濃度 $C(t)$ 的時間響應（非等間距數據），計算停留時間分布（RTD）函數 $E(t)$，並分析平均停留時間 $T_m$ 與方差 $\sigma^2$ 以判斷流動模式。

**數學模型**：RTD 分析的積分計算（非等間距數值積分）

$$
E(t) = \frac{C(t)}{\int_0^\infty C(t)\,dt}, \quad T_m = \int_0^\infty t \cdot E(t)\,dt, \quad \sigma^2 = \int_0^\infty (t-T_m)^2 \cdot E(t)\,dt
$$

**化工重點**：
- 使用 `scipy.integrate.trapezoid(y, t)` 處理**非等間距**實驗數據的積分
- 依序計算歸一化 $E(t)$、平均停留時間 $T_m$、方差 $\sigma^2$
- 流動模式判斷：$\sigma^2/T_m^2 \to 0$（平推流 PFR）、$\sigma^2/T_m^2 \to 1$（全混流 CSTR）
- 繪製 $E(t)$ 分布曲線與累積分布 $F(t)$ 曲線，分析反應器流動特性

---

### 🗼 Example 06 — 填充吸收塔傳遞單位數與塔高計算

**檔案**：[Unit08_Example_06.md](Unit08_Example_06.md) | [Unit08_Example_06.ipynb](Unit08_Example_06.ipynb)

**問題概述**：SO₂ 氣體的水洗吸收作業，計算整體傳遞單位數 $N_{OG}$（含插值的函數積分），結合傳遞單位高度 $H_{OG}$ 求解所需填充塔高度 $H$，並分析液氣比對塔高的影響。

**數學模型**：傳遞單位數積分（插值 + 數值積分的組合應用）

$$
N_{OG} = \int_{y_2}^{y_1} \frac{dy}{y - y^*}, \quad H = N_{OG} \cdot H_{OG}
$$

其中 $y^*(x)$ 由平衡曲線插值求得，$x$ 由操作線方程式計算。

**化工重點**：
- 先對溶解度（平衡）數據 $(x, y^*)$ 進行多項式擬合（`numpy.polyfit()`），建立解析平衡關係式
- 操作線方程式計算各氣相濃度對應的液相濃度 $x$，再由擬合函數求 $y^*$
- 使用 `scipy.integrate.quad()` 計算傳遞單位數積分（含奇異點附近的處理）
- 以液氣比（$L/G$）為參數進行敏感度分析，繪製 $N_{OG}$、$H$ vs $L/G$ 曲線
- **插值 + 積分組合應用**：本案例展示兩工具的協同使用，是 Unit08 的整合性案例

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit08_Interpolation_Differentiation_Integration.md](Unit08_Interpolation_Differentiation_Integration.md) | 📄 教學講義 | 插值、數值微分、數值積分理論與 SciPy 工具總覽 |
| [Unit08_Interpolation_Differentiation_Integration.ipynb](Unit08_Interpolation_Differentiation_Integration.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit08_Example_01.md](Unit08_Example_01.md) | 📄 案例講義 | 黏度物性數據一維插值（四法比較 + 逆向插值） |
| [Unit08_Example_01.ipynb](Unit08_Example_01.ipynb) | 💻 程式演練 | 黏度插值實作 |
| [Unit08_Example_02.md](Unit08_Example_02.md) | 📄 案例講義 | 擴散係數二維插值（溫度 × 濃度雙變數） |
| [Unit08_Example_02.ipynb](Unit08_Example_02.ipynb) | 💻 程式演練 | 二維插值實作 |
| [Unit08_Example_03.md](Unit08_Example_03.md) | 📄 案例講義 | 批次反應速率推斷（數值微分 + 動力學參數估計） |
| [Unit08_Example_03.ipynb](Unit08_Example_03.ipynb) | 💻 程式演練 | 反應速率推斷實作 |
| [Unit08_Example_04.md](Unit08_Example_04.md) | 📄 案例講義 | 流體化床壓力梯度與固體體積分率（數值微分應用） |
| [Unit08_Example_04.ipynb](Unit08_Example_04.ipynb) | 💻 程式演練 | 流體化床分析實作 |
| [Unit08_Example_05.md](Unit08_Example_05.md) | 📄 案例講義 | RTD 停留時間分布分析（非等間距數值積分） |
| [Unit08_Example_05.ipynb](Unit08_Example_05.ipynb) | 💻 程式演練 | RTD 分析實作 |
| [Unit08_Example_06.md](Unit08_Example_06.md) | 📄 案例講義 | 填充吸收塔塔高計算（插值 + 積分組合應用） |
| [Unit08_Example_06.ipynb](Unit08_Example_06.ipynb) | 💻 程式演練 | 吸收塔計算實作 |
| [Unit08_Homework.ipynb](Unit08_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit07 非線性方程式之求解](../Unit07/README.md)**：`scipy.optimize`、root_scalar、fsolve、CSTR 多重穩態

### ➡️ 下一單元
- **[Unit09 常微分方程式之求解](../Unit09/README.md)**：`scipy.integrate.solve_ivp`/`solve_bvp`、IVP/BVP、ODE 動態模擬

---

## 📈 本單元在課程中的定位

```
Unit07 (scipy.optimize) — 非線性方程式
      ↓
   Unit08 ← 你在這裡
 ┌──────────────────────────────────────────────────────────┐
 │  插值、微分與積分 (scipy.interpolate / scipy.integrate)   │
 │  插值工具：interp1d / CubicSpline / RegularGridInterpolator│
 │  微分工具：numpy.gradient / numpy.diff                   │
 │  積分工具：trapezoid / simpson / quad / dblquad           │
 └──────────────────────────────────────────────────────────┘
      ↓
 Unit09 (scipy.integrate.solve_ivp / solve_bvp) — ODE 求解
      ↓
 Unit10 (py-pde / scipy) — PDE 求解
      ↓
 ...（後續各數值計算單元）
```

**與化工問題的對應**：
```
物性數據估算（黏度、擴散係數）  → 插值 (Unit08)  ← 本單元
批次實驗數據分析（反應速率）    → 數值微分 (Unit08)  ← 本單元
反應熱 / RTD / 塔高計算        → 數值積分 (Unit08)  ← 本單元
反應器動態行為                 → ODE (Unit09)
熱傳 / 質傳場                  → PDE (Unit10)
```

**Unit08 的橫向聯繫**：
- **與 Unit07 的連結**：Unit08_Example_06 的吸收塔計算需要平衡曲線，可與 Unit07 的泡露點計算結合
- **與 Unit09 的連結**：Unit08_Example_03 的動力學參數估計（速率常數 $k$、反應階數 $N$）將在 Unit09 中用於建立 ODE 動態模型

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit08_Interpolation_Differentiation_Integration.md`（約 45–60 分鐘）建立三大工具的概念地圖
   - Step 2：執行 `Unit08_Interpolation_Differentiation_Integration.ipynb` 熟悉各工具的基本用法
   - Step 3：依序完成 Example 01–06（插值 → 微分 → 積分 → 組合應用）

2. **重點關注**：
   - **插值方法選擇**：化工物性數據通常使用 `CubicSpline()`，不建議使用 nearest 或 linear（物性數據應平滑）
   - **數值微分的步長選取**：步長過大截斷誤差大，過小會放大捨入誤差（float64 精度限制）
   - **積分工具選擇**：有實驗數據用 `trapezoid()`/`simpson()`；有解析函數用 `quad()`；注意是否需要廣義積分（無窮限）
   - **Example 06**（吸收塔）：本單元最複雜的整合案例，展示插值與積分的協同應用，值得重點學習

3. **常見錯誤提醒**：
   - `interp1d` 預設不允許外插（`bounds_error=True`），需設 `fill_value='extrapolate'` 才可外插（但需謹慎）
   - `numpy.gradient()` 返回的是梯度（已除以步長），**不是** `numpy.diff()` 的差分值（未除步長）
   - `scipy.integrate.quad()` 返回 `(積分值, 估計誤差)`，使用時需解包：`result, error = quad(...)`

4. **工具選擇速查**：

   | 情境 | 建議工具 |
   |------|---------|
   | 離散等間距數據插值（平滑要求高） | `CubicSpline()` |
   | 離散非等間距或散點數據 | `griddata()` |
   | 雙變數規則格點插值 | `RegularGridInterpolator()` |
   | 均勻間距數值微分 | `numpy.gradient()` |
   | 非等間距實驗數據積分 | `trapezoid(y, x)` |
   | 解析函數定積分（含廣義積分） | `quad(f, a, b)` |

5. **參考外部資源**：
   - [SciPy interpolate 官方文件](https://docs.scipy.org/doc/scipy/reference/interpolate.html)
   - [SciPy integrate 官方文件](https://docs.scipy.org/doc/scipy/reference/integrate.html)
   - [NumPy gradient 說明](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit08 插值、微分與積分之運算
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
