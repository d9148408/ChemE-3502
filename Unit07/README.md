# Unit07 非線性方程式之求解 (Solving Nonlinear Equations)

## 📚 單元簡介

化工工程中大多數真實問題都是**非線性**的：氣體狀態方程式（Van der Waals、PR EOS）、相平衡泡露點計算、化學反應平衡、CSTR 多重穩態、乾燥與蒸餾設計，無一不涉及非線性代數方程式的求解。相較於線性方程組有成熟的直接法，非線性問題的求解需要考量**初始猜測值的選取**、**多重解的偵測**，以及**收斂性與穩定性**等挑戰。

本單元以 **`scipy.optimize`** 模組為核心工具，系統性地介紹單變數與多變數非線性方程式的數值方法理論與 SciPy 實作，並透過六個豐富的化工實際案例，引導學生掌握從「識別非線性結構」到「驗證物理意義」的完整求解流程。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握非線性方程式求解的理論基礎**：了解 Bisection、Fixed-Point、Newton-Raphson、Secant、Brent 等數值方法的原理與收斂特性
2. **熟練使用 SciPy 求解工具**：`scipy.optimize.root_scalar()`、`scipy.optimize.fsolve()`、`scipy.optimize.root()`、`scipy.optimize.least_squares()` 的選用與參數設定
3. **解決多重解問題**：使用圖形法、多起始點搜尋、延續法偵測並確認所有物理意義解
4. **建立化工非線性問題的數學模型**：將狀態方程式、相平衡、反應平衡、反應器設計等化工問題轉化為方程式組
5. **驗證求解結果的正確性**：代回殘差計算、物理意義驗證（莫耳分率範圍、溫度合理性、穩態穩定性分析）

---

## 📖 單元內容架構

### 1️⃣ Unit07_Nonlinear_Equations — 非線性方程式理論與求解工具

**檔案**：
- 講義：[Unit07_Nonlinear_Equations.md](Unit07_Nonlinear_Equations.md)
- 程式範例：[Unit07_Nonlinear_Equations.ipynb](Unit07_Nonlinear_Equations.ipynb)

**內容重點**：

#### 非線性方程式系統基礎
- **單變數非線性方程式**：$f(x) = 0$，解的幾何意義（函數與 x 軸的交點）
- **多變數聯立非線性方程式**：$\mathbf{F}(\mathbf{x}) = \mathbf{0}$，解的存在性與多重解問題
- **收斂性與穩定性分析**：不同初始猜測值對解的影響

#### 數值方法理論基礎

| 方法 | 類型 | 優點 | 缺點 |
|------|------|------|------|
| **Bisection 法**（二分法） | 括號法 | 穩定保證收斂 | 收斂慢（線性） |
| **Fixed-Point Iteration** | 迭代法 | 概念簡單 | 收斂條件需驗證 |
| **Newton-Raphson 法** | 開放法 | 二次收斂（最快） | 需計算導數，可能發散 |
| **Secant 法**（割線法） | 開放法 | 超線性收斂，無需導數 | 需兩個初始點 |
| **Brent's Method** | 混合法 | 兼具穩定性與效率 | 需括號內有根 |

#### SciPy 求解工具總覽

| 函式 | 用途 | 適用場景 |
|------|------|---------|
| `scipy.optimize.root_scalar(..., method='brentq')` | 單變數，需區間 | 已知根的位置範圍 |
| `scipy.optimize.root_scalar(..., method='newton')` | 單變數，Newton/Secant | 可計算導數或有初始點 |
| `scipy.optimize.root_scalar(..., method='bisect')` | 單變數，二分法 | 需要穩定保證收斂 |
| `scipy.optimize.fsolve()` | 多變數，Powell's hybrid | 最常用的多變數求解器 |
| `scipy.optimize.root(..., method='hybr')` | 多變數，統一介面 | Levenberg-Marquardt |
| `scipy.optimize.root(..., method='broyden1')` | 多變數，擬牛頓法 | 大型系統（Jacobian 昂貴） |
| `scipy.optimize.least_squares()` | 非線性最小平方 | 過確定非線性系統 |

#### 起始猜測值策略
- **物理意義分析法**：使用簡化模型（如理想氣體、簡化平衡）估算
- **圖形法**：繪製 $f(x)$ 或 $f(x) - g(x)$，視覺化根的位置
- **多起始點搜尋法**：系統化掃描解空間，找出所有解
- **延續法（Continuation Method）**：逐步改變參數，追蹤解的分支

#### 多重解問題與穩定性分析
- **多重解的偵測**：相圖分析、特徵值分析（Jacobian 矩陣特徵值之正負）
- **穩定解 vs 不穩定解**：Jacobian 所有特徵值負實部 → 穩定；含正實部 → 不穩定
- **分岔圖（Bifurcation Diagram）**：參數掃描視覺化多重穩態的存在範圍

---

## 🧪 化工案例演練

### ⚗️ Example 01 — Van der Waals 狀態方程式

**檔案**：[Unit07_Example_01.md](Unit07_Example_01.md) | [Unit07_Example_01.ipynb](Unit07_Example_01.ipynb)

**問題概述**：給定溫度與壓力，求解 Van der Waals 狀態方程式中氣體的摩爾體積 $V_m$，並探討不同起始猜測值對解的影響（氣相根 vs 液相根）。

**數學模型**：單變數非線性方程式

$$
\left(P + \frac{a}{V_m^2}\right)(V_m - b) = RT \quad \Rightarrow \quad f(V_m) = 0
$$

**化工重點**：
- 等溫線分析：在臨界溫度以下出現三個實根（兩個物理根：氣相與液相）
- 使用 `scipy.optimize.root_scalar()` 多種方法（`brentq`、`newton`、`bisect`）比較收斂速度
- 繪製 $P$-$V$ 等溫線，視覺化多重根的位置
- 驗證理想氣體近似在高壓條件下的誤差

---

### 🌡️ Example 02 — 理想溶液之泡點計算

**檔案**：[Unit07_Example_02.md](Unit07_Example_02.md) | [Unit07_Example_02.ipynb](Unit07_Example_02.ipynb)

**問題概述**：多成分理想溶液（苯、甲苯、二甲苯三成分）在給定壓力下，使用 Raoult 定律與 Antoine 方程式計算泡點溫度 $T_{bp}$。

**數學模型**：單變數非線性方程式（Antoine 方程式應用）

$$
\sum_{i=1}^{n} x_i P_i^{sat}(T) = P_{total} \quad \Rightarrow \quad f(T) = \sum_{i} x_i P_i^{sat}(T) - P_{total} = 0
$$

**化工重點**：
- Antoine 方程式：$\log_{10} P_i^{sat} = A_i - \dfrac{B_i}{C_i + T}$ 的參數使用
- 泡點計算作為蒸餾設計（Unit07_Example_06）的基礎子程序
- 不同組成對泡點的影響分析
- 驗證計算結果：$\sum y_i = 1$（氣相莫耳分率總和）

---

### ⚖️ Example 03 — 化學反應平衡系統

**檔案**：[Unit07_Example_03.md](Unit07_Example_03.md) | [Unit07_Example_03.ipynb](Unit07_Example_03.ipynb)

**問題概述**：高溫氣相系統中同時發生兩個化學反應（水-氣轉換反應與水蒸氣重組反應），聯立平衡常數方程式與物料守恆，求各成分平衡莫耳分率。

**數學模型**：多變數聯立非線性方程式組

$$
K_1 = \frac{y_{CO_2} \cdot y_{H_2}}{y_{CO} \cdot y_{H_2O}}, \quad K_2 = \frac{y_{CO} \cdot y_{H_2}^3}{y_{CH_4} \cdot y_{H_2O}}, \quad \sum y_i = 1
$$

**化工重點**：
- 多個平衡常數與物料守恆的聯立求解
- 使用 `scipy.optimize.fsolve()` 求解，初始猜測值敏感性分析
- 莫耳分率的物理約束驗證（$0 \leq y_i \leq 1$，$\sum y_i = 1$）
- 溫度對化學平衡的影響（Van't Hoff 方程式）

---

### 🔄 Example 04 — CSTR 反應器多重穩態分析

**檔案**：[Unit07_Example_04.md](Unit07_Example_04.md) | [Unit07_Example_04.ipynb](Unit07_Example_04.ipynb)

**問題概述**：放熱一次不可逆反應 A → B 在連續攪拌槽反應器（CSTR）中，聯立穩態物料平衡與能量平衡，分析多重穩態的存在性與穩定性。

**數學模型**：雙變數聯立非線性方程式（物料 + 能量平衡）

$$
\begin{cases} F(C_A, T) = \tau k(T) C_A - (C_{A0} - C_A) = 0 \quad & \text{（物料平衡）} \\ G(C_A, T) = (-\Delta H_r) k(T) C_A - \dfrac{\rho C_p}{\tau}(T - T_0) - \dfrac{UA}{\tau V}(T - T_c) = 0 \quad & \text{（能量平衡）}\end{cases}
$$

**化工重點**：
- **多重穩態的成因**：放熱反應的「S 形曲線」（Sigmoidal）現象
- 圖形法（$C_A$-$T$ 相圖）輔助識別所有穩態點（通常 1 或 3 個）
- 多起始點系統化搜尋策略（掃描溫度網格）
- **穩定性分析**：計算 Jacobian 矩陣特徵值，判斷穩定/不穩定穩態
- 繪製分岔圖（對應給熱係數 $UA/V$ 之穩態轉化率）

---

### 🌬️ Example 05 — 熱傳導乾燥過程

**檔案**：[Unit07_Example_05.md](Unit07_Example_05.md) | [Unit07_Example_05.ipynb](Unit07_Example_05.ipynb)

**問題概述**：熱板與熱風乾燥系統中，聯立熱量平衡與質量平衡（Antoine 方程式計算飽和蒸氣壓），求解乾燥面溫度 $T_s$ 與恆率乾燥速率 $N_c$。

**數學模型**：單變數非線性方程式（耦合熱質傳方程式）

$$
h_c (T_{air} - T_s) = \lambda N_c = \lambda k_y (P_s^{sat}(T_s) - y_w P_{total}) \quad \Rightarrow \quad f(T_s) = 0
$$

**化工重點**：
- Antoine 方程式描述飽和蒸氣壓的非線性溫度依賴性
- 熱傳係數 $h_c$ 與質傳係數 $k_y$ 的物理意義
- 溫度範圍的分段函數處理（確保在合理範圍內求根）
- 乾燥速率 $N_c$ 的物理意義與恆率期的條件分析

---

### 🏭 Example 06 — 二元蒸餾塔設計計算

**檔案**：[Unit07_Example_06.md](Unit07_Example_06.md) | [Unit07_Example_06.ipynb](Unit07_Example_06.ipynb)

**問題概述**：使用 McCabe-Thiele 數值方法，對苯-甲苯二元混合物蒸餾塔進行逐級計算，求解理論板數、進料板位置，以及各板氣液相組成與溫度分布。

**數學模型**：複雜多變數非線性方程式組（逐級計算）

- **每一板**：泡點計算（求 $T_n$）+ 平衡關係（求 $y_n$）+ 操作線（求 $x_{n+1}$）
- **整合計算**：自塔頂逐板計算至塔底，判斷塔底規格是否滿足

**化工重點**：
- McCabe-Thiele 方法的數值實現（替代圖解法）
- 逐板計算流程整合多個非線性子問題（泡點、平衡計算）
- Murphree 效率修正：$y_n^* = y_{n,ideal} \times E_{MV}$
- 視覺化：$x$-$y$ 圖（操作線與平衡線）與各板溫度分布圖
- 程式架構設計：函數模組化（`bubble_point()`, `equilibrium_stage()`, `stage_count()`）

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit07_Nonlinear_Equations.md](Unit07_Nonlinear_Equations.md) | 📄 教學講義 | 非線性方程式理論、數值方法、SciPy 工具、求解策略 |
| [Unit07_Nonlinear_Equations.ipynb](Unit07_Nonlinear_Equations.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit07_Example_01.md](Unit07_Example_01.md) | 📄 案例講義 | Van der Waals 狀態方程式（單變數求解、多重根） |
| [Unit07_Example_01.ipynb](Unit07_Example_01.ipynb) | 💻 程式演練 | Van der Waals 方程式實作 |
| [Unit07_Example_02.md](Unit07_Example_02.md) | 📄 案例講義 | 理想溶液泡點計算（Antoine 方程式） |
| [Unit07_Example_02.ipynb](Unit07_Example_02.ipynb) | 💻 程式演練 | 泡點計算實作 |
| [Unit07_Example_03.md](Unit07_Example_03.md) | 📄 案例講義 | 化學反應平衡系統（多變數聯立求解） |
| [Unit07_Example_03.ipynb](Unit07_Example_03.ipynb) | 💻 程式演練 | 反應平衡系統實作 |
| [Unit07_Example_04.md](Unit07_Example_04.md) | 📄 案例講義 | CSTR 多重穩態分析（多重解、穩定性） |
| [Unit07_Example_04.ipynb](Unit07_Example_04.ipynb) | 💻 程式演練 | CSTR 多重穩態實作與分岔圖 |
| [Unit07_Example_05.md](Unit07_Example_05.md) | 📄 案例講義 | 熱傳導乾燥過程（耦合熱質傳方程式） |
| [Unit07_Example_05.ipynb](Unit07_Example_05.ipynb) | 💻 程式演練 | 乾燥過程實作 |
| [Unit07_Example_06.md](Unit07_Example_06.md) | 📄 案例講義 | 二元蒸餾塔設計計算（McCabe-Thiele 數值法） |
| [Unit07_Example_06.ipynb](Unit07_Example_06.ipynb) | 💻 程式演練 | 蒸餾塔逐板計算實作 |
| [Unit07_Homework.ipynb](Unit07_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit06 線性聯立方程式之求解](../Unit06/README.md)**：`scipy.linalg`、矩陣方程組、物料平衡

### ➡️ 下一單元
- **[Unit08 插值、微分與積分之運算](../Unit08/README.md)**：`scipy.interpolate`、`scipy.integrate`、物性數據插值、批次反應積分

---

## 📈 本單元在課程中的定位

```
Unit06 (scipy.linalg) — 線性方程組
      ↓
   Unit07 ← 你在這裡
 ┌──────────────────────────────────────────────────────────┐
 │  非線性方程式 (scipy.optimize)                            │
 │  單變數工具：root_scalar (brentq / newton / bisect)      │
 │  多變數工具：fsolve / root (hybr / broyden1 / anderson)  │
 │  多重解策略：圖形法 / 多起始點 / 延續法 / 穩定性分析       │
 └──────────────────────────────────────────────────────────┘
      ↓
 Unit08 (scipy.interpolate / scipy.integrate)
      ↓
 Unit09 (scipy.integrate.solve_ivp / solve_bvp) — ODE 求解
      ↓
 ...（後續各數值計算單元）
```

**與化工問題的對應**：
```
物料平衡（線性）       → 線性方程組 (Unit06)
狀態方程式 / 相平衡    → 非線性方程式 (Unit07)  ← 本單元
反應器動態模擬        → ODE (Unit09)
熱傳 / 質傳場         → PDE (Unit10)
```

**Unit07 與 Unit09 的連結**：Unit07 求解的 CSTR 多重穩態點（Example 04）將在 Unit09 中以動態 ODE 模擬驗證，從不同角度理解相同物理現象。

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit07_Nonlinear_Equations.md`（約 60–90 分鐘）理解各數值方法理論
   - Step 2：執行 `Unit07_Nonlinear_Equations.ipynb` 熟悉 `root_scalar`、`fsolve`、`root` 的用法
   - Step 3：依序完成 Example 01–06（建議 Example 01 → 02 → 03 → 04 → 05 → 06）

2. **重點關注**：
   - **多重解問題**：非線性方程式最重要也最難處理的挑戰，Example 01 和 04 是重點
   - **初始猜測值的重要性**：建議每個案例都先繪圖，再決定起始猜測值
   - **Example 04（CSTR 多重穩態）**：是 Unit07 最具挑戰性的案例，涵蓋多重解偵測、穩定性分析、分岔圖繪製
   - **Example 06（蒸餾塔）**：是最複雜的整合案例，展示如何將多個非線性子問題組合為完整計算流程

3. **求解器選擇速查**：
   - 單變數且已知根所在區間 → `root_scalar(method='brentq')`（最推薦）
   - 單變數且可提供導數 → `root_scalar(method='newton')`
   - 多變數（一般情況） → `fsolve()` 或 `root(method='hybr')`
   - 多變數且 Jacobian 計算昂貴（大型系統） → `root(method='broyden1')`

4. **參考外部資源**：
   - [SciPy optimize 官方文件](https://docs.scipy.org/doc/scipy/reference/optimize.html)
   - [SciPy root_scalar 說明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html)
   - [SciPy fsolve 說明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit07 非線性方程式之求解
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
