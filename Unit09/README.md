# Unit09 常微分方程式之求解 (Solving Ordinary Differential Equations)

## 📚 單元簡介

化工程序的**動態行為**——反應器的濃度與溫度隨時間的演變、管式反應器沿軸向的轉化率與溫度分布、非牛頓流體在管內的速度分布——幾乎都可以用**常微分方程式（ODE）**來描述。ODE 求解是化學工程數值計算的核心技能之一，其中涵蓋兩大問題類型：給定起始條件的**起始值問題（IVP）**，以及在兩端指定邊界條件的**邊界值問題（BVP）**。

本單元以 **`scipy.integrate.solve_ivp()`** 與 **`scipy.integrate.solve_bvp()`** 為核心工具，系統性介紹各類 ODE 求解器（RK45、Radau、BDF、LSODA）的特性與選用時機，並深入探討化工程序中常見的 **Stiff ODE** 問題與**微分代數系統（DAE）**的處理策略。透過六個化工實際案例，從 CSTR 動態模擬、PFR 軸向分布，到觸媒反應管的 BVP 求解與非牛頓流體速度分布，完整呈現 ODE 工具在化工動態模擬中的應用。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握 IVP 與 BVP 的理論基礎**：理解兩類問題的本質差異，以及不同求解器的收斂特性與適用範圍
2. **熟練使用 `scipy.integrate.solve_ivp()`**：設定 ODE 函數、起始條件、時間範圍、求解方法與精度參數（rtol/atol）
3. **識別並處理 Stiff ODE**：計算 Stiffness Ratio，選擇適當的隱式求解器（Radau/BDF），分析 Non-stiff 求解器的發散失效
4. **解決 BVP 問題**：使用 `scipy.integrate.solve_bvp()` 設定邊界條件函數與初始猜測網格，處理含待定參數的 BVP
5. **建立化工動態 ODE 模型**：將 CSTR、PFR、固定床反應器、生化反應器等化工問題轉化為規範的 ODE 方程式組，並驗證求解結果的物理合理性

---

## 📖 單元內容架構

### 1️⃣ Unit09_ODE — 常微分方程式理論與求解工具

**檔案**：
- 講義：[Unit09_ODE.md](Unit09_ODE.md)
- 程式範例：[Unit09_ODE.ipynb](Unit09_ODE.ipynb)

**內容重點**：

#### ODE 問題分類

| 問題類型 | 問題形式 | 特徵 | 求解工具 |
|---------|---------|------|---------|
| **IVP（起始值問題）** | $\dot{\mathbf{x}} = \mathbf{f}(t, \mathbf{x})$，$\mathbf{x}(t_0) = \mathbf{x}_0$ | 從初始條件向前積分 | `solve_ivp()` |
| **BVP（邊界值問題）** | $\dot{\mathbf{x}} = \mathbf{F}(t, \mathbf{x})$，$\mathbf{g}(\mathbf{x}(a), \mathbf{x}(b)) = 0$ | 兩端各有約束，需迭代 | `solve_bvp()` |

#### `solve_ivp()` 求解器比較

| 方法 | 類型 | 適用問題 | 特點 |
|------|------|---------|------|
| `RK45` | 顯式，4(5) 階 | Non-stiff（預設） | 最常用，自適應步長 |
| `RK23` | 顯式，2(3) 階 | Non-stiff，低精度 | 快速粗略估算 |
| `DOP853` | 顯式，8(5,3) 階 | Non-stiff，高精度 | 高精度要求 |
| `Radau` | 隱式，5 階 Radau IIA | **Stiff（推薦）** | 穩定性優異 |
| `BDF` | 隱式，1–5 階 | **Stiff，大規模** | 逆向微分公式 |
| `LSODA` | 自動切換 | 未知 stiffness | 自動偵測並切換 |

#### Stiff ODE 的識別與處理
- **Stiffness 定義**：系統 Jacobian 矩陣特徵值的實部差距懸殊
- **Stiffness Ratio**：$SR = \max|\mathrm{Re}(\lambda_i)| / \min|\mathrm{Re}(\lambda_i)|$，$SR > 1000$ 視為 Stiff
- **Non-stiff 求解器失效現象**：步長被迫縮至極小、計算時間劇增甚至解發散
- **化工 Stiff 問題來源**：快慢反應並存的化學反應動力學（如燃燒、酵素催化、生化反應）

#### `solve_bvp()` 使用要點
- **邊界條件函數** `bc(ya, yb)`：定義 $\mathbf{g}(\mathbf{x}(a), \mathbf{x}(b)) = 0$，返回殘差向量
- **初始猜測**：提供初始網格 `x` 與猜測解 `y`（常以線性插值建立）
- **含待定參數的 BVP**：引入參數向量 `p`，讓求解器同步估計未知參數
- **解的品質評估**：`sol.rms_residuals`（殘差均方根）、`sol.status`（收斂狀態）

#### 微分代數系統（DAE）
- **DAE 形式**：ODE 與代數約束方程式混合 $\mathbf{M}\dot{\mathbf{y}} = \mathbf{F}(t, \mathbf{y})$，$\mathbf{M}$ 為奇異矩陣
- **處理策略**：隱式 BDF 搭配 mass 矩陣、代數式視為極快速動力學（奇異攝動法）、強制一致性初始條件
- **化工 DAE 應用**：相平衡約束 + 動態質量平衡的聯立求解

#### 程式設計最佳實踐
- **高階 ODE 轉換**：$n$ 階 ODE → $n$ 個聯立一階 ODE（標準向量化寫法）
- **外部輸入函數（Forcing Function）**：常數、階梯函數、時變函數的處理（`lambda` 或 `partial`）
- **精度控制**：一般化工問題 `rtol=1e-6, atol=1e-8`；Stiff 問題可適度放寬
- **結果驗證**：殘差計算、與解析解對比、物理量守恆檢核（質量守恆、能量守恆）

---

## 🧪 化工案例演練

### 🔄 Example 01 — CSTR 反應器動態模擬

**檔案**：[Unit09_Example_01.md](Unit09_Example_01.md) | [Unit09_Example_01.ipynb](Unit09_Example_01.ipynb)

**問題概述**：放熱一次不可逆反應 A → B 在 CSTR 中的動態行為，求解無因次濃度 $c$ 與溫度 $\theta$ 的時間演變，並透過相圖分析三個穩態點（低、中、高轉化率）的穩定性。

**數學模型**：雙變數 IVP ODE（Non-stiff，`RK45`）

$$
\begin{cases}
\dot{c} = \frac{1}{\tau}(1 - c) - Da \cdot c \cdot e^{\gamma\theta/(1+\theta)} \\
\dot{\theta} = \frac{1}{\tau}(-\theta) + Da \cdot B \cdot c \cdot e^{\gamma\theta/(1+\theta)} - \beta(\theta - \theta_c)
\end{cases}
$$

**化工重點**：
- 使用 `solve_ivp(method='RK45')` 求解雙變數動態方程式
- 不同初始條件（$c_0, \theta_0$）對應不同穩態的動態軌跡
- **相圖分析（Phase Portrait）**：繪製 $c$-$\theta$ 相圖，識別三個穩態點的穩定性（穩定焦點/節點 vs 鞍點）
- **與 Unit07 的聯繫**：驗證動態模擬的最終穩態與 Unit07_Example_04 非線性方程式的靜態解一致

---

### 🏭 Example 02 — 非恆溫柱塞型反應器（PFR）溫度與轉化率分布

**檔案**：[Unit09_Example_02.md](Unit09_Example_02.md) | [Unit09_Example_02.ipynb](Unit09_Example_02.ipynb)

**問題概述**：丙酮裂解反應（$\mathrm{CH_3COCH_3 \to CH_2CO + CH_4}$）在有外部熱交換的 PFR 中進行，建立轉化率 $X_A$ 與溫度 $T$ 沿反應器長度 $z$ 變化的耦合 ODE 系統，分析操作條件對反應器性能的影響。

**數學模型**：雙變數 IVP ODE（以空間 $z$ 為獨立變數）

$$
\begin{cases}
\dfrac{dX_A}{dz} = \dfrac{A_c \cdot r_A(T, C_A)}{F_{A0}} \\[8pt]
\dfrac{dT}{dz} = \dfrac{U \cdot \pi D (T_a - T) + A_c \cdot (-\Delta H_r) \cdot r_A}{F_{A0} \cdot C_{p,mix}}
\end{cases}
$$

**化工重點**：
- ODE 以**空間位置**（而非時間）為獨立變數：PFR 的穩態軸向分布本質上是 IVP
- 外部熱交換溫度 $T_a$ 的改變對轉化率與溫度分布的敏感度分析
- 識別沿反應器的**溫度熱點**（Hot Spot）與最大反應速率位置
- 繪製轉化率 $X_A(z)$ 與溫度 $T(z)$ 的軸向分布圖

---

### 🦠 Example 03 — 批次反應器中之生化程序動態（Stiff ODE）

**檔案**：[Unit09_Example_03.md](Unit09_Example_03.md) | [Unit09_Example_03.ipynb](Unit09_Example_03.ipynb)

**問題概述**：細胞（Biomass）與基質（Substrate）濃度動態的 Monod 動力學模型，系統因快速細胞死亡動力學與慢速生長動力學共存而具有高 Stiffness，展示 Non-stiff 求解器的失效現象與 Stiff 求解器的正確求解。

**數學模型**：Stiff IVP ODE（Monod 動力學）

$$
\begin{cases}
\dot{X} = \mu_{max} \dfrac{S}{K_S + S} X - k_d X \\[8pt]
\dot{S} = -\dfrac{1}{Y_{X/S}} \mu_{max} \dfrac{S}{K_S + S} X
\end{cases}
$$

**化工重點**：
- **Stiffness 分析**：計算 Jacobian 矩陣特徵值，評估 Stiffness Ratio（$SR \gg 1000$）
- **方法對比實驗**：`RK45` 失效（步長縮至極小、計算耗時或發散）vs `Radau`/`BDF` 穩定快速求解
- `rtol`/`atol` 容差設定對計算時間與精度的影響分析
- 繪製細胞濃度 $X(t)$ 與基質濃度 $S(t)$ 的動態曲線，分析生化反應的延遲期、對數生長期與穩定期

---

### ⚗️ Example 04 — 觸媒反應管溫度及轉化率軸向分布（BVP）

**檔案**：[Unit09_Example_04.md](Unit09_Example_04.md) | [Unit09_Example_04.ipynb](Unit09_Example_04.ipynb)

**問題概述**：苯加氫製環己烷的外部熱交換式固定床觸媒反應器（Packed Bed Reactor），建立轉化率 $X_B$ 與溫度 $T$ 沿管長的耦合 BVP，邊界條件為入口轉化率與出口絕熱條件。

**數學模型**：雙變數 BVP ODE

$$
\begin{cases}
X_B'(z) = f_1(X_B, T) \\
T'(z) = f_2(X_B, T)
\end{cases}, \quad
\underbrace{X_B(0) = 0}_{\text{入口}}, \quad
\underbrace{T'(L) = 0}_{\text{出口絕熱}}
$$

**化工重點**：
- **BVP 與 IVP 的根本差異**：出口邊界條件 $T'(L) = 0$ 無法用 IVP 直接積分求解
- `solve_bvp()` 的初始猜測網格建立策略（線性插值、參考 IVP 估算）
- 識別反應器內**熱點位置**及其對反應器安全操作的意義
- 探討入口溫度與流量變化對轉化率與溫度分布的影響（敏感度分析）
- `sol.rms_residuals` 評估 BVP 求解品質

---

### 🌊 Example 05 — 非牛頓流體管內流動之速度分布（BVP）

**檔案**：[Unit09_Example_05.md](Unit09_Example_05.md) | [Unit09_Example_05.ipynb](Unit09_Example_05.ipynb)

**問題概述**：不可壓縮非牛頓流體（Carreau 黏度模式）在無限圓管中的穩態層流速度分布，由圓柱座標動量平衡建立含非線性特性黏度的 BVP，邊界條件為管壁無滑移與管中心對稱條件。

**數學模型**：Carreau 非牛頓流體速度 BVP（圓柱座標）

$$
\frac{d}{dr}\left[\mu(|\dot{\gamma}|) r \frac{dV_x}{dr}\right] = r \frac{dP}{dz}, \quad \mu/\mu_0 = \left[1 + (t_1 |\dot{\gamma}|)^2\right]^{(n-1)/2}
$$

$$
\underbrace{V_x(R) = 0}_{\text{管壁無滑移}}, \quad \underbrace{\frac{dV_x}{dr}\bigg|_{r=0} = 0}_{\text{中心對稱}}
$$

**化工重點**：
- Carreau 黏度模式的非線性特性：低剪率牛頓平台（$\mu_0$）→ 冪次律區（Power-Law）
- BVP 轉換：高階 ODE 轉為一階 ODE 系統，加入速度 $V_x$ 與其導數 $V_x'$ 為狀態變數
- 體積流率計算：`scipy.integrate.quad()` 對速度分布積分（與 Unit08 工具的應用）
- 與牛頓流體（Hagen-Poiseuille 解析解）的速度分布比較
- 探討流體指數 $n$ 對速度分布形狀的影響（剪切稀化 $n < 1$ vs 剪切增稠 $n > 1$）

---

### 🔥 Example 06 — 伴有輻射之平板穩態熱傳溫度分布（BVP）

**檔案**：[Unit09_Example_06.md](Unit09_Example_06.md) | [Unit09_Example_06.ipynb](Unit09_Example_06.ipynb)

**問題概述**：含熱傳導與輻射的平板穩態熱傳導方程式求解，輻射項引入四次方非線性 BVP，邊界條件為平板兩端固定溫度，分析不同輻射係數與邊界溫度對溫度分布的非線性影響。

**數學模型**：非線性 BVP（熱傳導 + Stefan-Boltzmann 輻射）

$$
\frac{d^2T}{dx^2} = \sigma (T^4 - T_s^4), \quad T(0) = T_L, \quad T(1) = T_R
$$

**化工重點**：
- 輻射項 $\sigma T^4$ 引入強非線性，使 BVP 可能有多解（初始猜測影響收斂解）
- 嘗試多組初始猜測值，探討不同收斂結果的物理意義
- **純熱傳導對比**（$\sigma = 0$，線性解析解 $T = T_L + (T_R - T_L)x$）：輻射對溫度分布的非線性扭曲
- 分析輻射係數 $\sigma$ 大小對溫度分布曲率的影響（高溫區輻射損失主導）
- `solve_bvp()` 的收斂診斷與初始猜測的調整策略

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit09_ODE.md](Unit09_ODE.md) | 📄 教學講義 | IVP/BVP 理論、Stiff ODE、DAE、`solve_ivp`/`solve_bvp` 工具總覽 |
| [Unit09_ODE.ipynb](Unit09_ODE.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit09_Example_01.md](Unit09_Example_01.md) | 📄 案例講義 | CSTR 動態模擬（Non-stiff IVP + 相圖分析） |
| [Unit09_Example_01.ipynb](Unit09_Example_01.ipynb) | 💻 程式演練 | CSTR 動態模擬實作 |
| [Unit09_Example_02.md](Unit09_Example_02.md) | 📄 案例講義 | 非恆溫 PFR 軸向分布（IVP，空間為獨立變數） |
| [Unit09_Example_02.ipynb](Unit09_Example_02.ipynb) | 💻 程式演練 | 非恆溫 PFR 實作 |
| [Unit09_Example_03.md](Unit09_Example_03.md) | 📄 案例講義 | 生化程序動態（Stiff ODE，RK45 vs Radau/BDF 比較） |
| [Unit09_Example_03.ipynb](Unit09_Example_03.ipynb) | 💻 程式演練 | Stiff ODE 實作與求解器比較 |
| [Unit09_Example_04.md](Unit09_Example_04.md) | 📄 案例講義 | 觸媒反應管軸向分布（BVP，熱點分析） |
| [Unit09_Example_04.ipynb](Unit09_Example_04.ipynb) | 💻 程式演練 | 固定床反應器 BVP 實作 |
| [Unit09_Example_05.md](Unit09_Example_05.md) | 📄 案例講義 | 非牛頓流體速度分布（BVP，Carreau 模式） |
| [Unit09_Example_05.ipynb](Unit09_Example_05.ipynb) | 💻 程式演練 | 非牛頓流體 BVP 實作 |
| [Unit09_Example_06.md](Unit09_Example_06.md) | 📄 案例講義 | 含輻射平板熱傳（非線性 BVP，多解探討） |
| [Unit09_Example_06.ipynb](Unit09_Example_06.ipynb) | 💻 程式演練 | 含輻射熱傳 BVP 實作 |
| [Unit09_Homework.ipynb](Unit09_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit08 插值、微分與積分之運算](../Unit08/README.md)**：`scipy.interpolate`、`scipy.integrate`（trapezoid/quad）、物性插值、RTD 分析

### ➡️ 下一單元
- **[Unit10 偏微分方程式之求解](../Unit10/README.md)**：`py-pde`、Method of Lines、熱傳/質傳/流動的 PDE 模擬

---

## 📈 本單元在課程中的定位

```
Unit08 (scipy.interpolate / scipy.integrate) — 插值、微分與積分
      ↓
   Unit09 ← 你在這裡
 ┌──────────────────────────────────────────────────────────────┐
 │  常微分方程式求解 (scipy.integrate)                            │
 │  IVP 工具：solve_ivp (RK45 / DOP853 / Radau / BDF / LSODA)  │
 │  BVP 工具：solve_bvp（兩端邊界條件、含待定參數）               │
 │  Stiff 問題：Stiffness Ratio 分析、隱式求解器選用              │
 │  DAE 系統：Mass 矩陣、一致性初始條件                           │
 └──────────────────────────────────────────────────────────────┘
      ↓
 Unit10 (py-pde + scipy) — PDE 求解（Method of Lines）
      ↓
 Unit11 (scipy.fft) — 傅立葉轉換與頻譜分析
      ↓
 ...（後續各數值計算單元）
```

**與化工問題的對應**：
```
物性數據估算                   → 插值 (Unit08)
反應速率 / RTD / 塔高計算      → 數值積分 (Unit08)
反應器動態 / 製程動態模擬      → ODE IVP (Unit09)  ← 本單元
反應管 / 流動速度分布          → ODE BVP (Unit09)  ← 本單元
熱傳 / 質傳 / 流體力學場       → PDE (Unit10)
```

**Unit09 的重要橫向聯繫**：
- **與 Unit07 的連結**：Example 01（CSTR 動態）驗證 Unit07_Example_04（CSTR 多重穩態靜態解），動態與靜態分析相互印證
- **與 Unit08 的連結**：Example 05（非牛頓流體 BVP）使用 `scipy.integrate.quad()` 計算體積流率（Unit08 積分工具的延伸應用）
- **與 Unit10 的連結**：ODE 的 Method of Lines 概念是 Unit10 中 PDE 空間離散化的基礎

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit09_ODE.md`（約 60–90 分鐘）建立 IVP/BVP/Stiff/DAE 的完整概念框架
   - Step 2：執行 `Unit09_ODE.ipynb` 熟悉 `solve_ivp()` 與 `solve_bvp()` 的基本用法
   - Step 3：依序完成 Example 01–06（IVP 案例 01→02→03，BVP 案例 04→05→06）

2. **重點關注**：
   - **Example 01（CSTR 相圖）**：掌握相圖分析是理解化工動態系統穩定性的關鍵技能
   - **Example 03（Stiff ODE）**：動手比較 `RK45` vs `Radau` 的速度差異，建立對 Stiff 問題的直覺
   - **Example 04（BVP）**：`solve_bvp()` 的初始猜測設定是 BVP 求解成功的關鍵，需要耐心調整

3. **求解器選擇速查**：

   | 情境 | 建議求解器 |
   |------|---------|
   | 一般 Non-stiff IVP | `method='RK45'`（預設）|
   | 高精度 Non-stiff IVP | `method='DOP853'` |
   | Stiff IVP（推薦） | `method='Radau'` |
   | Stiff IVP（大型系統） | `method='BDF'` |
   | 不確定 stiffness | `method='LSODA'` |
   | 兩點邊界值問題 | `solve_bvp()` |

4. **常見錯誤提醒**：
   - `solve_ivp()` 的 ODE 函數簽名必須為 `f(t, y)`，其中 `y` 為狀態向量，**不能**使用位置參數以外的不同順序
   - `solve_bvp()` 的 `bc(ya, yb)` 返回的殘差向量長度必須等於狀態變數個數
   - Stiff 問題若使用 `RK45` 且長時間不收斂，請**立即改用** `method='Radau'`，不要調整 tol 強行求解

5. **參考外部資源**：
   - [SciPy solve_ivp 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
   - [SciPy solve_bvp 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html)
   - [Stiff ODE 介紹（Mathworks）](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/ode.pdf)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit09 常微分方程式之求解
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
