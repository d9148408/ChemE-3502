# Unit10 偏微分方程式之求解 (Solving Partial Differential Equations)

## 📚 單元簡介

流體在管道中的流速分布、固體在淬火冷卻過程中的溫度場演變、薄膜中溶質的擴散濃度場——這些化工問題的特徵是：**物理量同時隨空間與時間變化**，無法以 ODE 描述，而必須使用**偏微分方程式（PDE）**。PDE 求解是化學工程輸送現象（Transport Phenomena）計算的核心，涵蓋熱量傳遞、質量傳遞與動量傳遞三大面向。

本單元以 **`py-pde`** 套件與 **`scipy`** 配合 **Method of Lines（MoL）** 為核心工具，介紹二階線性 PDE 三大類型（橢圓型、拋物線型、雙曲線型）的物理意義與求解策略，以及 py-pde 的 Grid 系統、場變數設定與邊界條件處理。透過六個標準化工案例（平板熱傳、擴散反應、球體冷卻、二維穩態熱傳、液膜質傳、觸媒反應管），以及三個深度特別案例（Navier-Stokes 方程式、Fick's Laws 質傳、Fourier's Law 熱傳），完整呈現 PDE 工具在化工輸送現象模擬中的應用與邊界。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握 PDE 分類與物理意義**：識別橢圓型（穩態）、拋物線型（非穩態擴散/熱傳）、雙曲線型（波動）三大類型，以及 Dirichlet、Neumann、Robin 三種邊界條件的化工對應情境
2. **熟練使用 `py-pde` 套件**：建立 Grid 物件、ScalarField 場變數、設定邊界條件、定義 PDE 方程式（`DiffusionPDE`、`PDE` 字串式、自訂 `PDEBase`）並執行求解
3. **實作 Method of Lines（MoL）方法**：手動空間差分離散化，搭配 `scipy.integrate.solve_ivp()` 進行時間積分，從底層理解 PDE 數值求解原理
4. **認識 py-pde 的能力邊界**：了解 py-pde 的適用範圍（規則幾何、結構化網格）與侷限，建立何時需要升級至 COMSOL/ANSYS 商業軟體的判斷能力
5. **求解化工輸送現象 PDE**：應用 py-pde 與 MoL 求解平板/圓柱/球體的熱傳與質傳問題，並驗證數值解與解析解（Fourier 級數、誤差函數解）的一致性

---

## 📖 單元內容架構

### 1️⃣ Unit10_PDE — 偏微分方程式理論與求解工具

**檔案**：
- 講義：[Unit10_PDE.md](Unit10_PDE.md)
- 程式範例：[Unit10_PDE.ipynb](Unit10_PDE.ipynb)

**內容重點**：

#### PDE 分類與邊界條件

| PDE 類型 | 方程特徵 | 化工對應問題 | 求解特質 |
|---------|---------|------------|---------|
| **橢圓型（Elliptic）** | $\nabla^2 u = f$ | 穩態熱傳、穩態質傳 | 全域耦合，需迭代 |
| **拋物線型（Parabolic）** | $\partial u/\partial t = \alpha \nabla^2 u$ | 非穩態熱傳、非穩態擴散 | 時間推進，好求解 |
| **雙曲線型（Hyperbolic）** | $\partial^2 u/\partial t^2 = c^2 \nabla^2 u$ | 聲波、衝擊波傳播 | 有限速度傳播 |

| 邊界條件類型 | 數學形式 | 化工對應情境 |
|------------|---------|------------|
| **Dirichlet** | $u\|_{\partial\Omega} = f$ | 壁面固定溫度/濃度 |
| **Neumann** | $\partial u/\partial n\|_{\partial\Omega} = g$ | 絕熱壁（$g=0$）、固定通量 |
| **Robin** | $k\partial u/\partial n + h(u - u_\infty) = 0$ | 對流換熱（Newton's Law） |

#### `py-pde` 核心物件

| 物件類型 | 常用類別 | 說明 |
|---------|---------|------|
| **Grid** | `CartesianGrid`, `CylindricalGrid`, `SphericalGrid` | 計算區域與座標系統 |
| **場變數** | `ScalarField`, `VectorField` | 溫度、濃度、速度場 |
| **PDE 方程式** | `DiffusionPDE`, `PDE`（字串式）, 繼承 `PDEBase` | 問題定義 |
| **求解器** | `ExplicitSolver`, `ImplicitSolver`, `ScipySolver` | Stiff 問題用 ImplicitSolver |
| **結果追蹤** | `StorageTracker`, `PlotTracker` | 記錄中間時刻結果 |

#### Method of Lines（MoL）方法
- **原理**：空間方向採有限差分離散化，將 PDE 轉化為大型 ODE 系統
- **時間積分**：搭配 `scipy.integrate.solve_ivp()` 進行（Non-stiff 用 RK45，Stiff 用 Radau）
- **邊界條件處理**：固定邊界值（Dirichlet）與通量條件（Neumann）的數值實作
- **與 py-pde 的比較**：MoL 更靈活（可整合任意函數），py-pde 更簡潔（自動處理邊界與網格）

#### py-pde + scipy 的能力與侷限

| 能力 | 侷限 |
|------|------|
| 1D/2D/3D 規則幾何（矩形、圓柱、球形）PDE 求解 | 無法處理任意形狀不規則幾何邊界 |
| Dirichlet、Neumann、Robin 邊界條件 | 不支援多物理場耦合（Multi-physics） |
| 非線性 PDE（自訂 `PDEBase`） | 非結構化網格（Unstructured Mesh）不支援 |
| 教學、概念驗證、工程快速估算 | 大規模 3D 問題計算效率受限 |

**商業軟體需求時機**：任意複雜幾何、多物理場耦合（流熱固化電磁）、紊流、多材質介面 → 使用 **COMSOL Multiphysics** 或 **ANSYS Fluent**

---

## 🧪 化工案例演練（標準案例）

### 🌡️ Example 01 — 平板之非穩態熱傳導

**檔案**：[Unit10_Example_01.md](Unit10_Example_01.md) | [Unit10_Example_01.ipynb](Unit10_Example_01.ipynb)

**問題概述**：初始溫度均勻的有限厚平板，兩端施加不同固定溫度邊界條件（Dirichlet），求非穩態溫度分布 $T(x, t)$ 的時空演變。

**數學模型**：一維拋物線型 PDE（Parabolic PDE）

$$
\rho C_p \frac{\partial T}{\partial t} = k \frac{\partial^2 T}{\partial x^2}, \quad T(0,t) = T_L, \quad T(L,t) = T_R
$$

**化工重點**：
- **方法一**：`py-pde` 的 `DiffusionPDE` + `CartesianGrid`（最簡潔，3 行程式碼）
- **方法二**：Method of Lines（MoL）空間差分 + `scipy.integrate.solve_ivp()`（理解底層原理）
- 兩種方法結果比較，與 Fourier 級數解析解驗證精度
- 網格取點密度 $N$ 與計算精度的系統性分析
- 繪製溫度場時空演變曲面圖及特定時刻的軸向分布

---

### 🧪 Example 02 — 氣體在液體中之擴散與反應

**檔案**：[Unit10_Example_02.md](Unit10_Example_02.md) | [Unit10_Example_02.ipynb](Unit10_Example_02.ipynb)

**問題概述**：靜止液柱上方曝露於氣體 A 中，A 沿液柱向下擴散。比較有無一階化學反應兩種情形下的濃度分布演變，並計算氣液界面的質量通量隨時間的變化。

**數學模型**：一維拋物線型 PDE（含反應 source 項，混合邊界條件）

$$
\frac{\partial C_A}{\partial t} = D_{AB} \frac{\partial^2 C_A}{\partial z^2} - k_1 C_A, \quad C_A(0,t) = C_{A0} \text{（Dirichlet）}, \quad \frac{\partial C_A}{\partial z}\bigg|_{z=L} = 0 \text{（Neumann）}
$$

**化工重點**：
- 使用 `py-pde` 的 `PDE` 類別（字串式）定義含反應項的擴散方程式
- 比較無反應（Case 1）與一階反應（Case 2）的濃度分布差異
- 界面通量計算：$N_{Az} = -D_{AB} \partial C_A / \partial z |_{z=0}$（使用數值微分）
- Hatta 數 $Ha = L\sqrt{k_1/D_{AB}}$ 對濃度分布形狀的影響

---

### ❄️ Example 03 — 固體球體之急速冷卻

**檔案**：[Unit10_Example_03.md](Unit10_Example_03.md) | [Unit10_Example_03.ipynb](Unit10_Example_03.ipynb)

**問題概述**：初溫均勻的熱球體急速投入冷水浴中，求球體內部非穩態溫度分布（球座標 PDE）。

**數學模型**：一維球座標非穩態熱傳 PDE

$$
\rho C_p \frac{\partial T}{\partial t} = \frac{k}{r^2} \frac{\partial}{\partial r}\left(r^2 \frac{\partial T}{\partial r}\right), \quad \frac{\partial T}{\partial r}\bigg|_{r=0} = 0 \text{（Neumann）}, \quad -k\frac{\partial T}{\partial r}\bigg|_{r=R} = h(T - T_\infty) \text{（Robin）}
$$

**化工重點**：
- **球座標轉換技巧**：令 $v = rT$，將球座標方程式轉為等效一維平板方程式（避免 $r=0$ 奇異點）
- 使用 `py-pde` 的 `SphericalGrid` 求解
- 驗證：Biot 數 $Bi = hR/k$ 對應的 Heisler chart 理論值比對
- 探討 $Bi$ 大小對球心/表面溫度趨近速度的影響

---

### 📐 Example 04 — 二維穩態熱傳導

**檔案**：[Unit10_Example_04.md](Unit10_Example_04.md) | [Unit10_Example_04.ipynb](Unit10_Example_04.ipynb)

**問題概述**：矩形平板四邊施加不同固定溫度邊界條件，求穩態二維溫度分布（Laplace 方程式）。

**數學模型**：二維橢圓型 PDE（Laplace 方程式）

$$
\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0, \quad T = T_i \text{ on each boundary}
$$

**化工重點**：
- **方法一**：`py-pde` 的 `CartesianGrid`（2D）時間推進法（令 $t \to \infty$ 求穩態）
- **方法二**：`scipy.linalg.solve()` 對二維有限差分線性系統直接求解（與 Unit06 的連結）
- 兩種方法結果比較，與 Fourier 二維解析解驗證
- **py-pde 侷限的展示**：矩形網格無法處理含孔洞幾何（引出 COMSOL/FEM 的需求）
- 繪製二維溫度等高線圖與三維曲面圖

---

### 💧 Example 05 — 液境膜中溶質滲透與 Sherwood 數計算

**檔案**：[Unit10_Example_05.md](Unit10_Example_05.md) | [Unit10_Example_05.ipynb](Unit10_Example_05.ipynb)

**問題概述**：液境膜理論（Penetration Theory）——氣液界面溶質向液相本體擴散，求無因次濃度分布隨無因次時間 $\theta$ 的演變，以及 Sherwood 數的時間衰減曲線。

**數學模型**：一維無因次拋物線型 PDE

$$
\frac{\partial C}{\partial \theta} = \frac{\partial^2 C}{\partial X^2}, \quad C(0,\theta) = 1, \quad C(1,\theta) = 0 \text{（Dirichlet 邊界）}
$$

**化工重點**：
- 計算界面 Sherwood 數：$Sh(\theta) = -\partial C / \partial X |_{X=0}$（數值微分求通量）
- 與滲透理論解析解 $Sh = 2 / \sqrt{\pi\theta}$ 比對驗證
- 探討 Fourier 數 $Fo$ 對濃度分布形狀的影響（擴散前緣深度）
- 繪製濃度時空演變圖與 Sherwood 數衰減曲線

---

### ⚗️ Example 06 — 管形觸媒反應器徑向溫度與轉化率分布

**檔案**：[Unit10_Example_06.md](Unit10_Example_06.md) | [Unit10_Example_06.ipynb](Unit10_Example_06.ipynb)

**問題概述**：苯加氫的固定床觸媒反應管，以徑向位置為空間變數，在固定軸向截面下求溫度 $T(r)$ 與轉化率 $f(r)$ 的穩態徑向分布（耦合橢圓型 PDE）。

**數學模型**：圓柱座標聯立 PDE（簡化稳態，溫度與轉化率耦合）

$$
\frac{d^2T}{dr^2} + \frac{1}{r}\frac{dT}{dr} + S_T(T, f) = 0, \quad \frac{d^2f}{dr^2} + \frac{1}{r}\frac{df}{dr} + S_f(T, f) = 0
$$

**化工重點**：
- 使用 `py-pde` 的 `CylindricalGrid` 與自訂 `PDEBase` 建立耦合 PDE 系統
- Langmuir-Hinshelwood 動力學：$r_A(T, f)$ 的非線性計算
- 徑向溫度分布的「熱點」現象與管壁熱傳（Robin BC）的影響
- 展示 py-pde 在複雜耦合 PDE 的應用極限（引出需要 COMSOL 的情境）

---

## 🔬 特別深度案例

Unit10 包含三個**深度特別案例**，涵蓋化工輸送現象的三大傳遞律，從 1D 到 3D 系統系統性介紹 py-pde 的完整應用能力與邊界。

### 🌊 特別案例：Navier-Stokes 方程式（動量傳遞）

**檔案**：[Unit10_Example_Navier_Stokes_Equation.md](Unit10_Example_Navier_Stokes_Equation.md) | [Unit10_Example_Navier_Stokes_Equation.ipynb](Unit10_Example_Navier_Stokes_Equation.ipynb)

**涵蓋範圍**：不可壓縮 N-S 方程式的動量傳遞，1D → 2D → 3D 三個維度遞進

$$
\rho\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}\right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \rho \mathbf{g}, \quad \nabla \cdot \mathbf{u} = 0
$$

| Part | 場景 | 方法 |
|------|------|------|
| **Part 1（1D）** | Stokes 第一問題（瞬間啟動平板）、Hagen-Poiseuille 管流、Burgers 方程式 | `DiffusionPDE` / 自訂 `PDEBase` |
| **Part 2（2D）** | 驅動蓋方腔流（Lid-Driven Cavity，CFD 經典 Benchmark）、平行平板 Poiseuille 流 | 渦度-流函數法，自訂 `PDEBase` |
| **Part 3（3D）** | 3D 矩形管道流動（演示 + 能力邊界說明） | `CartesianGrid`（3D），說明侷限 |

---

### 🧬 特別案例：Fick's Laws 質量傳遞方程式

**檔案**：[Unit10_Example_Ficks_Laws_Equation.md](Unit10_Example_Ficks_Laws_Equation.md) | [Unit10_Example_Ficks_Laws_Equation.ipynb](Unit10_Example_Ficks_Laws_Equation.ipynb)

**涵蓋範圍**：Fick 第一定律（穩態通量）與第二定律（非穩態擴散）的完整求解

$$
\frac{\partial C_A}{\partial t} = D_{AB} \nabla^2 C_A - r_A(C_A)
$$

| Part | 場景 | 特點 |
|------|------|------|
| **Part 1（1D）** | 半無限介質擴散、有限厚平板雙面擴散、1D 反應-擴散（Thiele 模數 $\phi$） | 與 erfc 解析解比對 |
| **Part 2（2D）** | 矩形區域 2D 非穩態擴散、圓柱座標軸對稱擴散、Gray-Scott 反應-擴散（Turing Pattern） | `CylindricalGrid`，耦合 PDE |
| **Part 3（3D）** | 3D 箱體擴散、球座標球形觸媒顆粒擴散（有效因子 $\eta$） | `SphericalGrid`，Thiele 曲線 |
| **Part 4** | Maxwell-Stefan vs Fick 定律、多孔介質有效擴散係數 | 延伸概念 |

---

### 🔥 特別案例：Fourier's Law 能量傳遞方程式

**檔案**：[Unit10_Example_Fouriers_Laws_Equation.md](Unit10_Example_Fouriers_Laws_Equation.md) | [Unit10_Example_Fouriers_Laws_Equation.ipynb](Unit10_Example_Fouriers_Laws_Equation.ipynb)

**涵蓋範圍**：Fourier 第一定律（穩態熱通量）與通用非穩態熱傳導方程式的完整求解

$$
\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + \dot{Q}, \quad \mathbf{q} = -k \nabla T
$$

| Part | 場景 | 特點 |
|------|------|------|
| **Part 1（1D）** | 非穩態平板熱傳（含 Robin BC）、含內熱源穩態 Poisson 方程式 | Heisler chart 驗證 |
| **Part 2（2D）** | 矩形板穩態 Laplace、非穩態 2D 熱傳、圓柱軸對稱 2D、散熱鰭片 Robin BC | `CylindricalGrid`，鰭片效率 $\eta_f$ |
| **Part 3（3D）** | 3D 立方體對流冷卻、球體衰火（含輻射 Robin BC） | 非線性 Robin BC，輻射效應 |
| **Part 4** | 溫度相依熱導率 $k(T)$（非線性 PDE）、熱質傳耦合（Dufour 效應） | 自訂 `PDEBase` |

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit10_PDE.md](Unit10_PDE.md) | 📄 教學講義 | PDE 分類、py-pde 工具、MoL 方法、商業軟體比較 |
| [Unit10_PDE.ipynb](Unit10_PDE.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit10_Example_01.md](Unit10_Example_01.md) | 📄 案例講義 | 平板非穩態熱傳（py-pde + MoL 雙方法，Fourier 解析解驗證） |
| [Unit10_Example_01.ipynb](Unit10_Example_01.ipynb) | 💻 程式演練 | 平板熱傳實作 |
| [Unit10_Example_02.md](Unit10_Example_02.md) | 📄 案例講義 | 液柱擴散與反應（混合邊界條件，有/無反應比較） |
| [Unit10_Example_02.ipynb](Unit10_Example_02.ipynb) | 💻 程式演練 | 擴散反應實作 |
| [Unit10_Example_03.md](Unit10_Example_03.md) | 📄 案例講義 | 球體急速冷卻（球座標 PDE，Heisler chart 驗證） |
| [Unit10_Example_03.ipynb](Unit10_Example_03.ipynb) | 💻 程式演練 | 球體冷卻實作 |
| [Unit10_Example_04.md](Unit10_Example_04.md) | 📄 案例講義 | 二維穩態熱傳（Laplace，py-pde + scipy 雙方法） |
| [Unit10_Example_04.ipynb](Unit10_Example_04.ipynb) | 💻 程式演練 | 二維穩態熱傳實作 |
| [Unit10_Example_05.md](Unit10_Example_05.md) | 📄 案例講義 | 液境膜滲透（無因次化，Sherwood 數計算） |
| [Unit10_Example_05.ipynb](Unit10_Example_05.ipynb) | 💻 程式演練 | 液膜滲透實作 |
| [Unit10_Example_06.md](Unit10_Example_06.md) | 📄 案例講義 | 觸媒反應管徑向分布（耦合 PDE，CylindricalGrid） |
| [Unit10_Example_06.ipynb](Unit10_Example_06.ipynb) | 💻 程式演練 | 觸媒反應管實作 |
| [Unit10_Example_Navier_Stokes_Equation.md](Unit10_Example_Navier_Stokes_Equation.md) | 📄 特別案例講義 | N-S 方程式 1D/2D/3D（動量傳遞，CFD 入門） |
| [Unit10_Example_Navier_Stokes_Equation.ipynb](Unit10_Example_Navier_Stokes_Equation.ipynb) | 💻 程式演練 | N-S 方程式實作 |
| [Unit10_Example_Ficks_Laws_Equation.md](Unit10_Example_Ficks_Laws_Equation.md) | 📄 特別案例講義 | Fick's Laws 1D/2D/3D（質量傳遞，有效因子） |
| [Unit10_Example_Ficks_Laws_Equation.ipynb](Unit10_Example_Ficks_Laws_Equation.ipynb) | 💻 程式演練 | Fick's Laws 實作 |
| [Unit10_Example_Fouriers_Laws_Equation.md](Unit10_Example_Fouriers_Laws_Equation.md) | 📄 特別案例講義 | Fourier's Law 1D/2D/3D（熱量傳遞，鰭片效率） |
| [Unit10_Example_Fouriers_Laws_Equation.ipynb](Unit10_Example_Fouriers_Laws_Equation.ipynb) | 💻 程式演練 | Fourier's Law 實作 |
| [Unit10_Homework.ipynb](Unit10_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit09 常微分方程式之求解](../Unit09/README.md)**：`scipy.integrate.solve_ivp`/`solve_bvp`、IVP/BVP、Stiff ODE

### ➡️ 下一單元
- **[Unit11 傅立葉轉換與頻譜分析](../Unit11/README.md)**：`scipy.fft`、FFT、頻譜分析、化工信號處理

---

## 📈 本單元在課程中的定位

```
Unit09 (scipy.integrate.solve_ivp / solve_bvp) — ODE 求解
      ↓
   Unit10 ← 你在這裡
 ┌────────────────────────────────────────────────────────────────┐
 │  偏微分方程式求解 (py-pde + scipy Method of Lines)               │
 │  py-pde 工具：CartesianGrid / CylindricalGrid / SphericalGrid  │
 │              DiffusionPDE / PDE(字串式) / 自訂 PDEBase          │
 │              ExplicitSolver / ImplicitSolver / ScipySolver     │
 │  MoL 方法：空間差分離散化 + scipy.integrate.solve_ivp()         │
 │  能力邊界：規則幾何 ✓，不規則幾何/多物理耦合 → 需 COMSOL/ANSYS  │
 └────────────────────────────────────────────────────────────────┘
      ↓
 Unit11 (scipy.fft) — 傅立葉轉換與頻譜分析
      ↓
 Unit12 (scipy.optimize) — 程序最適化
      ↓
 ...（後續各數值計算單元）
```

**化工輸送現象三大傳遞律的對應**：
```
動量傳遞（Fluid Mechanics）   → N-S 方程式 PDE (Unit10 特別案例)  ← 本單元
熱量傳遞（Heat Transfer）     → 熱傳 PDE / Fourier's Law (Unit10)  ← 本單元
質量傳遞（Mass Transfer）     → 質傳 PDE / Fick's Laws (Unit10)    ← 本單元
反應器動態（Reaction Eng.）   → ODE IVP (Unit09)
```

**Unit10 的重要橫向聯繫**：
- **與 Unit06 的連結**：Example 04（二維穩態熱傳）使用 `scipy.linalg.solve()` 求解有限差分線性系統（Unit06 工具的延伸應用）
- **與 Unit09 的連結**：Method of Lines 將 PDE 轉化為 ODE 系統，再用 `solve_ivp()` 求解——直接橋接兩個單元的核心工具

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit10_PDE.md`（約 60 分鐘）建立 PDE 分類、py-pde 工具、MoL 概念框架
   - Step 2：執行 `Unit10_PDE.ipynb` 熟悉 py-pde 的基本操作（Grid、ScalarField、DiffusionPDE）
   - Step 3：依序完成標準案例 Example 01–06（建議 01 → 02 → 03 → 04 → 05 → 06）
   - Step 4：視興趣與需求選讀特別案例（N-S、Fick's Laws、Fourier's Law）

2. **重點關注**：
   - **Example 01（平板熱傳）**：同時實作 py-pde 與 MoL 兩種方法，理解兩者的差異是本單元最核心的學習
   - **Example 04（二維穩態）**：橢圓型 PDE 與拋物線型 PDE 的求解策略差異（時間推進法 vs 直接法）
   - **py-pde 的侷限**：每個案例都注意 py-pde 「能做什麼」與「不能做什麼」，建立工具選擇的判斷力

3. **py-pde 工具選擇速查**：

   | 問題類型 | 推薦 py-pde 方式 |
   |---------|----------------|
   | 均勻擴散（1D/2D/3D 矩形） | `DiffusionPDE` + `CartesianGrid` |
   | 任意 PDE（含反應項） | `PDE(字串式)` |
   | 複雜耦合非線性 PDE | 繼承 `PDEBase` 自訂 |
   | 球座標問題 | `SphericalGrid`（利用對稱性降維） |
   | 圓柱座標軸對稱問題 | `CylindricalGrid` |
   | Stiff PDE（隱式求解） | `ImplicitSolver` 或 `ScipySolver` |

4. **常見錯誤提醒**：
   - `py-pde` 的 `DiffusionPDE` 的擴散係數是純數值（熱問題需先計算熱擴散率 $\alpha = k/(\rho C_p)$）
   - MoL 中 Dirichlet 邊界條件需在每次 ODE 函數調用中強制設定邊界點值，不能讓 ODE 求解器自由演進

5. **參考外部資源**：
   - [py-pde 官方文件](https://py-pde.readthedocs.io/)
   - [py-pde GitHub](https://github.com/zwicker-group/py-pde)
   - [SciPy solve_ivp（MoL 時間積分）](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit10 偏微分方程式之求解
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
