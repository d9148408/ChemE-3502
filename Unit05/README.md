# Unit05 SciPy 科學運算套件應用概述 (SciPy Scientific Computing Suite Overview)

## 📚 單元簡介

SciPy 是建立在 NumPy 之上的高階科學計算套件，也是本課程 **Unit06–Unit15** 所有數值計算單元的核心工具。本單元扮演「導覽地圖」的角色，分為兩個主題：

1. **SciPy 整體生態系統概述** (`Unit05_Scipy_Overview`)：介紹 SciPy 的架構、所有主要子模組的功能定位，以及各子模組在化工領域的典型應用，為後續各單元建立完整的先備認知框架。

2. **化工特殊數學函式** (`Unit05_Special_Functions`)：深入介紹 `scipy.special` 模組，並以化工八大學科（單操、傳輸、反應、熱力學、物化、程序控制、工程統計、數值方法）為分類，系統化整理各領域常用的特殊函式及其應用場景，搭配大量程式範例與執行結果。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解 SciPy 生態系統的架構**：掌握 SciPy 與 NumPy 的關係，以及各主要子模組的功能分工
2. **掌握 SciPy 基本使用模式**：子模組載入方式、函式文件查詢、常見回傳值結構、容差參數設定
3. **了解各子模組的化工應用場景**：根據化工問題類型選擇適合的 SciPy 子模組
4. **認識化工領域常用的特殊數學函式**：Bessel 函式、誤差函式、Gamma/Beta 函式等在化工各學科的應用
5. **能夠查閱並正確使用 `scipy.special` 函式**：根據化工問題情境選擇適當的特殊函式

---

## 📖 單元內容架構

### 1️⃣ Unit05_Scipy_Overview — SciPy 科學運算套件應用概述

**檔案**：
- 講義：[Unit05_Scipy_Overview.md](Unit05_Scipy_Overview.md)

**內容重點**：

#### SciPy 生態系統介紹
- **SciPy 與 NumPy 的關係**：NumPy 為多維陣列運算的「地基」，SciPy 為建立在其上的「高樓層」，提供針對特定科學領域的進階演算法
- **SciPy 的主要特色**：開源免費（BSD 授權）、高效能 FORTRAN/C 底層、化工應用深厚
- **安裝與版本管理**：`pip install scipy`、`conda install scipy`、`scipy.__version__`

#### 主要子模組功能概述與課程對應

| 子模組 | 功能說明 | 課程對應單元 |
|--------|---------|------------|
| `scipy.linalg` | 線性代數（矩陣分解、線性方程組求解、特徵值分析） | **Unit06** |
| `scipy.optimize` | 最適化與方程式求解（根求解、函式最小化、曲線擬合） | **Unit07, Unit12, Unit13** |
| `scipy.interpolate` | 插值法（線性插值、樣條插值、多維插值） | **Unit08** |
| `scipy.integrate` | 數值積分與 ODE/BVP 求解 (`quad`, `solve_ivp`, `solve_bvp`) | **Unit08, Unit09** |
| `scipy.sparse` / `scipy.sparse.linalg` | 稀疏矩陣儲存與大型方程組求解 | **Unit06, Unit10** |
| `scipy.fft` | 快速傅立葉轉換（FFT/IFFT、頻率分析） | **Unit11** |
| `scipy.stats` | 統計分析（機率分布、假設檢定、描述統計） | **Unit14** |
| `scipy.signal` | 信號模擬與處理（濾波器、系統建模、頻率響應） | **Unit15** |
| `scipy.special` | 特殊數學函式（Bessel、Gamma、誤差函式等） | **Unit05 本單元** |
| `scipy.spatial` | 空間資料結構與幾何計算（KD-tree、Delaunay 三角化） | 延伸應用 |
| `scipy.io` | 資料讀寫（MATLAB `.mat`、NetCDF 格式） | 延伸應用 |

#### SciPy 基本使用模式
- **子模組載入方式**：`from scipy import linalg`、`from scipy.optimize import root_scalar` 等命名慣例
- **函式文件查詢方法**：`help()`、`?`、官方文件 `docs.scipy.org`
- **常見回傳值結構**：`OptimizeResult`（optimize 系列）、`OdeResult`（solve_ivp）、`BunchResult` 等
- **數值精度與容差參數**：`rtol`（相對容差）、`atol`（絕對容差）、`xtol`（根的精度）等設定原則

#### 化工領域應用分類導覽（含程式碼範例）

| 化工領域 | 主要 SciPy 工具 | 典型應用 |
|---------|----------------|---------|
| **熱力學計算** | `scipy.optimize` | Van der Waals、PR 狀態方程式求解；泡露點計算 |
| **反應工程** | `scipy.integrate`, `scipy.optimize` | 反應動力學 ODE 求解；速率參數 `curve_fit()` 估計 |
| **傳輸現象** | `scipy.integrate`, `scipy.sparse` | 非穩態熱傳/質傳 PDE；大型差分矩陣求解 |
| **程序控制** | `scipy.signal` | 頻率響應分析（Bode 圖）；系統識別與模擬 |
| **數據分析** | `scipy.stats`, `scipy.optimize` | 製程統計分析（SPC）；非線性模型擬合 |

---

### 2️⃣ Unit05_Special_Functions — 化工特殊數學函式

**檔案**：
- 講義：[Unit05_Special_Functions.md](Unit05_Special_Functions.md)
- 程式範例：[Unit05_Special_Functions.ipynb](Unit05_Special_Functions.ipynb)

**內容重點**：

本講義以化工八大學科為分類架構，系統化整理 `scipy.special` 中與化工應用相關的特殊函式：

#### 🔧 2.1 單操（Unit Operations）— 分離、過濾、沉降、吸收的數學支援

| 特殊函式 | 應用場景 |
|---------|---------|
| `gammainc(k, x)` | 連續反應器穿透曲線 F(t)；RTD 分析 |
| `gammaincinv(k, p)` | 由目標穿透比例反推操作時間 |
| `beta(a, b)` / `betaln(a, b)` | 效率分布函式建模（分離效率、Rosin-Rammler 粒徑分布） |
| `betainc(a, b, x)` | 不均一分離效率的累積分布 |
| `erf(x)` / `erfc(x)` | 層析分離前緣展寬（穿透曲線） |
| `exp1(x)` | 停留時間分布計算（指數積分） |
| `logsumexp(a)` | 多機制穿透或多成分 RTD 混合模型 |

#### 🌡️ 2.2 傳輸現象（Transport Phenomena）— 熱、質、動量傳遞

| 特殊函式 | 應用場景 |
|---------|---------|
| `erf(x)` / `erfc(x)` | 半無限介質非穩態熱傳/質傳（平板解析解） |
| `erfcx(x)` | 尾端濃度高精度計算（避免數值溢位） |
| `erfcinv(x)` | 由量測濃度反推穿透深度 |
| `jv(n, x)` / `j0(x)` / `j1(x)` | 圓柱座標熱傳/質傳 PDE 特徵值問題解析解 |
| `yv(n, x)` / `y0(x)` / `y1(x)` | 空心圓柱（管壁）熱傳，含 Y 型解 |
| `iv(n, x)` / `i0(x)` | 圓柱擴散含一階反應（修正 Bessel I 型） |
| `kv(n, x)` / `k0(x)` | 無限介質圓柱熱源衰減場解（修正 Bessel K 型） |
| `expi(x)` | 對流-擴散解析解（Ei 函式） |

#### ⚗️ 2.3 反應工程（Reaction Engineering）— 動力學、反應器設計、觸媒

| 特殊函式 | 應用場景 |
|---------|---------|
| `lambertw(z)` | 具解析解的動力學封閉解（Lambert W 函式） |
| `iv(1,φ)` / `iv(0,φ)` | 圓柱形觸媒顆粒效率因子（Thiele 模數） |
| `gammainc(a, kt)` | 累積轉化率 = $P(a, kt)$；剩餘轉化潛力 = $Q(a, kt)$ |
| `exprel(x)` | 小擾動動力學修正（$e^x-1)/x$ 數值穩定計算） |
| `xlogy(x, y)` | 動力學 log-likelihood（含零點穩定計算） |

#### 🌡️ 2.4 熱力學（Thermodynamics）— 工程熱力學、統計量

| 特殊函式 | 應用場景 |
|---------|---------|
| `logsumexp(a)` | 統計熱力學配分函數 $\ln Z$ 穩定計算 |
| `expm1(x)` | 小 $\Delta G$ 的指數差精確計算（自由能修正） |
| `gamma(n)` | 半整數 Gamma 積分常數（Stirling 近似相關） |
| `gammaln(n)` | 大參數 Gamma 比值（對數域計算，避免溢位） |
| `psi(x)` / `polygamma(n, x)` | $\ln\Gamma$ 的導數（Gibbs 自由能偏導）；Hessian 矩陣 |

#### 🔬 2.5 物化（Physical Chemistry）— 統計熱力學、量子、分布

| 特殊函式 | 應用場景 |
|---------|---------|
| `zeta(s, q)` / `zetac(s)` | Riemann zeta 函式（Debye 固體熱容模型） |
| `spence(x)` | Spence 函式（二聚體自由能計算） |
| `gamma(1.5)` | Maxwell-Boltzmann 速率分布 3D 積分常數 |
| `hyp2f1(a, b, c, z)` | Gauss 超幾何函式（分子軌道計算） |
| `erfc(x)` | Maxwell-Boltzmann 分布中超過臨界速率的分子分率 |

#### 🎛️ 2.6 程序控制（Process Control）— 系統辨識、噪聲模型、狀態估測

| 特殊函式 | 應用場景 |
|---------|---------|
| `logsumexp(a)` | 穩定 log-likelihood（混合模型 / IMM 濾波器） |
| `ndtr(x)` / `ndtri(x)` | 正態 CDF / 分位數（Kalman 更新、控制限設定） |
| `gammainc(a, x)` | Gamma 分布 CDF（到達事件機率建模） |
| `psi(x)` | Digamma 函式（Bayesian 估參 / MCMC 收斂分析） |

#### 📊 2.7 工程統計（Engineering Statistics）— 分布、估參、假設檢定

| 特殊函式 | 應用場景 |
|---------|---------|
| `ndtr(x)` / `ndtri(x)` | 正態分布 CDF 與分位數（製程能力指數 $C_p$, $C_{pk}$） |
| `erf(x)` | 正態分布 CDF（$\Phi(x) = \frac{1}{2}[1 + \mathrm{erf}(x/\sqrt{2})]$） |
| `gammainc(a, x)` | 卡方分布、Poisson 分布 CDF |
| `betainc(a, b, x)` | Beta 分布、F 分布 CDF（ANOVA 假設檢定） |
| `psi(x)` / `polygamma(n, x)` | Gamma/Beta 分布 MLE 估參（含 Hessian 不確定度） |

#### 📐 2.8 數值方法（Numerical Methods）— 譜法、展開、特殊函數解

| 特殊函式 | 應用場景 |
|---------|---------|
| `eval_legendre(n, x)` | Legendre 多項式（球座標 PDE 的角向基底函式） |
| `eval_chebyt(n, x)` / `eval_chebyu(n, x)` | Chebyshev 多項式（譜元素法、最小誤差近似） |
| `eval_hermite(n, x)` | Hermite 多項式（Gauss-Hermite 積分節點與權重） |
| `eval_laguerre(n, x)` | Laguerre 多項式（半無限域積分、量子化學） |
| `airy(x)` | Airy 函式（邊界層問題的特殊解） |
| `hyp1f1(a, b, x)` | 合流超幾何函式 1F1（Kummer 函式，含反應項 ODE） |
| `factorial(n)` / `comb(n, k)` | 組合計算（Taylor 展開係數） |

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit05_Scipy_Overview.md](Unit05_Scipy_Overview.md) | 📄 教學講義 | SciPy 生態系統概述、子模組功能、化工應用導覽 |
| [Unit05_Special_Functions.md](Unit05_Special_Functions.md) | 📄 教學講義 | 含 `scipy.special` 化工特殊數學函式完整介紹與程式範例 |
| [Unit05_Special_Functions.ipynb](Unit05_Special_Functions.ipynb) | 💻 程式演練 | 化工八大學科特殊函式的互動式程式範例 |
| [Unit05_Homework.ipynb](Unit05_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

> **注意**：`Unit05_Scipy_Overview` 為概念性總覽講義，不附程式演練 (`.ipynb`)。動手操作的程式範例集中在 `Unit05_Special_Functions.ipynb`。

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit04 Matplotlib 與 Seaborn 資料視覺化](../Unit04/README.md)**：資料視覺化工具、圖表製作、統計圖表

### ➡️ 下一單元
- **[Unit06 線性聯立方程式之求解](../Unit06/README.md)**：`scipy.linalg`、密集/稀疏矩陣求解、化工物料平衡

---

## 📈 本單元在課程中的定位

```
Unit03 (NumPy/Pandas)
Unit04 (Matplotlib)
      ↓
   Unit05 ← 你在這裡
 ┌──────────────────────────────────────────────┐
 │   SciPy 整體導覽（scipy.special 深度應用）    │
 │   建立 Unit06–Unit15 的共同先備知識框架       │
 └──────────────────────────────────────────────┘
      ↓
 ┌── Unit06 (scipy.linalg) ─ 線性方程組
 ├── Unit07 (scipy.optimize) ─ 非線性方程式
 ├── Unit08 (scipy.interpolate / scipy.integrate) ─ 插值積分
 ├── Unit09 (scipy.integrate) ─ ODE
 ├── Unit10 (py-pde / MoL) ─ PDE
 ├── Unit11 (scipy.fft) ─ 傅立葉轉換
 ├── Unit12 (scipy.optimize) ─ 程序最適化
 ├── Unit13 (scipy.optimize) ─ 參數估計
 ├── Unit14 (scipy.stats) ─ 統計分析
 └── Unit15 (scipy.signal) ─ 信號處理
```

---

## 💡 學習建議

1. **閱讀順序**：
   - 先閱讀 `Unit05_Scipy_Overview.md`（約 45–60 分鐘）了解 SciPy 全貌
   - 再閱讀 `Unit05_Special_Functions.md` 並同步執行 `.ipynb`（約 60–90 分鐘）

2. **重點章節**：
   - 不熟悉 SciPy 架構 → 重點閱讀 **SciPy 子模組功能對照表**
   - 要解化工 ODE 問題 → 提前預習 **`scipy.integrate` 小節**（Unit09 先備）
   - 要做統計分析 → 提前預習 **`scipy.stats` 小節**（Unit14 先備）

3. **特殊函式使用訣竅**：
   - 不知道用哪個函式 → 先查 `Unit05_Special_Functions.md` 的化工學科分類表
   - 懷疑計算結果 → 使用各章節的「驗證程式碼」交叉比對

4. **參考外部資源**：
   - [SciPy 官方文件](https://docs.scipy.org/doc/scipy/)：最權威的參考資料
   - [scipy.special 函式清單](https://docs.scipy.org/doc/scipy/reference/special.html)：完整函式索引
   - [SciPy Lectures](https://scipy-lectures.org/)：深入淺出的 SciPy 教學電子書

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit05 SciPy 科學運算套件應用概述
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
