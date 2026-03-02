# Unit12 程序最適化 (Process Optimization)

## 📚 單元簡介

化工程序設計與操作中，**「最適化（Optimization）」**無所不在——管徑選擇、操作溫度決策、原料配比規劃、整廠獲利最大化……幾乎所有工程決策都可以用**最適化問題**的數學框架來描述和求解。本單元系統性地介紹從單變數到多變數、從無約束到有約束、從局部最適到全域最適的完整最適化工具體系。

本單元以 **`scipy.optimize`** 的最適化模組為核心，涵蓋 `minimize_scalar()`（單變數）、`minimize()`（多變數，含 SLSQP 有約束求解）、`linprog()`（線性規劃）、`milp()`（混合整數規劃）以及 `differential_evolution()`、`dual_annealing()` 等全域最適化演算法。透過六個化工實際案例，從晶圓製程膜厚最佳化、生產流程獲利最大化（線性規劃）、化學平衡 Gibbs 自由能最小化、烷化程序複雜非線性約束最適化、管型反應器動態最適化，到差分進化全域最適化比較，完整呈現最適化工具在化工設計與操作中的應用。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **識別並建立化工最適化問題的數學模型**：正確定義目標函數、決策變數、等式/不等式/邊界限制條件，區分線性 vs 非線性、局部 vs 全域最適化問題
2. **熟練使用 `scipy.optimize` 最適化工具**：依問題類型選用 `minimize_scalar()`、`minimize()`、`linprog()`、`milp()`，設定 `bounds` 與 `constraints` 參數
3. **解決有限制條件的非線性規劃（NLP）問題**：以 `SLSQP` 或 `trust-constr` 求解含等式/不等式限制的化工設計問題
4. **認識並應用全域最適化策略**：理解局部最適化的限制，使用 `differential_evolution()` 或 `dual_annealing()` 避免陷入局部最小值
5. **整合最適化與其他數值工具**：結合 `scipy.integrate.solve_ivp()` 求解動態最適化問題（最適操作溫度分布），掌握跨單元的工具整合能力

---

## 📖 單元內容架構

### 1️⃣ Unit12_Optimization — 程序最適化理論與工具總覽

**檔案**：
- 講義：[Unit12_Optimization.md](Unit12_Optimization.md)
- 程式範例：[Unit12_Optimization.ipynb](Unit12_Optimization.ipynb)

**內容重點**：

#### 最適化問題分類

| 分類維度 | 類型 | 求解工具 |
|---------|------|---------|
| **變數數量** | 單變數 / 多變數 | `minimize_scalar()` / `minimize()` |
| **目標函數** | 線性 / 非線性 | `linprog()` / `minimize()` |
| **限制條件** | 無約束 / 有約束 | `minimize(method='BFGS')` / `SLSQP` |
| **變數性質** | 連續 / 整數 | `linprog()` / `milp()` |
| **解的範圍** | 局部 / 全域 | `minimize()` / `differential_evolution()` |

**通用最適化問題標準型**（NLP）：

$$
\min_{\mathbf{x}} f(\mathbf{x})
$$

$$
\text{s.t.} \quad \mathbf{c}(\mathbf{x}) \leq 0, \quad \mathbf{c}_{eq}(\mathbf{x}) = 0, \quad \mathbf{x}_L \leq \mathbf{x} \leq \mathbf{x}_U
$$

> ⚠️ **SciPy 不等式符號約定**：`scipy.optimize` 的 `'ineq'` 限制約定為 $c(\mathbf{x}) \geq 0$，與 MATLAB 的 $c(\mathbf{x}) \leq 0$ **相反**，使用時需特別注意符號轉換。

#### `scipy.optimize` 最適化工具總覽

| 函式 | 用途 | 對應 MATLAB 指令 |
|------|------|----------------|
| `minimize_scalar(method='bounded')` | 單變數有界最適化 | `fminbnd` |
| `minimize(method='BFGS')` | 無約束多變數最適化 | `fminunc` |
| `minimize(method='Nelder-Mead')` | 無導數多變數最適化 | `fminsearch` |
| `minimize(method='SLSQP')` | 有約束非線性規劃（NLP） | `fmincon` |
| `minimize(method='trust-constr')` | 有約束 NLP（新式推薦） | `fmincon` |
| `linprog()` | 線性規劃（LP） | `linprog` |
| `milp()` | 混合整數線性規劃（MILP） | `intlinprog` |
| `differential_evolution()` | 差分進化全域最適化 | GA（Real-Coded） |
| `dual_annealing()` | 模擬退火全域最適化 | Simulated Annealing |
| `basinhopping()` | 盆地跳躍全域最適化 | — |
| `shgo()` | 簡單同調全域最適化 | — |

#### 多變數最適化求解器比較（`minimize()` 方法）

| 方法 | 需要導數 | 特點 | 適用問題 |
|------|---------|------|---------|
| `Nelder-Mead` | 否 | 無導數，Simplex 法 | 不連續、不可微函數 |
| `Powell` | 否 | 共軛方向法 | 光滑但難以求導 |
| `BFGS` | 一階（可數值近似）| 擬牛頓，收斂快 | **一般用途推薦** |
| `L-BFGS-B` | 一階 | 有限記憶 BFGS | 大型問題、含邊界限制 |
| `SLSQP` | 一階 | 序列二次規劃 | **有約束 NLP 推薦** |
| `trust-constr` | 一階 | Trust-Region | **SciPy 新式推薦** |

#### 全域最適化方法比較

| 方法 | 原理 | 需設定邊界 | 支援約束 | 建議場景 |
|------|------|----------|---------|---------|
| `differential_evolution()` | 差分進化（族群搜尋）| ✅ 必要 | ✅ | 多峰函數，主要推薦 |
| `dual_annealing()` | 雙退火（隨機跳躍）| ✅ 必要 | 有限 | 單目標全域搜尋 |
| `basinhopping()` | 盆地跳躍 | 否（需合理 x0）| 否 | 搭配局部最適化 |
| `shgo()` | 簡單同調 | ✅ 必要 | ✅ | 小型複雜問題 |

#### 方法選擇決策流程

```
目標函數是否線性?
├─ 是 → 是否含整數變數?
│       ├─ 是 → milp()
│       └─ 否 → linprog()
└─ 否 → 是否有限制條件?
         ├─ 是 → minimize(method='SLSQP') 或 'trust-constr'
         └─ 否 → minimize_scalar()（單變數）
                 minimize(method='BFGS')（多變數）
         是否需要全域最適?
         └─ 是 → differential_evolution() 或 dual_annealing()
```

---

## 🧪 化工案例演練

### 💡 Example 01 — 晶圓製程最佳光阻劑膜厚

**檔案**：[Unit12_Example_01.md](Unit12_Example_01.md) | [Unit12_Example_01.ipynb](Unit12_Example_01.ipynb)

**問題概述**：晶圓製程中，光阻劑膜厚 $d$ 同時影響缺損率（過薄→缺損增加）與產量（過厚→產量下降），決定使每小時完好晶片產量 $f(d)$ 最大的最佳膜厚。

**數學模型**：單變數最適化（`minimize_scalar`）

$$
\max f = 100 V(d) \cdot \eta(d) \quad \Leftrightarrow \quad \min [-f(d)], \quad d \in [0.5, 2.5]
$$

$$
V = 125 - 50d + 5d^2, \quad D_0 = 1.5 d^{-3}, \quad \eta = \frac{1}{(1 + \beta D_0 a)^4}
$$

**化工重點**：
- `minimize_scalar(method='bounded', bounds=(0.5, 2.5))` 求解單峰最佳化問題
- 比較 Brent 法與 Golden Section Search 法的收斂速度
- 繪製目標函數曲線，標示最佳膜厚位置
- 敏感度分析：參數 $\beta$（製程因子）與 $a$（晶片面積）對最佳膜厚的影響

---

### 🏭 Example 02 — 生產流程最大獲利之操作條件（線性規劃）

**檔案**：[Unit12_Example_02.md](Unit12_Example_02.md) | [Unit12_Example_02.ipynb](Unit12_Example_02.ipynb)

**問題概述**：某化學工廠以有限量原料 A、B、C（每日最大供應量限制）生產三種產品 E、F、G。建立線性目標函數（每日總獲利 = 收入 + 原料成本 + 操作成本）並以線性規劃求最佳操作條件。

**數學模型**：線性規劃（`linprog`）

$$
\max \mathbf{c}^T \mathbf{x} \equiv \min (-\mathbf{c}^T \mathbf{x}), \quad \text{s.t.} \quad \mathbf{A}_{eq} \mathbf{x} = \mathbf{b}_{eq}, \quad \mathbf{x}_L \leq \mathbf{x} \leq \mathbf{x}_U
$$

**化工重點**：
- 建立線性目標函數（獲利函數線性化）與等式限制（質量平衡）
- `linprog()` 使用 HiGHS 求解器，最大化問題目標係數取負號
- 解讀最適化結果：各原料最佳用量、各產品最佳產量、每日最大獲利
- 敏感度分析：原料供應量變化或售價波動對最大獲利的影響（影子價格 Shadow Price）

---

### ⚖️ Example 03 — 化學平衡 Gibbs 自由能最小化

**檔案**：[Unit12_Example_03.md](Unit12_Example_03.md) | [Unit12_Example_03.ipynb](Unit12_Example_03.ipynb)

**問題概述**：十成分化學平衡計算，以最小化 Gibbs 自由能的方法取代傳統平衡常數法，受制於原子平衡等式限制與各成分莫耳數非負邊界。

**數學模型**：有等式限制的非線性最適化（`minimize(method='SLSQP')`）

$$
\min f(\mathbf{x}) = \sum_{i=1}^{10} x_i \left(w_i + \ln P + \ln\frac{x_i}{\sum_j x_j}\right)
$$

$$
\text{s.t.} \quad \mathbf{A}_{eq} \mathbf{x} = \mathbf{b}_{eq} \text{（原子平衡）}, \quad x_i \geq \varepsilon > 0
$$

**化工重點**：
- `bounds` 設定各成分莫耳數下限取 `np.finfo(float).eps`（避免 $\ln 0$ 數值錯誤）
- `constraints` 字典定義多條等式限制（各元素原子守恆）
- 驗證解的物理合理性：原子平衡殘差、莫耳分率總和為 1
- 不同起始猜測值對收斂結果的影響（凸 vs 非凸函數的差異）
- 繪製十成分平衡濃度分布柱狀圖

---

### 🏗️ Example 04 — 烷化程序最大獲利（複雜非線性約束 NLP）

**檔案**：[Unit12_Example_04.md](Unit12_Example_04.md) | [Unit12_Example_04.ipynb](Unit12_Example_04.ipynb)

**問題概述**：烷化（Alkylation）程序含 10 個決策變數（烯烴進料量、異丁烷回流量、酸液添加率、產品產量、辛烷值等），同時含非線性等式限制（酸強度與流量關係）、非線性不等式限制（產量與辛烷值迴歸關係的上下限）及線性操作範圍限制。

**數學模型**：有非線性等式 + 不等式限制的 NLP

$$
\max f = c_1 x_4 x_7 - c_2 x_1 - c_3 x_2 - c_4 x_3 - c_5 x_5
$$

**化工重點**：
- 同時使用 `Bounds` 物件（10 個變數上下限）與多個 `constraints` 字典（等式 + 不等式限制混合）
- `minimize(method='SLSQP')` 求解複雜 NLP（本單元最具挑戰性的局部最適化案例）
- 驗證所有限制條件的滿足度（殘差檢查，確認約束有效）
- 探討不同起始猜測值對收斂結果的穩定性（局部最適化的陷阱）
- 報告最大獲利與各操作變數的最佳值

---

### 🔥 Example 05 — 管型反應器最佳操作溫度分布（動態最適化）

**檔案**：[Unit12_Example_05.md](Unit12_Example_05.md) | [Unit12_Example_05.ipynb](Unit12_Example_05.ipynb)

**問題概述**：管型反應器中連串反應 $A \xrightarrow{k_1} B \xrightarrow{k_3} C$，各反應速率常數隨溫度呈 Arrhenius 關係。將反應時間分割為 $n$ 段，以各段溫度為決策變數，求最大化中間產物 B 的最佳溫度分布。

**數學模型**：動態最適化（ODE 約束 + SLSQP，整合 Unit09 工具）

$$
\max x_2(t_f), \quad \text{s.t.} \quad \dot{x}_1 = -k_1 x_1^2 + k_2 x_2, \quad \dot{x}_2 = k_1 x_1^2 - (k_2+k_3) x_2
$$

$$
k_i = k_{i0} \exp\left(-\frac{E_i}{RT}\right), \quad T_{min} \leq T_j \leq T_{max}
$$

**化工重點**：
- **動態最適化策略**：將連續溫度分布離散為 $n$ 段均一溫度（階梯型控制）
- 目標函數：對每組溫度序列 $\{T_j\}$，以 `scipy.integrate.solve_ivp()` 積分 ODE → 取 $-x_2(t_f)$
- `minimize(method='SLSQP')` 搭配溫度邊界限制（Bounds），迭代尋找最佳溫度分布
- **與 Unit09 的整合**：ODE 求解器作為最適化的內層計算工具
- 探討時間分割段數 $n$ 對最佳解品質的影響（$n$ 越大越接近真實連續最佳解）
- 繪製系統狀態 ($x_1$, $x_2$) 時間響應圖與最佳操作溫度分布階梯圖

---

### 🌍 Example 06 — 全域最適化與差分進化演算法比較

**檔案**：[Unit12_Example_06.md](Unit12_Example_06.md) | [Unit12_Example_06.ipynb](Unit12_Example_06.ipynb)

**問題概述**：含五個決策變數的非線性約束最適化問題，目標函數為 Miele-Cantrell 型多峰非線性函數，受制於非線性不等式限制組（上下限型）。比較局部最適化（SLSQP）與全域最適化（差分進化、雙退火）的結果差異。

**數學模型**：非線性約束多峰最適化

$$
f(\mathbf{x}) = 5.3578547 x_3^2 + 0.8356891 x_1 x_5 + 37.293239 x_1 - 40792.141
$$

**化工重點**：
- **方法一（局部 SLSQP）**：結果取決於起始猜測值，不同初始點得到不同局部解
- **方法二（差分進化）**：`differential_evolution()` 族群搜尋，突破局部最小值陷阱：
  - 關鍵參數：`popsize`、`maxiter`、`mutation`、`recombination`、`seed`
  - 設定 `constraints=NonlinearConstraint(...)` 處理非線性約束
- **方法三（雙退火）**：`dual_annealing()` 與差分進化結果對比
- 比較三種方法的目標函數值（全域最小值的接近程度）與計算時間
- 驗證所有限制條件的滿足度
- 繪製差分進化收斂曲線圖（目標函數值 vs 迭代次數）

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit12_Optimization.md](Unit12_Optimization.md) | 📄 教學講義 | 最適化理論、`scipy.optimize` 工具總覽、方法選擇決策流程 |
| [Unit12_Optimization.ipynb](Unit12_Optimization.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit12_Example_01.md](Unit12_Example_01.md) | 📄 案例講義 | 晶圓膜厚最適化（單變數，`minimize_scalar`） |
| [Unit12_Example_01.ipynb](Unit12_Example_01.ipynb) | 💻 程式演練 | 晶圓膜厚最適化實作 |
| [Unit12_Example_02.md](Unit12_Example_02.md) | 📄 案例講義 | 生產獲利最大化（線性規劃，`linprog`） |
| [Unit12_Example_02.ipynb](Unit12_Example_02.ipynb) | 💻 程式演練 | 線性規劃實作 |
| [Unit12_Example_03.md](Unit12_Example_03.md) | 📄 案例講義 | 化學平衡 Gibbs 自由能最小化（有等式約束 NLP） |
| [Unit12_Example_03.ipynb](Unit12_Example_03.ipynb) | 💻 程式演練 | Gibbs 自由能最適化實作 |
| [Unit12_Example_04.md](Unit12_Example_04.md) | 📄 案例講義 | 烷化程序獲利最大化（複雜非線性約束 NLP） |
| [Unit12_Example_04.ipynb](Unit12_Example_04.ipynb) | 💻 程式演練 | 烷化程序最適化實作 |
| [Unit12_Example_05.md](Unit12_Example_05.md) | 📄 案例講義 | 管型反應器最佳溫度分布（動態最適化，ODE + SLSQP） |
| [Unit12_Example_05.ipynb](Unit12_Example_05.ipynb) | 💻 程式演練 | 動態最適化實作 |
| [Unit12_Example_06.md](Unit12_Example_06.md) | 📄 案例講義 | 全域最適化比較（差分進化 vs 雙退火 vs SLSQP） |
| [Unit12_Example_06.ipynb](Unit12_Example_06.ipynb) | 💻 程式演練 | 全域最適化實作與方法比較 |
| [Unit12_Homework.ipynb](Unit12_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit11 傅立葉轉換與頻譜分析](../Unit11/README.md)**：`scipy.fft`、FFT、PSD、STFT、化工訊號頻譜診斷

### ➡️ 下一單元
- **[Unit13 參數估計](../Unit13/README.md)**：`scipy.optimize.curve_fit()`、`least_squares()`、線性/非線性最小平方法、置信區間

---

## 📈 本單元在課程中的定位

```
Unit11 (scipy.fft) — 傅立葉轉換與頻譜分析
      ↓
   Unit12 ← 你在這裡
 ┌──────────────────────────────────────────────────────────────────┐
 │  程序最適化 (scipy.optimize)                                      │
 │  局部最適化：minimize_scalar / minimize(BFGS / SLSQP / trust-constr)│
 │  線性規劃：linprog() / milp()（整數規劃）                          │
 │  全域最適化：differential_evolution / dual_annealing / shgo       │
 │  動態最適化：solve_ivp() + minimize() 整合應用                    │
 └──────────────────────────────────────────────────────────────────┘
      ↓
 Unit13 (scipy.optimize.curve_fit / least_squares) — 參數估計
      ↓
 Unit14 (scipy.stats) — 統計分析
      ↓
 ...（後續各應用單元）
```

**與化工問題的對應**：
```
操作條件單一決策（膜厚、流量）     → 單變數最適化 minimize_scalar() (Unit12 Ex01)
生產計畫、原料配比（線性關係）     → 線性規劃 linprog() (Unit12 Ex02)
化學平衡（Gibbs 自由能最小化）     → 有約束 NLP，SLSQP (Unit12 Ex03)
複雜製程設計（多變數多約束）       → 有約束 NLP，SLSQP (Unit12 Ex04)
反應器最佳操作策略（動態控制）     → 動態最適化 ODE + SLSQP (Unit12 Ex05)
多峰問題（全廠全域最佳化）         → 全域 DE / dual_annealing (Unit12 Ex06)
```

**Unit12 的重要橫向聯繫**：
- **與 Unit07 的連結**：Unit07 的 `scipy.optimize.root`/`fsolve` 求解方程式的根；Unit12 的 `minimize()` 求解目標函數的最小值，兩者都使用 `scipy.optimize`，但解題概念不同
- **與 Unit09 的連結**：Example 05（動態最適化）直接整合 `solve_ivp()` 作為最適化目標函數的計算工具
- **與 Unit13 的連結**：Unit13 的參數估計本質上是一種非線性最小平方最適化，使用 `curve_fit()` 與 `least_squares()` 是 Unit12 工具的專門化版本

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit12_Optimization.md`（約 60–90 分鐘）建立最適化問題分類與工具選擇框架
   - Step 2：執行 `Unit12_Optimization.ipynb` 熟悉各類最適化函式的基本用法
   - Step 3：依序完成 Example 01–06（由簡到難：單變數 → 線性規劃 → 有約束 NLP → 動態最適化 → 全域最適化）

2. **重點關注**：
   - **SciPy 不等式符號**：`'ineq'` 約定 $c(\mathbf{x}) \geq 0$（與 MATLAB 相反），是最常見的錯誤來源
   - **Example 03（Gibbs 自由能）**：`bounds` 下限為 `eps` 而非 `0`，避免對數函數發散（數值細節）
   - **Example 05（動態最適化）**：最適化 + ODE 整合是化工領域非常實用的進階技能
   - **Example 06（全域最適化）**：實際操作比較 SLSQP vs DE vs 雙退火，建立對「局部 vs 全域」差異的直覺

3. **求解器選擇速查**：

   | 情境 | 推薦工具 |
   |------|---------|
   | 單變數有界搜尋 | `minimize_scalar(method='bounded')` |
   | 無約束多變數（一般用途） | `minimize(method='BFGS')` |
   | 無約束多變數（函數不可微） | `minimize(method='Nelder-Mead')` |
   | 有約束 NLP（推薦） | `minimize(method='SLSQP')` |
   | 線性規劃 | `linprog(method='highs')` |
   | 混合整數線性規劃 | `milp()` |
   | 全域最適化（主要推薦） | `differential_evolution()` |
   | 全域最適化（替代） | `dual_annealing()` |

4. **常見錯誤提醒**：
   - `linprog()` 只求最小值，最大化問題必須將目標係數 `c` 取負號：`linprog(-c, ...)`
   - `constraints` 中 `'ineq'` 型限制的函數必須在可行解時返回**非負值**（$\geq 0$），不是 $\leq 0$
   - 全域最適化（DE、dual_annealing）每次執行結果可能略有不同（隨機性），建議設定 `seed` 確保可重現性

5. **參考外部資源**：
   - [SciPy optimize 官方文件](https://docs.scipy.org/doc/scipy/reference/optimize.html)
   - [SciPy minimize 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
   - [SciPy differential_evolution 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit12 程序最適化
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
