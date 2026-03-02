# Unit06 線性聯立方程式之求解 (Solving Systems of Linear Equations)

## 📚 單元簡介

線性聯立方程式是化工工程中最基礎且最廣泛應用的數學工具，物料平衡、能量平衡、穩態製程模擬等問題，幾乎都可歸結為求解 $\mathbf{Ax} = \mathbf{b}$ 的矩陣方程式。本單元以 **SciPy 線性代數工具（`scipy.linalg`）** 為核心，不使用 NumPy 的線性代數求解工具，系統性地介紹各類線性方程組的特性判斷、求解方法選擇，以及解的驗證策略。

透過一個主題講義與六個化工實際案例，本單元引導學生掌握從「建立物料平衡方程組」到「驗證求解結果」的完整計算流程，涵蓋液體摻合、蒸餾塔、反應器網絡、批次反應、多板吸收塔、含循環流製程等豐富的化工應用場景。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握線性方程組的理論基礎**：矩陣形式表示法、秩判定、三種系統類型（唯一解、無窮多解、無解）
2. **熟練使用 SciPy 線性代數工具**：`scipy.linalg.solve()`、`lu()`、`lu_factor()`/`lu_solve()`、`lstsq()`、`pinv()`、`block_diag()` 等
3. **處理大型稀疏矩陣**：`scipy.sparse.linalg.spsolve()`、`cg()`、`gmres()` 迭代求解器
4. **建立化工物料/能量平衡方程組**：將化工問題系統性轉化為 $\mathbf{Ax} = \mathbf{b}$ 矩陣形式
5. **驗證求解結果的正確性**：殘差計算、條件數評估、質量守恆檢查、物理合理性驗證

---

## 📖 單元內容架構

### 1️⃣ Unit06_Linear_Equations — 線性聯立方程式理論與求解工具

**檔案**：
- 講義：[Unit06_Linear_Equations.md](Unit06_Linear_Equations.md)
- 程式範例：[Unit06_Linear_Equations.ipynb](Unit06_Linear_Equations.ipynb)

**內容重點**：

#### 線性聯立方程式系統基礎
- **矩陣形式表示法**：$\mathbf{Ax} = \mathbf{b}$，其中 $\mathbf{A}$ 為係數矩陣，$\mathbf{x}$ 為未知數向量，$\mathbf{b}$ 為右端向量
- **解的存在性與唯一性**：Rouché–Capelli 定理，使用 `np.linalg.matrix_rank()` 進行秩判定
- **三種系統類型及其處理策略**：

  | 系統類型 | 判定條件 | 求解策略 |
  |---------|---------|---------|
  | **唯一解** | rank(**A**) = rank([**A**\|**b**]) = n | `scipy.linalg.solve()` |
  | **無窮多解（低確定）** | rank(**A**) = rank([**A**\|**b**]) < n | `scipy.linalg.pinv()` 求最小範數解 |
  | **無解（過確定）** | rank(**A**) < rank([**A**\|**b**]) | `scipy.linalg.lstsq()` 求最小平方解 |

- **病態系統（ill-conditioned）**：條件數 $\kappa(\mathbf{A})$ 分析與數值穩定性評估

#### SciPy 線性代數求解工具

| 函式 | 用途 | 適用場景 |
|------|------|---------|
| `scipy.linalg.solve(A, b)` | 一般密集矩陣求解 | 非奇異方陣，最常用 |
| `scipy.linalg.lu(A)` | LU 分解（P, L, U） | 學習矩陣分解原理 |
| `scipy.linalg.lu_factor(A)` + `lu_solve()` | 預分解 + 多次求解 | 相同係數矩陣，不同 **b** |
| `scipy.linalg.lstsq(A, b)` | 最小平方解 | 過確定系統（方程數 > 未知數） |
| `scipy.linalg.pinv(A)` | 虛擬反矩陣 | 低確定系統，取最小範數解 |
| `scipy.linalg.block_diag(*mats)` | 建立塊對角矩陣 | 獨立子系統合併求解 |
| `scipy.sparse.linalg.spsolve()` | 稀疏矩陣直接法 | 大型稀疏系統 |
| `scipy.sparse.linalg.cg()` | 共軛梯度迭代法 | 大型對稱正定稀疏系統 |
| `scipy.sparse.linalg.gmres()` | GMRES 迭代法 | 大型非對稱稀疏系統 |

#### 化工問題中的應用
- **物料平衡方程組**：摻合問題、分離程序、反應器網絡
- **能量平衡方程組**：熱交換器網絡、製程熱整合
- **穩態製程模擬**：循環流處理與收斂策略
- **解的物理意義驗證**：質量守恆、濃度範圍、流率非負

#### 程式設計最佳實踐
- **求解器選擇決策樹**：密集 vs 稀疏、直接法 vs 迭代法
- **數值精度評估**：條件數 $\kappa < 10^3$（良態）、$\kappa \approx 10^6$（警示）、$\kappa > 10^{12}$（病態）
- **殘差驗證標準**：$\|\mathbf{Ax} - \mathbf{b}\|_2$ 與 $\|\mathbf{Ax} - \mathbf{b}\|_2 / \|\mathbf{b}\|_2$ 的解釋

---

## 🧪 化工案例演練

### 📋 Example 01 — 液體摻合問題

**檔案**：[Unit06_Example_01.md](Unit06_Example_01.md) | [Unit06_Example_01.ipynb](Unit06_Example_01.ipynb)

**問題概述**：某工廠有 5 個儲存槽，各槽含有不同比例的 5 種成分（A、B、C、D、E）。需決定各槽的用量以摻合成 40 公升的指定組成產品。

**數學模型**：5×5 線性方程組（各成分體積平衡）

$$
\sum_{k=1}^{5} a_{jk} V_k = b_j, \quad j = A, B, C, D, E
$$

**化工重點**：
- 建構物料平衡係數矩陣與目標向量
- 秩判定確認唯一解存在（rank = 5）
- 條件數評估（$\kappa = 5.04$，良態系統）
- 驗證各槽用量均為正值（物理合理性）

**求解結果**：殘差達機器精度（$\sim 10^{-13}$），所有成分組成完全匹配目標。

---

### 🏭 Example 02 — 蒸餾塔組之成分分析

**檔案**：[Unit06_Example_02.md](Unit06_Example_02.md) | [Unit06_Example_02.ipynb](Unit06_Example_02.ipynb)

**問題概述**：三塔串聯蒸餾系統（第一塔塔頂送第二塔，塔底送第三塔）分離 para-xylene、styrene、toluene、benzene 四成分混合物。進料 70 mol/hr，求各出口股流流率及第一塔出口組成。

**數學模型**：4×4 線性方程組（各成分整體物料平衡）

$$
\mathbf{A}\begin{bmatrix}D_2\\B_2\\D_3\\B_3\end{bmatrix} = \mathbf{z} \cdot F
$$

**化工重點**：
- 多塔串聯系統的物料平衡建立方式
- 由出口股流流率推算中間股流（$D_1$、$B_1$）組成
- 分析條件數（$\kappa = 154.3$）與求解穩定性的關係

**求解結果**：D2 = 26.25、B2 = 17.50、D3 = 8.75、B3 = 17.50（mol/hr），總流率守恆 ✓

---

### ⚗️ Example 03 — 連續攪拌反應槽組之出口成分分析

**檔案**：[Unit06_Example_03.md](Unit06_Example_03.md) | [Unit06_Example_03.ipynb](Unit06_Example_03.ipynb)

**問題概述**：四槽 CSTR 反應器網絡（含回流），各槽進行液相一次不可逆反應 A → B。進料 1000 L/h（$C_{A0} = 1.0$ mol/L），求各槽出口濃度與整體轉化率。

**數學模型**：4×4 三對角（tridiagonal）線性方程組（各槽穩態物料平衡）

$$
(F_{in} + V_i k_i) C_{Ai} - \text{（相鄰槽流率貢獻）} = \text{進料項}
$$

**化工重點**：
- 含回流的反應器網絡物料平衡建立
- 三對角帶狀矩陣結構的識別（大型系統可用稀疏矩陣求解器）
- 整體轉化率的計算與各槽貢獻分析

**求解結果**：$C_{A1}=0.909$、$C_{A2}=0.697$、$C_{A3}=0.665$、$C_{A4}=0.586$（mol/L），整體轉化率 41.4%，與教科書參考值完全一致。

---

### 🔬 Example 04 — 批次反應系統之穩態成分分析

**檔案**：[Unit06_Example_04.md](Unit06_Example_04.md) | [Unit06_Example_04.ipynb](Unit06_Example_04.ipynb)

**問題概述**：封閉批次系統含六成分（A–F），各成分間發生 7 組可逆一次反應。從動力學 ODE 推導穩態齊次線性方程組，但係數矩陣奇異（rank = 5 < 6），需補充總莫耳數守恆約束求唯一解。

**數學模型**：6×6 線性方程組（穩態條件 $\dot{\mathbf{x}} = \mathbf{A}_{kin}\mathbf{x} = \mathbf{0}$ 加上守恆約束）

$$
\mathbf{A}_{kin}\mathbf{x}_{ss} = \mathbf{0} \quad \Rightarrow \quad \text{（奇異！rank = 5）} \quad \Rightarrow \quad \text{補充} \sum x_i = 2.0
$$

**化工重點**：
- **重要概念**：封閉系統動力學矩陣必然奇異（各行和為零 → 總莫耳守恆造成線性相依）
- 如何以物理守恆方程式取代線性相依方程式，使方程組有唯一解
- 特徵值分析識別零特徵值（對應守恆律）
- 此策略可推廣至管路網絡、混合槽組、電化學系統等奇異問題

**求解結果**：穩態濃度 $x_{ss} = [0.104, 0.174, 0.278, 0.556, 0.389, 0.500]$（mol/L），動力學殘差達機器精度（$\sim 10^{-17}$）。

---

### 🗼 Example 05 — 多成分吸收塔穩態成分分析

**檔案**：[Unit06_Example_05.md](Unit06_Example_05.md) | [Unit06_Example_05.ipynb](Unit06_Example_05.ipynb)

**問題概述**：4 個理論平衡板的多成分吸收塔，吸收三種成分（丙酮、乙醇、丙醇）。依 Henry 定律建立各板物料平衡，形成三對角線性方程組，使用 `scipy.linalg.block_diag()` 合併為 12×12 方程組，並與 Kremser 解析解驗證。

**數學模型**：每成分 4×4 三對角矩陣，合併為 12×12 block-diagonal 系統

$$
\underbrace{\begin{bmatrix}\mathbf{A}_A & & \\ & \mathbf{A}_B & \\ & & \mathbf{A}_C\end{bmatrix}}_{\mathbf{A}_{all} \in \mathbb{R}^{12\times12}} \mathbf{x} = \mathbf{b}
$$

**化工重點**：
- 吸收因子 $A_j = L/(Vm_j)$ 的物理意義：$A_j > 1$ 吸收佳、$A_j < 1$ 吸收差
- Block-diagonal 矩陣結構的識別與 `scipy.linalg.block_diag()` 的使用
- 數值解與 Kremser 解析解完全吻合（差值 = 0），驗證方程組建立的正確性

**求解結果**：丙酮吸收率 96.77%（$A=2.0$）、乙醇 80.00%（$A=1.0$）、丙醇 48.39%（$A=0.5$）。

---

### 🏗️ Example 06 — 含循環排放之完整化工製程穩態物料平衡

**檔案**：[Unit06_Example_06.md](Unit06_Example_06.md) | [Unit06_Example_06.ipynb](Unit06_Example_06.ipynb)

**問題概述**：甲醇生產製程，包含混合器（Mixer）、反應器（Reactor）、分離器（Separator）、分流器（Splitter）四個單元，共 7 個流股（含循環流 S7、排放流 S6）。建立 19×19 線性方程組求解各流股流率，並進行排放比例與轉化率的敏感度分析。

**數學模型**：19×19 線性方程組（19 個未知數：6 個未知流股 × 3 成分 + 反應程度 ξ）

| 單元 | 方程式條數 |
|:----:|:---------:|
| 混合器 | 3 |
| 反應器 | 4（含轉化率規格） |
| 分離器 | 6 |
| 分流器 | 6 |
| **合計** | **19** |

**化工重點**：
- 含循環流的複雜製程可直接整理為線性方程組（無需迭代），是線性方法的重要優勢
- 函式化設計（`build_system(p, X)`）使參數掃描輕鬆實現
- 敏感度分析：排放比例 $p \downarrow$ → 整體轉化率 $\uparrow$ 但循環流量急遽增大（trade-off）

**求解結果**（$X=0.5$, $p=0.2$）：整體轉化率 83.33%，循環流量 106.67 mol/s，排放流量 26.67 mol/s。

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit06_Linear_Equations.md](Unit06_Linear_Equations.md) | 📄 教學講義 | 線性聯立方程式理論、SciPy 工具、求解策略 |
| [Unit06_Linear_Equations.ipynb](Unit06_Linear_Equations.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit06_Example_01.md](Unit06_Example_01.md) | 📄 案例講義 | 液體摻合問題 |
| [Unit06_Example_01.ipynb](Unit06_Example_01.ipynb) | 💻 程式演練 | 液體摻合問題實作 |
| [Unit06_Example_02.md](Unit06_Example_02.md) | 📄 案例講義 | 蒸餾塔組成分分析 |
| [Unit06_Example_02.ipynb](Unit06_Example_02.ipynb) | 💻 程式演練 | 蒸餾塔組成分分析實作 |
| [Unit06_Example_03.md](Unit06_Example_03.md) | 📄 案例講義 | CSTR 反應器網絡物料平衡 |
| [Unit06_Example_03.ipynb](Unit06_Example_03.ipynb) | 💻 程式演練 | CSTR 反應器網絡實作 |
| [Unit06_Example_04.md](Unit06_Example_04.md) | 📄 案例講義 | 批次反應系統穩態成分分析（奇異矩陣處理） |
| [Unit06_Example_04.ipynb](Unit06_Example_04.ipynb) | 💻 程式演練 | 批次反應穩態分析實作 |
| [Unit06_Example_05.md](Unit06_Example_05.md) | 📄 案例講義 | 多成分吸收塔穩態分析（block-diagonal） |
| [Unit06_Example_05.ipynb](Unit06_Example_05.ipynb) | 💻 程式演練 | 多成分吸收塔實作 |
| [Unit06_Example_06.md](Unit06_Example_06.md) | 📄 案例講義 | 含循環流完整化工製程物料平衡（19×19） |
| [Unit06_Example_06.ipynb](Unit06_Example_06.ipynb) | 💻 程式演練 | 含循環流製程實作與敏感度分析 |
| [Unit06_Homework.ipynb](Unit06_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit05 SciPy 科學運算套件應用概述](../Unit05/README.md)**：SciPy 生態系統、`scipy.special` 特殊函式

### ➡️ 下一單元
- **[Unit07 非線性方程式之求解](../Unit07/README.md)**：`scipy.optimize`、Newton-Raphson、Brent 法、CSTR 多重穩態

---

## 📈 本單元在課程中的定位

```
Unit05 (SciPy 概述)
      ↓
   Unit06 ← 你在這裡
 ┌──────────────────────────────────────────────────┐
 │  線性聯立方程式 (scipy.linalg)                    │
 │  核心工具：solve / lu / lstsq / pinv             │
 │  稀疏工具：spsolve / cg / gmres                  │
 └──────────────────────────────────────────────────┘
      ↓
 Unit07 (scipy.optimize) ─ 非線性方程式
      ↓
 Unit08 (scipy.interpolate / scipy.integrate)
      ↓
 ...（後續各數值計算單元）
```

**與化工問題的對應**：
```
物料平衡          → 線性方程組 (Unit06)
狀態方程式         → 非線性方程式 (Unit07)
反應器動態        → ODE (Unit09)
熱傳/質傳場       → PDE (Unit10)
```

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit06_Linear_Equations.md`（約 45–60 分鐘）理解理論框架
   - Step 2：執行 `Unit06_Linear_Equations.ipynb` 熟悉各求解器用法
   - Step 3：依序閱讀並執行 Example 01–06（每個約 20–30 分鐘）

2. **重點關注**：
   - **秩判定**是每個案例的第一步，養成先判斷解的存在性再求解的習慣
   - **條件數評估**：$\kappa < 100$ 良態、$100 < \kappa < 10^6$ 可接受、$\kappa > 10^6$ 警示
   - **Example 04** 的奇異矩陣處理技巧是較特殊的概念，值得重點學習

3. **求解器選擇速查**：
   - 一般化工問題（< 1000 維）→ `scipy.linalg.solve()`
   - 蒸餾/吸收塔（三對角矩陣）→ `scipy.linalg.solve()` 或 `solve_banded()`
   - PDE 差分法（大型稀疏）→ `scipy.sparse.linalg.spsolve()` 或 `cg()`/`gmres()`
   - 過確定系統（實驗數據擬合）→ `scipy.linalg.lstsq()`

4. **參考外部資源**：
   - [SciPy linalg 官方文件](https://docs.scipy.org/doc/scipy/reference/linalg.html)
   - [SciPy sparse.linalg 官方文件](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit06 線性聯立方程式之求解
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
