# Unit05 SciPy 科學運算套件應用概述

## 學習目標

完成本單元後，學生將能夠：
- 理解 SciPy 生態系統的架構與其與 NumPy 的關係
- 掌握 SciPy 各主要子模組的功能與應用情境
- 熟悉 SciPy 子模組的載入方式與基本使用模式
- 了解數值精度、容差參數設定與常見回傳值結構
- 認識 SciPy 在化工領域各專業科目中的實際應用
- 建立 SciPy 課程學習地圖，掌握 Unit06–Unit15 的延伸方向
- 初步認識 `scipy.special` 特殊數學函式及其化工應用

---

## 1. SciPy 生態系統介紹

### 1.1 什麼是 SciPy？

SciPy（Scientific Python）是 Python 科學計算生態系統中最核心的套件之一，建立在 NumPy 的陣列運算基礎之上，提供大量高階的科學與工程計算功能。

**SciPy 的主要特點：**
- **建立在 NumPy 之上**：直接操作 NumPy 陣列，無縫整合
- **功能廣泛**：涵蓋線性代數、最佳化、積分、插值、統計、信號處理等
- **經過驗證**：許多演算法源自高品質的 Fortran/C 函式庫（如 LAPACK、FITPACK）
- **開源免費**：BSD 授權，社群活躍，持續維護
- **化工應用深厚**：可直接求解化工中的 ODE、代數方程、最佳化等問題

### 1.2 SciPy 與 NumPy 的關係

SciPy 與 NumPy 的關係可以用以下比喻來理解：

- **NumPy** = 提供基礎的多維陣列（`ndarray`）與基本數學運算，是科學計算的「大樓地基」
- **SciPy** = 在 NumPy 地基上構建的「高樓層」，提供針對特定科學領域的進階演算法

```
Python 科學計算生態系統架構：
┌─────────────────────────────────────────┐
│         應用層: matplotlib, pandas      │
├─────────────────────────────────────────┤
│         SciPy  (高階科學計算演算法)      │
├─────────────────────────────────────────┤
│         NumPy  (陣列運算、基礎數學)      │
├─────────────────────────────────────────┤
│ C/Fortran 函式庫: LAPACK, BLAS, FITPACK │
└─────────────────────────────────────────┘
```

**兩者的互補關係：**

| 功能 | NumPy | SciPy |
|------|-------|-------|
| 陣列操作 | ✅ 核心 | ✅ 繼承 |
| 基礎線性代數 | ✅ `np.linalg` | ✅ `scipy.linalg`（更完整） |
| 矩陣分解 | 部分 | ✅ 更多選項（LU, QR, SVD等） |
| 方程式求解 | ❌ | ✅ `scipy.optimize` |
| 數值積分 | ❌ | ✅ `scipy.integrate` |
| 統計分析 | 基礎 | ✅ `scipy.stats`（完整） |
| 特殊函式 | 基礎 | ✅ `scipy.special`（豐富） |

> **建議**：在本課程中，線性代數求解使用 `scipy.linalg` 而非 `numpy.linalg`，以獲得更完整的功能與更佳的數值穩定性。

### 1.3 安裝方式與版本管理

**使用 pip 安裝：**

```bash
pip install scipy
```

**使用 conda 安裝（推薦）：**

```bash
conda install scipy
```

**本課程環境（PY310）已包含 SciPy**，無需另行安裝。

**查詢 SciPy 版本：**

```python
import scipy
print(scipy.__version__)   # 查詢版本號

import numpy as np
print(np.__version__)      # 查詢 NumPy 版本
```

**典型輸出：**
```
1.14.0
2.1.0
```

> 本課程以 SciPy ≥ 1.7 為基礎，部分較新語法（如 `scipy.integrate.solve_ivp` 的進階選項）需要 ≥ 1.9。

### 1.4 SciPy 子模組整體架構總覽

SciPy 採用**子模組（submodule）**架構，將不同功能分類組織。以下是完整子模組列表與課程對應：

| 子模組 | 功能說明 | 本課程對應 |
|--------|---------|-----------|
| `scipy.linalg` | 線性代數（矩陣分解、特徵值、LU/QR/SVD 分解） | **Unit06** |
| `scipy.optimize` | 最適化與方程式求解（根求解、最小化、曲線擬合） | **Unit07, Unit12, Unit13** |
| `scipy.interpolate` | 插值法（線性、樣條、多維插值） | **Unit08** |
| `scipy.integrate` | 數值積分與 ODE/BVP 求解 | **Unit08, Unit09** |
| `scipy.sparse` | 稀疏矩陣結構與操作 | **Unit06, Unit10** |
| `scipy.sparse.linalg` | 大型稀疏方程組求解（共軛梯度、GMRES） | **Unit06, Unit10** |
| `scipy.fft` | 快速傅立葉轉換（FFT/IFFT、頻率分析） | **Unit11** |
| `scipy.stats` | 統計分析（機率分布、假設檢定、迴歸） | **Unit14** |
| `scipy.signal` | 信號處理（濾波、系統模型、頻域分析） | **Unit15** |
| `scipy.special` | 特殊數學函式（Bessel、Gamma、誤差函式等） | **Unit05（本單元）** |
| `scipy.spatial` | 空間資料結構（KD-tree、Delaunay、凸包） | 延伸應用 |
| `scipy.io` | 資料讀寫（MATLAB .mat、NetCDF 等格式） | 延伸應用 |

```python
import scipy
print(dir(scipy))  # 列出所有可用子模組
```

---

## 2. 主要子模組功能概述

本節逐一介紹 SciPy 各主要子模組的核心功能、關鍵函式，以及在化工領域的典型應用情境。

### 2.1 scipy.linalg — 線性代數（→ Unit06）

`scipy.linalg` 提供完整的線性代數運算功能，建議優先使用本模組而非 `numpy.linalg`，因為它直接調用底層的 LAPACK/BLAS 函式庫，功能更豐富、數值穩定性更好。

**核心函式：**

| 函式 | 功能說明 |
|------|---------|
| `scipy.linalg.solve()` | 求解線性方程組 $\mathbf{Ax} = \mathbf{b}$ |
| `scipy.linalg.lu()` | LU 分解 |
| `scipy.linalg.lu_factor()` / `lu_solve()` | 同一係數矩陣的多次求解 |
| `scipy.linalg.lstsq()` | 最小平方解（過確定系統） |
| `scipy.linalg.pinv()` | 虛擬反矩陣 |
| `scipy.linalg.eig()` | 特徵值與特徵向量 |
| `scipy.linalg.svd()` | 奇異值分解 |

**快速示例：**

```python
import numpy as np
from scipy import linalg

# 求解線性方程組 Ax = b
A = np.array([[3, 1], [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)

x = linalg.solve(A, b)
print("解向量 x =", x)
print("驗證 Ax =", A @ x)
```

**化工應用**：物料平衡方程組（摻合問題、反應器網絡）、能量平衡（熱交換器網絡）

---

### 2.2 scipy.optimize — 最適化與方程式求解（→ Unit07, Unit12, Unit13）

`scipy.optimize` 是 SciPy 中最豐富的子模組之一，提供方程式求解、函式極值尋找與曲線擬合等功能。

**核心函式：**

| 函式 | 功能說明 |
|------|---------|
| `root_scalar()` | 單變數方程式 $f(x) = 0$ 求解 |
| `root()` | 多變數非線性方程組 $\mathbf{F}(\mathbf{x}) = \mathbf{0}$ 求解 |
| `minimize_scalar()` | 單變數函式極小值 |
| `minimize()` | 多變數函式極小值（支援梯度、約束、邊界） |
| `curve_fit()` | 非線性曲線擬合（最小平方參數估計） |
| `linprog()` | 線性規劃 |
| `differential_evolution()` | 全域最佳化（差分演化） |

**快速示例：**

```python
from scipy import optimize
import numpy as np

# 求解 van der Waals 方程式的莫耳體積
# (P + a/V^2)(V - b) = RT
R, T, P = 8.314, 500, 2e6  # 單位: J/(mol·K), K, Pa
a, b = 0.3640, 4.267e-5     # CO2 van der Waals 常數

def vdw_eq(V):
    return (P + a/V**2) * (V - b) - R * T

result = optimize.root_scalar(vdw_eq, bracket=[b*1.01, 0.01], method='brentq')
print(f"莫耳體積 V = {result.root:.6f} m³/mol")
```

**化工應用**：狀態方程式求解（vdW/SRK/PR EOS）、相平衡計算、反應器設計最佳化、參數估計

---

### 2.3 scipy.interpolate — 插值法（→ Unit08）

`scipy.interpolate` 提供一維與多維插值功能，適用於實驗數據的內插與數值函式的平滑化。

**核心函式：**

| 函式 | 功能說明 |
|------|---------|
| `interp1d()` | 一維插值（線性、二次、三次樣條等） |
| `CubicSpline()` | 三次樣條插值（更精確的控制） |
| `RegularGridInterpolator()` | 規則網格多維插值 |
| `griddata()` | 非規則散點多維插值 |
| `splrep()` / `splev()` | B-Spline 擬合與求值 |

**快速示例：**

```python
import numpy as np
from scipy.interpolate import CubicSpline

# 物性數據：水的黏度 vs 溫度
T_data = np.array([20, 40, 60, 80, 100])       # °C
mu_data = np.array([1.002, 0.653, 0.467, 0.354, 0.282])  # mPa·s

cs = CubicSpline(T_data, mu_data)
print(f"50°C 時黏度 = {cs(50):.4f} mPa·s")
```

**化工應用**：物性數據插值（黏度、密度、熱導率 vs 溫度）、蒸餾操作線補插

---

### 2.4 scipy.integrate — 數值積分與 ODE 求解（→ Unit08, Unit09）

`scipy.integrate` 提供數值積分（定積分）與常微分方程式（ODE/BVP）的求解工具。

**核心函式：**

| 函式 | 功能說明 |
|------|---------|
| `quad()` | 一維定積分 $\int_a^b f(x)\,dx$ |
| `dblquad()` | 二維定積分 |
| `tplquad()` | 三維定積分 |
| `solve_ivp()` | 初始值問題 ODE 求解（現代介面） |
| `solve_bvp()` | 邊界值問題 BVP 求解 |
| `odeint()` | ODE 求解（傳統介面，仍常用） |

**快速示例：**

```python
from scipy import integrate
import numpy as np

# 求解批次反應器動力學 ODE: dCA/dt = -k * CA
k = 0.2   # min^-1
CA0 = 1.0  # mol/L

def batch_reactor(t, y):
    CA = y[0]
    dCAdt = -k * CA
    return [dCAdt]

t_span = (0, 20)
t_eval = np.linspace(0, 20, 100)
sol = integrate.solve_ivp(batch_reactor, t_span, [CA0], t_eval=t_eval)
print(f"20 min 時轉化率 = {(1 - sol.y[0, -1]/CA0)*100:.1f}%")
```

**化工應用**：反應動力學 ODE（批次反應器、PFR）、非穩態熱傳/質傳、控制系統模擬

---

### 2.5 scipy.sparse / scipy.sparse.linalg — 稀疏矩陣（→ Unit06, Unit10）

`scipy.sparse` 處理大型稀疏矩陣（大部分元素為零），可節省大量記憶體並加速計算。在 PDE 差分法或大型製程網絡中不可或缺。

**核心格式與函式：**

| 功能 | 說明 |
|------|------|
| `csr_matrix()` | 壓縮行格式（Compressed Sparse Row），適合矩陣-向量乘法 |
| `csc_matrix()` | 壓縮列格式，適合列切片操作 |
| `lil_matrix()` | 鏈結串列格式，適合逐元素賦值 |
| `spsolve()` | 直接求解稀疏線性方程組 |
| `cg()` | 共軛梯度法（大型對稱正定系統） |
| `gmres()` | GMRES 迭代法（大型非對稱系統） |

**快速示例：**

```python
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# 一維穩態熱傳差分方程（Tridiagonal 系統）
n = 5
diag_main = np.full(n, 2.0)
diag_off  = np.full(n-1, -1.0)
A_sparse = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csr')
b = np.zeros(n)
b[0] = 100.0  # 左邊界溫度（°C）
b[-1] = 0.0   # 右邊界溫度（°C）

T = spsolve(A_sparse, b)
print("穩態溫度分布 T =", T)
```

**化工應用**：大型物料平衡方程組、PDE 有限差分矩陣（Unit10）

---

### 2.6 scipy.fft — 快速傅立葉轉換（→ Unit11）

`scipy.fft` 提供 FFT 與 IFFT 運算，用於頻率分析與信號處理。化工製程中常用於振動分析、週期性製程雜訊識別。

**核心函式：**

| 函式 | 功能說明 |
|------|---------|
| `fft()` | 一維離散傅立葉轉換 |
| `ifft()` | 逆傅立葉轉換 |
| `fftfreq()` | 計算對應頻率軸 |
| `rfft()` | 實數輸入的 FFT（效率更高） |
| `fft2()` | 二維 FFT |

**化工應用**：製程信號的頻譜分析（壓力振動特徵）、設備故障診斷（FFT 特徵頻率）

---

### 2.7 scipy.stats — 統計分析（→ Unit14）

`scipy.stats` 提供完整的機率分布、假設檢定與描述統計工具，是化工製程數據分析與品質管制的重要工具。

**核心功能：**

| 功能分類 | 代表函式/類別 |
|---------|-------------|
| 連續分布 | `norm`, `t`, `chi2`, `f`, `expon`, `lognorm` 等 |
| 離散分布 | `binom`, `poisson`, `nbinom` 等 |
| 假設檢定 | `ttest_1samp()`, `ttest_ind()`, `f_oneway()`, `normaltest()` |
| 描述統計 | `describe()`, `skew()`, `kurtosis()` |
| 相關與迴歸 | `pearsonr()`, `spearmanr()`, `linregress()` |

**化工應用**：製程統計製程控制（SPC）、品質管制圖、實驗數據假設檢定、可靠度分析

---

### 2.8 scipy.signal — 信號處理（→ Unit15）

`scipy.signal` 提供數位/類比濾波器設計、系統模型與時域/頻域模擬，化工製程控制中的 Bode 圖分析即使用此模組。

**核心函式：**

| 函式 | 功能說明 |
|------|---------|
| `lti()` / `TransferFunction()` | 建立線性時不變系統模型 |
| `step()` / `impulse()` | 步階/脈衝響應 |
| `freqs()` / `freqz()` | 頻率響應（類比/數位） |
| `butter()` / `cheby1()` | 濾波器設計（Butterworth/Chebyshev） |
| `filtfilt()` | 零相位雙向濾波 |
| `bode()` | Bode 圖分析 |

**化工應用**：程序控制系統的頻率響應分析（Bode 圖）、工廠系統辨識、量測信號去雜訊

---

### 2.9 scipy.special — 特殊數學函式（→ Unit05 本單元重點）

`scipy.special` 收錄了大量無法用基礎函式封閉表示的特殊函式，這些函式廣泛出現在化工各領域的物理方程中。

**代表函式分類：**

| 類別 | 代表函式 | 化工應用 |
|------|---------|---------|
| 誤差函式 | `erf()`, `erfc()`, `erfinv()` | 非穩態質傳、熱傳滲透 |
| Gamma 函式 | `gamma()`, `gammaln()`, `gammainc()` | 停留時間分布（RTD） |
| Bessel 函式 | `j0()`, `j1()`, `jn()`, `kn()` | 圓柱坐標熱傳/質傳 |
| Beta 函式 | `beta()`, `betainc()` | 分布函數計算 |
| 超幾何函式 | `hyp2f1()`, `hyp1f1()` | 特殊 ODE 解析解 |

詳細內容請參閱本單元第 5 節與 `Unit05_Special_Functions.ipynb` 程式演練筆記本。

---

### 2.10 scipy.spatial — 空間資料結構（延伸應用）

提供 KD-tree 近鄰搜尋、Delaunay 三角化、Voronoi 圖與凸包等空間幾何計算，適用於粒子模擬、網格生成等進階應用。

---

### 2.11 scipy.io — 資料讀寫（延伸應用）

提供與 MATLAB `.mat`、NetCDF 等格式的資料讀寫介面，方便與其他科學計算軟體交換數據。

```python
from scipy.io import loadmat, savemat
import numpy as np

# 讀取 MATLAB 資料
data = loadmat('process_data.mat')
T_data = data['Temperature']

# 儲存為 MATLAB 格式
savemat('results.mat', {'concentration': np.array([0.5, 0.8, 1.2])})
```

---

## 3. SciPy 基本使用模式

### 3.1 子模組載入方式與命名慣例

SciPy 的子模組**不會隨 `import scipy` 自動載入**，需要明確匯入所需的子模組。

**建議的載入方式（推薦）：**

```python
# 方式一：from scipy import 子模組名稱（最常用）
from scipy import linalg
from scipy import optimize
from scipy import integrate
from scipy import interpolate
from scipy import stats
from scipy import special
from scipy import signal
from scipy import fft
from scipy.sparse import linalg as spla  # 避免與 scipy.linalg 命名衝突

# 方式二：import scipy.子模組 as 別名
import scipy.linalg as la
import scipy.optimize as opt
import scipy.integrate as integ

# 方式三：直接引入特定函式（適合只用少數函式時）
from scipy.linalg import solve, lu
from scipy.optimize import root_scalar, curve_fit
from scipy.integrate import solve_ivp, quad
```

**命名慣例總結：**

| 子模組 | 推薦別名 |
|--------|---------|
| `scipy.linalg` | `from scipy import linalg` |
| `scipy.optimize` | `from scipy import optimize` |
| `scipy.integrate` | `from scipy import integrate` |
| `scipy.interpolate` | `from scipy import interpolate` |
| `scipy.stats` | `from scipy import stats` |
| `scipy.special` | `from scipy import special` |
| `scipy.signal` | `from scipy import signal` |
| `scipy.fft` | `from scipy import fft` |
| `scipy.sparse.linalg` | `from scipy.sparse import linalg as spla` |

> **注意**：`scipy.sparse.linalg` 與 `scipy.linalg` 都有 `solve()` 等同名函式，載入時建議使用不同的別名加以區分。

---

### 3.2 函式文件查詢方法

在學習或使用 SciPy 函式時，有多種方式可以快速查詢函式說明。

**方法 1：使用 Python 內建 `help()`**

```python
from scipy import optimize
help(optimize.root_scalar)
```

**方法 2：在 Jupyter Notebook 中使用 `?`**

```python
from scipy.integrate import solve_ivp
solve_ivp?    # 顯示簡要說明
solve_ivp??   # 顯示完整原始碼
```

**方法 3：使用 `tab` 鍵自動補全**

```python
from scipy import linalg
linalg.  # 按 Tab 鍵，列出所有可用函式
```

**方法 4：官方線上文件**

- SciPy 官方文件：[https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/)
- SciPy API 參考：[https://docs.scipy.org/doc/scipy/reference/](https://docs.scipy.org/doc/scipy/reference/)

> **學習技巧**：遇到不熟悉的函式，養成先查看 `Parameters`（參數說明）、`Returns`（回傳值說明）與 `Examples`（使用範例）三個部分的習慣。

---

### 3.3 常見回傳值結構

SciPy 各子模組的回傳值結構不盡相同，以下列出常見的結構型態。

**類型一：`OptimizeResult`（scipy.optimize 系列）**

```python
from scipy import optimize

def f(x):
    return x**2 - 2*x - 3

result = optimize.root_scalar(f, bracket=[2, 5], method='brentq')

print(type(result))       # <class 'scipy.optimize.RootResults'>
print(result.root)        # 解值（float）
print(result.converged)   # 是否收斂（bool）
print(result.iterations)  # 迭代次數（int）
print(result.function_calls)  # 函式呼叫次數
```

**類型二：`OdeResult`（scipy.integrate.solve_ivp）**

```python
from scipy import integrate
import numpy as np

def ode(t, y):
    return [-0.5 * y[0]]

sol = integrate.solve_ivp(ode, [0, 10], [1.0], t_eval=np.linspace(0, 10, 50))

print(type(sol))     # <class 'scipy.integrate._ivp.ivp.OdeResult'>
print(sol.t)         # 時間點陣列
print(sol.y)         # 解的陣列（shape: [n_states, n_timepoints]）
print(sol.status)    # 求解狀態（0=成功）
print(sol.message)   # 求解訊息
print(sol.success)   # 是否成功（bool）
```

**類型三：元組（tuple）回傳**

許多函式回傳包含多個值的 `tuple`：

```python
from scipy import linalg, integrate
import numpy as np

# linalg.solve 直接回傳解向量
A = np.array([[2, 1], [1, 3]], dtype=float)
b = np.array([5, 10], dtype=float)
x = linalg.solve(A, b)    # 直接回傳解向量 ndarray

# quad 回傳 (積分值, 誤差估計)
result, error = integrate.quad(np.sin, 0, np.pi)
print(f"∫sin(x)dx [0,π] = {result:.6f}, 估計誤差 = {error:.2e}")

# LU 分解回傳 (P, L, U) 三個矩陣
P, L, U = linalg.lu(A)
print("P =\n", P)
print("L =\n", L)
print("U =\n", U)
```

---

### 3.4 數值精度與容差參數設定

數值方法的精度由容差（tolerance）參數控制。理解容差的含義有助於在精度與計算效率之間做出平衡。

**常用容差參數：**

| 參數名稱 | 含義 | 出現函式 |
|---------|------|---------|
| `tol` | 一般容差（絕對或相對） | `root_scalar()`, `root()` |
| `rtol` | 相對容差（Relative Tolerance）| `solve_ivp()` |
| `atol` | 絕對容差（Absolute Tolerance） | `solve_ivp()` |
| `xtol` | 根的迭代精度 | `root_scalar()` |
| `ftol` | 函式值的精度 | `minimize()` |
| `limit` | 積分子區間數上限 | `quad()` |

**容差概念說明：**

對於 ODE 求解器，誤差控制條件為：

$$
\text{誤差} \leq \mathrm{atol} + \mathrm{rtol} \times |y|
$$

其中 $y$ 是解的當前值。

**典型容差設定範例：**

```python
from scipy import integrate, optimize
import numpy as np

# ODE 求解：鬆散容差（快速）
sol_fast = integrate.solve_ivp(
    lambda t, y: [-0.5*y[0]],
    [0, 10], [1.0],
    rtol=1e-3, atol=1e-6   # 預設值
)

# ODE 求解：嚴格容差（精確）
sol_accurate = integrate.solve_ivp(
    lambda t, y: [-0.5*y[0]],
    [0, 10], [1.0],
    rtol=1e-8, atol=1e-10  # 更嚴格
)

print(f"快速版最終值: {sol_fast.y[0,-1]:.8f}")
print(f"精確版最終值: {sol_accurate.y[0,-1]:.8f}")

# 方程式求解：設定容差
result = optimize.root_scalar(
    lambda x: x**3 - 2,
    bracket=[1, 2],
    method='brentq',
    xtol=1e-12   # 要求根的精度達到 1e-12
)
print(f"∛2 = {result.root:.12f}")
```

> **化工實務建議**：一般工程計算使用 `rtol=1e-6` 至 `1e-8` 即已足夠；若需要高精度的物性計算或相平衡求解，可加嚴至 `rtol=1e-10`。過嚴的容差會顯著增加計算時間，應視需求調整。

---

## 4. 化工領域應用分類導覽

本節依化工專業科目分類，說明如何將 SciPy 各子模組應用於化工典型問題的求解。

### 4.1 熱力學計算

化工熱力學中許多問題需要求解隱含的代數方程式或系統方程組。

**典型應用：**
- 求解狀態方程式（Van der Waals、SRK、Peng-Robinson EOS）的莫耳體積
- 相平衡計算（泡點、露點溫度/壓力）
- 混合物 fugacity 計算

**使用的 SciPy 工具**：`scipy.optimize.root_scalar()`、`scipy.optimize.root()`

**PR EOS 示例：**

```python
from scipy import optimize
import numpy as np

# Peng-Robinson EOS 求解壓縮因子 Z
# Z^3 - (1-B)*Z^2 + (A-3B^2-2B)*Z - (AB-B^2-B^3) = 0
# A = aP/(RT)^2, B = bP/(RT)

R = 8.314  # J/(mol·K)
# CO2 at T=350K, P=7MPa
T, P = 350, 7e6
Tc, Pc, omega = 304.2, 7.39e6, 0.228

kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2
a = 0.45724 * R**2 * Tc**2 / Pc * alpha
b = 0.07780 * R * Tc / Pc
A = a*P/(R*T)**2
B = b*P/(R*T)

def pr_eos(Z):
    return Z**3 - (1-B)*Z**2 + (A-3*B**2-2*B)*Z - (A*B-B**2-B**3)

# 找氣相根（Z > B）
result = optimize.root_scalar(pr_eos, bracket=[B+0.001, 1.2], method='brentq')
Z_gas = result.root
print(f"氣相壓縮因子 Z = {Z_gas:.4f}")
print(f"莫耳體積 V = {Z_gas*R*T/P*1000:.4f} L/mol")
```

---

### 4.2 反應工程

反應工程中的動力學模型通常以 ODE 形式表示，SciPy 提供高效的 ODE 求解器。

**典型應用：**
- 批次反應器（Batch Reactor）動力學模擬
- 推流反應器（PFR）濃度/溫度分布計算
- 連續攪拌槽反應器（CSTR）多重穩態分析
- 反應速率參數估計（ $k$, $E_a$ 等）

**使用的 SciPy 工具**：`scipy.integrate.solve_ivp()`、`scipy.optimize.curve_fit()`、`scipy.optimize.root()`

**批次反應器 A → B，二級反應示例：**

```python
from scipy import integrate
import numpy as np

# 二級不可逆反應 A → B: -rA = k*CA^2
k = 0.05   # L/(mol·min)
CA0 = 2.0  # mol/L
CB0 = 0.0

def batch_2nd_order(t, y):
    CA, CB = y
    rA = k * CA**2
    return [-rA, rA]

t_span = (0, 60)
t_eval = np.linspace(0, 60, 200)
sol = integrate.solve_ivp(
    batch_2nd_order,
    t_span, [CA0, CB0],
    t_eval=t_eval,
    method='RK45', rtol=1e-8
)

# 計算轉化率
X_A = (CA0 - sol.y[0, -1]) / CA0
print(f"60 min 時 A 的轉化率 = {X_A*100:.1f}%")
print(f"60 min 時 CB = {sol.y[1, -1]:.4f} mol/L")
```

**速率參數估計示例：**

```python
from scipy import optimize
import numpy as np

# 實驗數據：時間 vs 濃度
t_exp = np.array([0, 5, 10, 20, 30, 45, 60])
CA_exp = np.array([2.0, 1.667, 1.429, 1.111, 0.909, 0.727, 0.606])

# 解析解: CA(t) = CA0/(1 + k*CA0*t)
def model(t, k):
    CA0 = 2.0
    return CA0 / (1 + k * CA0 * t)

popt, pcov = optimize.curve_fit(model, t_exp, CA_exp, p0=[0.05])
k_est = popt[0]
k_std = np.sqrt(pcov[0, 0])
print(f"估計速率常數 k = {k_est:.4f} ± {k_std:.4f} L/(mol·min)")
```

---

### 4.3 傳輸現象

傳輸現象（熱/質/動量傳遞）中的非穩態問題與多維問題需要積分、ODE 與稀疏矩陣工具。

**典型應用：**
- 非穩態熱傳（Fourier 方程）的有限差分模擬
- 球體/圓柱體中的非穩態質傳（Fick 方程）
- 管道流的邊界層問題（BVP）

**使用的 SciPy 工具**：`scipy.integrate.solve_ivp()`、`scipy.integrate.solve_bvp()`、`scipy.sparse.linalg.spsolve()`、`scipy.special`（Bessel 函式）

**非穩態熱傳解析解（使用 Bessel 函式）：**

以無限長圓柱體對流冷卻為例，解析解涉及 Bessel 函式 $J_0(\zeta_n r/R)$：

$$
\frac{T(r,t) - T_\infty}{T_0 - T_\infty} = \sum_{n=1}^{\infty} C_n \cdot J_0\!\left(\frac{\zeta_n r}{R}\right) \exp\!\left(-\zeta_n^2 \mathrm{Fo}\right)
$$

其中 $\mathrm{Fo} = \alpha t / R^2$ 為 Fourier 數，$\zeta_n$ 為特徵值。

```python
from scipy import special
import numpy as np

# 圓柱體中心溫度（r=0）的前 10 項級數解
def cylinder_center_temp(Fo, Bi, n_terms=10):
    """計算圓柱體中心無因次溫度"""
    theta = 0.0
    # 特徵方程: zeta * J1(zeta) = Bi * J0(zeta)
    from scipy.optimize import root_scalar
    for n in range(1, n_terms + 1):
        # 初步搜尋特徵值（第 n 個正根）
        left = (n - 1) * np.pi + 0.01
        right = n * np.pi - 0.01
        def char_eq(z):
            return z * special.j1(z) - Bi * special.j0(z)
        zeta_n = root_scalar(char_eq, bracket=[left, right]).root
        C_n = 2 / zeta_n * special.j1(zeta_n) / (special.j0(zeta_n)**2 + special.j1(zeta_n)**2)
        theta += C_n * special.j0(0) * np.exp(-zeta_n**2 * Fo)
    return theta

Bi = 5.0  # 畢歐數
Fo = 0.3  # 傅立葉數
theta = cylinder_center_temp(Fo, Bi)
print(f"Fo={Fo}, Bi={Bi} 時，中心無因次溫度 θ = {theta:.4f}")
```

---

### 4.4 程序控制

程序控制中的頻率響應分析與系統辨識是 `scipy.signal` 的核心應用。

**典型應用：**
- 傳遞函式模型的 Bode 圖繪製
- 系統步階響應的模擬
- PID 控制器設計與性能分析

**使用的 SciPy 工具**：`scipy.signal.TransferFunction()`、`scipy.signal.step()`、`scipy.signal.bode()`

**一階系統步階響應示例：**

```python
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# 一階加迟延系統傳遞函式：G(s) = Kp / (tau*s + 1)
Kp = 2.0    # 穩態增益
tau = 5.0   # 時間常數（min）

# 建立傳遞函式 G(s) = 2 / (5s + 1)
sys = signal.TransferFunction([Kp], [tau, 1])

# 計算步階響應
t, y = signal.step(sys, T=np.linspace(0, 30, 300))

# 繪圖
plt.figure(figsize=(10, 5))
plt.plot(t, Kp * y / y[-1], 'b-', linewidth=2, label='Step Response')
plt.axhline(y=Kp, color='r', linestyle='--', linewidth=1.5, label=f'Steady State = {Kp}')
plt.title('First-Order System Step Response', fontsize=14)
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
```

---

### 4.5 數據分析

在化工製程數據分析中，統計工具與曲線擬合是最常見的需求。

**典型應用：**
- 實驗數據的描述統計分析
- 製程能力指數（ $C_p$, $C_{pk}$ ）計算
- 正態性檢定、t 檢定
- 速率方程式、關聯式的參數擬合

**使用的 SciPy 工具**：`scipy.stats`、`scipy.optimize.curve_fit()`

```python
from scipy import stats
import numpy as np

# 模擬製程數據：產品純度（%）
np.random.seed(42)
purity = np.random.normal(loc=98.5, scale=0.8, size=100)

# 描述統計
desc = stats.describe(purity)
print(f"樣本數: {desc.nobs}")
print(f"平均值: {desc.mean:.3f}%")
print(f"變異數: {desc.variance:.4f}")
print(f"偏態: {desc.skewness:.4f}")
print(f"峰態: {desc.kurtosis:.4f}")

# 正態性檢定（Shapiro-Wilk）
stat, p_value = stats.shapiro(purity)
print(f"\nShapiro-Wilk 正態性檢定: p = {p_value:.4f}")
if p_value > 0.05:
    print("→ 無法拒絕正態假設（數據符合正態分布）")

# 製程能力指數
USL, LSL = 100.0, 97.0  # 規格上下限
mu, sigma = np.mean(purity), np.std(purity)
Cp = (USL - LSL) / (6 * sigma)
Cpk = min((USL - mu), (mu - LSL)) / (3 * sigma)
print(f"\nCp = {Cp:.3f}, Cpk = {Cpk:.3f}")
```

---

## 5. scipy.special — 特殊數學函式概述（本單元重點）

### 5.1 為什麼需要特殊函式？

在化工工程計算中，許多物理方程的解析解涉及「特殊函式」——這些函式沒有簡單的代數封閉式，但數學家已對其做了深入研究，建立了完整的數值計算方法。

**特殊函式出現的典型情境：**

- **非穩態熱傳/質傳**：圓柱/球體幾何中出現 **Bessel 函式**
- **停留時間分布（RTD）**：Gamma 分布的 CDF 使用**不完全 Gamma 函式**
- **質傳與滲透**：誤差函式 **erf** 描述半無限介質的濃度/溫度分布
- **統計分布**：正態分布的 CDF 與 **erf** 掛鉤
- **量子與統計熱力學**：Fermi-Dirac 分布使用**多重對數函式**

### 5.2 化工科目相關特殊函式分類

以下依化工各專業科目分類列出常用特殊函式：

#### 5.2.1 傳輸現象（熱傳、質傳、動量傳遞）

| 函式 | scipy.special 函式名稱 | 應用場合 |
|------|----------------------|---------|
| 誤差函式 $\mathrm{erf}(x)$ | `erf(x)` | 半無限介質非穩態熱/質傳（平板） |
| 互補誤差函式 $\mathrm{erfc}(x)$ | `erfc(x)` | 同上，補函式形式 |
| 第一類 Bessel 函式 $J_0(x)$, $J_1(x)$ | `j0(x)`, `j1(x)` | 圓柱/圓管非穩態熱傳解析解 |
| 第一類 Bessel 函式 $J_n(x)$ | `jn(n, x)` | 任意階 Bessel 函式 |
| 第二類修正 Bessel 函式 $K_0(x)$, $K_1(x)$ | `kn(0, x)`, `kn(1, x)` | 圓管壁稳態熱傳（絕熱外壁） |
| 第一類球 Bessel 函式 $j_n(x)$ | `spherical_jn(n, x)` | 球體幾何非穩態熱傳 |

**示例：半無限介質非穩態質傳**

$$
C(x, t) = C_s \cdot \mathrm{erfc}\!\left(\frac{x}{2\sqrt{Dt}}\right)
$$

```python
from scipy import special
import numpy as np

# 非穩態質傳：半無限介質中的濃度分布
D = 1e-9    # 擴散係數 (m^2/s)
Cs = 1.0    # 表面濃度 (mol/m^3)
x = np.linspace(0, 0.01, 100)  # 位置 (m)

for t in [60, 300, 1800]:  # 時間 (s)
    C = Cs * special.erfc(x / (2 * np.sqrt(D * t)))
    print(f"t={t}s, 1mm 處濃度 = {Cs * special.erfc(0.001/(2*np.sqrt(D*t))):.4f} mol/m³")
```

---

#### 5.2.2 反應工程（動力學、反應器設計、觸媒）

| 函式 | scipy.special 函式名稱 | 應用場合 |
|------|----------------------|---------|
| 第一類修正 Bessel 函式 $I_0(x)$, $I_1(x)$ | `iv(0, x)`, `iv(1, x)` | 球形/柱形觸媒顆粒內傳遞效率因子 |
| 誤差函式 `erf` | `erf(x)` | 觸媒孔道質傳分析 |
| 指數積分 $E_1(x)$ | `exp1(x)` | 反應器停留時間分布計算 |

**示例：球形觸媒的 Thiele 效率因子**

觸媒效率因子 $\eta$ 與 Thiele 模數 $\phi$ 的關係：

$$
\eta = \frac{3}{\phi^2}\left[\phi \coth(\phi) - 1\right]
$$

```python
from scipy import special
import numpy as np

def thiele_efficiency(phi):
    """計算球形觸媒的 Thiele 效率因子"""
    if phi < 1e-6:
        return 1.0
    return (3 / phi**2) * (phi / np.tanh(phi) - 1)

phi_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
for phi in phi_values:
    eta = thiele_efficiency(phi)
    print(f"φ = {phi:5.1f}, η = {eta:.4f}")
```

**示例：停留時間分布（RTD）— Gamma 分布**

$$
E(t) = \frac{t^{n-1} e^{-t/\tau}}{\tau^n \, \Gamma(n)}
$$

其中 $\Gamma(n)$ 為 Gamma 函式，CDF 用不完全 Gamma 函式表示：

```python
from scipy import special
import numpy as np

# n 個完全混合槽串聯的停留時間分布
n = 3.0      # 槽數（可為非整數）
tau = 2.0    # 平均停留時間 (min)
t = np.array([1, 2, 4, 6, 8.0])

# E(t) 機率密度函數
E_t = t**(n-1) * np.exp(-t/tau) / (tau**n * special.gamma(n))

# F(t) 累積分布函數（不完全 Gamma 函式的正規化形式）
F_t = special.gammainc(n, t/tau)

print("t(min)   E(t)      F(t)")
for ti, Ei, Fi in zip(t, E_t, F_t):
    print(f"{ti:6.1f}   {Ei:.4f}    {Fi:.4f}")
```

---

#### 5.2.3 單元操作（分離、過濾、吸收等）

| 函式 | scipy.special 函式名稱 | 應用場合 |
|------|----------------------|---------|
| 不完全 Beta 函式 $I_x(a,b)$ | `betainc(a, b, x)` | 分離效率分布建模 |
| Gamma 函式分位數反函式 | `gammaincinv(a, p)` | RTD 反推操作時間 |
| 誤差函式 | `erf(x)` | 吸附穿透曲線建模 |

**示例：穿透曲線（Breakthrough Curve）**

填充床吸附的穿透曲線可用互補誤差函式近似表示：

$$
\frac{C}{C_0} \approx \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{t - t_{1/2}}{\sigma\sqrt{2}}\right)
$$

```python
from scipy import special
import numpy as np

# 填充床吸附穿透曲線
t = np.linspace(0, 200, 300)  # 時間 (min)
t_half = 100.0   # 穿透時間 (t at C/C0 = 0.5)
sigma = 10.0     # 分散參數

C_over_C0 = 0.5 * special.erfc((t_half - t) / (sigma * np.sqrt(2)))

# 計算 10% 和 90% 穿透時間
t10 = t_half - sigma * np.sqrt(2) * special.erfinv(0.8)
t90 = t_half - sigma * np.sqrt(2) * special.erfinv(-0.8)
print(f"10% 穿透時間 t₁₀ = {t10:.1f} min")
print(f"90% 穿透時間 t₉₀ = {t90:.1f} min")
```

---

#### 5.2.4 工程統計（分布、推估、假設檢定）

`scipy.stats` 內部大量使用 `scipy.special` 的函式來計算機率分布的 CDF 與 PDF。了解其底層特殊函式有助於深入理解統計分布的數學本質。

| 統計分布 | 對應的特殊函式 |
|---------|-------------|
| 正態分布 CDF | $\Phi(x) = \frac{1}{2}\left[1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$ |
| 卡方分布 CDF | 不完全 Gamma 函式 $\gamma(k/2,\, x/2)$ |
| Beta 分布 CDF | 正規化不完全 Beta 函式 $I_x(a,b)$ |
| F 分布 CDF | 正規化不完全 Beta 函式 |

```python
from scipy import special
import numpy as np

# 正態分布 CDF 使用誤差函式（與 scipy.stats.norm.cdf 等價）
x = 1.645  # 對應 95% 單尾信賴水準
Phi_x = 0.5 * (1 + special.erf(x / np.sqrt(2)))
print(f"Φ({x}) = {Phi_x:.4f}  (應接近 0.9500)")

# 等同於：
from scipy import stats
print(f"stats.norm.cdf({x}) = {stats.norm.cdf(x):.4f}")
```

---

#### 5.2.5 數值方法（譜法、特殊函數解）

在使用正交多項式展開（如 Chebyshev、Legendre 展開）作為譜方法的基底函式時，`scipy.special` 提供了這些多項式的直接計算。

| 函式 | scipy.special 函式名稱 | 應用場合 |
|------|----------------------|---------|
| Legendre 多項式 $P_n(x)$ | `legendre(n)` | Gauss 積分權重/節點計算 |
| Chebyshev 多項式 $T_n(x)$ | `chebyt(n)` | 譜方法基底函式 |
| Laguerre 多項式 $L_n^k(x)$ | `genlaguerre(n, k)` | 無界域積分 |
| 階乘 $n!$ | `factorial(n)` | 組合計算 |
| 二項式係數 $\binom{n}{k}$ | `comb(n, k)` | 組合數計算 |

```python
from scipy import special
import numpy as np

# Gauss-Legendre 積分節點與權重
n_points = 5
x_nodes, w_weights = special.roots_legendre(n_points)
print("Gauss-Legendre 積分節點:", x_nodes)
print("對應權重:", w_weights)

# 驗證：∫sin(x)dx 從 0 到 π ≈ 2
# 需將 [-1,1] 轉換到 [0,π]
a, b = 0, np.pi
x_mapped = (b - a)/2 * x_nodes + (a + b)/2
integral = (b - a)/2 * np.sum(w_weights * np.sin(x_mapped))
print(f"\n∫sin(x)dx [0,π] ≈ {integral:.8f}  (精確值 = 2.00000000)")
```

---

### 5.3 特殊函式使用注意事項

**1. 數值溢位（Overflow）問題**

Gamma 函式 $\Gamma(n)$ 在 $n$ 較大時會極快增大，甚至超出浮點數範圍（Overflow）。此時應使用對數形式：

```python
from scipy import special

# 計算大數的 Gamma 函式（避免溢位）
n = 200  # Γ(200) 超出 float64 範圍
try:
    val = special.gamma(n)
    print(f"gamma(200) = {val}")  # 會顯示 inf
except:
    pass

# 使用對數形式（不會溢位）
log_val = special.gammaln(n)
print(f"ln(Γ(200)) = {log_val:.4f}")
print(f"Γ(200) = exp({log_val:.1f}) ≈ 10^{log_val/np.log(10):.0f}")
```

**2. 引數範圍檢查**

不完全 Gamma 函式 `gammainc(a, x)` 要求 $a > 0$ 且 $x \geq 0$，在使用前應驗證物理量的合理性。

**3. 版本差異**

部分函式名稱在新版 SciPy 中有所調整（如 `jn()` → `jv()`），建議查閱對應版本的官方文件。

---

## 6. 學習路徑與延伸資源

### 6.1 課程學習地圖

本課程以 SciPy 各子模組為主軸，從 Unit06 到 Unit15 依序深入各領域。以下學習地圖可幫助你掌握整體架構：

```
Unit05（本單元）
  ├─ SciPy 生態系統總覽
  ├─ 各子模組功能概述
  └─ scipy.special 特殊函式導論
         ↓
Unit06：scipy.linalg + scipy.sparse.linalg
  └─ 線性聯立方程式求解（物料平衡、能量平衡）
         ↓
Unit07：scipy.optimize（根求解）
  └─ 非線性方程式求解（狀態方程式、相平衡）
         ↓
Unit08：scipy.interpolate + scipy.integrate（數值積分）
  └─ 插值法、數值微分與積分
         ↓
Unit09：scipy.integrate（ODE）
  └─ 常微分方程式求解（反應動力學、熱傳）
         ↓
Unit10：scipy.sparse + PDE 求解
  └─ 偏微分方程式求解（有限差分法）
         ↓
Unit11：scipy.fft
  └─ 傅立葉轉換與頻譜分析
         ↓
Unit12：scipy.optimize（最佳化）
  └─ 程序最適化（線性規劃、非線性最佳化）
         ↓
Unit13：scipy.optimize（參數估計）
  └─ 曲線擬合、速率參數估計
         ↓
Unit14：scipy.stats
  └─ 統計分析（假設檢定、製程統計）
         ↓
Unit15：scipy.signal
  └─ 信號處理（濾波、程序控制、Bode 圖）
```

### 6.2 各單元前置知識對應

| 單元 | 主要 SciPy 模組 | 化工科目連結 | 前置知識要求 |
|------|---------------|------------|------------|
| Unit06 | `scipy.linalg` | 化工系統工程 | 線性代數基礎 |
| Unit07 | `scipy.optimize` | 熱力學、分離工程 | 非線性方程式概念 |
| Unit08 | `scipy.interpolate`, `scipy.integrate` | 傳質、化工數學 | 微積分 |
| Unit09 | `scipy.integrate` | 反應工程、傳輸現象 | ODE 基礎 |
| Unit10 | `scipy.sparse` | 傳輸現象、工程數學 | PDE 基礎 |
| Unit11 | `scipy.fft` | 程序控制 | 傅立葉分析概念 |
| Unit12 | `scipy.optimize` | 程序設計與分析 | 最佳化基礎 |
| Unit13 | `scipy.optimize` | 反應工程、程序控制 | 統計基礎、ODE |
| Unit14 | `scipy.stats` | 工程統計、SPC | 機率統計基礎 |
| Unit15 | `scipy.signal` | 程序控制、動態模擬 | 傳遞函式概念 |

### 6.3 SciPy 官方文件與學習資源

**官方文件：**
- [SciPy 官方文件](https://docs.scipy.org/doc/scipy/)
- [SciPy API 參考](https://docs.scipy.org/doc/scipy/reference/)
- [SciPy 教學範例庫](https://docs.scipy.org/doc/scipy/tutorial/index.html)
- [scipy.special API](https://docs.scipy.org/doc/scipy/reference/special.html)

**推薦學習資源：**
- *Scientific Python Lectures* — [https://scipy-lectures.org/](https://scipy-lectures.org/)
- *Python for Scientists and Engineers* — SciPy 原始文件
- Numpy/SciPy 中文文件 — 網路社群維護

**化工應用參考：**
- Rawlings, J.B. & Ekerdt, J.G., *Chemical Reactor Analysis and Design Fundamentals*
- Chapra, S.C., *Numerical Methods for Engineers*

---

## 7. 小結

本單元介紹了 SciPy 科學計算套件的整體架構與化工應用概觀：

1. **生態系統**：SciPy 建立在 NumPy 之上，提供高階的科學計算演算法庫
2. **子模組架構**：11 個主要子模組各司其職，對應課程 Unit06–Unit15
3. **基本使用模式**：掌握子模組載入方式、文件查詢方法、回傳值結構與容差設定
4. **化工應用地圖**：熱力學、反應工程、傳輸現象、程序控制、數據分析均有 SciPy 的身影
5. **特殊函式**：`scipy.special` 提供誤差函式、Bessel 函式、Gamma 函式等，廣泛應用於化工解析解

### 學習要點

- **不要直接 `import scipy`**：需明確匯入所需子模組，如 `from scipy import linalg`
- **優先查官方文件**：SciPy 文件包含豐富的使用範例，是最可靠的學習資源
- **理解回傳值結構**：不同函式的回傳型態不同（元組、OptimizeResult、OdeResult 等）
- **合理設定容差**：根據工程精度需求選擇容差，避免過嚴導致計算緩慢

### 下一步

- 完成 `Unit05_Special_Functions.ipynb` 程式演練
- 開始 Unit06 線性聯立方程式求解

---

## 8. 參考資源

### 官方文件
- [SciPy 官方文件](https://docs.scipy.org/doc/scipy/)
- [NumPy 官方文件](https://numpy.org/doc/stable/)
- [SciPy GitHub 儲存庫](https://github.com/scipy/scipy)

### 化工計算參考書
- Felder, R.M. & Rousseau, R.W., *Elementary Principles of Chemical Processes*
- Bird, R.B., Stewart, W.E. & Lightfoot, E.N., *Transport Phenomena*
- Fogler, H.S., *Elements of Chemical Reaction Engineering*

### 線上學習資源
- [Real Python - SciPy Tutorial](https://realpython.com/python-scipy-cluster-graph/)
- [SciPy Lecture Notes](https://scipy-lectures.org/)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit 05 SciPy 科學運算套件應用概述
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
