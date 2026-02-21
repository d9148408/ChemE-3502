# Unit08 插值、微分與積分之運算

## 課程簡介

在化學工程計算中，工程師經常需要面對以下三類問題：

1. **插值 (Interpolation)**：已知某物質在若干溫度下的黏度數據，如何估計其他溫度的黏度值？
2. **數值微分 (Numerical Differentiation)**：已知批次反應器中的濃度-時間數據，如何計算各時間點的反應速率？
3. **數值積分 (Numerical Integration)**：已知填充塔吸收器的氣相組成分布，如何計算所需的傳質單元數 (NOG)？

這三類運算是連結「實驗量測數據」與「工程設計計算」的核心橋樑。本單元以 Python 的 `numpy` 與 `scipy` 函式庫為工具，系統性地介紹這三類數值方法的原理、語法與化工應用。

### 學習目標

完成本單元後，學生應能夠：

1. 區分並選用適合的一維與二維插值方法（`interp1d`、`CubicSpline`、`RegularGridInterpolator`、`griddata`）
2. 理解外插的風險，並正確設定邊界條件
3. 使用有限差分法（前向、後向、中心差分）實作數值微分
4. 應用 `numpy.diff`、`numpy.gradient` 處理離散數據的微分問題
5. 選用適當的數值積分方法（`trapezoid`、`simpson`、`quad`、`dblquad`）
6. 將插值、微分、積分整合應用於黏度分析、反應動力學推算、RTD 分析、填充塔設計等化工問題

### 本單元內容架構

```
Unit08 插值、微分與積分
├── 1. 插值法基礎
│   ├── 1.1 一維插值（interp1d, CubicSpline）
│   ├── 1.2 二維插值（RegularGridInterpolator, griddata）
│   └── 1.3 外插的風險
├── 2. 數值微分
│   ├── 2.1 有限差分近似原理
│   ├── 2.2 NumPy 數值微分工具
│   └── 2.3 高階微分與偏微分
├── 3. 數值積分
│   ├── 3.1 離散數據積分（trapezoid, simpson）
│   ├── 3.2 函數積分（quad）
│   └── 3.3 重積分（dblquad, nquad）
├── 4. SciPy 工具總覽
├── 5. 化工應用
│   ├── 5.1 黏度插值與反插值
│   ├── 5.2 批次反應器反應速率推算
│   ├── 5.3 RTD 分析
│   └── 5.4 填充塔 NOG 計算
└── 6. 程式設計最佳實踐
```

---

## 1. 插值法基礎 (Interpolation)

### 1.1 一維插值方法

插值的核心問題是：給定 $n$ 個資料點 $(x_0, y_0), (x_1, y_1), \ldots, (x_{n-1}, y_{n-1})$，估計 $x$ 在兩點之間的 $y$ 值。

#### `scipy.interpolate.interp1d`

`interp1d` 是最常用的一維插值函式，支援多種插值方式：

| `kind` 參數 | 方法 | 特性 |
|-----------|------|------|
| `'nearest'` | 最近鄰插值 | 僅適用於階梯狀數據 |
| `'linear'` | 線性插值 | 最簡單，C⁰ 連續 |
| `'quadratic'` | 二次樣條 | C¹ 連續 |
| `'cubic'` | 三次樣條 | C¹ 連續（非自然樣條） |

```python
from scipy.interpolate import interp1d

# 建立插值函數
f_lin  = interp1d(x_data, y_data, kind='linear')
f_cub  = interp1d(x_data, y_data, kind='cubic')

# 查詢新 x 值對應的 y 值
y_query = f_lin(x_query)   # x_query 必須在 x_data 的範圍內
```

#### `scipy.interpolate.CubicSpline`

`CubicSpline` 建立自然三次樣條（C² 連續），即在每個節點處一、二階導數均連續。相較於 `interp1d(kind='cubic')`，`CubicSpline` 行為更可預測、無 Runge 震盪，是一維平滑插值的首選：

```python
from scipy.interpolate import CubicSpline

cs = CubicSpline(x_data, y_data)
y_interp = cs(x_query)

# CubicSpline 額外功能：直接求導數
dy_dx   = cs(x_query, 1)   # 一階導數
d2y_dx2 = cs(x_query, 2)   # 二階導數
```

> **重要提醒**：`CubicSpline` 支援 `nu` 參數直接計算導數（`cs(x, 1)`），這在需要同時插值與求導時非常方便（如牛頓法求反插值）。

### 1.2 二維插值方法

當數據為二維表格時（如熱力學性質，以溫度與壓力為自變數），需使用二維插值。

| 函式 | 數據要求 | 適用場景 |
|-----|---------|---------|
| `RegularGridInterpolator` | 規則矩形網格 | 蒸汽表、查表計算 |
| `RectBivariateSpline` | 規則矩形網格 | 需要平滑導數 |
| `griddata` | 任意散點 | 實驗量測不規則數據 |

```python
from scipy.interpolate import RegularGridInterpolator

# x_grid, y_grid 為一維陣列；z_data 為二維陣列 shape=(len(x_grid), len(y_grid))
rgi = RegularGridInterpolator((x_grid, y_grid), z_data, method='linear')

# 查詢任意點
z_query = rgi([[x_q, y_q]])   # 輸入需為 (N, 2) 陣列

# 批次查詢（向量化）
pts = np.column_stack([x_pts.ravel(), y_pts.ravel()])
z_batch = rgi(pts).reshape(x_pts.shape)
```

### 1.3 外插的風險

插值方法僅保證在資料範圍「內」的準確性。查詢範圍「外」的點稱為**外插 (extrapolation)**，可能導致嚴重誤差：

```python
# interp1d 外插設定
f1 = interp1d(x, y, bounds_error=False, fill_value=np.nan)           # 超界回傳 NaN
f2 = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))    # 超界填充端點值
f3 = interp1d(x, y, bounds_error=False, fill_value='extrapolate')    # 外插（危險！）
```

> **最佳實踐**：永遠先檢查查詢值是否在原始數據範圍內。如必須外插，應選取在物理上有意義的模型（如 Arrhenius、Antoine 方程式）直接外推，而非用純數學插值。

---

## 2. 數值微分 (Numerical Differentiation)

### 2.1 有限差分近似原理

對函數 $f(x)$，利用 Taylor 展開推導出三種有限差分公式：

**前向差分 (Forward Difference)**（截斷誤差 $O(h)$）：

$$
f'(x) \approx \frac{f(x+h) - f(x)}{h}
$$

**後向差分 (Backward Difference)**（截斷誤差 $O(h)$）：

$$
f'(x) \approx \frac{f(x) - f(x-h)}{h}
$$

**中心差分 (Central Difference)**（截斷誤差 $O(h^2)$，**精度更高**）：

$$
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
$$

其中 $h$ 為步長。步長的選擇需在**截斷誤差**（步長太大）與**捨入誤差**（步長太小）間取得平衡：

- 最佳步長（中心差分，倍精度浮點數）：$h^* \approx \epsilon_{\text{mach}}^{1/3} \approx 6 \times 10^{-6}$

### 2.2 NumPy 數值微分工具

#### `numpy.diff`

`np.diff(y, n)` 計算陣列的 $n$ 階差分（相鄰元素相減）：

```python
import numpy as np

t = np.array([0, 1, 2, 3, 4])     # 時間 (s)
C = np.array([1.0, 0.6, 0.36, 0.22, 0.13])  # 濃度 (mol/L)

dC = np.diff(C)                    # shape: (n-1,)
dt = np.diff(t)                    # shape: (n-1,)
dC_dt = dC / dt                    # 一階差商（前向差分）
t_mid = (t[:-1] + t[1:]) / 2      # 對應的中間時間點
```

> **注意**：`np.diff` 回傳長度比原陣列少 1，適合等間距或非等間距數據。

#### `numpy.gradient`

`np.gradient(y, x)` 使用**中心差分**自動計算梯度，回傳與輸入**相同長度**的陣列（邊界採用一階差分）：

```python
dC_dt = np.gradient(C, t)   # 支援非等間距 t
```

| 工具 | 長度 | 精度 | 邊界 |
|-----|------|------|------|
| `np.diff(y)/np.diff(x)` | $n-1$ | $O(h)$ | 無特殊處理 |
| `np.gradient(y, x)` | $n$ | $O(h^2)$ | 邊界自動降為 $O(h)$ |

### 2.3 高階微分與偏微分

**二階導數**（中心差分公式，截斷誤差 $O(h^2)$）：

$$
f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}
$$

也可使用兩次 `np.gradient`，但精度略低：

```python
d2C_dt2 = np.gradient(np.gradient(C, t), t)
```

**偏微分**：對二維陣列 $Z[i,j] = f(x_i, y_j)$：

```python
dZ_dx = np.gradient(Z, x_vals, axis=0)   # ∂Z/∂x，沿第 0 軸
dZ_dy = np.gradient(Z, y_vals, axis=1)   # ∂Z/∂y，沿第 1 軸
```

---

## 3. 數值積分 (Numerical Integration)

### 3.1 離散數據積分

當數據以表格形式給出（如流量計讀數、感測器數據），無法得到解析函數，必須使用**數值正交 (numerical quadrature)**。

#### 梯形法則 (Trapezoid Rule)

$$
\int_a^b f(x)\,dx \approx \sum_{i=0}^{n-1} \frac{h_i}{2}(f_i + f_{i+1})
$$

截斷誤差 $O(h^2)$，支援**非等間距**數據：

```python
from scipy.integrate import trapezoid

result = trapezoid(y_data, x_data)   # 非等間距自動處理
```

#### Simpson 法則 (Simpson's Rule)

$$
\int_a^b f(x)\,dx \approx \frac{h}{3}\bigl[f_0 + 4f_1 + 2f_2 + 4f_3 + \cdots + f_n\bigr]
$$

截斷誤差 $O(h^4)$（需偶數個區間），精度遠高於梯形法：

```python
from scipy.integrate import simpson

result = simpson(y_data, x=x_data)   # 自動偵測是否等間距
```

> **建議**：若數據等間距且點數足夠，優先使用 `simpson`；若數據非等間距或點數奇偶不確定，使用 `trapezoid`。

### 3.2 函數積分 (`scipy.integrate.quad`)

當積分對象為**解析函數**時，`quad` 是最強大的工具。它採用自適應 Gauss-Kronrod 積分法：

```python
from scipy.integrate import quad

# 基本用法
result, error = quad(f, a, b)

# 傳遞額外參數
result, error = quad(lambda x: x**n * np.exp(-x), 0, np.inf, args=(n,))

# 無窮積分
result, error = quad(f, 0, np.inf)        # 0 到 +∞
result, error = quad(f, -np.inf, np.inf)  # 全實數軸
```

`quad` 回傳值說明：
- `result`：積分值估計
- `error`：積分誤差上界（非相對誤差）

精度控制參數：
- `epsabs`：絕對誤差容限（默認 `1.49e-8`）
- `epsrel`：相對誤差容限（默認 `1.49e-8`）
- `limit`：最大子區間數（默認 50，複雜積分可增大）

### 3.3 重積分

| 函式 | 積分維度 | 語法特點 |
|-----|---------|---------|
| `dblquad(f, a, b, gfun, hfun)` | 二重 | 內層積分限可為 $x$ 的函數 |
| `tplquad(f, a, b, gfun, hfun, qfun, rfun)` | 三重 | 最多 3 維 |
| `nquad(f, ranges)` | $n$ 重 | 最靈活，支援任意維度 |

**重要：變數積分順序**（由內而外）

```python
from scipy.integrate import dblquad

# ∫∫ f(x,y) dy dx，x 從 a 到 b，y 從 gfun(x) 到 hfun(x)
# 注意！scipy 的 dblquad 第一個自變數是「內層積分變數」
result, err = dblquad(
    lambda y, x: x**2 + y**2,  # f(y, x) 注意順序！y 在前
    0, 2,                        # x 的積分範圍
    0, lambda x: x              # y 的積分範圍（可為 x 的函數）
)
```

> **常見錯誤**：`dblquad` 的 callable 參數順序是 `f(inner_var, outer_var)`，即 `f(y, x)` 而非 `f(x, y)`。

---

## 4. SciPy 工具總覽

### 4.1 `scipy.interpolate` 常用函式

| 函式 | 輸入 | 輸出 | 導數支援 |
|-----|------|------|---------|
| `interp1d(x, y, kind)` | 1D 數組 | 插值物件 | 否 |
| `CubicSpline(x, y)` | 1D 數組 | 插值物件 | ✓ `cs(x, nu)` |
| `RegularGridInterpolator((x_g, y_g), z)` | 2D 規則網格 | 插值物件 | 否 |
| `RectBivariateSpline(x_g, y_g, z)` | 2D 規則網格 | 插值物件 | ✓ `.ev(x,y,dx,dy)` |
| `griddata(points, values, xi)` | 2D 散點 | numpy 陣列 | 否 |

### 4.2 `scipy.integrate` 常用函式

| 函式 | 輸入類型 | 維度 | 自適應 |
|-----|---------|------|---------|
| `trapezoid(y, x)` | 離散數據 | 1D | 否 |
| `simpson(y, x)` | 離散數據（等間距） | 1D | 否 |
| `quad(f, a, b)` | 解析函數 | 1D | ✓ |
| `dblquad(f, a, b, g, h)` | 解析函數 | 2D | ✓ |
| `tplquad(f, ...)` | 解析函數 | 3D | ✓ |
| `nquad(f, ranges)` | 解析函數 | nD | ✓ |
| `fixed_quad(f, a, b, n)` | 解析函數 | 1D | 否（固定點數） |

### 4.3 `numpy` 微分工具

| 函式 | 輸出長度 | 差分方式 | 備註 |
|-----|---------|---------|------|
| `np.diff(y, n=1)` | N-n | 前向差分 | 支援高階，需另除以 Δx |
| `np.gradient(y, x)` | N | 中心差分（邊界除外） | 直接支援非等間距 x |

---

## 5. 化工應用

### 5.1 黏度插值與反插值

**問題**：已知甲苯在不同溫度下的動力黏度，求指定溫度的黏度；以及黏度達到指定值時的溫度（反插值）。

**方法**：
- `CubicSpline` 建立黏度插值模型
- 反插值轉化為非線性方程求根問題： `mu_cs(T) - mu_target = 0`
- 使用 `scipy.optimize.brentq` 求根

**Andrade 方程式**（化工常用黏度模型）：

}
\ln \mu = A + \frac{B}{T}
}

用 `numpy.polyfit` 進行線性回歸（對 1/T 線性化）。

### 5.2 批次反應器反應速率推算

**問題**：已知批次反應器中的濃度-時間數據，推算反應速率 r = -dC_A/dt。

**方法**：
1. 使用 `np.gradient(C_A, t)` 計算 dC_A/dt（非均勻時間間隔自動處理）
2. 速率 r = -dC_A/dt
3. 假設冪次定律 r = k * C_A^n，取對數回歸求 n（反應級數）與 k（速率常數）

### 5.3 RTD 分析（停留時間分布）

**問題**：從脈衝追蹤劑實驗數據中提取 RTD 特性，評估反應器的非理想流動行為。

**關鍵公式**：

正規化 E(t) 曲線：

}
E(t) = \frac{C(t)}{\int_0^\infty C(t)\,dt}
}

平均滯留時間：

}
\bar{t} = \int_0^\infty t \cdot E(t)\,dt
}

無因次方差（流動模型指標）：

}
\sigma_\theta^2 = \frac{\sigma^2}{\bar{t}^2}
}

| $\sigma_\theta^2$ 值 | 對應流動模型 |
|---------------------|-------------|
| 0 | 理想平推流 (PFR) |
| 1 | 理想全混流 (CSTR) |
| 0 ~ 1 | 介於兩者之間（軸向擴散模型） |

### 5.4 填充塔吸收器 NOG 計算

**傳質單元數**：

}
N_{OG} = \int_{y_2}^{y_1} \frac{dy}{y - y^*}
}

**計算步驟**：

1. 由平衡數據建立 CubicSpline： ^* = f_{\text{eq}}(x)$
2. 由物料平衡操作線求 x：  = (y - y_2)/(L/G) + x_2$
3. 建立積分函數 (y) = 1/(y - y^*(x(y)))$
4. `quad(g, y2, y1)` 求 N_OG
5.  = N_{OG} \cdot H_{OG}$

---

## 6. 程式設計最佳實踐

### 6.1 插值方法選擇

**一維數據**：
- 需要導數 → `CubicSpline`
- 大量等間距數據 → `interp1d(kind='linear')`（快速）
- 一般平滑估計 → `CubicSpline`（首選）

**二維數據**：
- 規則網格（蒸汽表、查表）→ `RegularGridInterpolator`
- 規則網格（需要導數）→ `RectBivariateSpline`
- 散點數據 → `griddata`

### 6.2 數值微分注意事項

1. **步長選擇**：中心差分最佳步長約 ^{-5} \sim 10^{-6}$
2. **雜訊數據**：先 Savitzky-Golay 平滑，再用 `np.gradient` 微分
3. **非等間距**：優先使用 `np.gradient(y, x)`，不要自行除以固定步長
4. **邊界處理**：`np.gradient` 邊界自動降為一階差分，應避免直接使用邊界點的導數值

### 6.3 數值積分選擇

1. **離散數據**：`trapezoid`（萬用）> `simpson`（等間距且點數多時更精確）
2. **解析函數**：`quad`（自適應，首選）> `fixed_quad`（固定 n 點 Gauss 積分）
3. **多維積分**：`nquad` 比巢狀 `quad` 更清晰，建議使用
4. **精度驗證**：檢查 `quad` 的相對誤差 `error / abs(result)` 是否小於 ^{-6}$

---

## 7. 結語

本單元介紹了 Python 中插值、數值微分、數值積分的完整工具鏈：

| 問題類型 | 核心工具 | 化工應用範例 |
|---------|---------|------------|
| 1D 插值 | `CubicSpline`, `interp1d` | 黏度、蒸汽壓查表 |
| 2D 插值 | `RegularGridInterpolator`, `griddata` | 蒸汽表、相圖 |
| 外插邊界 | `fill_value`, `bounds_error` | 超出量測範圍的估計 |
| 數值微分 | `np.gradient`, `np.diff` | 反應速率推算、流量計算 |
| 離散積分 | `trapezoid`, `simpson` | RTD 分析 |
| 函數積分 | `quad`, `dblquad`, `nquad` | NOG 計算、能量平衡 |

這些工具共同構成化工數值計算的重要基礎。後續在**聯立方程求解 (Unit09)**、**常微分方程 (Unit10)** 以及**機器學習建模 (Unit12-15)** 中，都將大量應用本單元所介紹的數值技術。

---

**課程資訊**
- 課程名稱：電腦在化工上之應用
- 課程單元：Unit08 插值、微分與積分之運算
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2025-06-01

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---