# Unit06 線性聯立方程式之求解

## 學習目標

在本單元中，我們將學習如何使用 Python 的 NumPy 和 SciPy 套件來求解線性聯立方程式系統。這些技術在化工領域中有廣泛的應用，包括物料平衡、能量平衡、反應器網絡分析等問題。

學習完本單元後，您將能夠：

- 理解線性聯立方程式的數學基礎與解的存在性條件
- 使用 NumPy 的線性代數工具進行基本的矩陣運算與求解
- 運用 SciPy 的進階求解器處理各種類型的線性系統
- 分析並處理不同類型的線性系統（唯一解、無窮多解、無解、病態系統）
- 將線性方程式求解技術應用於實際化工問題
- 撰寫穩健的程式碼並進行結果驗證

---

## 1. 線性聯立方程式系統基礎

### 1.1 矩陣形式表示法

線性聯立方程式可以表示為以下形式：

$$
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}
$$

使用矩陣表示法，可以簡潔地寫成：

$$
\mathbf{Ax} = \mathbf{b}
$$

其中：
- $\mathbf{A}$ 是 $m \times n$ 的係數矩陣 (coefficient matrix)
- $\mathbf{x}$ 是 $n \times 1$ 的未知數向量 (unknown vector)
- $\mathbf{b}$ 是 $m \times 1$ 的常數向量 (constant vector)

具體來說：

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ b_m
\end{bmatrix}
$$

### 1.2 解的存在性與唯一性

線性聯立方程式 $\mathbf{Ax} = \mathbf{b}$ 的解取決於係數矩陣 $\mathbf{A}$ 和擴增矩陣 $[\mathbf{A} \mid \mathbf{b}]$ 的**秩** (rank)。

**定義**：矩陣的秩是指矩陣中線性獨立的行（或列）向量的最大數目。

**定理**：線性聯立方程式 $\mathbf{Ax} = \mathbf{b}$ 有解的充要條件為：

$$
\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])
$$

令 $r = \mathrm{rank}(\mathbf{A})$ ，則解的情況可分為以下三種類型。

### 1.3 三種系統類型

#### 類型一：唯一解系統 (Unique Solution System)

**條件**：
- $\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ 且 $r = n$
- 獨立方程式數目等於未知數數目
- 係數矩陣為方陣且 $\det(\mathbf{A}) \neq 0$

**特性**：
- 方程組有唯一解
- 矩陣 $\mathbf{A}$ 可逆 (invertible)
- 解可表示為 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$

**範例**：

$$
\begin{aligned}
2x_1 + x_2 &= 5 \\
x_1 + 3x_2 &= 6
\end{aligned}
$$

矩陣形式：

$$
\begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix} =
\begin{bmatrix}
5 \\ 6
\end{bmatrix}
$$

此系統有唯一解： $x_1 = 1.8$ ， $x_2 = 1.4$ 。

#### 類型二：無窮多解系統（低確定系統，Underdetermined System）

**條件**：
- $\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ 且 $r < n$
- 獨立方程式數目少於未知數數目
- 存在 $n - r$ 個自由變數 (free variables)

**特性**：
- 方程組有無窮多組解
- $r$ 個未知數可表示為其他 $n - r$ 個未知數的線性組合
- 可以找到最小範數解 (minimum norm solution)

**範例**：

$$
\begin{aligned}
x_1 + 2x_2 + 3x_3 &= 6 \\
2x_1 + 4x_2 + 6x_3 &= 12
\end{aligned}
$$

第二個方程式是第一個的兩倍，因此 $\mathrm{rank}(\mathbf{A}) = 1 < 3$ ，有無窮多組解。

#### 類型三：無解系統（過確定系統，Overdetermined System）

**條件**：
- $\mathrm{rank}(\mathbf{A}) < \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$
- 方程組不一致 (inconsistent)
- 獨立方程式數目大於未知數數目，但存在矛盾

**特性**：
- 方程組無精確解
- 可以求最小平方解 (least-squares solution)
- 最小平方解使殘差 $\|\mathbf{Ax} - \mathbf{b}\|^2$ 最小

**範例**：

$$
\begin{aligned}
x_1 + x_2 &= 1 \\
x_1 + 2x_2 &= 3 \\
x_1 + 5x_2 &= 10
\end{aligned}
$$

若前兩式有解 $x_1 = -1, x_2 = 2$ ，但不滿足第三式，故無精確解。可求最小平方解。

### 1.4 齊次與非齊次方程組的特性

#### 齊次方程組 (Homogeneous System)

當 $\mathbf{b} = \mathbf{0}$ 時，線性聯立方程式組為齊次方程組：

$$
\mathbf{Ax} = \mathbf{0}
$$

**特性**：
- 必定有解（至少有零解 $\mathbf{x} = \mathbf{0}$ ）
- 若 $\mathrm{rank}(\mathbf{A}) = n$ ，則只有零解（顯解，trivial solution）
- 若 $\mathrm{rank}(\mathbf{A}) < n$ ，則有非零解（非顯解，non-trivial solution），且有無窮多組解

**應用**：齊次方程組常出現在系統穩定性分析、特徵值問題等。

#### 非齊次方程組 (Non-homogeneous System)

當 $\mathbf{b} \neq \mathbf{0}$ 時，則為非齊次方程組。解的存在性需要檢查秩條件。

**通解結構**：非齊次方程組的通解等於特解加上對應齊次方程組的通解：

$$
\mathbf{x}_{\mathrm{general}} = \mathbf{x}_{\mathrm{particular}} + \mathbf{x}_{\mathrm{homogeneous}}
$$

### 1.5 小結

線性聯立方程式系統的特性總結如下表：

| 情況 | 條件 | 非齊次式 $\mathbf{Ax} = \mathbf{b}$ | 齊次式 $\mathbf{Ax} = \mathbf{0}$ |
|------|------|-------------------------------------|----------------------------------|
| 一 | $\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ <br> $\mathrm{rank}(\mathbf{A}) = n$ | 唯一解 | 唯一解 $\mathbf{x} = \mathbf{0}$ （顯解） |
| 二 | $\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ <br> $\mathrm{rank}(\mathbf{A}) < n$ | 無窮多組解 | 無窮多組解（含非零解） |
| 三 | $\mathrm{rank}(\mathbf{A}) < \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ | 無解 | （不適用） |

---

## 2. NumPy 線性代數基礎工具

NumPy 的 `numpy.linalg` 子模組提供了一系列線性代數運算工具。在求解線性聯立方程式時，我們常用以下函數。

### 2.1 numpy.linalg.matrix_rank() - 計算矩陣秩數

**功能**：計算矩陣的秩（線性獨立的行或列的數目）。

**語法**：
```python
numpy.linalg.matrix_rank(A, tol=None)
```

**參數**：
- `A`：輸入矩陣
- `tol`：容許誤差，用於判斷奇異值是否為零（預設為自動計算）

**範例**：
```python
import numpy as np

A = np.array([[1, 2, 3],
              [2, 4, 6],
              [1, 1, 1]])

r = np.linalg.matrix_rank(A)
print(f"Rank of A: {r}")  # 輸出：2
```

### 2.2 numpy.linalg.det() - 計算行列式值

**功能**：計算方陣的行列式 (determinant)。

**語法**：
```python
numpy.linalg.det(A)
```

**參數**：
- `A`：方陣（必須是正方形矩陣）

**用途**：
- 檢查矩陣是否可逆（ $\det(\mathbf{A}) \neq 0$ 則可逆）
- 計算矩陣的體積縮放因子

**範例**：
```python
A = np.array([[2, 1],
              [1, 3]])

det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}")  # 輸出：5.0

# 檢查可逆性
if det_A != 0:
    print("Matrix A is invertible")
```

### 2.3 numpy.linalg.inv() - 計算反矩陣

**功能**：計算方陣的反矩陣 (inverse matrix)。

**語法**：
```python
numpy.linalg.inv(A)
```

**參數**：
- `A`：可逆方陣

**注意事項**：
- 僅適用於可逆矩陣（ $\det(\mathbf{A}) \neq 0$ ）
- 對於奇異矩陣會產生 `LinAlgError`
- 數值穩定性問題：不建議用於直接求解線性方程式

**範例**：
```python
A = np.array([[2, 1],
              [1, 3]])

A_inv = np.linalg.inv(A)
print("Inverse of A:")
print(A_inv)

# 驗證 A * A_inv = I
I = A @ A_inv
print("A @ A_inv:")
print(I)  # 應接近單位矩陣
```

### 2.4 numpy.linalg.solve() - 求解線性方程組

**功能**：求解線性聯立方程式 $\mathbf{Ax} = \mathbf{b}$ （唯一解情況）。

**語法**：
```python
numpy.linalg.solve(A, b)
```

**參數**：
- `A`：係數矩陣（必須是可逆方陣）
- `b`：常數向量或矩陣

**適用情況**：
- 係數矩陣為方陣且可逆
- 系統有唯一解

**範例**：
```python
# 求解 2x + y = 5, x + 3y = 6
A = np.array([[2, 1],
              [1, 3]])
b = np.array([5, 6])

x = np.linalg.solve(A, b)
print("Solution:")
print(x)  # 輸出：[1.8 1.4]

# 驗證解
residual = A @ x - b
print(f"Residual: {np.linalg.norm(residual)}")  # 應接近 0
```

**優點**：
- 比使用 `inv()` 計算 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ 更快且數值穩定
- 內部使用 LU 分解法

### 2.5 numpy.linalg.lstsq() - 最小平方解

**功能**：求解線性聯立方程式的最小平方解（適用於過確定系統）。

**語法**：
```python
x, residuals, rank, s = numpy.linalg.lstsq(A, b, rcond=None)
```

**參數**：
- `A`：係數矩陣（可以是非方陣）
- `b`：常數向量
- `rcond`：截斷奇異值的臨界值

**返回值**：
- `x`：最小平方解
- `residuals`：殘差平方和（僅當 $m > n$ 且 $\mathrm{rank}(\mathbf{A}) = n$ 時返回）
- `rank`：矩陣 $\mathbf{A}$ 的秩
- `s`：矩陣 $\mathbf{A}$ 的奇異值

**適用情況**：
- 過確定系統（ $m > n$ ）
- 無精確解的系統
- 需要找到使 $\|\mathbf{Ax} - \mathbf{b}\|^2$ 最小的解

**範例**：
```python
# 過確定系統：3 個方程式，2 個未知數
A = np.array([[1, 1],
              [1, 2],
              [1, 5]])
b = np.array([1, 3, 10])

x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print("Least-squares solution:")
print(x)
print(f"Residual sum of squares: {residuals[0]}")
print(f"Rank: {rank}")
```

### 2.6 numpy.linalg.pinv() - 虛擬反矩陣

**功能**：計算矩陣的 Moore-Penrose 虛擬反矩陣 (pseudo-inverse)。

**語法**：
```python
numpy.linalg.pinv(A, rcond=1e-15)
```

**參數**：
- `A`：任意矩陣（可以是非方陣或奇異矩陣）
- `rcond`：截斷奇異值的臨界值

**適用情況**：
- 低確定系統（找到最小範數解）
- 奇異矩陣或非方陣

**範例**：
```python
# 低確定系統：1 個方程式，3 個未知數
# x1 + 2*x2 + 3*x3 = 6
A = np.array([[1, 2, 3]])
b = np.array([6])

A_pinv = np.linalg.pinv(A)
x = A_pinv @ b
print("Minimum norm solution:")
print(x)  # 最接近原點的解

# 驗證解
print(f"A @ x = {A @ x}")  # 應等於 [6]
print(f"Norm of x: {np.linalg.norm(x)}")  # 最小範數
```

### 2.7 NumPy 工具選擇指南

根據不同的問題類型，選擇適當的 NumPy 函數：

| 系統類型 | 條件 | 建議函數 | 說明 |
|----------|------|----------|------|
| 唯一解系統 | $\mathbf{A}$ 為可逆方陣 | `np.linalg.solve()` | 最快且數值穩定 |
| 低確定系統 | $m < n$ 或 $\mathrm{rank}(\mathbf{A}) < n$ | `np.linalg.pinv()` | 找到最小範數解 |
| 過確定系統 | $m > n$ | `np.linalg.lstsq()` | 找到最小平方解 |
| 檢查秩 | 任意矩陣 | `np.linalg.matrix_rank()` | 判斷解的存在性 |
| 檢查可逆性 | 方陣 | `np.linalg.det()` | 行列式非零則可逆 |

**重要提示**：
- **避免使用** `inv()`：直接計算反矩陣數值穩定性較差，優先使用 `solve()`
- **選擇合適函數**：根據系統特性選擇，可提高效率和準確性
- **檢查秩**：求解前先檢查秩，可預測解的類型

---

## 3. SciPy 進階求解方法

SciPy 的 `scipy.linalg` 和 `scipy.sparse.linalg` 子模組提供了更進階的線性代數工具，特別適合處理大型系統、稀疏矩陣和需要特定分解方法的情況。

### 3.1 scipy.linalg.solve() - 一般密集矩陣求解

**功能**：求解線性聯立方程式，支援多種矩陣類型和分解方法。

**語法**：
```python
from scipy import linalg

x = linalg.solve(A, b, assume_a='gen', check_finite=True)
```

**參數**：
- `A`：係數矩陣
- `b`：常數向量或矩陣
- `assume_a`：假設矩陣類型
  - `'gen'`：一般矩陣（default）
  - `'sym'`：對稱矩陣
  - `'her'`：Hermitian 矩陣
  - `'pos'`：正定矩陣
- `check_finite`：是否檢查輸入包含無窮大或 NaN

**範例**：
```python
from scipy import linalg
import numpy as np

A = np.array([[3, 2, 0],
              [1, -1, 0],
              [0, 5, 1]])
b = np.array([2, 4, -1])

x = linalg.solve(A, b)
print("Solution:")
print(x)

# 驗證
print(f"Verification: A @ x = {A @ x}")
```

**與 NumPy 的差異**：
- SciPy 版本提供更多選項（矩陣類型假設、分解方法選擇）
- 對特定類型矩陣可能更快（如對稱矩陣、正定矩陣）
- 功能更豐富，但對簡單問題 `np.linalg.solve()` 已足夠

### 3.2 scipy.linalg.lu() - LU 分解

**功能**：執行 LU 分解 (LU decomposition)，將矩陣分解為下三角矩陣和上三角矩陣的乘積。

**數學原理**：

$$
\mathbf{A} = \mathbf{PLU}
$$

其中：
- $\mathbf{P}$ ：置換矩陣 (permutation matrix)
- $\mathbf{L}$ ：下三角矩陣 (lower triangular matrix)
- $\mathbf{U}$ ：上三角矩陣 (upper triangular matrix)

**語法**：
```python
P, L, U = linalg.lu(A)
```

**用途**：
- 理解矩陣結構
- 重複求解具有相同係數矩陣但不同 $\mathbf{b}$ 的系統
- 計算行列式： $\det(\mathbf{A}) = \det(\mathbf{P}) \det(\mathbf{L}) \det(\mathbf{U})$

**範例**：
```python
A = np.array([[2, 5, 8, 7],
              [5, 2, 2, 8],
              [7, 5, 6, 6],
              [5, 4, 4, 8]])

P, L, U = linalg.lu(A)

print("P (permutation matrix):")
print(P)
print("\nL (lower triangular):")
print(L)
print("\nU (upper triangular):")
print(U)

# 驗證 A = PLU
print("\nVerify PLU = A:")
print(np.allclose(P @ L @ U, A))  # 應輸出 True
```

### 3.3 scipy.linalg.lu_factor() 與 lu_solve() - 多次求解

**功能**：當需要對同一係數矩陣但不同常數向量多次求解時，使用 LU 分解可大幅提升效率。

**語法**：
```python
# 第一步：分解（只需執行一次）
lu, piv = linalg.lu_factor(A)

# 第二步：求解（可多次執行）
x1 = linalg.lu_solve((lu, piv), b1)
x2 = linalg.lu_solve((lu, piv), b2)
x3 = linalg.lu_solve((lu, piv), b3)
```

**效率優勢**：
- `lu_factor()` 的時間複雜度為 $O(n^3)$
- `lu_solve()` 的時間複雜度為 $O(n^2)$
- 若需求解 $k$ 次，總時間為 $O(n^3 + kn^2)$ 遠小於 $O(kn^3)$

**範例**：
```python
A = np.array([[3, 2, 0],
              [1, -1, 0],
              [0, 5, 1]])

# 多個不同的 b 向量
b1 = np.array([2, 4, -1])
b2 = np.array([1, 0, 3])
b3 = np.array([5, -2, 1])

# LU 分解（只執行一次）
lu, piv = linalg.lu_factor(A)

# 求解多個系統
x1 = linalg.lu_solve((lu, piv), b1)
x2 = linalg.lu_solve((lu, piv), b2)
x3 = linalg.lu_solve((lu, piv), b3)

print("Solution for b1:", x1)
print("Solution for b2:", x2)
print("Solution for b3:", x3)
```

### 3.4 scipy.sparse.linalg.spsolve() - 稀疏矩陣求解器

**功能**：專為稀疏矩陣設計的直接法求解器。

**稀疏矩陣**：大部分元素為零的矩陣。在化工問題中常見於：
- 有限差分法求解偏微分方程式
- 大型反應器網絡
- 製程流程模擬

**語法**：
```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 建立稀疏矩陣
A_sparse = csr_matrix(A)
x = spsolve(A_sparse, b)
```

**稀疏矩陣格式**：
- CSR (Compressed Sparse Row)：適合矩陣-向量乘法
- CSC (Compressed Sparse Column)：適合列存取
- COO (Coordinate format)：適合建構矩陣

**範例**：
```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

# 建立稀疏矩陣（大部分為 0）
A_dense = np.array([[10, 0, 0, 0, -2],
                    [0, 11, 0, -1, 0],
                    [0, 0, 12, 0, 0],
                    [0, -1, 0, 13, 0],
                    [-2, 0, 0, 0, 14]])
b = np.array([3, 1, 2, 4, 5])

# 轉換為稀疏格式
A_sparse = csr_matrix(A_dense)

# 求解
x = spsolve(A_sparse, b)
print("Solution:")
print(x)

# 記憶體使用比較
print(f"\nDense matrix memory: {A_dense.nbytes} bytes")
print(f"Sparse matrix memory: {A_sparse.data.nbytes + A_sparse.indices.nbytes + A_sparse.indptr.nbytes} bytes")
```

**優勢**：
- 大幅減少記憶體使用
- 對大型稀疏系統計算速度快
- 適合 $n > 10000$ 的系統

### 3.5 scipy.sparse.linalg.cg() - 共軛梯度迭代法

**功能**：使用共軛梯度法 (Conjugate Gradient method) 求解對稱正定線性系統。

**適用條件**：
- 係數矩陣必須是對稱正定的
- 適合大型稀疏系統
- 迭代法，不需完整矩陣分解

**語法**：
```python
from scipy.sparse.linalg import cg

x, info = cg(A, b, tol=1e-5, maxiter=None)
```

**參數**：
- `A`：係數矩陣（可以是稀疏矩陣或 LinearOperator）
- `b`：常數向量
- `tol`：收斂容許誤差
- `maxiter`：最大迭代次數
- `x0`：初始猜測值（選擇性）

**返回值**：
- `x`：解向量
- `info`：收斂資訊
  - `0`：成功收斂
  - `> 0`：未收斂，返回迭代次數

**範例**：
```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
import numpy as np

# 建立對稱正定矩陣
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

# 使用共軛梯度法
x, info = cg(A, b, tol=1e-8)

if info == 0:
    print("Converged successfully")
    print("Solution:", x)
else:
    print(f"Not converged after {info} iterations")

# 驗證解
residual = np.linalg.norm(A @ x - b)
print(f"Residual: {residual}")
```

**優勢**：
- 對大型系統記憶體需求低
- 每次迭代僅需矩陣-向量乘法
- 對某些問題收斂快速

### 3.6 scipy.sparse.linalg.gmres() - GMRES 迭代法

**功能**：廣義最小殘差法 (Generalized Minimal Residual method)，適用於一般非對稱線性系統。

**適用條件**：
- 不要求係數矩陣對稱或正定
- 適合大型稀疏系統
- 比共軛梯度法更通用但可能較慢

**語法**：
```python
from scipy.sparse.linalg import gmres

x, info = gmres(A, b, tol=1e-5, restart=None, maxiter=None)
```

**參數**：
- `restart`：重啟之前的迭代次數（影響記憶體使用）
- 其他參數類似 `cg()`

**範例**：
```python
from scipy.sparse.linalg import gmres
import numpy as np

# 非對稱矩陣
A = np.array([[3, 2, 0],
              [1, -1, 0],
              [0, 5, 1]], dtype=float)
b = np.array([2, 4, -1], dtype=float)

# 使用 GMRES
x, info = gmres(A, b, tol=1e-8)

if info == 0:
    print("Converged successfully")
    print("Solution:", x)
else:
    print(f"Not converged after {info} iterations")

# 驗證
residual = np.linalg.norm(A @ x - b)
print(f"Residual: {residual}")
```

### 3.7 SciPy 求解器選擇指南

| 情況 | 矩陣特性 | 建議方法 | 函數 |
|------|----------|----------|------|
| 小型密集系統 | $n < 1000$ | 直接法 | `linalg.solve()` |
| 大型密集系統 | $n \geq 1000$ | 直接法 | `linalg.solve()` |
| 多次求解（相同 A） | 任意大小 | LU 分解 | `lu_factor()` + `lu_solve()` |
| 大型稀疏系統 | 稀疏度 > 90% | 稀疏直接法 | `sparse.linalg.spsolve()` |
| 大型對稱正定 | 對稱正定 | 共軛梯度 | `sparse.linalg.cg()` |
| 大型非對稱 | 非對稱 | GMRES | `sparse.linalg.gmres()` |

**關鍵決策因素**：
1. **矩陣大小**： $n < 1000$ 用密集法， $n \geq 10000$ 考慮稀疏/迭代法
2. **稀疏度**：若非零元素 < 10%，使用稀疏格式
3. **矩陣特性**：對稱正定用 CG，一般用 GMRES
4. **求解次數**：多次求解用 LU 分解

---

## 4. 不同系統類型的處理策略

針對不同類型的線性系統，我們需要採用不同的求解策略和驗證方法。

### 4.1 唯一解系統的處理

**特徵**：
- 係數矩陣為可逆方陣
- $\mathrm{rank}(\mathbf{A}) = n$ 且 $\det(\mathbf{A}) \neq 0$

**求解步驟**：

```python
import numpy as np
from scipy import linalg

def solve_unique_system(A, b):
    """求解唯一解系統"""
    # 步驟 1: 檢查矩陣是否為方陣
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix A must be square for unique solution")
    
    # 步驟 2: 檢查秩
    rank = np.linalg.matrix_rank(A)
    if rank < n:
        print(f"Warning: Matrix is singular (rank={rank} < n={n})")
        return None
    
    # 步驟 3: 求解
    try:
        x = linalg.solve(A, b)
    except linalg.LinAlgError:
        print("Matrix is singular")
        return None
    
    # 步驟 4: 驗證解
    residual = np.linalg.norm(A @ x - b)
    print(f"Residual: {residual:.2e}")
    
    if residual < 1e-10:
        print("Solution verified successfully")
    else:
        print("Warning: Large residual, solution may be inaccurate")
    
    return x

# 範例
A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]])
b = np.array([4, 5, 6])

x = solve_unique_system(A, b)
print("Solution:", x)
```

**驗證方法**：
1. **殘差檢查**： $\|\mathbf{Ax} - \mathbf{b}\| < \epsilon$
2. **代回驗證**：將解代回原方程式檢查
3. **條件數檢查**：評估數值穩定性

### 4.2 低確定系統的處理

**特徵**：
- 方程式數少於未知數數目
- $\mathrm{rank}(\mathbf{A}) < n$
- 有無窮多組解

**求解策略**：
- 找到最小範數解（最接近原點的解）
- 或者參數化表示通解

```python
def solve_underdetermined_system(A, b):
    """求解低確定系統，返回最小範數解"""
    m, n = A.shape
    rank = np.linalg.matrix_rank(A)
    
    print(f"System: {m} equations, {n} unknowns")
    print(f"Rank: {rank}")
    print(f"Degrees of freedom: {n - rank}")
    
    # 檢查相容性
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    if rank != rank_Ab:
        print("System is inconsistent (no solution)")
        return None
    
    # 使用虛擬反矩陣求最小範數解
    A_pinv = np.linalg.pinv(A)
    x_min = A_pinv @ b
    
    print(f"Minimum norm solution found")
    print(f"Norm of solution: {np.linalg.norm(x_min):.4f}")
    
    # 驗證
    residual = np.linalg.norm(A @ x_min - b)
    print(f"Residual: {residual:.2e}")
    
    return x_min

# 範例：2 個方程式，3 個未知數
A = np.array([[1, 2, 3],
              [2, 4, 5]])
b = np.array([6, 11])

x = solve_underdetermined_system(A, b)
print("Minimum norm solution:", x)
```

**參數化解**：
若想找通解，可將某些變數設為參數：

```python
# 通解示例
# x1 + 2*x2 + 3*x3 = 6
# 令 x3 = t（參數），則可解出 x1, x2 為 t 的函數
```

### 4.3 過確定系統的處理

**特徵**：
- 方程式數多於未知數數目
- 通常無精確解
- 需找最小平方解

**最小平方法**：找到使殘差平方和最小的解

$$
\min_{\mathbf{x}} \|\mathbf{Ax} - \mathbf{b}\|^2
$$

**求解步驟**：

```python
def solve_overdetermined_system(A, b):
    """求解過確定系統，返回最小平方解"""
    m, n = A.shape
    
    print(f"System: {m} equations, {n} unknowns")
    
    # 使用 lstsq 求最小平方解
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    print(f"Rank: {rank}")
    
    if len(residuals) > 0:
        print(f"Sum of squared residuals: {residuals[0]:.4e}")
        rms_error = np.sqrt(residuals[0] / m)
        print(f"RMS error: {rms_error:.4e}")
    
    # 計算各方程式的殘差
    res_vector = A @ x - b
    print(f"Max residual: {np.max(np.abs(res_vector)):.4e}")
    
    return x, residuals

# 範例：4 個方程式，2 個未知數（擬合直線）
A = np.array([[1, 1],
              [1, 2],
              [1, 3],
              [1, 4]])
b = np.array([2.1, 3.9, 6.2, 7.8])

x, residuals = solve_overdetermined_system(A, b)
print("Least-squares solution:", x)
```

**應用範例：直線擬合**
```python
import matplotlib.pyplot as plt

# 資料點
x_data = np.array([1, 2, 3, 4])
y_data = np.array([2.1, 3.9, 6.2, 7.8])

# 建立設計矩陣（y = a + b*x）
A = np.column_stack([np.ones_like(x_data), x_data])

# 求解
coeffs, _ = solve_overdetermined_system(A, y_data)
a, b = coeffs

print(f"Fitted line: y = {a:.2f} + {b:.2f}*x")

# 繪圖
plt.scatter(x_data, y_data, label='Data points')
x_fit = np.linspace(0, 5, 100)
y_fit = a + b * x_fit
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y={a:.2f}+{b:.2f}x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

### 4.4 病態系統的處理

**病態系統** (ill-conditioned system)：係數矩陣的條件數很大，導致數值解不穩定。

**條件數** (condition number)：

$$
\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|
$$

**判斷標準**：
- $\kappa(\mathbf{A}) \approx 1$ ：良態 (well-conditioned)
- $\kappa(\mathbf{A}) > 10^{10}$ ：病態 (ill-conditioned)

**檢查條件數**：

```python
def check_condition_number(A):
    """檢查矩陣的條件數"""
    cond = np.linalg.cond(A)
    
    print(f"Condition number: {cond:.2e}")
    
    if cond < 100:
        print("System is well-conditioned")
    elif cond < 1e10:
        print("System is moderately conditioned")
    else:
        print("System is ill-conditioned (numerical instability expected)")
    
    # 估計精度損失
    digits_lost = np.log10(cond)
    print(f"Expected precision loss: ~{digits_lost:.1f} decimal digits")
    
    return cond

# 範例：病態矩陣（Hilbert matrix）
n = 5
H = np.array([[1/(i+j-1) for j in range(1, n+1)] for i in range(1, n+1)])
print("Hilbert matrix (ill-conditioned):")
cond = check_condition_number(H)
```

**改善策略**：
1. **重新縮放**：調整方程式的尺度
2. **正規化**：在最小平方問題中加入正規化項（Tikhonov regularization）
3. **奇異值分解 (SVD)**：使用 SVD 進行穩健求解
4. **提高精度**：使用更高精度的數值型別

```python
# 使用 SVD 處理病態系統
def solve_with_svd(A, b, rcond=1e-10):
    """使用奇異值分解求解病態系統"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # 截斷小奇異值
    s_inv = np.where(s > rcond * s[0], 1/s, 0)
    
    # 計算解
    x = Vt.T @ np.diag(s_inv) @ U.T @ b
    
    print(f"Number of singular values used: {np.sum(s > rcond * s[0])}/{len(s)}")
    
    return x
```

---

## 5. 化工問題中的應用

線性方程式求解在化工領域有廣泛的應用，以下介紹幾個典型場景。

### 5.1 物料平衡方程組

**物料平衡**是化工計算的基礎，在穩態過程中，各成分的質量守恆可表示為線性方程組。

**摻合問題範例**：

某工廠有三個儲存槽，欲混合成特定組成的產品。

| 槽號 | 成分 A (%) | 成分 B (%) | 成分 C (%) |
|------|-----------|-----------|-----------|
| 1 | 50 | 30 | 20 |
| 2 | 20 | 60 | 20 |
| 3 | 30 | 10 | 60 |

目標產品 100 kg，組成為 A: 35%、B: 35%、C: 30%。求各槽所需用量。

**解法**：

```python
# 物料平衡方程式
# 成分 A: 0.50*V1 + 0.20*V2 + 0.30*V3 = 35
# 成分 B: 0.30*V1 + 0.60*V2 + 0.10*V3 = 35
# 成分 C: 0.20*V1 + 0.20*V2 + 0.60*V3 = 30
# 總量:   V1 + V2 + V3 = 100

A = np.array([[0.50, 0.20, 0.30],
              [0.30, 0.60, 0.10],
              [0.20, 0.20, 0.60],
              [1.00, 1.00, 1.00]])
b = np.array([35, 35, 30, 100])

# 過確定系統，使用最小平方法
V, residuals = np.linalg.lstsq(A, b, rcond=None)[:2]

print("Required volumes from each tank:")
print(f"Tank 1: {V[0]:.2f} kg")
print(f"Tank 2: {V[1]:.2f} kg")
print(f"Tank 3: {V[2]:.2f} kg")

# 驗證質量守恆
if len(residuals) > 0 and residuals[0] < 1e-6:
    print("Mass balance satisfied")
```

### 5.2 反應器網絡分析

**化學反應器網絡**中的穩態物料平衡也可表示為線性方程組。

考慮連續攪拌槽反應器 (CSTR) 網絡，反應 $\mathrm{A} \rightarrow \mathrm{B}$ ，一階反應。

```python
# 三個串聯 CSTR 的穩態物料平衡
# 對成分 A:
# Reactor 1: F0*CA0 - F1*CA1 - V1*k*CA1 = 0
# Reactor 2: F1*CA1 - F2*CA2 - V2*k*CA2 = 0
# Reactor 3: F2*CA2 - F3*CA3 - V3*k*CA3 = 0

# 參數設定
F0 = 100  # Feed flow rate (L/min)
CA0 = 2.0  # Feed concentration (mol/L)
k = 0.5    # Reaction rate constant (1/min)
V1 = V2 = V3 = 50  # Reactor volumes (L)

# 假設流率不變 F0 = F1 = F2 = F3 = F
F = F0

# 線性方程組 A*C = b
# (F + V*k)*CA1 = F*CA0
# (F + V*k)*CA2 = F*CA1
# (F + V*k)*CA3 = F*CA2

# 改寫為標準形式
A = np.array([[F + V1*k, 0, 0],
              [-F, F + V2*k, 0],
              [0, -F, F + V3*k]])
b = np.array([F*CA0, 0, 0])

C = np.linalg.solve(A, b)

print("Concentrations in each reactor:")
print(f"Reactor 1: CA = {C[0]:.4f} mol/L")
print(f"Reactor 2: CA = {C[1]:.4f} mol/L")
print(f"Reactor 3: CA = {C[2]:.4f} mol/L")

# 計算總轉化率
conversion = (CA0 - C[2]) / CA0 * 100
print(f"Overall conversion: {conversion:.2f}%")
```

### 5.3 熱交換器網絡能量平衡

**熱交換器網絡**的能量平衡可建立線性方程組求解各股流溫度。

```python
# 三個熱交換器的能量平衡範例  
# 已知進料溫度，求各出口溫度

# T1: HX1 熱流出口溫度
# T2: HX2 熱流出口溫度
# T3: HX3 冷流出口溫度

# 能量平衡方程式（簡化模型）
A_heat = np.array([[1, 0, -0.3],
                   [-0.5, 1, 0],
                   [0, -0.4, 1]])
b_heat = np.array([80, 60, 40])

T = np.linalg.solve(A_heat, b_heat)

print("Outlet temperatures:")
print(f"T1 = {T[0]:.2f} °C")
print(f"T2 = {T[1]:.2f} °C")
print(f"T3 = {T[2]:.2f} °C")
```

### 5.4 循環流處理

**循環流** (recycle stream) 是化工製程中常見的情況，會形成隱式方程組。

```python
# 具有循環流的系統
# Fresh feed + Recycle → Reactor → Separator → Product + Recycle

# 變數：F_feed, F_recycle, F_reactor, F_product
# 方程式：
# 1. F_reactor = F_feed + F_recycle
# 2. F_product = 0.8 * F_reactor (80% 分離效率)
# 3. F_recycle = 0.2 * F_reactor (20% 循環)
# 4. F_feed = 100 (已知)

A_recycle = np.array([[-1, 1, 1, 0],
                      [0, 0, 0.8, -1],
                      [0, -1, 0.2, 0],
                      [1, 0, 0, 0]])
b_recycle = np.array([0, 0, 0, 100])

flows = np.linalg.solve(A_recycle, b_recycle)

print("Flow rates:")
print(f"Feed: {flows[0]:.2f} kg/hr")
print(f"Recycle: {flows[1]:.2f} kg/hr")
print(f"To reactor: {flows[2]:.2f} kg/hr")
print(f"Product: {flows[3]:.2f} kg/hr")
```

### 5.5 解的物理意義驗證

在化工問題中，數學解必須符合物理限制：

```python
def validate_chemical_solution(x, var_names):
    """驗證化工問題的解是否合理"""
    valid = True
    
    for i, (val, name) in enumerate(zip(x, var_names)):
        # 檢查物理量是否為非負
        if val < -1e-10:
            print(f"Warning: {name} = {val:.4f} is negative (unphysical)")
            valid = False
            
        # 檢查濃度是否在 [0, 1] 範圍
        if 'composition' in name.lower() or 'fraction' in name.lower():
            if val < 0 or val > 1:
                print(f"Warning: {name} = {val:.4f} out of [0,1] range")
                valid = False
    
    # 檢查質量守恆
    if 'flow' in ' '.join(var_names).lower():
        total_in = sum(x[i] for i, name in enumerate(var_names) if 'in' in name.lower())
        total_出 = sum(x[i] for i, name in enumerate(var_names) if 'out' in name.lower())
        if abs(total_in - total_out) > 1e-6:
            print(f"Warning: Mass balance violated (in={total_in:.4f}, out={total_out:.4f})")
            valid = False
    
    if valid:
        print("All physical constraints satisfied")
    
    return valid
```

---

## 6. 程式設計最佳實踐

在實際應用中，撰寫穩健、高效的線性方程式求解程式需要遵循一些最佳實踐。

### 6.1 求解器選擇決策樹

```python
def choose_solver(A, b):
    """根據系統特性自動選擇最佳求解器"""
    import scipy.sparse as sp
    from scipy import linalg
    
    m, n = A.shape
    
    # 檢查是否為稀疏矩陣
    if sp.issparse(A):
        sparsity = 1 - A.nnz / (m * n)
        print(f"Sparse matrix detected (sparsity: {sparsity:.2%})")
        
        if m == n:
            from scipy.sparse.linalg import spsolve
            print("Using sparse direct solver (spsolve)")
            return lambda: spsolve(A, b)
        else:
            from scipy.sparse.linalg import lsqr
            print("Using sparse least-squares (lsqr)")
            return lambda: lsqr(A, b)[0]
    
    # 密集矩陣
    if m == n:
        # 方陣
        rank = np.linalg.matrix_rank(A)
        if rank == n:
            # 檢查條件數
            cond = np.linalg.cond(A)
            if cond < 1e10:
                print(f"Well-conditioned square system (cond={cond:.2e})")
                print("Using numpy.linalg.solve")
                return lambda: np.linalg.solve(A, b)
            else:
                print(f"Ill-conditioned system (cond={cond:.2e})")
                print("Using SVD-based solver")
                return lambda: np.linalg.lstsq(A, b, rcond=1e-10)[0]
        else:
            print(f"Singular square matrix (rank={rank} < {n})")
            print("Using pseudo-inverse")
            return lambda: np.linalg.pinv(A) @ b
    elif m < n:
        # 低確定系統
        print(f"Underdetermined system ({m} eq, {n} unknowns)")
        print("Using pseudo-inverse for minimum norm solution")
        return lambda: np.linalg.pinv(A) @ b
    else:
        # 過確定系統
        print(f"Overdetermined system ({m} eq, {n} unknowns)")
        print("Using least-squares solver")
        return lambda: np.linalg.lstsq(A, b, rcond=None)[0]

# 使用範例
A = np.random.rand(100, 100)
b = np.random.rand(100)

solver = choose_solver(A, b)
x = solver()
print(f"Solution norm: {np.linalg.norm(x):.4f}")
```

### 6.2 稀疏矩陣的建立與使用

對於大型稀疏系統，正確建立稀疏矩陣可大幅提升效率。

```python
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import spsolve
import time

# 方法 1: 從密集矩陣轉換（不推薦用於大矩陣）
A_dense = np.array([[10, 0, 0, 0, -2],
                    [0, 11, 0, -1, 0],
                    [0, 0, 12, 0, 0],
                    [0, -1, 0, 13, 0],
                    [-2, 0, 0, 0, 14]])
A_csr = csr_matrix(A_dense)

# 方法 2: 使用 COO 格式逐步建立（推薦）
row = np.array([0, 0, 1, 1, 2, 3, 3, 4, 4])
col = np.array([0, 4, 1, 3, 2, 1, 3, 0, 4])
data = np.array([10, -2, 11, -1, 12, -1, 13, -2, 14])
from scipy.sparse import coo_matrix
A_coo = coo_matrix((data, (row, col)), shape=(5, 5))
A_csr = A_coo.tocsr()

# 方法 3: 使用 lil_matrix 逐個設定元素（適合動態建立）
n = 1000
A_lil = lil_matrix((n, n))
for i in range(n):
    A_lil[i, i] = 10 + i  # 對角線
    if i > 0:
        A_lil[i, i-1] = -1  # 下對角線
    if i < n-1:
        A_lil[i, i+1] = -1  # 上對角線
A_csr_large = A_lil.tocsr()

# 方法 4: 使用 diags 建立帶狀矩陣（最快）
diagonals = [np.arange(1, n+1) + 10, -np.ones(n-1), -np.ones(n-1)]
A_diag = diags(diagonals, [0, -1, 1], format='csr')

# 比較求解速度
b_large = np.random.rand(n)

start = time.time()
x_sparse = spsolve(A_csr_large, b_large)
time_sparse = time.time() - start

# 對比密集矩陣（不建議用於大型系統）
# A_dense_large = A_csr_large.toarray()
# start = time.time()
# x_dense = np.linalg.solve(A_dense_large, b_large)
# time_dense = time.time() - start

print(f"Sparse solver time: {time_sparse:.4f} seconds")
# print(f"Dense solver time: {time_dense:.4f} seconds")
# print(f"Speedup: {time_dense/time_sparse:.2f}x")
```

### 6.3 結果驗證與錯誤檢查

**完整的驗證流程**：

```python
def solve_and_validate(A, b, method='auto', tol=1e-10):
    """
    求解線性方程組並進行完整驗證
    
    Parameters:
    -----------
    A : array_like
        係數矩陣
    b : array_like
        常數向量
    method : str
        求解方法 ('auto', 'solve', 'lstsq', 'pinv')
    tol : float
        容許誤差
    
    Returns:
    --------
    x : ndarray
        解向量
    info : dict
        診斷資訊
    """
    info = {}
    
    # 1. 基本檢查
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array")
    
    m, n = A.shape
    if m != len(b):
        raise ValueError(f"Inconsistent dimensions: A is {m}x{n}, b has length {len(b)}")
    
    info['shape'] = (m, n)
    info['method'] = method
    
    # 2. 秩分析
    rank_A = np.linalg.matrix_rank(A, tol=tol)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]), tol=tol)
    
    info['rank_A'] = rank_A
    info['rank_Ab'] = rank_Ab
    
    # 判斷系統類型
    if rank_A < rank_Ab:
        info['system_type'] = 'inconsistent'
        print("Warning: System is inconsistent (no solution)")
    elif rank_A == rank_Ab == n and m == n:
        info['system_type'] = 'unique'
    elif rank_A == rank_Ab < n:
        info['system_type'] = 'underdetermined'
    elif m > n:
        info['system_type'] = 'overdetermined'
    else:
        info['system_type'] = 'unknown'
    
    print(f"System type: {info['system_type']}")
    
    # 3. 條件數檢查（僅對方陣）
    if m == n and rank_A == n:
        cond = np.linalg.cond(A)
        info['condition_number'] = cond
        print(f"Condition number: {cond:.2e}")
        
        if cond > 1e12:
            print("Warning: System is severely ill-conditioned")
    
    # 4. 選擇求解方法
    if method == 'auto':
        if info['system_type'] == 'unique':
            method = 'solve'
        elif info['system_type'] == 'underdetermined':
            method = 'pinv'
        elif info['system_type'] == 'overdetermined':
            method = 'lstsq'
        else:
            method = 'lstsq'
    
    # 5. 求解
    try:
        if method == 'solve':
            x = np.linalg.solve(A, b)
        elif method == 'lstsq':
            result = np.linalg.lstsq(A, b, rcond=None)
            x = result[0]
            if len(result[1]) > 0:
                info['residual_norm_squared'] = result[1][0]
        elif method == 'pinv':
            A_pinv = np.linalg.pinv(A, rcond=tol)
            x = A_pinv @ b
        else:
            raise ValueError(f"Unknown method: {method}")
        
        info['solution_found'] = True
        
    except np.linalg.LinAlgError as e:
        print(f"Error during solving: {e}")
        info['solution_found'] = False
        info['error'] = str(e)
        return None, info
    
    # 6. 驗證解
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)
    relative_residual = residual_norm / (np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b))
    
    info['residual_norm'] = residual_norm
    info['relative_residual'] = relative_residual
    info['solution_norm'] = np.linalg.norm(x)
    
    print(f"Residual norm: {residual_norm:.2e}")
    print(f"Relative residual: {relative_residual:.2e}")
    
    if relative_residual < tol:
        print("✓ Solution verified successfully")
        info['verified'] = True
    else:
        print("✗ Warning: Large residual, solution may be inaccurate")
        info['verified'] = False
    
    return x, info

# 使用範例
A_test = np.array([[3, 2, -1],
                   [2, -2, 4],
                   [-1, 0.5, -1]])
b_test = np.array([1, -2, 0])

x, info = solve_and_validate(A_test, b_test)
print("\nSolution:", x)
print("\nDiagnostics:")
for key, value in info.items():
    print(f"  {key}: {value}")
```

### 6.4 錯誤處理與除錯技巧

```python
class LinearSystemSolver:
    """線性方程組求解器類別，包含完整錯誤處理"""
    
    def __init__(self, A, b, name="System"):
        self.name = name
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.x = None
        self.diagnostics = {}
        
    def diagnose(self):
        """診斷系統特性"""
        print(f"\n{'='*50}")
        print(f"Diagnosing: {self.name}")
        print(f"{'='*50}")
        
        m, n = self.A.shape
        print(f"Matrix size: {m} × {n}")
        print(f"Number of equations: {m}")
        print(f"Number of unknowns: {n}")
        
        # 檢查秩
        rank_A = np.linalg.matrix_rank(self.A)
        rank_Ab = np.linalg.matrix_rank(np.column_stack([self.A, self.b]))
        print(f"Rank of A: {rank_A}")
        print(f"Rank of [A|b]: {rank_Ab}")
        
        # 檢查行列式（方陣）
        if m == n:
            det_A = np.linalg.det(self.A)
            print(f"Determinant: {det_A:.2e}")
            
            if abs(det_A) < 1e-10:
                print("⚠ Matrix is nearly singular")
            
            # 條件數
            if rank_A == n:
                cond = np.linalg.cond(self.A)
                print(f"Condition number: {cond:.2e}")
                
                if cond > 1e10:
                    print("⚠ System is ill-conditioned")
        
        # 檢查對稱性
        if m == n:
            is_symmetric = np.allclose(self.A, self.A.T)
            print(f"Symmetric: {is_symmetric}")
        
        # 檢查稀疏度
        nonzero = np.count_nonzero(self.A)
        sparsity = 1 - nonzero / (m * n)
        print(f"Sparsity: {sparsity:.2%} ({nonzero} non-zero elements)")
        
        if sparsity > 0.9:
            print("💡 Consider using sparse matrix format")
        
        return rank_A, rank_Ab
    
    def solve(self, method='auto', verbose=True):
        """求解系統"""
        try:
            rank_A, rank_Ab = self.diagnose() if verbose else (
                np.linalg.matrix_rank(self.A),
                np.linalg.matrix_rank(np.column_stack([self.A, self.b]))
            )
            
            m, n = self.A.shape
            
            if rank_A < rank_Ab:
                raise ValueError("System is inconsistent (no solution)")
            
            # 選擇方法
            if method == 'auto':
                if m == n and rank_A == n:
                    self.x = np.linalg.solve(self.A, self.b)
                    method_used = 'solve'
                elif m > n:
                    self.x = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
                    method_used = 'lstsq'
                else:
                    self.x = np.linalg.pinv(self.A) @ self.b
                    method_used = 'pinv'
            else:
                method_used = method
                if method == 'solve':
                    self.x = np.linalg.solve(self.A, self.b)
                elif method == 'lstsq':
                    self.x = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
                elif method == 'pinv':
                    self.x = np.linalg.pinv(self.A) @ self.b
            
            if verbose:
                print(f"\n✓ Solution found using '{method_used}'")
            
            return self.x
            
        except np.linalg.LinAlgError as e:
            print(f"\n✗ Linear algebra error: {e}")
            raise
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            raise
    
    def verify(self, tol=1e-10):
        """驗證解"""
        if self.x is None:
            print("No solution to verify")
            return False
        
        print(f"\n{'='*50}")
        print("Verification")
        print(f"{'='*50}")
        
        residual = self.A @ self.x - self.b
        res_norm = np.linalg.norm(residual)
        res_max = np.max(np.abs(residual))
        
        print(f"Residual norm (L2): {res_norm:.2e}")
        print(f"Max residual (L∞): {res_max:.2e}")
        
        if res_norm < tol:
            print("✓ Verification PASSED")
            return True
        else:
            print("✗ Verification FAILED")
            print("  Equation-wise residuals:")
            for i, r in enumerate(residual):
                status = "✓" if abs(r) < tol else "✗"
                print(f"    Eq {i+1}: {r:+.2e} {status}")
            return False

# 使用範例
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]])
b = np.array([1, 2, 3])

solver = LinearSystemSolver(A, b, name="Test Chemical System")
x = solver.solve()
solver.verify()

print(f"\nFinal solution:\n{x}")
```

### 6.5 性能優化技巧總結

1. **矩陣格式選擇**
   - 密集矩陣：NumPy array
   - 稀疏矩陣（> 90% 零元素）：SciPy sparse matrix (CSR/CSC format)

2. **求解器選擇**
   - 小型系統 ($n < 1000$)：`np.linalg.solve()`
   - 大型密集系統：`scipy.linalg.solve()`
   - 大型稀疏系統：`scipy.sparse.linalg.spsolve()` 或迭代法

3. **重複求解優化**
   - 使用 LU 分解：`lu_factor()` + `lu_solve()`
   - 避免重複計算矩陣反矩陣

4. **數值穩定性**
   - 避免直接計算反矩陣
   - 檢查條件數
   - 對病態系統使用正規化或 SVD

5. **記憶體優化**
   - 使用稀疏矩陣格式
   - 原地運算 (in-place operations)
   - 避免不必要的矩陣複製

---

## 7. 總結

本單元介紹了使用 Python 求解線性聯立方程式的完整方法，包括：

**理論基礎**：
- 線性方程組的矩陣表示
- 解的存在性與唯一性判定（秩的概念）
- 三種系統類型：唯一解、無窮多解、無解

**NumPy 工具**：
- 基本函數：`solve()`, `lstsq()`, `pinv()`, `matrix_rank()`, `det()`, `inv()`
- 適用於小到中型密集矩陣系統

**SciPy 進階方法**：
- 密集矩陣：`linalg.solve()`, LU 分解
- 稀疏矩陣：`sparse.linalg.spsolve()`
- 迭代法：`cg()`, `gmres()`

**實務應用**：
- 化工物料平衡（摻合問題）
- 反應器網絡分析
- 熱交換器能量平衡
- 循環流處理

**最佳實踐**：
- 根據系統特性選擇合適求解器
- 完整的驗證與錯誤處理
- 性能優化技巧

**下一步學習**：
- 實際化工案例演練（Example 01-06）
- 非線性方程式求解（Unit 07）
- 常微分方程式求解（Unit 09）

---

## 參考資料

1. NumPy Documentation: Linear Algebra
   - https://numpy.org/doc/stable/reference/routines.linalg.html

2. SciPy Documentation: Linear Algebra
   - https://docs.scipy.org/doc/scipy/reference/linalg.html

3. SciPy Sparse Linear Algebra
   - https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

4. Cutlip, M. B., & Shacham, M. (1999). *Problem Solving in Chemical and Biochemical Engineering with POLYMATH, Excel, and MATLAB*. Prentice Hall.

5. Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing*. Cambridge University Press.

---

**下一個單元**：[Unit06_Example_01 - 液體摻合問題](Unit06_Example_01.md)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用
- 課程單元：Unit 06 線性聯立方程式求解
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-02-18

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---