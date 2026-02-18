# Unit06 Example 01 - 液體摻合問題

## 學習目標

在本範例中，我們將探討化工製程中常見的液體摻合問題。透過建立物料平衡方程式，將實際化工問題轉化為線性聯立方程組，並應用 NumPy 與 SciPy 的求解工具來計算各儲存槽的所需用量。

學習完本範例後，您將能夠：

- 建立多成分液體摻合的物料平衡方程式
- 將化工問題轉化為標準矩陣形式 $\mathbf{Ax} = \mathbf{b}$
- 使用 `numpy.linalg.solve()` 求解線性方程組
- 使用 `scipy.linalg.solve()` 進行求解並比較結果
- 驗證解的唯一性（秩判定）
- 檢查解的正確性（質量守恆、殘差分析）
- 解釋解的物理意義與實際應用

---

## 1. 問題描述

### 1.1 化工情境

某化工廠生產特殊配方的液體產品，需要從三個不同的儲存槽中取出原料進行摻合。每個儲存槽中的液體含有三種成分（A、B、C），但各槽的成分比例不同。

**儲存槽組成資料**：

| 槽號 | 成分 A (wt%) | 成分 B (wt%) | 成分 C (wt%) |
|------|-------------|-------------|-------------|
| 槽 1 | 50 | 30 | 20 |
| 槽 2 | 20 | 60 | 20 |
| 槽 3 | 30 | 10 | 60 |

**生產目標**：

- 目標產品總重量：100 kg
- 目標產品組成：
  - 成分 A：35 wt%（35 kg）
  - 成分 B：35 wt%（35 kg）
  - 成分 C：30 wt%（30 kg）

**求解問題**：需要從各槽取出多少重量的液體（ $V_1, V_2, V_3$ ），才能混合出符合目標組成的產品？

### 1.2 問題示意圖
![問題示意圖](outputs/figs/exam01_01.png)

---

## 2. 數學模型建立

### 2.1 物料平衡原理

根據質量守恆定律，對於每一個成分，其在產品中的總量等於各槽貢獻量的總和。

**成分 A 的物料平衡**：

$$
0.50 V_1 + 0.20 V_2 + 0.30 V_3 = 35
$$

**成分 B 的物料平衡**：

$$
0.30 V_1 + 0.60 V_2 + 0.10 V_3 = 35
$$

**成分 C 的物料平衡**：

$$
0.20 V_1 + 0.20 V_2 + 0.60 V_3 = 30
$$

**總質量守恆**：

$$
V_1 + V_2 + V_3 = 100
$$

### 2.2 矩陣形式表示

將上述方程式整理為標準的線性聯立方程組 $\mathbf{Ax} = \mathbf{b}$ ：

**係數矩陣 $\mathbf{A}$ （4×3）**：

$$
\mathbf{A} = \begin{bmatrix}
0.50 & 0.20 & 0.30 \\
0.30 & 0.60 & 0.10 \\
0.20 & 0.20 & 0.60 \\
1.00 & 1.00 & 1.00
\end{bmatrix}
$$

**未知數向量 $\mathbf{x}$ （3×1）**：

$$
\mathbf{x} = \begin{bmatrix}
V_1 \\ V_2 \\ V_3
\end{bmatrix}
$$

**常數向量 $\mathbf{b}$ （4×1）**：

$$
\mathbf{b} = \begin{bmatrix}
35 \\ 35 \\ 30 \\ 100
\end{bmatrix}
$$

### 2.3 系統類型分析

觀察係數矩陣：
- 方程式數量 $m = 4$
- 未知數數量 $n = 3$
- 系統類型：**過確定系統** (overdetermined system)，因為 $m > n$

對於過確定系統，可能出現兩種情況：
1. **相容系統**：存在精確解（當額外方程式是其他方程式的線性組合時）
2. **不相容系統**：不存在精確解（需要使用最小平方法）

**判斷準則**：檢查係數矩陣 $\mathbf{A}$ 與擴增矩陣 $[\mathbf{A} \mid \mathbf{b}]$ 的秩：
- 若 $\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ ，系統相容
- 若 $\mathrm{rank}(\mathbf{A}) < \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ ，系統不相容

---

## 3. NumPy 求解方法

### 3.1 使用 np.linalg.lstsq() 求解

由於這是過確定系統，我們使用 `numpy.linalg.lstsq()` 函數來求解。此函數會找到使殘差平方和 $\|\mathbf{Ax} - \mathbf{b}\|^2$ 最小的解。

**函數語法**：

```python
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

**返回值說明**：
- `x`：最小平方解
- `residuals`：殘差平方和（僅當 $m > n$ 且 $\mathrm{rank}(\mathbf{A}) = n$ 時返回）
- `rank`：矩陣 $\mathbf{A}$ 的秩
- `s`：矩陣 $\mathbf{A}$ 的奇異值

**程式碼範例**：

```python
import numpy as np

# 建立係數矩陣 A
A = np.array([[0.50, 0.20, 0.30],
              [0.30, 0.60, 0.10],
              [0.20, 0.20, 0.60],
              [1.00, 1.00, 1.00]])

# 建立常數向量 b
b = np.array([35, 35, 30, 100])

# 求解
result = np.linalg.lstsq(A, b, rcond=None)
V = result[0]
residuals = result[1]
rank = result[2]

print("NumPy 最小平方解:")
print(f"V1 = {V[0]:.4f} kg")
print(f"V2 = {V[1]:.4f} kg")
print(f"V3 = {V[2]:.4f} kg")
print(f"\n矩陣秩: {rank}")
print(f"殘差平方和: {residuals[0] if len(residuals) > 0 else 0:.2e}")
```

### 3.2 驗證解的唯一性

檢查係數矩陣的秩：

```python
# 檢查秩
rank_A = np.linalg.matrix_rank(A)
rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))

print(f"rank(A) = {rank_A}")
print(f"rank([A|b]) = {rank_Ab}")

if rank_A == rank_Ab:
    if rank_A == 3:
        print("✓ 系統相容，有唯一解")
    else:
        print("⚠ 系統相容，但有無窮多組解")
else:
    print("✗ 系統不相容，無精確解（使用最小平方解）")
```

### 3.3 質量守恆檢查

驗證解是否滿足物料平衡：

```python
# 驗證各成分物料平衡
comp_A = 0.50*V[0] + 0.20*V[1] + 0.30*V[2]
comp_B = 0.30*V[0] + 0.60*V[1] + 0.10*V[2]
comp_C = 0.20*V[0] + 0.20*V[1] + 0.60*V[2]
total = V[0] + V[1] + V[2]

print("\n物料平衡驗證:")
print(f"成分 A: {comp_A:.4f} kg (目標: 35 kg)")
print(f"成分 B: {comp_B:.4f} kg (目標: 35 kg)")
print(f"成分 C: {comp_C:.4f} kg (目標: 30 kg)")
print(f"總重量: {total:.4f} kg (目標: 100 kg)")

# 計算誤差
tol = 1e-6
errors = [
    abs(comp_A - 35),
    abs(comp_B - 35),
    abs(comp_C - 30),
    abs(total - 100)
]

if all(e < tol for e in errors):
    print("\n✓ 所有物料平衡方程式滿足！")
else:
    print("\n⚠ 存在物料平衡誤差")
```

---

## 4. SciPy 求解方法

### 4.1 使用 scipy.linalg.lstsq() 求解

SciPy 也提供了 `lstsq()` 函數，功能類似但提供更多控制選項。

**程式碼範例**：

```python
from scipy import linalg

# 使用 SciPy 求解
V_scipy, residuals_scipy, rank_scipy, s_scipy = linalg.lstsq(A, b)

print("SciPy 最小平方解:")
print(f"V1 = {V_scipy[0]:.4f} kg")
print(f"V2 = {V_scipy[1]:.4f} kg")
print(f"V3 = {V_scipy[2]:.4f} kg")
print(f"\n矩陣秩: {rank_scipy}")
print(f"殘差平方和: {residuals_scipy[0] if len(residuals_scipy) > 0 else 0:.2e}")
```

### 4.2 NumPy 與 SciPy 結果比較

```python
# 比較兩種方法的結果
diff = np.abs(V - V_scipy)

print("\nNumPy vs SciPy 差異:")
print(f"V1 差異: {diff[0]:.2e}")
print(f"V2 差異: {diff[1]:.2e}")
print(f"V3 差異: {diff[2]:.2e}")

if np.allclose(V, V_scipy, rtol=1e-10):
    print("\n✓ NumPy 與 SciPy 結果一致")
```

### 4.3 條件數分析

檢查系統的數值穩定性：

```python
# 計算條件數（使用前三列組成方陣）
A_square = A[:3, :]
cond = np.linalg.cond(A_square)

print(f"\n係數矩陣條件數: {cond:.2e}")

if cond < 1e3:
    print("✓ 系統數值穩定")
elif cond < 1e6:
    print("⚠ 系統稍微病態")
else:
    print("✗ 系統嚴重病態，解可能不可靠")
```

---

## 5. 結果分析與討論

### 5.1 解的物理意義

從各儲存槽取出的重量必須滿足：
- **非負性**： $V_1, V_2, V_3 \geq 0$ （不能取出負重量）
- **可行性**：各槽有足夠的液體可供取用
- **操作性**：實際操作時的精度與誤差範圍

```python
# 檢查解的物理合理性
print("\n解的物理性檢查:")
for i, v in enumerate(V, 1):
    if v >= 0:
        print(f"✓ V{i} = {v:.4f} kg (非負)")
    else:
        print(f"✗ V{i} = {v:.4f} kg (負值，物理上不合理)")
```

### 5.2 敏感度分析

如果目標組成稍有變化，各槽用量會如何改變？

**範例**：將成分 A 的目標從 35% 改為 40%

```python
# 調整目標組成
b_new = np.array([40, 35, 25, 100])  # A增加，C減少以保持總量

V_new = np.linalg.lstsq(A, b_new, rcond=None)[0]

print("\n目標組成變化後的用量:")
print(f"V1: {V[0]:.2f} → {V_new[0]:.2f} kg (變化 {V_new[0]-V[0]:+.2f})")
print(f"V2: {V[1]:.2f} → {V_new[1]:.2f} kg (變化 {V_new[1]-V[1]:+.2f})")
print(f"V3: {V[2]:.2f} → {V_new[2]:.2f} kg (變化 {V_new[2]-V[2]:+.2f})")
```

### 5.3 過確定系統的意義

本問題有 4 個方程式但只有 3 個未知數，為什麼？

- **3 個成分平衡**：對每個成分的質量守恆
- **1 個總質量平衡**：總重量約束

實際上，如果成分比例設定正確（A% + B% + C% = 100%），總質量方程式可以由三個成分方程式相加得到，因此這 4 個方程式並非完全獨立。

**驗證線性相依性**：

```python
# 檢查：三個成分方程式相加是否等於總質量方程式
sum_of_comp = A[0] + A[1] + A[2]
total_eqn = A[3]

print("\n方程式相依性檢查:")
print(f"成分方程式係數和: {sum_of_comp}")
print(f"總質量方程式係數: {total_eqn}")
print(f"是否相等: {np.allclose(sum_of_comp, total_eqn)}")

# 這表示第 4 個方程式可由前 3 個線性組合得到
# 因此 rank(A) = 3，系統實際上有唯一解
```

---

## 6. 完整程式碼示範

以下是完整的求解與驗證程式：

```python
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# ==================== 問題定義 ====================
print("="*60)
print("液體摻合問題 - 物料平衡求解")
print("="*60)

# 各槽組成（重量百分比 → 小數）
tank_compositions = {
    1: {'A': 0.50, 'B': 0.30, 'C': 0.20},
    2: {'A': 0.20, 'B': 0.60, 'C': 0.20},
    3: {'A': 0.30, 'B': 0.10, 'C': 0.60}
}

# 目標產品
target_total = 100  # kg
target_composition = {'A': 35, 'B': 35, 'C': 30}  # kg

print("\n各儲存槽組成:")
for tank_id, comp in tank_compositions.items():
    print(f"槽 {tank_id}: A={comp['A']*100:.0f}%, "
          f"B={comp['B']*100:.0f}%, C={comp['C']*100:.0f}%")

print(f"\n目標產品 ({target_total} kg):")
for component, amount in target_composition.items():
    print(f"成分 {component}: {amount} kg ({amount/target_total*100:.0f}%)")

# ==================== 建立方程組 ====================
A = np.array([
    [0.50, 0.20, 0.30],  # 成分 A
    [0.30, 0.60, 0.10],  # 成分 B
    [0.20, 0.20, 0.60],  # 成分 C
    [1.00, 1.00, 1.00]   # 總質量
])

b = np.array([35, 35, 30, 100])

print("\n係數矩陣 A (4×3):")
print(A)
print("\n常數向量 b:")
print(b)

# ==================== 秩分析 ====================
rank_A = np.linalg.matrix_rank(A)
rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))

print(f"\nrank(A) = {rank_A}")
print(f"rank([A|b]) = {rank_Ab}")

if rank_A == rank_Ab == A.shape[1]:
    print("✓ 系統相容且有唯一解")
elif rank_A == rank_Ab < A.shape[1]:
    print("⚠ 系統相容但有無窮多組解")
else:
    print("✗ 系統不相容，僅能求最小平方解")

# ==================== NumPy 求解 ====================
print("\n" + "="*60)
print("NumPy 求解")
print("="*60)

V_numpy, residuals_numpy, rank_numpy, s_numpy = np.linalg.lstsq(A, b, rcond=None)

print(f"\n各槽所需用量:")
print(f"槽 1: {V_numpy[0]:.4f} kg")
print(f"槽 2: {V_numpy[1]:.4f} kg")
print(f"槽 3: {V_numpy[2]:.4f} kg")
print(f"總計: {V_numpy.sum():.4f} kg")

if len(residuals_numpy) > 0:
    print(f"\n殘差平方和: {residuals_numpy[0]:.2e}")

# ==================== SciPy 求解 ====================
print("\n" + "="*60)
print("SciPy 求解")
print("="*60)

V_scipy, residuals_scipy, rank_scipy, s_scipy = linalg.lstsq(A, b)

print(f"\n各槽所需用量:")
print(f"槽 1: {V_scipy[0]:.4f} kg")
print(f"槽 2: {V_scipy[1]:.4f} kg")
print(f"槽 3: {V_scipy[2]:.4f} kg")

# ==================== 結果比較 ====================
print("\n" + "="*60)
print("NumPy vs SciPy 比較")
print("="*60)

diff = np.abs(V_numpy - V_scipy)
print(f"\n最大差異: {np.max(diff):.2e}")

if np.allclose(V_numpy, V_scipy, rtol=1e-10):
    print("✓ 兩種方法結果一致")

# ==================== 物料平衡驗證 ====================
print("\n" + "="*60)
print("物料平衡驗證")
print("="*60)

V = V_numpy  # 使用 NumPy 結果

actual_A = A[0] @ V
actual_B = A[1] @ V
actual_C = A[2] @ V
actual_total = A[3] @ V

print(f"\n成分 A: {actual_A:.4f} kg (目標: 35 kg, 誤差: {abs(actual_A-35):.2e})")
print(f"成分 B: {actual_B:.4f} kg (目標: 35 kg, 誤差: {abs(actual_B-35):.2e})")
print(f"成分 C: {actual_C:.4f} kg (目標: 30 kg, 誤差: {abs(actual_C-30):.2e})")
print(f"總重量: {actual_total:.4f} kg (目標: 100 kg, 誤差: {abs(actual_total-100):.2e})")

# 殘差向量
residual_vector = A @ V - b
residual_norm = np.linalg.norm(residual_vector)

print(f"\n殘差向量範數: {residual_norm:.2e}")

if residual_norm < 1e-10:
    print("✓ 所有物料平衡方程式精確滿足！")

# ==================== 物理性檢查 ====================
print("\n" + "="*60)
print("解的物理性檢查")
print("="*60)

all_positive = True
for i, v in enumerate(V, 1):
    if v >= -1e-10:  # 允許微小的數值誤差
        print(f"✓ V{i} = {v:.4f} kg (非負)")
    else:
        print(f"✗ V{i} = {v:.4f} kg (負值，物理上不合理)")
        all_positive = False

if all_positive:
    print("\n✓ 所有解均為非負值，物理上合理")

# ==================== 視覺化結果 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子圖 1: 各槽用量
ax1 = axes[0]
tanks = ['Tank 1', 'Tank 2', 'Tank 3']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax1.bar(tanks, V, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Required Amount (kg)', fontsize=12)
ax1.set_title('Required Amount from Each Tank', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 在長條上標註數值
for bar, val in zip(bars, V):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f} kg',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 子圖 2: 成分組成驗證
ax2 = axes[1]
components = ['A', 'B', 'C']
target_amounts = [35, 35, 30]
actual_amounts = [actual_A, actual_B, actual_C]

x = np.arange(len(components))
width = 0.35

bars1 = ax2.bar(x - width/2, target_amounts, width, label='Target', 
                color='lightblue', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, actual_amounts, width, label='Actual', 
                color='lightcoral', edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Component', fontsize=12)
ax2.set_ylabel('Amount (kg)', fontsize=12)
ax2.set_title('Component Mass Balance Verification', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(components)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ 視覺化圖表已生成")

# ==================== 條件數分析 ====================
print("\n" + "="*60)
print("數值穩定性分析")
print("="*60)

A_square = A[:3, :]  # 取前三列作為方陣
cond_num = np.linalg.cond(A_square)

print(f"\n係數矩陣條件數: {cond_num:.2e}")

if cond_num < 1e3:
    print("✓ 系統數值穩定 (條件數 < 10³)")
elif cond_num < 1e6:
    print("⚠ 系統稍微病態 (10³ ≤ 條件數 < 10⁶)")
else:
    print("✗ 系統嚴重病態 (條件數 ≥ 10⁶)")

print("\n" + "="*60)
print("求解完成！")
print("="*60)
```

---

## 7. 總結

### 重點回顧

**1. 問題建模**
- 從實際化工問題建立物料平衡方程式
- 將物料平衡轉化為線性聯立方程組
- 識別系統類型（過確定系統）

**2. 求解方法**
- NumPy: `np.linalg.lstsq()` 求最小平方解
- SciPy: `scipy.linalg.lstsq()` 提供相同功能
- 兩種方法結果一致，驗證正確性

**3. 解的驗證**
- 秩分析：判斷解的存在性與唯一性
- 殘差檢查：確認解的精確度
- 物理性檢查：確認解的實際可行性（非負值）
- 質量守恆：驗證物料平衡

**4. 數值穩定性**
- 條件數分析：評估系統的病態程度
- 本問題條件數小，系統數值穩定

### 延伸思考

1. **如果某個槽的庫存不足怎麼辦？**
   - 需要增加不等式約束： $V_i \leq V_{i,\max}$
   - 轉化為線性規劃問題

2. **如果目標組成無法精確達成怎麼辦？**
   - 使用最小平方法找到最接近的解
   - 分析殘差，調整可接受的誤差範圍

3. **如果有更多儲存槽可以選擇？**
   - 增加決策變數，變成低確定系統
   - 可能有無窮多組解，需要額外的優化目標（如成本最小化）

### 實際應用

液體摻合問題廣泛應用於：
- **石油煉製**：調配不同規格的汽油、柴油
- **塗料工業**：混合顏料達到特定色彩
- **飲料製造**：調配飲料成分比例
- **化妝品產業**：混合原料達到配方要求

---

**課程資訊**
- 課程名稱：電腦在化工上之應用
- 課程單元：Unit06 線性聯立方程式之求解
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-02-18

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
