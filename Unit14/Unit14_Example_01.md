# Unit14 Example 01 - 製程品質數據描述統計與常態性分析

## 學習目標

本範例以**製程品質數據描述統計與常態性分析**為題，示範如何使用 `scipy.stats` 對化工廠連續採樣的產品純度數據進行全面的統計分析，包含描述統計量計算、機率分布鑑別、常態性檢定與圖形診斷。

學習完本範例後，您將能夠：

- 使用 `scipy.stats.describe()` 一次輸出樣本數、最大最小值、均值、變異數、偏態與峰態等描述統計量
- 了解偏態係數 (Skewness) 與峰態係數 (Kurtosis) 的物理意義，並判斷數據分布形狀
- 以 `scipy.stats.norm.fit()` 估計常態分布參數，並繪製直方圖疊加 PDF 曲線
- 使用 `scipy.stats.probplot()` 繪製 Q-Q 圖，以視覺化方式評估常態性
- 執行 Shapiro-Wilk 檢定 `scipy.stats.shapiro()` 與 D'Agostino-Pearson 檢定 `scipy.stats.normaltest()`
- 使用 `scipy.stats.weibull_min.fit()` 擬合 Weibull 分布，並以 AIC 比較兩種分布的擬合優劣
- 繪製箱型圖並識別 IQR 離群值，與 $3\sigma$ 法則結果進行比較

---

## 目錄

1. [問題描述與實驗數據](#1-問題描述與實驗數據)
2. [描述統計量計算](#2-描述統計量計算)
3. [直方圖與常態分布 PDF 疊加](#3-直方圖與常態分布-pdf-疊加)
4. [Q-Q 圖 — 常態性視覺診斷](#4-q-q-圖--常態性視覺診斷)
5. [正式常態性檢定](#5-正式常態性檢定)
6. [分布擬合比較：常態 vs Weibull (AIC)](#6-分布擬合比較常態-vs-weibull-aic)
7. [箱型圖與離群值分析](#7-箱型圖與離群值分析)
8. [綜合結論](#8-綜合結論)

---

## 1. 問題描述與實驗數據

### 1.1 背景說明

某化工廠連續生產高純度化學品，為確保出廠產品品質符合規格，品管部門對連續生產的 50 批產品各取一個樣本，量測其純度（wt%）。已知工廠設計目標純度為 **95 wt%**，規格範圍為 $93 \sim 97$ wt%。

此次分析的目標為：

1. 計算並解讀這 50 批次數據的完整描述統計量
2. 判斷數據是否符合常態分布（為後續假設檢定建立前提）
3. 比較常態分布與 Weibull 分布對此數據的擬合優劣

### 1.2 模擬數據說明

由於此為教學範例，我們使用 `numpy.random.default_rng()` 生成具有**輕微右偏態**的模擬純度數據，以模擬真實製程中可能出現的分布偏移情形。

- 基礎分布：常態分布 $\mathcal{N}(\mu=95, \sigma=1.2)$
- 加入輕微右偏：混入少量高值樣本（模擬偶發的高純度批次）
- 隨機種子：`seed=42`，確保結果可重現

> **注意**：真實案例中，數據應來自實驗量測。此處模擬數據僅供教學示範使用。

### 1.3 原始數據

以下為 50 筆模擬純度數據（wt%）：

```python
# 隨機種子固定為 42，可自行重現
rng = numpy.random.default_rng(42)
# 主體：常態分布 N(95, 1.2^2)，45 筆
# 右尾：N(97.5, 0.5^2)，5 筆（模擬高純度偶發批次）
purity = np.concatenate([
    rng.normal(loc=95.0, scale=1.2, size=45),
    rng.normal(loc=97.5, scale=0.5, size=5)
])
```

---

## 2. 描述統計量計算

### 2.1 `scipy.stats.describe()` 函式說明

`scipy.stats.describe(a)` 函式一次計算並回傳 6 項統計量，使用方式如下：

```python
from scipy import stats
result = stats.describe(data)
```

回傳值為具名元組 (NamedTuple)，包含：

| 屬性 | 說明 |
|------|------|
| `nobs` | 樣本數 $n$ |
| `minmax` | (最小值, 最大值) |
| `mean` | 樣本均值 $\bar{x}$ |
| `variance` | 樣本變異數 $s^2$（分母為 $n-1$） |
| `skewness` | 偏態係數 $g_1$ |
| `kurtosis` | 峰態係數 $g_2$（超額峰態，常態分布為 0） |

### 2.2 偏態係數 (Skewness)

偏態係數 $g_1$ 描述分布的左右對稱性：

$$
g_1 = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3
$$

- $g_1 = 0$：對稱分布（常態分布）
- $g_1 > 0$：右偏（正偏），右尾較長，均值 > 中位數
- $g_1 < 0$：左偏（負偏），左尾較長，均值 < 中位數

### 2.3 峰態係數 (Kurtosis)

峰態係數 $g_2$（超額峰態）描述分布的尖峰與厚尾程度：

$$
g_2 = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^4 - 3
$$

- $g_2 = 0$：常態分布（中峰態，Mesokurtic）
- $g_2 > 0$：高峰態（Leptokurtic），尖峰且厚尾
- $g_2 < 0$：低峰態（Platykurtic），平坦且薄尾

### 2.4 百分位數與四分位距

百分位數可用 `numpy.percentile()` 計算：

```python
Q1 = np.percentile(data, 25)   # 第 25 百分位數
Q3 = np.percentile(data, 75)   # 第 75 百分位數
IQR = Q3 - Q1                  # 四分位距
```

---

## 3. 直方圖與常態分布 PDF 疊加

### 3.1 `scipy.stats.norm.fit()` 估計分布參數

使用最大概似估計法 (MLE) 估計常態分布的均值 $\hat{\mu}$ 與標準差 $\hat{\sigma}$：

```python
mu_hat, sigma_hat = stats.norm.fit(data)
```

### 3.2 繪製直方圖與 PDF 疊加

繪製直方圖時，需設定 `density=True` 使直方圖縱軸為機率密度，才能與 PDF 曲線比較：

```python
plt.hist(data, bins=12, density=True, alpha=0.6, label='Sample Histogram')

x = np.linspace(data.min() - 1, data.max() + 1, 300)
pdf_fitted = stats.norm.pdf(x, mu_hat, sigma_hat)
plt.plot(x, pdf_fitted, 'r-', lw=2, label=f'Normal PDF (μ={mu_hat:.2f}, σ={sigma_hat:.2f})')
```

### 3.3 常態分布 PDF 公式

$$
f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

---

## 4. Q-Q 圖 — 常態性視覺診斷

### 4.1 Q-Q 圖原理

Q-Q 圖（Quantile-Quantile Plot）是比較樣本分位數與理論分布分位數的散佈圖。若數據完全符合常態分布，所有點應緊密落在對角參考線上。

- **偏離參考線**：偏離程度越大，常態性越差
- **S 形曲線**：表示厚尾分布
- **上揚/下彎**：表示分布有偏態

### 4.2 `scipy.stats.probplot()` 使用方式

```python
fig, ax = plt.subplots()
stats.probplot(data, dist="norm", plot=ax)
ax.set_title('Normal Q-Q Plot')
```

`probplot()` 會自動：
1. 計算樣本的理論常態分位數（橫軸）
2. 計算樣本的實際排序值（縱軸）
3. 繪製最佳擬合線並顯示 $R^2$ 值

---

## 5. 正式常態性檢定

### 5.1 假設設定

$$
H_0: \text{數據來自常態分布}
$$

$$
H_1: \text{數據不來自常態分布}
$$

決策準則：若 $p < 0.05$，則在 5% 顯著水準下拒絕 $H_0$，認為數據顯著偏離常態分布。

### 5.2 Shapiro-Wilk 檢定

適用於小樣本（建議 $n < 50$），對常態分布偏離具有較高的統計能力：

```python
stat_sw, p_sw = stats.shapiro(data)
```

- `stat_sw`：W 統計量（越接近 1.0 越符合常態）
- `p_sw`：p 值

### 5.3 D'Agostino-Pearson 檢定（`normaltest`）

結合偏態與峰態的組合檢定，適用於較大樣本：

```python
stat_dp, p_dp = stats.normaltest(data)
```

- `stat_dp`：$K^2$ 統計量，由偏態 $z_1^2$ 與峰態 $z_2^2$ 組合而成（$K^2 = z_1^2 + z_2^2$）
- `p_dp`：p 值（基於卡方分布，自由度為 2）

### 5.4 兩種檢定方法比較

| 方法 | 檢定統計量 | 原假設 | 建議樣本量 |
|------|-----------|--------|-----------|
| Shapiro-Wilk | W | 數據來自常態分布 | $8 \le n \le 2000$ |
| D'Agostino-Pearson | $K^2$ | 偏態 = 0 且峰態 = 0 | $n \ge 8$ |

---

## 6. 分布擬合比較：常態 vs Weibull (AIC)

### 6.1 Weibull 分布擬合

Weibull 分布常用於描述工程材料壽命或製程失效時間。此處嘗試以其擬合純度數據，觀察是否比常態分布更適合：

```python
c_hat, loc_hat, scale_hat = stats.weibull_min.fit(data, floc=data.min()-0.01)
```

> **注意**：`floc` 參數固定位置參數，確保所有數據值大於 `loc`，避免 MLE 數值問題。

Weibull 分布的 PDF（三參數形式，位置參數固定）：

$$
f(x; c, \lambda) = \frac{c}{\lambda} \left(\frac{x - \mu_0}{\lambda}\right)^{c-1} \exp\left[-\left(\frac{x - \mu_0}{\lambda}\right)^c\right]
$$

其中 $c$ 為形狀參數、 $\lambda$ 為比例參數、 $\mu_0$ 為位置參數（固定）。

### 6.2 AIC 準則比較模型

赤池資訊準則 (Akaike Information Criterion, AIC) 是常用的模型選擇工具，平衡模型的擬合優度與參數個數：

$$
\mathrm{AIC} = 2k - 2\ln(\hat{L})
$$

其中：
- $k$：模型自由參數個數
- $\hat{L}$：資料在最佳參數下的最大概似值

**AIC 越小，模型越優。** 計算方式：

```python
# 常態分布：k = 2 (mu, sigma)
log_lik_norm = np.sum(stats.norm.logpdf(data, mu_hat, sigma_hat))
AIC_norm = 2 * 2 - 2 * log_lik_norm

# Weibull 分布：k = 2 (c, scale，loc 固定)
log_lik_weibull = np.sum(stats.weibull_min.logpdf(data, c_hat, loc_hat, scale_hat))
AIC_weibull = 2 * 2 - 2 * log_lik_weibull
```

---

## 7. 箱型圖與離群值分析

### 7.1 箱型圖 (Box Plot) 結構說明

箱型圖以五數摘要（Five-Number Summary）呈現數據分布：

| 線段/點 | 說明 |
|---------|------|
| 箱體下緣 | 第 25 百分位數 $Q_1$ |
| 箱體中線 | 中位數 $Q_2$（第 50 百分位數） |
| 箱體上緣 | 第 75 百分位數 $Q_3$ |
| 下鬚線末端 | $\max(x_{\min},\ Q_1 - 1.5 \times \mathrm{IQR})$ |
| 上鬚線末端 | $\min(x_{\max},\ Q_3 + 1.5 \times \mathrm{IQR})$ |
| 菱形點（⬦） | 超出鬚線範圍的離群值 |

### 7.2 IQR 法離群值定義

四分位距法（Tukey's Fence）：

- **下界**：$Q_1 - 1.5 \times \mathrm{IQR}$
- **上界**：$Q_3 + 1.5 \times \mathrm{IQR}$

超出此範圍的數據點即被判定為**離群值 (Outlier)**。

### 7.3 $3\sigma$ 法則

對符合常態分布的數據，約有 **99.73%** 的數值落在 $[\bar{x} - 3s,\ \bar{x} + 3s]$ 範圍內。超出此範圍者視為異常點。

$$
P(\bar{x} - 3s \le X \le \bar{x} + 3s) \approx 99.73\%
$$

兩種方法的比較：

| 方法 | 適用情境 | 抗干擾性 |
|------|---------|---------|
| IQR 法 | 一般數據，不假設分布 | 較佳（不受極端值影響） |
| $3\sigma$ 法 | 假設數據近似常態分布 | 較差（均值與標準差受離群值影響） |

---

## 8. 綜合結論

本範例展示了完整的製程品質數據統計分析流程：

1. **描述統計**：`scipy.stats.describe()` 快速輸出 6 項核心統計量；偏態 $g_1 > 0$ 反映右偏特性，源於混入少量高純度批次
2. **直方圖分析**：疊加常態 PDF 曲線後，可目視分布右尾較長呈輕微偏態
3. **Q-Q 圖**：右尾部分偏離參考線，視覺確認右偏特性
4. **常態性檢定**：Shapiro-Wilk 與 D'Agostino-Pearson 均對右偏數據靈敏，小樣本建議以 Shapiro-Wilk 為主要參考
5. **AIC 比較**：對於輕微偏態數據，常態分布與 Weibull 分布的 AIC 相近；若 Weibull AIC 明顯較小，則建議改用 Weibull 模型
6. **箱型圖**：IQR 法識別出右尾少數高純度批次為離群值；$3\sigma$ 法因受這些點的拉升影響，識別結果略有差異

### 關鍵 Python 函式速查

| 函式 | 用途 |
|------|------|
| `scipy.stats.describe(data)` | 一次輸出 6 項描述統計量 |
| `scipy.stats.norm.fit(data)` | MLE 估計常態分布參數 |
| `scipy.stats.probplot(data, plot=ax)` | 繪製 Q-Q 圖 |
| `scipy.stats.shapiro(data)` | Shapiro-Wilk 常態性檢定 |
| `scipy.stats.normaltest(data)` | D'Agostino-Pearson 常態性檢定 |
| `scipy.stats.weibull_min.fit(data)` | MLE 估計 Weibull 分布參數 |
| `scipy.stats.weibull_min.logpdf(data, ...)` | 計算 Weibull 對數概似值 |

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit14 統計分析 — 化工案例一
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
