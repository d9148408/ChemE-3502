# Unit14 統計分析 (Statistical Analysis)

## 📚 單元簡介

在化工程序設計與操作中，**「統計分析（Statistical Analysis）」**是從量測數據中萃取有意義資訊、量化不確定性並支援工程決策的核心工具。製程品質變異、設備可靠度評估、催化劑效能比較、操作條件最適化等，無一不需要嚴謹的統計方法作為依據。

本單元以 **`scipy.stats`** 模組為核心，從描述統計、機率分布、信賴區間，到假設檢定（t 檢定、ANOVA、卡方檢定、無母數檢定）與相關分析，完整涵蓋化工數據分析的統計方法論。透過六個化工實際案例，從製程品質數據常態性分析、催化劑批次比較、多溫度條件 ANOVA、Arrhenius 回歸統計推論，到設備可靠度 Weibull 分析與製程能力指標計算，完整呈現統計分析在化工決策中的廣泛應用。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **計算並解讀描述統計量**：平均值、標準差、偏態係數、峰態係數，以及箱型圖中的 IQR 離群值判斷
2. **使用 `scipy.stats` 分布物件**：熟練運用 `rvs()`、`pdf()`、`cdf()`、`ppf()`、`sf()`、`fit()` 介面操作常態、t、F、卡方、指數、Weibull 等分布
3. **進行常態性檢定**：以 Shapiro-Wilk（小樣本）與 D'Agostino-Pearson（大樣本）檢定，並搭配 Q-Q 圖視覺化評估
4. **建構與解讀信賴區間**：母體均值（t 分布）與母體變異數（卡方分布）之信賴區間計算與圖形化解讀
5. **執行假設檢定並作正確統計決策**：單樣本/獨立兩樣本/配對 t 檢定、變異數齊一性檢定、單因子 ANOVA、卡方檢定、Mann-Whitney U 與 Kruskal-Wallis 無母數檢定
6. **進行相關分析與線性回歸統計推論**：Pearson/Spearman 相關係數、`scipy.stats.linregress()` 斜率與截距的 t 檢定、 $R^2$ 解讀、95% 信賴區間估算
7. **應用 Weibull 分布進行設備可靠度分析**：MLE 參數估計、可靠度函數 $R(t)$、故障率函數 $h(t)$、MTTF 與 $B_{10}$ 壽命計算
8. **計算製程能力指標**：$C_p$ 與 $C_{pk}$ 公式、良率計算（`scipy.stats.norm.cdf()`）、$\bar{X}$ 管制圖繪製

---

## 📖 單元內容架構

### 1️⃣ Unit14_Statistics — 統計分析理論與工具總覽

**檔案**：
- 講義：[Unit14_Statistics.md](Unit14_Statistics.md)
- 程式範例：[Unit14_Statistics.ipynb](Unit14_Statistics.ipynb)

**內容重點**：

#### 統計分析問題分類

| 分析類型 | 方法概述 | Python 工具 |
|---------|---------|------------|
| **描述統計** | 集中趨勢、離散程度、偏態/峰態 | `scipy.stats.describe()` |
| **機率分布** | 分布物件介面、PDF/CDF/PPF、參數估計 | `scipy.stats.norm`、`weibull_min` 等 |
| **常態性檢定** | Shapiro-Wilk、D'Agostino-Pearson、Q-Q 圖 | `scipy.stats.shapiro()`、`normaltest()` |
| **信賴區間** | 均值（t 分布）、變異數（卡方分布） | `scipy.stats.t.interval()` |
| **假設檢定** | t 檢定、ANOVA、卡方檢定、無母數 | `ttest_ind()`、`f_oneway()` 等 |
| **相關與回歸** | Pearson/Spearman 相關、線性回歸統計推論 | `pearsonr()`、`linregress()` |

**`scipy.stats` 分布物件統一介面**：

| 方法 | 用途 |
|------|------|
| `rvs(size)` | 隨機樣本生成 |
| `pdf(x)` / `pmf(x)` | 機率密度/質量函數 |
| `cdf(x)` | 累積分布函數 $F(x) = P(X \leq x)$ |
| `ppf(q)` | 分位數函數（`cdf` 之反函數） |
| `sf(x)` | 存活函數 $S(x) = 1 - F(x)$ |
| `fit(data)` | 最大概似估計（MLE）參數估計 |
| `interval(alpha)` | 計算包含 $\alpha$ 機率之對稱信賴區間 |

#### 假設檢定架構

$$
\text{決策準則：} p < \alpha \Rightarrow \text{拒絕 } H_0
$$

| 檢定類型 | 適用情境 | 函式 |
|---------|---------|------|
| 單樣本 t 檢定 | 樣本均值是否等於指定值 | `scipy.stats.ttest_1samp()` |
| 獨立兩樣本 t 檢定 | 兩獨立組均值比較 | `scipy.stats.ttest_ind()` |
| 配對樣本 t 檢定 | 同批樣本不同條件比較 | `scipy.stats.ttest_rel()` |
| 變異數齊一性檢定 | 各組變異數是否相等 | `scipy.stats.levene()` / `bartlett()` |
| 單因子 ANOVA | 三組以上均值比較 | `scipy.stats.f_oneway()` |
| 卡方適合度 | 觀測頻率vs理論分布 | `scipy.stats.chisquare()` |
| 卡方獨立性 | 兩類別變數獨立性 | `scipy.stats.chi2_contingency()` |
| Mann-Whitney U | 兩樣本中位數（無母數） | `scipy.stats.mannwhitneyu()` |
| Kruskal-Wallis | ANOVA 無母數版本 | `scipy.stats.kruskal()` |

#### 信賴區間公式

$$
\text{均值信賴區間（}\sigma\text{ 未知）：} \bar{x} \pm t_{\alpha/2,\, n-1} \cdot \frac{s}{\sqrt{n}}
$$

$$
\text{變異數信賴區間：} \left[\frac{(n-1)s^2}{\chi^2_{\alpha/2}},\ \frac{(n-1)s^2}{\chi^2_{1-\alpha/2}}\right]
$$

---

## 🧪 化工案例演練

### 📊 Example 01 — 製程品質數據描述統計與常態性分析

**檔案**：[Unit14_Example_01.md](Unit14_Example_01.md) | [Unit14_Example_01.ipynb](Unit14_Example_01.ipynb)

**問題概述**：某化工廠連續採樣 50 批產品純度數據 (wt%)，進行全面描述統計分析、分布鑑別，並以圖形判斷數據是否符合常態分布。

**數學工具**：`scipy.stats.describe()`、`shapiro()`、`normaltest()`、`probplot()`

**化工重點**：
- 使用 `scipy.stats.describe()` 輸出完整描述統計量（n, min, max, mean, variance, skewness, kurtosis）
- 繪製直方圖並疊加常態分布 PDF 曲線，以 `scipy.stats.norm.fit()` 以 MLE 估計參數
- 以 `scipy.stats.probplot()` 繪製 Q-Q 圖視覺化常態性評估
- 分別使用 `scipy.stats.shapiro()`（$n < 50$）與 `scipy.stats.normaltest()` 進行正式常態性檢定
- 使用 `scipy.stats.weibull_min.fit()` 嘗試 Weibull 分布擬合，以 AIC 比較常態 vs Weibull 擬合優劣
- 繪製箱型圖，標示 IQR 離群值，並與 $3\sigma$ 法則比較

---

### 🧫 Example 02 — 不同催化劑批次反應收率之假設檢定

**檔案**：[Unit14_Example_02.md](Unit14_Example_02.md) | [Unit14_Example_02.ipynb](Unit14_Example_02.ipynb)

**問題概述**：工廠欲評估新型催化劑 (B) 是否顯著優於現有催化劑 (A)，分別取 20 批實驗數據進行統計比較，顯著水準 $\alpha = 0.05$。

**數學工具**：`scipy.stats.t.interval()`、`scipy.stats.levene()`、`scipy.stats.ttest_ind()`

**化工重點**：
- 計算各組之 95% 信賴區間（以 `scipy.stats.t.interval()`），並視覺化區間重疊狀況
- 以 `scipy.stats.levene()` 進行 Levene 變異數齊一性檢定，判斷是否使用 Welch's t 或 Pooled t
- 執行獨立兩樣本 t 檢定 `scipy.stats.ttest_ind()`，解讀 t 統計量、自由度 df 與 p 值
- 明確陳述統計結論：是否有統計顯著依據（$p < 0.05$）認為催化劑 B 優於 A
- 計算效果量 (Effect Size) Cohen's $d$，區分「統計顯著」與「實際顯著」之差異

$$
d = \frac{\bar{x}_B - \bar{x}_A}{s_{pooled}}
$$

---

### 🌡️ Example 03 — 多種操作溫度對反應轉化率的影響 — 單因子 ANOVA

**檔案**：[Unit14_Example_03.md](Unit14_Example_03.md) | [Unit14_Example_03.ipynb](Unit14_Example_03.ipynb)

**問題概述**：在五種不同反應溫度（300, 320, 340, 360, 380°C）下各進行 8 次批次反應，量測轉化率 $X_A$ (%)，以 ANOVA 判斷溫度是否顯著影響轉化率。

**數學工具**：`scipy.stats.f_oneway()`、`scipy.stats.levene()`、`scipy.stats.kruskal()`

**F 統計量公式**：

$$
F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between} / (k-1)}{SS_{within} / (N-k)}
$$

**化工重點**：
- 各組先執行 `scipy.stats.shapiro()` 常態性檢定（ANOVA 前提：各組數據須近似常態）
- 執行 `scipy.stats.levene()` 變異數齊一性檢定（ANOVA 前提：各組變異數相近）
- 執行單因子 ANOVA `scipy.stats.f_oneway()`，計算 F 統計量與 p 值
- 手動建構 ANOVA 表（SS_between, SS_within, df, MS, F, p），加深對 F 統計量公式的理解
- 以 Tukey HSD 概念說明多重比較的必要性（簡介 `statsmodels` 的 `pairwise_tukeyhsd`）
- 若 ANOVA 假設不成立則改用 Kruskal-Wallis 無母數版本 `scipy.stats.kruskal()`
- 繪製各組轉化率之箱型圖（含顯著差異標記）與殘差常態 Q-Q 圖

---

### ⚗️ Example 04 — 反應速率常數與溫度之相關分析與 Arrhenius 回歸

**檔案**：[Unit14_Example_04.md](Unit14_Example_04.md) | [Unit14_Example_04.ipynb](Unit14_Example_04.ipynb)

**問題概述**：由不同溫度下量測之反應速率常數 $k$（含量測誤差），驗證 Arrhenius 方程式，估計活化能 $E_a$ 與指前因子 $A$，並對回歸係數進行 t 檢定。

**Arrhenius 方程式線性化**：

$$
\ln k = \ln A - \frac{E_a}{R} \cdot \frac{1}{T}
$$

**數學工具**：`scipy.stats.pearsonr()`、`scipy.stats.linregress()`

**化工重點**：
- 計算 Pearson 相關係數 `scipy.stats.pearsonr()` 並解讀 p 值（線性相關之顯著性）
- 使用 `scipy.stats.linregress()` 進行一元線性回歸，輸出斜率 $-E_a/R$、截距 $\ln A$、 $R^2$、p 值、標準誤
- 由斜率計算活化能 $E_a$ 與其 95% 信賴區間（$b \pm t_{\alpha/2,\,n-2} \cdot SE_b$）
- 對回歸係數（斜率、截距）各自執行 t 檢定，解讀 p 值的物理意義
- **比較**：本例採用 `scipy.stats.linregress()` 著重統計推論；Unit13 的 `scipy.linalg.lstsq()` 著重矩陣求解；兩者結果一致
- 繪製 $1/T$ vs $\ln k$ 散佈圖、最佳擬合線與 95% 預測區間

---

### 🔬 Example 05 — 化工設備故障時間之可靠度分析與 Weibull 分布擬合

**檔案**：[Unit14_Example_05.md](Unit14_Example_05.md) | [Unit14_Example_05.ipynb](Unit14_Example_05.ipynb)

**問題概述**：某化工廠記錄 50 台泵浦的無故障運行時間（hours），以 Weibull 分布模型估計設備可靠度函數 $R(t)$，並計算平均故障間隔時間（MTTF）與特定時刻之故障機率。

**Weibull 分布模型**：

$$
R(t) = e^{-(t/\lambda)^k}, \quad h(t) = \frac{k}{\lambda^k} t^{k-1}
$$

**數學工具**：`scipy.stats.weibull_min.fit()`、`scipy.stats.kstest()`

**化工重點**：
- 使用 `scipy.stats.weibull_min.fit()` 以 MLE 估計 Weibull 形狀 $k$ 與比例 $\lambda$ 參數
- 以 `scipy.stats.kstest()` (Kolmogorov-Smirnov 檢定) 驗證數據與所擬合 Weibull 分布之適合度
- 計算可靠度函數 $R(t)$ 與故障率函數（Hazard Function）$h(t)$
- 計算 $B_{10}$ 壽命（10% 故障機率對應之時間，即 `ppf(0.1)`）與 MTTF（$= \lambda \, \Gamma(1+1/k)$）
- 形狀參數 $k$ 的物理意義：$k < 1$ 早夭型、$k = 1$ 隨機型（指數分布）、$k > 1$ 老化型
- 繪製故障時間直方圖與 Weibull PDF、存活函數 $R(t)$ 曲線、故障率函數 $h(t)$ 曲線（浴缸曲線）

---

### 📐 Example 06 — 製程能力分析與統計製程管制指標

**檔案**：[Unit14_Example_06.md](Unit14_Example_06.md) | [Unit14_Example_06.ipynb](Unit14_Example_06.ipynb)

**問題概述**：化工廠某產品黏度規格為 $100 \pm 10$ cP，收集 100 筆連續批次量測數據，計算製程能力指標，評估製程是否滿足規格要求，並繪製 $\bar{X}$ 管制圖識別特殊原因變異。

**製程能力指標**：

$$
C_p = \frac{USL - LSL}{6\sigma}, \quad C_{pk} = \min\!\left(\frac{USL - \bar{x}}{3\sigma},\ \frac{\bar{x} - LSL}{3\sigma}\right)
$$

**數學工具**：`scipy.stats.norm.cdf()`、`scipy.stats.norm.interval()`、`scipy.stats.ttest_1samp()`

**化工重點**：
- 計算 $C_p$（製程精密度，不考慮對中）與 $C_{pk}$（製程準確度，考慮偏移）
- 使用 `scipy.stats.norm.cdf()` 計算在規格限內之產品比例（良率）與不良率（ppm）
- 以 `scipy.stats.norm.interval(0.9973)` 對應 $\pm 3\sigma$ 管制界限
- 繪製 $\bar{X}$ 管制圖（Individual Moving Range Chart）：標示中心線 CL、上下管制界限 UCL/LCL
- 以 `scipy.stats.ttest_1samp()` 檢定製程均值是否顯著偏離目標值 100 cP
- 繪製製程能力圖（Capability Plot）：直方圖 + 常態 PDF 曲線 + LSL/USL 規格線

---

## 📁 本單元檔案列表

| 檔案名稱 | 類型 | 說明 |
|---------|------|------|
| [Unit14_Statistics.md](Unit14_Statistics.md) | 📄 教學講義 | 統計分析理論、`scipy.stats` 工具總覽、假設檢定架構 |
| [Unit14_Statistics.ipynb](Unit14_Statistics.ipynb) | 💻 程式演練 | 主題講義互動式程式範例 |
| [Unit14_Example_01.md](Unit14_Example_01.md) | 📄 案例講義 | 製程品質描述統計與常態性分析 |
| [Unit14_Example_01.ipynb](Unit14_Example_01.ipynb) | 💻 程式演練 | 描述統計與常態性分析實作 |
| [Unit14_Example_02.md](Unit14_Example_02.md) | 📄 案例講義 | 催化劑批次收率假設檢定（獨立兩樣本 t 檢定） |
| [Unit14_Example_02.ipynb](Unit14_Example_02.ipynb) | 💻 程式演練 | 假設檢定與信賴區間實作 |
| [Unit14_Example_03.md](Unit14_Example_03.md) | 📄 案例講義 | 多操作溫度轉化率單因子 ANOVA |
| [Unit14_Example_03.ipynb](Unit14_Example_03.ipynb) | 💻 程式演練 | 單因子 ANOVA 實作 |
| [Unit14_Example_04.md](Unit14_Example_04.md) | 📄 案例講義 | Arrhenius 回歸統計推論（相關分析與線性回歸） |
| [Unit14_Example_04.ipynb](Unit14_Example_04.ipynb) | 💻 程式演練 | 相關分析與線性回歸統計推論實作 |
| [Unit14_Example_05.md](Unit14_Example_05.md) | 📄 案例講義 | 設備可靠度分析與 Weibull 分布擬合 |
| [Unit14_Example_05.ipynb](Unit14_Example_05.ipynb) | 💻 程式演練 | Weibull 可靠度分析實作 |
| [Unit14_Example_06.md](Unit14_Example_06.md) | 📄 案例講義 | 製程能力分析與 $\bar{X}$ 管制圖 |
| [Unit14_Example_06.ipynb](Unit14_Example_06.ipynb) | 💻 程式演練 | 製程能力分析實作 |
| [Unit14_Homework.ipynb](Unit14_Homework.ipynb) | 📝 作業 | 學生課堂練習題 |

---

## 🗺️ 課程單元導覽

### ⬅️ 前一單元
- **[Unit13 參數估計](../Unit13/README.md)**：`scipy.optimize`、`lstsq()`、`curve_fit()`、`least_squares()`、ODE 嵌套參數估計

### ➡️ 下一單元
- **[Unit15 信號模擬與處理](../Unit15/README.md)**：`scipy.signal`、數位濾波器設計、頻譜分析、信號生成

---

## 📈 本單元在課程中的定位

```
Unit13 (scipy.optimize + scipy.linalg) — 參數估計
      ↓
   Unit14 ← 你在這裡
 ┌──────────────────────────────────────────────────────────────┐
 │  統計分析 (scipy.stats)                                       │
 │  描述統計：scipy.stats.describe()                             │
 │  機率分布：norm / t / f / chi2 / weibull_min 等物件            │
 │  常態性檢定：shapiro() / normaltest() / probplot()            │
 │  信賴區間：t.interval() / chi2 分位數                          │
 │  假設檢定：ttest_*() / f_oneway() / levene() / kruskal()     │
 │  相關分析：pearsonr() / spearmanr() / linregress()           │
 └──────────────────────────────────────────────────────────────┘
      ↓
 Unit15 (scipy.signal) — 信號模擬與處理
      ↓
 ...（後續各應用單元）
```

**與化工問題的對應**：
```
製程品質數據分布鑑別（純度、轉化率、黏度）    → 描述統計 + 常態性檢定 (Unit14 Ex01, Ex06)
催化劑/操作條件效能比較                      → t 檢定 / ANOVA (Unit14 Ex02, Ex03)
Arrhenius 反應動力學參數統計推論              → 線性回歸統計推論 (Unit14 Ex04)
化工設備壽命與可靠度評估                     → Weibull 分析 (Unit14 Ex05)
製程管制與品質工程                          → C_p / C_pk / 管制圖 (Unit14 Ex06)
```

**Unit14 的重要橫向聯繫**：
- **與 Unit13 的連結**：Unit13 `scipy.linalg.lstsq()` 著重最優參數求解；Unit14 `scipy.stats.linregress()` 著重統計推論（t 檢定、p 值、信賴區間）；兩者在線性回歸上互補
- **與 Unit12 的連結**：Unit12 最適化的目標函數最小化與 Unit14 最小平方回歸在數學上同根
- **與 Unit16 的連結**：Unit16 的 `sklearn.linear_model` 機器學習線性回歸（最小化 MSE）與 Unit14 的 `scipy.stats.linregress()`（統計推論框架）形成呼應，統計方法是機器學習的理論基礎

---

## 💡 學習建議

1. **建議學習順序**：
   - Step 1：閱讀 `Unit14_Statistics.md`（約 60 分鐘）建立 `scipy.stats` 模組的整體架構認識
   - Step 2：執行 `Unit14_Statistics.ipynb` 熟悉各統計工具的基本介面
   - Step 3：依序完成 Example 01–06（由敘述性分析 → 推論統計 → 多組比較 → 回歸推論 → 分布擬合 → 製程管制）

2. **重點關注**：
   - **Example 01（描述統計）**：`scipy.stats.describe()` 的六個輸出欄位、Shapiro-Wilk p 值解讀、Q-Q 圖偏離直線的意義
   - **Example 02（t 檢定）**：Levene 前置檢定的重要性、`equal_var=False` 使用時機、Cohen's $d$ 效果量與統計顯著的差異
   - **Example 03（ANOVA）**：F 統計量分子/分母的物理意義、多重比較問題（為何不直接做多次 t 檢定）、Kruskal-Wallis 作為 ANOVA 的無母數替代
   - **Example 04（Arrhenius 回歸）**：`linregress()` 輸出的六個值（slope, intercept, r, pvalue, stderr, intercept_stderr）、R² vs r 的關係、預測區間與信賴區間的差異
   - **Example 05（Weibull）**：形狀參數 $k$ 的物理詮釋（浴缸曲線三段）、KS 檢定驗證擬合品質、`ppf(0.1)` 對應 $B_{10}$ 壽命
   - **Example 06（製程能力）**：$C_p = 1.33$ 為四西格瑪水準的依據、$C_{pk} < C_p$ 代表有製程偏移、管制圖中管制界限（UCL/LCL）與規格限（USL/LSL）的本質區別

3. **`scipy.stats` 函式速查**：

   | 需求 | 推薦函式 |
   |------|---------| 
   | 描述統計一次輸出 | `scipy.stats.describe(data)` |
   | 常態性檢定（小樣本） | `scipy.stats.shapiro(data)` |
   | 常態性檢定（大樣本） | `scipy.stats.normaltest(data)` |
   | Q-Q 圖 | `scipy.stats.probplot(data, plot=ax)` |
   | 均值信賴區間 | `scipy.stats.t.interval(0.95, df, loc, scale)` |
   | 獨立兩樣本比較 | `scipy.stats.ttest_ind(a, b, equal_var=)` |
   | 多組均值比較 | `scipy.stats.f_oneway(*groups)` |
   | 線性回歸統計推論 | `scipy.stats.linregress(x, y)` |
   | Weibull 參數估計 | `scipy.stats.weibull_min.fit(data)` |
   | KS 擬合優度 | `scipy.stats.kstest(data, cdf_func)` |

4. **常見錯誤提醒**：
   - 假設檢定前務必確認前提假設（常態性、變異數齊一性），不滿足時應改用無母數方法
   - p 值只能說明是否有統計顯著差異，無法說明差異的**實際大小**，需搭配效果量（如 Cohen's $d$）解讀
   - `scipy.stats.kurtosis()` 預設回傳超額峰態（excess kurtosis = 峰態 − 3），常態分布為 0 而非 3
   - Weibull 分布的 `scipy.stats.weibull_min(c, loc, scale)` 中 `c` 對應形狀參數 $k$，`scale` 對應比例參數 $\lambda$；`loc` 通常固定為 0
   - `scipy.stats.linregress()` 回傳的 `r` 是相關係數，決定係數 $R^2 = r^2$（需自行平方）

5. **參考外部資源**：
   - [scipy.stats 官方文件](https://docs.scipy.org/doc/scipy/reference/stats.html)
   - [scipy.stats 分布物件列表](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions)
   - [scipy.stats.linregress 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)
   - [scipy.stats.weibull_min 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit14 統計分析
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-03

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
