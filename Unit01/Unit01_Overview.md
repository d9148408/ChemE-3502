# Unit01 電腦在化工上的應用概述

本單元介紹電腦在化學工程領域中的廣泛應用，包括過程模擬與優化、資料分析與視覺化、機器學習與人工智慧等主題，並帶領學生認識 Python 程式語言在化工領域的重要性與優勢，同時概述本課程所有單元的學習路徑。

## 目標
- 了解電腦在化工領域的廣泛應用範疇
- 掌握 Python 在化工問題求解中的優勢與定位
- 認識化工領域常用的套裝軟體（過程模擬、CFD、資料分析工具）
- 建立整個課程的學習路徑圖，理解各單元間的關聯性

---

## 1. 電腦在化工領域的應用概述

### 1.1 化工領域中的計算需求

化學工程是一門複雜的工程學科，涵蓋化學反應、熱力學、物質傳遞、流體力學等多個面向。現代化工問題的規模與複雜度已遠超人工計算的能力，電腦輔助計算已成為化工工程師不可或缺的工具。

化工領域中常見的計算需求包括：

| 應用領域 | 典型問題 | 計算需求 |
|----------|----------|----------|
| 反應工程 | 反應速率方程式求解、反應器設計 | 常微分方程式求解、非線性方程組 |
| 熱力學 | 相平衡計算、狀態方程式求解 | 非線性代數方程式 |
| 傳質傳熱 | 溫度與濃度場分布 | 偏微分方程式 |
| 製程設計 | 物料與能量平衡 | 線性聯立方程式 |
| 程序控制 | 動態響應分析 | ODE、頻率域分析 |
| 資料分析 | 製程監控、品質預測 | 統計學、機器學習 |

### 1.2 電腦應用的主要類別

電腦在化工領域的應用可以分為以下幾個主要類別：

#### (1) 程序模擬與設計 (Process Simulation & Design)

程序模擬軟體能夠對整個化工製程進行穩態或動態的模擬，協助工程師在不實際建造工廠的情況下進行製程設計與優化。

**應用範例：**
- 蒸餾塔的理論板數與操作條件設計
- 熱交換器網絡的最適化配置
- 反應器序列的轉化率與選擇率計算
- 製程流程圖 (Process Flow Diagram, PFD) 的模擬

#### (2) 數值方法求解 (Numerical Methods)

化工問題常涉及解析解難以或無法求得的數學方程式，數值方法提供了系統化的計算策略。

**常見問題類型：**
- 線性聯立方程式 → 物料平衡、能量平衡
- 非線性方程式 → 狀態方程式、化學平衡
- 常微分方程式 → 反應器動態、批次操作
- 偏微分方程式 → 擴散、熱傳導、流體流動

#### (3) 資料分析與視覺化 (Data Analytics & Visualization)

現代化工廠配備大量感測器，每秒產生海量的製程數據。有效分析這些數據，可以提升製程效率、預測設備故障、改善產品品質。

**應用場景：**
- 製程監控與異常偵測
- 產品品質預測模型
- 實驗數據分析與模型擬合
- 製程數據視覺化與趨勢分析

#### (4) 機器學習與人工智慧 (Machine Learning & AI)

近年來，機器學習與深度學習在化工領域展現出強大的潛力，特別是在複雜非線性關係建模和大量數據挖掘方面。

**典型應用：**
- 催化劑性能預測
- 分子性質 (物性) 的 QSPR 模型
- 製程軟感測器 (Soft Sensor)
- 反應條件最佳化推薦

#### (5) 計算流體力學 (Computational Fluid Dynamics, CFD)

CFD 利用數值方法求解 Navier-Stokes 方程式，模擬流體在複雜幾何結構中的流動、熱傳與質傳行為。

**化工應用：**
- 攪拌槽內的流場與混合效率分析
- 換熱器管路的熱傳強化
- 反應器內的流動死區偵測
- 分離設備的氣液分布分析

---

## 2. Python 程式語言在化工領域的應用

### 2.1 Python 的全球崛起：為何全世界都在學 Python？

#### 2.1.1 TIOBE Index：連年蟬聯第一的程式語言

**TIOBE Programming Community Index** 是目前最具公信力的程式語言熱度排行榜，透過搜尋引擎（Google、Bing、Wikipedia、Amazon 等）對各程式語言的搜尋量進行加權統計，每月更新一次，反映全球開發者社群的關注程度。

Python 自 2021 年起長期穩居 **TIOBE Index 第一名**，這並非偶然，而是多重因素長期累積的結果：

| 年份 | Python TIOBE 排名 | 說明 |
|------|------------------|--------------|
| 2010 | #6 | 開始受到科學社群關注 |
| 2015 | #5 | 機器學習熱潮帶動成長 |
| 2018 | #3 | 取代 C++ 躋身前三 |
| 2020 | #3 → #2 | 超越 Java 的時代來臨 |
| 2021 | **#1** | 首次登頂，並持續領先至今 |
| 2024–2025 | **#1**（穩居） | 與第二名差距持續擴大 |

> **延伸閱讀**：[TIOBE Index 官方網站](https://www.tiobe.com/tiobe-index/) 每月更新，可查看即時排行。

#### 2.1.2 為何 Python 能超越所有競爭對手？

Python 之所以能在眾多程式語言中脫穎而出，並不是因為它執行速度最快（C/C++ 更快），也不是因為它是最早的語言（Fortran、COBOL 年代久遠），而是因為它完美地滿足了**「現代資料科學與 AI 時代」的核心需求**：

**① 學習門檻極低，接受度廣**

Python 的語法設計理念是「可讀性第一」，接近英語自然語言：

```python
# Python：接近自然語言的語法
for student in class_list:
    if student.grade >= 60:
        print(f"{student.name} 及格！")
```

相比之下，C++ 實現相同邏輯需要更多語法結構（指標、型別宣告等），而 Java 需要定義類別與方法。Python 讓「非電腦科系的工程師、科學家、統計學家」也能快速上手，這是其他語言難以企及的優勢。

**② AI / 機器學習生態系統的「通用語」**

當 Google 在 2015 年開源 **TensorFlow**，Facebook（Meta）在 2016 年開源 **PyTorch**，這兩個最重要的深度學習框架都選擇了 Python 作為主要介面語言。此後：
- **Scikit-learn**（傳統機器學習）
- **Hugging Face Transformers**（大型語言模型）
- **OpenAI API**（ChatGPT 背後技術）
- **LangChain**（LLM 應用開發）

幾乎所有 AI/ML 工具都以 Python 為核心，形成強大的「網路效應」——越多工具支援 Python，就吸引越多使用者；越多使用者，就促使更多工具選擇支援 Python，形成正向循環。

**③ 「膠水語言」的角色：整合一切**

Python 被稱為「膠水語言」(Glue Language)，因為它能輕易呼叫 C/C++ 程式庫（透過 Cython、ctypes）、連接資料庫（SQLAlchemy）、呼叫 Web API（requests）、控制 Excel（openpyxl）、甚至驅動商業軟體（Aspen Plus COM API）。這種「整合一切」的能力，使 Python 在工程應用中具有不可替代的地位。

**④ 開源免費，社群龐大**

Python 完全開源且免費，任何人都可以使用和貢獻。**PyPI（Python Package Index）** 目前收錄超過 **530,000 個套件**（截至 2025 年），涵蓋幾乎所有可以想到的應用場景。全球有超過 **1,500 萬名**活躍的 Python 開發者，是 Stack Overflow 等技術社群中最熱門的問答語言。

#### 2.1.3 各行各業的 Python 採用浪潮

Python 的影響力早已超越學術研究，滲透到現代社會幾乎所有重要行業：

| 行業 | 代表性應用 | 使用的 Python 工具 |
|------|------------|-------------------|
| **科技業** | 自動化測試、後端服務、資料工程 | Flask/Django、Airflow、Pytest |
| **金融業** | 量化交易、風險模型、衍生品定價 | NumPy、QuantLib、Zipline |
| **生醫製藥** | 基因組分析、藥物分子設計、臨床試驗分析 | Biopython、RDKit、PyMol |
| **化學工程** | 製程模擬、反應動力學建模、資料驅動控制 | SciPy、Cantera、py-pde |
| **航太國防** | 飛行模擬、訊號處理、影像辨識 | NumPy、OpenCV、TensorFlow |
| **氣象環境** | 氣候模型分析、衛星資料處理 | Xarray、Cartopy、netCDF4 |
| **製造業** | 預測性維護、視覺檢測、供應鏈分析 | Scikit-learn、OpenCV、Pandas |
| **政府公部門** | 大數據政策分析、智慧城市應用 | Pandas、GeoPandas、Plotly |

**特別值得關注：** 在全球最大的職缺搜尋平台 LinkedIn 與 Indeed 中，「Python」是**出現頻率最高的技術關鍵字**之一，遠超過其他程式語言，且持續多年維持此地位。這意味著學習 Python 不僅是學術需求，更是提升就業競爭力的重要投資。

#### 2.1.4 全球頂尖大學的 Python 必修課趨勢

近年來，全球各大知名大學已紛紛將 Python 程式設計列為**工程學院甚至全校大一必修課程**，取代了傳統的 C/C++ 入門程式語言課程：

**國際知名大學：**
- **MIT（麻省理工學院）**：`6.0001 Introduction to Computer Science and Programming in Python` 是全校最早期的必修或選修計算課，開放式課程（MIT OpenCourseWare）全球免費存取
- **Stanford（史丹佛大學）**：`CS106A Programming Methodologies` 已改以 Python 為主要教學語言
- **Harvard（哈佛大學）**：`CS50P: Introduction to Programming with Python` 是 edX 平台上報名人數最多的線上課程之一
- **UC Berkeley（加州大學柏克萊分校）**：`Data 8: The Foundations of Data Science` 以 Python 為核心，已成為全系所最多學生選修的課程
- **Carnegie Mellon University（CMU）**：在工程和科學院系廣泛推行 Python 入門課程

**台灣國內大學：**
- **國立台灣大學**：多個學院（理、工、醫、農）皆設有 Python 計算相關必修或通識課程
- **國立清華大學**：工程學院大一「程式設計」課程以 Python 為主
- **國立成功大學**：各工程學系推行 Python 計算程式課
- **逢甲大學**：本課程「電腦在化工上之應用」即以 Python 為核心工具，培養化工學生的計算程式能力

> **為什麼是 Python，而不是傳統的 C/C++ 或 Java？**
> 
> 傳統上，大學工程入門程式課多以 C/C++ 為主，因其執行速度快且能培養底層記憶體管理的理解。然而，隨著資料科學、人工智慧的主導地位確立，學界逐漸意識到：**對於大多數工程師與科學家而言，「快速實現想法、分析數據」的能力遠比「手動管理記憶體」更為重要**。Python 讓學生在入門課程中就能完成真實問題的解決（如資料分析、視覺化），大幅提升學習動機與成就感，這是 C++ 難以提供的體驗。

### 2.2 為什麼選擇 Python 作為化工計算工具？

Python 是當今科學計算與資料科學領域最廣泛使用的程式語言之一，在化工領域同樣具有舉足輕重的地位。

**Python 的主要優勢（化工視角）：**

| 特點 | 說明 |
|------|------|
| **語法簡潔易學** | 接近自然語言的語法，學習曲線平緩，適合工程師快速上手 |
| **豐富的科學套件生態** | NumPy、SciPy、Pandas、Matplotlib 等套件提供完整的科學計算工具鏈 |
| **強大的資料處理能力** | Pandas 提供類似 Excel 的資料操作，支援大規模數據分析 |
| **機器學習整合** | Scikit-learn、TensorFlow、PyTorch 等頂尖 ML 框架皆支援 Python |
| **開源免費** | 無授權費用，且擁有龐大的開源社群持續維護與更新 |
| **跨平台相容** | 可在 Windows、macOS、Linux 及雲端環境（如 Google Colab）運行 |
| **互動式開發** | Jupyter Notebook 提供邊寫邊測的互動式開發體驗 |

### 2.3 化工領域的核心科學套件

以下介紹本課程中會使用到的主要 Python 科學套件：

#### NumPy — 數值運算基石
```python
import numpy as np

# 建立陣列與進行矩陣運算
A = np.array([[2, 1], [1, 3]])  # 係數矩陣
b = np.array([5, 10])           # 右側向量
x = np.linalg.solve(A, b)       # 求解線性方程組
print(f"Solution: {x}")
```
NumPy 提供高效能的多維陣列操作，是所有科學計算套件的基礎。

#### SciPy — 科學計算工具箱
```python
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# 非線性方程式求解
def equations(x):
    return [x[0]**2 + x[1] - 4,
            x[0] + x[1]**2 - 6]

solution = fsolve(equations, [1, 2])
print(f"Root: {solution}")
```
SciPy 在 NumPy 基礎上提供線性代數、最佳化、ODE 求解、插值、統計等高階功能，是本課程 Unit06–Unit12 的主要計算工具。

#### Pandas — 資料處理分析
```python
import pandas as pd

# 讀取製程數據並進行統計
df = pd.read_csv('process_data.csv')
print(df.describe())         # 基本統計量
print(df.corr())             # 相關性矩陣
```
Pandas 提供 DataFrame 資料結構，專為結構化資料的讀取、清理、分析設計，特別擅長時間序列數據處理。

#### Matplotlib & Seaborn — 資料視覺化
```python
import matplotlib.pyplot as plt

# 繪製溫度-轉化率曲線
plt.figure(figsize=(8, 5))
plt.plot(temperature, conversion, 'b-o', label='Experimental')
plt.xlabel('Temperature (K)')
plt.ylabel('Conversion (-)')
plt.title('Conversion vs Temperature in CSTR')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
Matplotlib 是 Python 最基本的繪圖套件，Seaborn 則提供更精美的統計圖表。

#### Scikit-learn — 機器學習
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 建立線性回歸預測模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
print(f"R² Score: {r2_score(y_test, model.predict(X_test)):.4f}")
```
Scikit-learn 提供標準化的機器學習工具，涵蓋回歸、分類、聚類、降維等演算法。

### 2.4 Python 在化工領域的實際應用案例

#### 案例一：CSTR 反應器穩態分析

以連續攪拌槽反應器 (CSTR) 的穩態物料平衡為例：

$$
0 = F_{A0} - F_A - r_A V
$$

其中反應速率 $r_A = k C_A$ ，可化簡為：

$$
C_A = \frac{C_{A0}}{1 + k\tau}
$$

使用 Python 可以快速繪製不同操作條件下的轉化率曲線，並進行敏感度分析。

#### 案例二：批次反應器動態模擬

批次反應器中的繁複反應動力學涉及聯立常微分方程組，可使用 `scipy.integrate.solve_ivp()` 直接求解，並視覺化各成分濃度隨時間的變化。

#### 案例三：製程數據異常偵測

透過 Scikit-learn 的主成分分析 (PCA) 對高維度製程數據降維，再搭配統計控制圖 (Hotelling's T² 與 Q 統計量) 偵測製程異常，是現代智慧型製程監控的典型應用。

### 2.5 Python 的未來發展趨勢

化工領域中 Python 的應用正在快速拓展，以下是幾個重要的未來趨勢：

#### (1) 機器學習輔助材料與製程設計
利用深度學習模型 (DNN、GNN) 加速催化劑篩選、新材料預測，以及製程操作條件的推薦，縮短傳統試誤法所需的大量時間與成本。

#### (2) 數位孿生 (Digital Twin)
結合物理模型 (First-principles model) 與數據驅動模型，建立工廠製程的即時虛擬鏡像，用於即時監控、故障預測與操作優化。

#### (3) 智慧型過程控制
以強化學習 (Reinforcement Learning) 配合製程動態模擬環境，訓練自適應控制器，取代傳統固定參數的 PID 控制器。

#### (4) 大數據分析平台整合
透過 Python 與雲端大數據平台（如 Apache Spark、AWS、Google Cloud）的整合，處理大型化工廠產生的海量感測器數據。

#### (5) 物理訊息神經網路 (Physics-Informed Neural Networks, PINNs)
將化工基理方程式（物料平衡、能量平衡）嵌入神經網路的損失函數中，確保模型預測結果符合化工物理約束，同時提升數據稀缺場景下的泛化能力。

---

## 3. 化工領域其他常用軟體工具概述

### 3.1 資料分析與科學計算工具比較

除 Python 外，化工領域中還有幾種常見的資料分析與科學計算工具：

| 工具 | 開發語言 | 費用 | 優勢 | 劣勢 | 適用場景 |
|------|----------|------|------|------|----------|
| **Python** | Python | 免費開源 | 語法簡潔、套件豐富、ML整合強 | 執行速度較慢（需Cython/JIT優化） | 資料分析、ML、ODE/PDE求解 |
| **MATLAB** | MATLAB | 商業付費 | 矩陣運算高效、工具箱豐富、Simulink動態仿真 | 授權費昂貴，封閉生態 | 控制系統、訊號處理、動態模擬 |
| **R** | R | 免費開源 | 統計分析功能強大、視覺化套件精美（ggplot2） | 非統計計算能力較弱 | 實驗設計、統計分析、生物統計 |
| **Microsoft Excel** | VBA/公式 | 商業付費 | 使用門檻低、GUI友好、廣泛普及 | 大數據處理能力差、可重現性差 | 小型數據整理、報表製作 |

#### MATLAB 的特色與化工應用
- **Simulink**：圖形化動態系統建模與模擬，適合化工製程控制系統設計
- **Control System Toolbox**：PID設計、頻率響應分析、根軌跡法
- **Optimization Toolbox**：線性規劃、非線性規劃、整數規劃
- **Statistics and Machine Learning Toolbox**：統計分析與ML演算法

**Python vs MATLAB 的選擇建議：**
> 若學校或公司已購買 MATLAB 授權，MATLAB 在控制系統和動態模擬方面的工具箱相當完整。然而，對於資料科學、機器學習，以及開源協作場景，Python 有著更廣泛的生態系統支援和零授權成本的優勢。本課程選擇 Python 作為主要工具，是考量到其開源免費、跨領域通用性、以及在產業界（特別是 AI/ML 領域）快速增長的應用比例。

#### R 的特色與化工應用
- 在化學計量學 (Chemometrics) 領域有豐富的套件（如 `caret`、`mlr3`）
- 多變量統計分析：PCA、PLS、PCR
- 實驗設計與統計假設檢定
- 製程數據的探索性分析 (EDA)

#### Excel 的定位
Excel 雖然功能有限，但在化工領域仍然廣泛用於：
- 小批量實驗數據記錄與初步整理
- 工程報表製作與溝通
- 簡單的物料平衡試算表
- 與客戶或管理層溝通的數據呈現

---

## 4. 化工過程模擬軟體介紹

### 4.1 什麼是過程模擬軟體？

過程模擬軟體 (Process Simulation Software) 是專為化工製程設計的計算工具，能夠對包含眾多單元操作（反應器、蒸餾塔、換熱器、泵等）的複雜製程進行穩態或動態的全流程模擬。

這類軟體內建大量的化學物性資料庫（如 NIST、DIPPR），並整合了熱力學模型（狀態方程式、活度係數模型），使工程師能夠直接進行全廠模擬而無需手動輸入物性數據。

### 4.2 主要過程模擬軟體比較

| 軟體 | 開發商 | 主要功能 | 適用場景 |
|------|--------|----------|----------|
| **Aspen Plus** | AspenTech | 穩態流程模擬（石化、化工、精煉） | 石化廠設計、能量整合 |
| **Aspen HYSYS** | AspenTech | 石油天然氣穩態/動態模擬 | 天然氣處理、煉油廠 |
| **PRO/II** | AVEVA (SimSci) | 穩態流程模擬 | 石化、精煉 |
| **CHEMCAD** | Chemstations | 穩態/動態流程模擬 | 中小型化工廠 |

#### Aspen Plus
Aspen Plus 是目前業界最廣泛使用的化工流程模擬軟體，特別適合石化與精煉領域的設計計算。

**主要功能：**
- 穩態物料與能量平衡計算
- 內建豐富的熱力學物性模型（PR、SRK、NRTL、UNIFAC 等）
- 電解質系統模擬（酸鹼、鹽溶液）
- 固體處理單元（結晶、研磨、旋風分離）
- 靈敏度分析與設計規格計算
- **與 Python 整合**：透過 `aspen` 或 COM API，Python 可驅動 Aspen Plus 批次執行模擬，用於製程優化或機器學習的訓練數據生成

#### Aspen HYSYS
Aspen HYSYS 特別針對石油天然氣行業設計，提供穩態與動態模擬能力。

**主要功能：**
- 天然氣管線與處理廠模擬
- 原油蒸餾與精煉流程
- 動態模擬：製程控制系統設計與評估
- 安全洩壓 (Relief System) 分析
- **與 Python 整合**：HYSYS 提供 COM 介面，可使用 Python 的 `win32com.client` 模組進行自動化操作

#### PRO/II
PRO/II（隸屬 AVEVA SimSci 產品線）是另一款廣泛應用於石化精煉業的穩態流程模擬軟體，功能與 Aspen Plus 相近。

#### CHEMCAD
CHEMCAD 由 Chemstations 開發，定位於中小型化工廠的設計計算，提供較低的授權費用，適合教育培訓用途。

### 4.3 Python 與過程模擬軟體的整合趨勢

現代化工數位化轉型中，Python 與商業過程模擬軟體的整合越來越受到重視：

```python
# 示意：使用 Python 透過 COM API 控制 Aspen Plus（需安裝 Aspen Plus）
import win32com.client

# 建立 Aspen Plus 連線
aspen = win32com.client.Dispatch("Apwn.Document")
aspen.Open("my_simulation.bkp")

# 設定操作條件
aspen.Tree.FindNode(r"\Data\Streams\FEED\Input\TOTFLOW\MIXED").Value = 100.0

# 執行模擬
aspen.Engine.Run2()

# 讀取結果
result = aspen.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFLOW\MIXED").Value
print(f"Product molar flow: {result:.2f} kmol/hr")
```

這種整合方式可以：
- 自動執行數千次模擬以生成機器學習訓練數據
- 結合 Python 的最佳化演算法對製程條件進行系統性搜尋
- 建立製程的「代理模型」(Surrogate Model) 用於快速計算

---

## 5. 計算流體力學 (CFD) 軟體介紹

### 5.1 CFD 在化工應用中的重要性

計算流體力學 (Computational Fluid Dynamics, CFD) 是求解流體力學方程式（通常為 Navier-Stokes 方程組）的數值計算技術，能夠在電腦上模擬各種幾何形狀和邊界條件下的流體行為。

在化工領域，CFD 尤其適用於：
- **攪拌槽**：分析攪拌器附近的流場、混合效率、局部 shear rate 分布
- **管式反應器**：溫度與濃度的軸向/徑向不均勻分布
- **換熱器**：流道內的熱傳增強結構設計
- **填充塔**：填充物局部流場與壓降計算
- **旋風分離器**：顆粒分離效率與流場特性

### 5.2 主要 CFD 軟體介紹

#### ANSYS Fluent
ANSYS Fluent 是全球最廣泛使用的商業 CFD 軟體之一，以有限體積法 (Finite Volume Method, FVM) 為核心，具有以下特色：

**主要功能：**
- 穩態與非穩態流場模擬
- 多種紊流模型：$k$-$\varepsilon$ 、$k$-$\omega$ SST、LES、DNS
- 多相流模擬：VOF、Mixture、Euler 模型
- 傳熱耦合：對流、輻射、導熱
- 化學反應流：燃燒、催化反應
- 粒子追蹤：DPM (Discrete Phase Model)
- **與 Python 整合**：Fluent 提供 Python Journal API，可用 Python 腳本自動化網格生成、邊界條件設定與後處理

#### COMSOL Multiphysics
COMSOL 是以有限元素法 (Finite Element Method, FEM) 為核心的多重物理耦合模擬平台，特點是不同物理場的耦合能力強：

**主要功能：**
- 流體流動、傳熱、質傳的多物理耦合
- 任意不規則幾何的自動網格剖分（非結構化網格）
- 電磁場、結構力學、聲學的整合模擬
- 生醫工程（藥物釋放、生物反應器）應用
- **與 Python 整合**：COMSOL 提供 LiveLink for MATLAB，亦可透過 `mph` Python 套件進行參數化掃描與結果提取

### 5.3 Python 在 CFD 中的角色

雖然 ANSYS Fluent 和 COMSOL 是工業級 CFD 的主流工具，Python 在 CFD 領域同樣有重要定位：

**開源 CFD 工具鏈：**
- **OpenFOAM**：全球最廣泛使用的開源 CFD 平台，可搭配 Python 進行前後處理
- **py-pde**：本課程 Unit10 使用的 Python PDE 求解套件，適合規則幾何的教學示範
- **FEniCS / FEniCSx**：有限元素法開源套件，適合自訂 PDE 問題求解

**Python 在 CFD 工作流中的位置：**

```
幾何建模 → 網格生成 → CFD 求解 → 後處理與視覺化
              ↓                         ↓
        Python 腳本自動化          Python 數據分析
        (Fluent Journal API)       (matplotlib, VTK)
```

---

## 6. 本課程學習路徑

### 6.1 課程架構總覽

本課程「電腦在化工上之應用」的內容依照學習進度分為以下幾個主要模組：

#### 模組一：Python 基礎與科學套件 (Unit00–Unit04)

| 單元 | 主題 | 核心內容 |
|------|------|----------|
| Unit00 | 環境設定 | Google Colab 與本地 Python 環境配置 |
| Unit01 | 應用概述 | 電腦在化工的角色、Python 與軟體工具簡介、課程路徑 |
| Unit02 | Python 基礎 | 變數、控制流程、函式、例外處理 |
| Unit03 | NumPy & Pandas | 陣列運算、資料處理與時序分析 |
| Unit04 | 視覺化 | Matplotlib & Seaborn 圖表製作 |

#### 模組二：科學計算套件 SciPy (Unit05–Unit12)

| 單元 | 主題 | 核心工具 |
|------|------|----------|
| Unit05 | SciPy 概述 | 各子模組功能導覽 |
| Unit06 | 線性聯立方程式 | `scipy.linalg.solve()`, 稀疏矩陣求解 |
| Unit07 | 非線性方程式 | `scipy.optimize.root_scalar()`, `fsolve()` |
| Unit08 | 插值、微分、積分 | `scipy.interpolate`, `numpy.gradient()`, `scipy.integrate.quad()` |
| Unit09 | 常微分方程式 (ODE) | `scipy.integrate.solve_ivp()`, `solve_bvp()` |
| Unit10 | 偏微分方程式 (PDE) | `py-pde`, Method of Lines |
| Unit11 | 傅立葉轉換 | `scipy.fft`, 頻譜分析 |
| Unit12 | 程序最適化 | `scipy.optimize.minimize()`, 全域最佳化 |

#### 模組三：參數估計與機器學習 (Unit13–Unit16)

| 單元 | 主題 | 核心工具 |
|------|------|----------|
| Unit13 | 參數估計 | `scipy.optimize.curve_fit()`, 最小平方法 |
| Unit14 | Scikit-learn 基礎 | 機器學習工作流、特徵工程、模型評估 |
| Unit15 | 線性回歸模型 | 多元線性回歸、正則化、多項式回歸 |
| Unit16 | 非線性回歸模型 | 決策樹、隨機森林、支援向量機、神經網路 |

### 6.2 課程學習路徑圖

```
           ┌─────────────────────────────────────────────┐
           │    模組一：Python 基礎與科學套件              │
           │  Unit00 → Unit01 → Unit02 → Unit03 → Unit04 │
           └─────────────────────┬───────────────────────┘
                                 │
                                 ▼
           ┌─────────────────────────────────────────────┐
           │    模組二：SciPy 科學計算                    │
           │  Unit05 (SciPy 概述)                        │
           │    │                                        │
           │    ├─ Unit06 (線性方程) ─ 物料/能量平衡      │
           │    ├─ Unit07 (非線性) ─ 狀態方程/反應平衡    │
           │    ├─ Unit08 (插值積分) ─ 數據插值/反應熱    │
           │    ├─ Unit09 (ODE) ─ 反應器動態              │
           │    ├─ Unit10 (PDE) ─ 熱傳/質傳場            │
           │    ├─ Unit11 (FFT) ─ 訊號頻譜分析            │
           │    └─ Unit12 (最適化) ─ 製程條件優化          │
           └─────────────────────┬───────────────────────┘
                                 │
                                 ▼
           ┌─────────────────────────────────────────────┐
           │    模組三：機器學習應用                       │
           │  Unit13 (參數估計) → Unit14 (Scikit-learn)  │
           │       → Unit15 (線性回歸) → Unit16 (非線性) │
           └─────────────────────────────────────────────┘
```

### 6.3 各模組的化工應用連結

每個計算模組都對應到具體的化工問題類型：

**模組二（數值方法）與化工問題的對應：**

```
線性方程 (Unit06)  →  物料/能量平衡、混合問題、蒸餾塔成分計算
非線性方程 (Unit07) →  狀態方程式求解、反應器多重穩態、泡露點計算
ODE (Unit09)       →  批次反應器動態、CSTR動態響應、PFR轉化率分布
PDE (Unit10)       →  非穩態熱傳、擴散反應、管流速度場
最適化 (Unit12)    →  製程條件最佳化、能量整合、操作成本最小化
```

**模組三（機器學習）與化工問題的對應：**

```
參數估計 (Unit13) →  動力學常數估計、傳遞係數擬合、QSPR 模型
線性回歸 (Unit15) →  製程性能預測、品質線性模型、製程變數相關性
非線性回歸 (Unit16) →  軟感測器開發、複雜非線性系統建模、異常偵測
```

### 6.4 學習建議

1. **循序漸進**：Unit02–Unit04 是後續所有單元的基礎，請確保紮實掌握 Python 基本語法與科學套件操作。

2. **動手實作**：每個單元都有對應的 `.ipynb` 程式演練，建議自行執行程式碼並嘗試修改參數，觀察結果變化。

3. **化工問題連結**：學習每個數值方法時，要主動思考它能解決哪些化工問題，這樣才能真正活用這些工具。

4. **版本管理**：建議使用 Google Colab 或架設好本地 Jupyter 環境（參考 Unit00）來保存學習成果。

5. **查閱官方文件**：SciPy、Scikit-learn 等套件都有完整的官方文件，培養查閱文件的習慣是工程師最重要的技能之一。

---

## 7. 電腦輔助化工設計的工具選擇指南

在實際工作中，面對不同的化工問題，選擇合適的計算工具是非常重要的：

### 7.1 工具選擇決策流程

```
問題類型判斷
     │
     ├─ 全廠穩態/動態流程模擬？
     │        └─→ Aspen Plus / Aspen HYSYS / PRO/II / CHEMCAD
     │
     ├─ 複雜幾何流場/多物理耦合？
     │        └─→ ANSYS Fluent / COMSOL Multiphysics / OpenFOAM
     │
     ├─ 單元操作數值計算（方程求解/ODE/PDE）？
     │        └─→ Python (SciPy) / MATLAB
     │
     ├─ 資料分析/機器學習？
     │        └─→ Python (Pandas, Scikit-learn, TensorFlow)
     │
     └─ 快速試算/報表？
              └─→ Microsoft Excel
```

### 7.2 電腦輔助化工設計的未來展望

隨著數位孿生技術、工業物聯網 (IIoT)、生成式 AI 的快速發展，電腦輔助化工設計正在朝向以下方向演進：

1. **即時數位孿生**：工廠製程的即時虛擬映射，支援操作決策輔助
2. **AI 輔助過程設計**：生成式 AI 加速新程序的概念設計與條件篩選
3. **自動化製程優化**：機器學習模型持續學習製程數據，自動推薦操作調整
4. **跨軟體整合工作流**：Python 作為「膠水語言」串聯 Aspen、COMSOL、Fluent 等工具的自動化計算工作流

掌握 Python 程式設計能力，將成為未來化工工程師在數位時代最核心的技術競爭力。

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit01 電腦在化工上的應用概述
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-02-28

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
