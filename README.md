# 電腦在化工上之應用課程 (Computer Applications in Chemical Engineering)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.x-green.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green.svg)
![Course](https://img.shields.io/badge/Course-ChemE--3502-blue.svg)

---

## 📚 課程簡介

本課程專為**化學工程學系大二**學生設計，旨在培養學生運用 Python 程式語言與科學計算套件解決化工領域實際問題的能力。課程涵蓋 Python 基礎程式設計、NumPy/Pandas/Matplotlib 等科學計算與視覺化工具，以及 SciPy 數值計算套件的完整應用，並延伸至 scikit-learn 基礎機器學習模型，透過豐富的化工實際案例進行教學。

- **課程名稱**：電腦在化工上之應用
- **課程代碼**：3502
- **課程製作**：逢甲大學 化工系 智慧程序系統工程實驗室
- **授課教師**：[莊曜禎 助理教授](https://sites.google.com/thaiche.tw/ipse/)
- **適合對象**：化工系大二學生
- **前置課程**：建議已完成大一 Python 程式設計課程
- **課程授權**：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 🎯 課程目標

完成本課程後，您將能夠：

1. **紮實的 Python 科學計算基礎**：熟練運用 NumPy、Pandas、Matplotlib、Seaborn 進行化工數據處理與視覺化
2. **SciPy 數值方法應用**：掌握 scipy.linalg、scipy.optimize、scipy.integrate、scipy.interpolate、scipy.stats、scipy.fft、scipy.signal 等子模組，解決化工中的數值計算問題
3. **數值方法理論與實作**：理解並實作線性/非線性方程式求解、插值、數值微積分、ODE/PDE 求解等核心數值方法
4. **化工問題建模能力**：能以數學模型描述化工程序，選擇合適的數值方法求解，並解讀計算結果的物理意義
5. **基礎 AI 模型入門**：初步認識並應用 scikit-learn 線性回歸、非線性回歸與神經網路模型於化工數據分析

---

## 📖 課程結構

本課程共 19 個單元 (Unit00–Unit18)，涵蓋 Python 環境設定、資料科學工具、SciPy 數值計算到基礎 AI 模型的完整學習路徑。

### Part 0：Python 學習環境設定

- **[Unit00: 環境設定教學](Unit00/)** — Google Colab 雲端環境設定、Windows 本地環境 (Miniconda + Jupyter) 建立、套件安裝與環境驗證

### Part 1：Python 程式設計與資料科學工具

- **[Unit01: 電腦在化工上的應用概述](Unit01/)** — Python 在化工領域的應用場景、課程學習路徑導覽、ASPEN/Fluent/COMSOL 等商業軟體概述
- **[Unit02: Python 程式語言基礎](Unit02/)** — 變數與資料型態、控制流程、函式與模組、例外處理
- **[Unit03: NumPy 與 Pandas 資料處理](Unit03/)** — 陣列運算、矩陣操作、DataFrame 處理、時間序列分析、化工批次數據應用
- **[Unit04: Matplotlib 與 Seaborn 資料視覺化](Unit04/)** — 折線圖、散佈圖、長條圖、統計圖表、多圖佈局，化工數據視覺化實作

### Part 2：SciPy 科學計算套件

- **[Unit05: SciPy 科學運算套件應用概述](Unit05/)** — SciPy 架構總覽、各子模組功能介紹、化工領域應用分類導覽、特殊數學函數
- **[Unit06: 線性聯立方程式之求解](Unit06/)** — scipy.linalg (solve, LU, lstsq, pinv)、scipy.sparse.linalg；物料平衡、能量平衡、反應器網絡 (6 個化工案例)
- **[Unit07: 非線性方程式之求解](Unit07/)** — scipy.optimize (root_scalar, fsolve, root)；狀態方程式、泡點計算、CSTR 多重穩態 (6 個化工案例)
- **[Unit08: 插值、微分與積分之運算](Unit08/)** — scipy.interpolate、numpy.gradient、scipy.integrate；物性估算、反應速率推斷、RTD 分析、傳遞單位數 (6 個化工案例)
- **[Unit09: 常微分方程式之求解](Unit09/)** — scipy.integrate.solve_ivp (IVP/BVP)、Stiff ODE；CSTR 動態、PFR 溫度分布、觸媒反應器、非牛頓流體 (6 個化工案例)
- **[Unit10: 偏微分方程式之求解](Unit10/)** — py-pde、scipy.linalg、Method of Lines；非穩態熱傳導、擴散反應、Fick's Laws、Fourier's Laws、Navier-Stokes 方程式 (6 個化工案例 + 3 個特別案例)
- **[Unit11: 傅立葉轉換與頻譜分析](Unit11/)** — scipy.fft；FFT/IFFT、PSD 分析、STFT 時頻分析；製程訊號分析、液泛偵測、泵浦 BPF 識別、Bode 圖 (6 個化工案例)

### Part 3：程序分析與最適化

- **[Unit12: 程序最適化](Unit12/)** — scipy.optimize (minimize, linprog, milp, differential_evolution)；單/多變數、線性規劃、全域最適化；光阻製程、化學平衡、烷化程序、反應器最適溫度 (6 個化工案例)
- **[Unit13: 參數估計](Unit13/)** — scipy.optimize (curve_fit, least_squares)、scipy.linalg.lstsq；線性/非線性最小平方法、置信區間；速率式擬合、吸附等溫線、ODE 嵌套估計 (7 個化工案例)
- **[Unit14: 統計分析](Unit14/)** *(建構中)* — scipy.stats；描述統計、機率分布 (常態/t/F/卡方/Weibull)、常態性檢定、信賴區間、假設檢定 (t 檢定/ANOVA/卡方)、相關分析、線性回歸推論、製程能力分析 (6 個化工案例：製程品質描述統計、催化劑收率假設檢定、操作溫度 ANOVA、Arrhenius 回歸、設備可靠度分析、製程能力 SPC)
- **[Unit15: 信號模擬與處理](Unit15/)** *(規劃中)* — scipy.signal；濾波器設計、系統模型、時域模擬、頻率響應分析

### Part 4：基礎 AI 模型入門

- **[Unit16: 線性回歸模型](Unit16/)** *(規劃中)* — sklearn.linear_model；線性/多元回歸、Ridge/Lasso 正規化、模型評估
- **[Unit17: 非線性回歸模型](Unit17/)** *(規劃中)* — PolynomialFeatures + linear_model；多項式回歸、化工非線性製程建模
- **[Unit18: 神經網路模型](Unit18/)** *(規劃中)* — sklearn.neural_network.MLPRegressor；多層感知機、超參數調整、化工軟感測器應用

---

## 課程安排

| 週次 | 主題與單元 | 教學與學習活動 | 面授 (小時) |
|---|---|---|---|
| 1 | Unit00 Python 學習環境設定<br>Unit01 電腦在化工上的應用概述 | 完成 Google Colab 環境設定<br>完成本機環境設定<br>完成課堂作業 | 3 |
| 2 | Unit02 Python 程式語言基礎 | 完成課堂作業 | 3 |
| 3 | Unit03 NumPy 與 Pandas 資料處理 | 完成課堂作業 | 3 |
| 4 | Unit04 Matplotlib 與 Seaborn 資料視覺化 | 完成課堂作業 | 3 |
| 5 | Unit05 SciPy 科學運算套件概述 | 完成課堂作業 | 3 |
| 6 | Unit06 線性聯立方程式之求解 | 完成課堂作業 | 3 |
| 7 | Unit07 非線性方程式之求解 | 完成課堂作業 | 3 |
| 8 | Unit08 插值、微分與積分之運算 | 完成課堂作業 | 3 |
| 9 | Unit09 常微分方程式 (ODE) 之求解 | 完成課堂作業 | 3 |
| 10 | Unit10 偏微分方程式 (PDE) 之求解 | 完成課堂作業 | 3 |
| 11 | Unit11 傅立葉轉換與頻譜分析 | 完成課堂作業 | 3 |
| 12 | Unit12 程序最適化 | 完成課堂作業 | 3 |
| 13 | Unit13 參數估計 | 完成課堂作業 | 3 |
| 14 | Unit14 統計分析 | 完成課堂作業 | 3 |
| 15 | Unit15 信號模擬與處理 | 完成課堂作業 | 3 |
| 16 | Unit16 基礎 AI — 線性回歸模型 | 完成課堂作業 | 3 |
| 17 | Unit17 基礎 AI — 非線性回歸模型 | 完成課堂作業 | 3 |
| 18 | Unit18 基礎 AI — 神經網路模型 | 完成課堂作業 | 3 |

---

## 🛠️ 技術棧

### 核心工具
- **Python 3.10** — 主要程式語言
- **Jupyter Notebook / JupyterLab** — 互動式開發環境
- **Google Colab** — 雲端運算平台 (免安裝，建議新手使用)

### 科學計算與資料處理
- **NumPy** — 數值運算、陣列操作、矩陣計算
- **Pandas** — 資料處理、表格分析、時間序列
- **SciPy** — 核心數值計算套件，涵蓋：
  - `scipy.linalg` — 線性代數
  - `scipy.optimize` — 最適化與方程式求解
  - `scipy.integrate` — 數值積分與 ODE/BVP 求解
  - `scipy.interpolate` — 插值法
  - `scipy.stats` — 統計分析
  - `scipy.fft` — 快速傅立葉轉換
  - `scipy.signal` — 信號處理
  - `scipy.sparse` — 稀疏矩陣
  - `scipy.special` — 特殊數學函數

### 資料視覺化
- **Matplotlib** — 基礎繪圖
- **Seaborn** — 統計視覺化

### 偏微分方程式求解
- **py-pde** — 有限差分法 PDE 數值模擬

### 基礎機器學習
- **scikit-learn** — 線性回歸、多項式回歸、神經網路 (MLPRegressor)

---

## 🚀 快速開始

### 方法一：使用 Google Colab (推薦新手)

1. 瀏覽至 [Google Colab](https://colab.research.google.com/)
2. 開啟本專案的任一 `.ipynb` 檔案
3. 點選「在 Colab 中開啟」
4. 開始學習！

**優點**：免費、無需安裝、瀏覽器即可使用，適合快速上手

### 課程資料下載
```python
# 連結個人Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 複製課程資料
%cd "/content/drive/MyDrive/Colab Notebooks"
!git clone https://github.com/d9148408/ChemE-3502.git
```

### 更新課程資料
```python
# 連結個人Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 更新課程資料
!git fetch origin
!git reset --hard origin/master
```

---

### 方法二：本地環境設定 (推薦長期學習)

#### 步驟 1：安裝 Miniconda

下載並安裝 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

#### 步驟 2：建立虛擬環境

```powershell
# 複製專案
git clone https://github.com/fcuycchuang/ChemE-3502.git
cd ChemE-3502

# 建立 Python 3.10 虛擬環境
conda env create -f Unit00/PY310_environment.yml

# 啟動環境
conda activate PY310
```

#### 步驟 3：啟動 Jupyter

```powershell
# 啟動 Jupyter Notebook
jupyter notebook

# 或啟動 Jupyter Lab (推薦)
jupyter lab
```

#### 步驟 4：開啟 Notebook 開始學習

瀏覽至對應的 Unit 資料夾，開啟 `.ipynb` 檔案即可開始學習。

---

## 📚 每個單元包含什麼？

每個單元通常包含以下內容：

1. **📄 教學講義 (.md)** — 完整理論說明、數學推導、演算法原理、化工應用案例介紹
2. **📓 範例程式碼 (.ipynb)** — 完整程式實作，含主題 Notebook 與多個化工案例 Notebook
3. **📝 作業練習 (Homework.ipynb)** — 引導式實作練習，提供部分程式碼框架
4. **📊 輸出資料夾 (outputs/)** — 程式執行結果圖表 (部分單元)

---

## 💡 學習建議

### 理論與實作並重
- 先閱讀 `.md` 講義理解理論，再執行 `.ipynb` 範例程式碼
- 嘗試修改程式碼參數，觀察結果改變
- 完成每單元 Homework 作業鞏固學習

### 化工應用導向
- 每個單元都包含多個化工實際案例 (物料平衡、反應器設計、熱傳導、製程最適化等)
- 思考數值方法如何對應到您的化工課程知識

### 建立連結
- Unit06–Unit10 的數值方法與工程數學、輸送現象、反應工程等課程密切相關
- Unit12–Unit13 的最適化與參數估計是程序控制、反應動力學的重要工具

---

## 🤝 貢獻

歡迎提出問題、建議或貢獻：

1. **Issue**：發現錯誤或有建議，請開 Issue
2. **Pull Request**：歡迎提交程式碼改進或新範例
3. **討論**：歡迎在討論區分享學習心得與應用案例

---

## 📧 聯絡方式

**授課教師**：[莊曜禎 助理教授](https://sites.google.com/thaiche.tw/ipse/)
**單位**：[逢甲大學 化學工程學系](https://che.fcu.edu.tw/)
**Email**：yaocchuang@fcu.edu.tw

---

## 📈 課程更新記錄

- **2026-01** — Unit00–Unit13 完成初版
- **2026-03** — Unit05 完成初版；Unit14 教學講義與 6 個化工案例 Notebook 建構中；Unit00–Unit13 各單元 README 完成

---

## 📄 授權

本課程內容採用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 授權。
允許他人以非商業方式使用、修改、再創作作品，但必須標明原作者，並以相同的 CC BY-NC-SA 授權條款發布衍生作品。

---

## 🙏 致謝

感謝以下開源社群與工具：
- Python Software Foundation
- NumPy、SciPy、Pandas、Matplotlib、Seaborn、py-pde、scikit-learn 開發團隊
- Jupyter Project
- 所有提供開源數據集的研究機構
- ChatGPT、Gemini、Claude 和 GitHub Copilot 提供的技術支援、內容校對以及內容優化

---

**祝您學習順利！💪**

如有任何問題，歡迎隨時聯繫或在討論區提問。讓我們一起善用 Python 解決化工工程問題！

---

**課程資訊**
- 課程名稱：電腦在化工上之應用 (ChemE 3502)
- 課程單元：Unit00–Unit18
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-03-02

**課程授權 [CC BY-NC-SA 4.0]**
- 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。
