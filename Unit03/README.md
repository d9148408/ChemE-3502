# Unit03 Numpy與Pandas (Numpy and Pandas)

## 📚 單元簡介

在資料科學與機器學習的工作流程中，約有70-80%的時間花費在資料處理與清理上。NumPy與Pandas是Python生態系統中最核心的兩個資料處理套件，掌握這兩個工具是進入機器學習領域的必要基礎。

**NumPy (Numerical Python)** 提供高效能的多維陣列物件與數值計算功能，是Python科學計算的基石。其向量化運算能力比原生Python快10-100倍，適合處理大規模數值資料。

**Pandas (Panel Data)** 建立在NumPy之上，提供DataFrame與Series兩種強大的資料結構，讓表格式資料的處理變得直觀且高效。Pandas整合了資料讀取、清理、轉換、分析等完整功能，是資料分析的標準工具。

本單元將帶您深入學習這兩個套件的核心功能，並透過化工領域的實例（製程數據處理、實驗數據分析、時間序列處理等）展示其實際應用。完成本單元後，您將具備處理化工大數據的能力，為後續的機器學習實作奠定堅實基礎。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **掌握NumPy陣列操作**：建立、索引、切片、變形多維陣列，運用向量化運算提升效能
2. **運用NumPy數學函式**：進行統計分析、線性代數運算、隨機數生成等科學計算
3. **理解Pandas資料結構**：熟悉Series與DataFrame的特性與使用時機
4. **執行資料IO操作**：讀取與寫入CSV、Excel、JSON等多種格式的資料檔案
5. **進行資料處理**：選取、篩選、排序、處理缺失值、資料合併與分組聚合
6. **處理時間序列資料**：進行日期時間格式轉換、重新取樣、滾動窗口計算等操作
7. **應用於化工資料分析**：將所學技術應用於製程數據、實驗數據、品質數據的實際分析

---

## 📖 單元內容架構

### 1️⃣ Numpy 數值計算基礎 ⭐

**檔案**：
- 投影片檔案：[Unit03_Numpy.pdf](Unit03_Numpy.pdf)
- 講義檔案：[Unit03_Numpy.md](Unit03_Numpy.md)
- 程式範例：[Unit03_Numpy.ipynb](Unit03_Numpy.ipynb)

**內容重點**：
- **NumPy基本概念**：
  - NumPy陣列 vs Python List的差異
  - 向量化運算的效能優勢
  - ndarray的基本屬性（ndim、shape、size、dtype）
  
- **陣列建立**：
  - 從Python List建立（array()）
  - 內建函式建立（zeros、ones、empty、arange、linspace）
  - 隨機陣列生成（random模組）
  - 特殊矩陣（單位矩陣、對角矩陣）
  
- **陣列索引與切片**：
  - 一維陣列索引
  - 多維陣列索引與切片
  - 布林索引（條件篩選）
  - 花式索引（整數陣列索引）
  
- **陣列運算**：
  - 向量化運算（element-wise operations）
  - 廣播機制（Broadcasting）
  - 通用函式（ufunc）
  - 陣列方法（sum、mean、std、min、max等）
  
- **陣列變形**：
  - reshape() - 改變陣列形狀
  - flatten() / ravel() - 陣列展平
  - transpose() - 陣列轉置
  - concatenate() / stack() - 陣列合併
  
- **線性代數運算**：
  - 矩陣乘法（dot、@運算子）
  - 行列式、特徵值、逆矩陣
  - 線性方程組求解
  
- **化工應用範例**：
  - 溫度壓力數據的向量化計算
  - 多變數製程數據的矩陣運算
  - 反應器溫度場的二維陣列處理
  - 批次實驗數據的統計分析

**適合場景**：需要高效能數值計算、矩陣運算、科學計算的場合

---

### 2️⃣ Pandas 資料處理與分析 ⭐

**檔案**：
- 投影片檔案：[Unit03_Pandas.pdf](Unit03_Pandas.pdf)
- 講義檔案：[Unit03_Pandas.md](Unit03_Pandas.md)
- 程式範例：[Unit03_Pandas.ipynb](Unit03_Pandas.ipynb)

**內容重點**：
- **Pandas核心資料結構**：
  - Series：一維標籤陣列
  - DataFrame：二維表格結構
  - Index：標籤索引系統
  
- **資料讀取與寫入**：
  - CSV檔案處理（read_csv、to_csv）
  - Excel檔案處理（read_excel、to_excel）
  - JSON、HTML、SQL資料庫等多種格式
  - 讀取參數優化（指定欄位、跳過列、日期解析等）
  
- **資料選取與索引**：
  - 欄位選取（單欄、多欄）
  - 列選取（loc、iloc、布林索引）
  - 條件篩選（單條件、多條件、query方法）
  - 多層索引（MultiIndex）
  
- **資料清理**：
  - 缺失值處理（檢測、刪除、填補）
  - 重複值處理
  - 資料型態轉換
  - 字串處理（str accessor）
  - 異常值處理
  
- **資料轉換**：
  - 新增與刪除欄位
  - apply()、map()、applymap()函式應用
  - 資料排序（sort_values、sort_index）
  - 資料重塑（pivot、melt、stack、unstack）
  
- **資料合併**：
  - concat()：串接資料
  - merge()：SQL式合併
  - join()：基於索引的合併
  
- **分組聚合**：
  - groupby()分組操作
  - 聚合函式（sum、mean、count等）
  - 多重聚合（agg）
  - 轉換與過濾
  
- **時間序列處理**：
  - 日期時間解析（to_datetime）
  - DatetimeIndex的時間切片
  - 重新取樣（resample）
  - 滾動窗口統計（rolling）
  - 時間偏移（shift、diff）
  
- **化工應用範例**：
  - 製程數據的CSV讀取與清理
  - 多批次實驗數據的合併分析
  - 時間序列製程數據的重新取樣
  - 品質數據的分組統計分析
  - 缺失值填補策略（前向填補、插值）

**適合場景**：處理表格式資料、時間序列資料、需要標籤索引與SQL式操作的場合

---

### 3️⃣ Numpy實作練習

**檔案**：[Unit03_Numpy_Homework.ipynb](Unit03_Numpy_Homework.ipynb)

**練習內容**：
- NumPy陣列建立與操作
- 向量化運算效能比較
- 矩陣運算應用
- 化工數據的NumPy處理

---

### 4️⃣ Pandas實作練習

**檔案**：[Unit03_Pandas_Homework.ipynb](Unit03_Pandas_Homework.ipynb)

**練習內容**：
- DataFrame的建立與基本操作
- 資料讀取、清理與篩選
- 分組聚合分析
- 時間序列資料處理
- 綜合化工資料分析案例

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
openpyxl >= 3.0.0  # Excel檔案支援
xlrd >= 2.0.0      # 舊版Excel檔案支援
```

### 選用套件
```python
matplotlib >= 3.4.0  # 視覺化（Unit04會詳細學習）
scipy >= 1.7.0       # 進階科學計算
```

### 安裝指令
```bash
# 使用pip安裝
pip install numpy pandas openpyxl

# 使用conda安裝
conda install numpy pandas openpyxl
```

---

## 📈 學習路徑建議

### 第一階段：NumPy基礎
1. 閱讀 [Unit03_Numpy.md](Unit03_Numpy.md) 第1-4節
2. 執行 [Unit03_Numpy.ipynb](Unit03_Numpy.ipynb) 對應章節
3. 重點掌握：
   - NumPy陣列的建立方法
   - 陣列索引與切片技巧
   - 向量化運算的概念與應用
   - 理解廣播機制

### 第二階段：NumPy進階
1. 閱讀講義第5-6節
2. 執行對應的程式範例
3. 重點掌握：
   - 陣列變形操作
   - 線性代數運算
   - 統計函式的應用
   - 隨機數生成

### 第三階段：Pandas基礎
1. 閱讀 [Unit03_Pandas.md](Unit03_Pandas.md) 第1-4節
2. 執行 [Unit03_Pandas.ipynb](Unit03_Pandas.ipynb) 對應章節
3. 重點掌握：
   - Series與DataFrame的特性
   - 資料讀取（read_csv）
   - 資料選取（loc、iloc、布林索引）
   - 缺失值處理方法

### 第四階段：Pandas進階
1. 閱讀講義第5-8節
2. 執行對應的程式範例
3. 重點掌握：
   - 資料合併（concat、merge）
   - 分組聚合（groupby）
   - 時間序列處理
   - apply()等高階函式

### 第五階段：綜合練習
1. 完成 [Unit03_Numpy_Homework.ipynb](Unit03_Numpy_Homework.ipynb)
2. 完成 [Unit03_Pandas_Homework.ipynb](Unit03_Pandas_Homework.ipynb)
3. 將NumPy與Pandas應用於化工資料處理
4. 建立個人的資料處理工作流程

---

## 🎓 NumPy vs Pandas 比較與選擇指南

| 比較維度 | NumPy | Pandas |
|---------|-------|--------|
| **資料結構** | ndarray（多維陣列） | Series、DataFrame（表格） |
| **資料型態** | 同質資料（所有元素相同型態） | 異質資料（不同欄位可不同型態） |
| **索引系統** | 整數索引 | 標籤索引 |
| **效能** | 非常快（C語言實作） | 快（基於NumPy） |
| **功能定位** | 數值計算 | 資料分析 |
| **適用資料** | 數值矩陣、科學計算 | 表格資料、時間序列 |

**選擇建議**：
1. **純數值計算（矩陣運算、線性代數）** → NumPy
2. **表格式資料（CSV、Excel、資料庫）** → Pandas
3. **時間序列分析** → Pandas
4. **需要標籤索引** → Pandas
5. **高效能數值運算** → NumPy
6. **資料清理與轉換** → Pandas

**實務上**：通常兩者搭配使用，Pandas用於資料前處理，NumPy用於底層數值計算。

---

## 🔍 化工領域核心應用

### 1. 製程數據處理與分析 ⭐
- **目標**：處理DCS系統採集的大量時間序列製程數據
- **使用工具**：Pandas
- **關鍵技術**：
  - CSV檔案讀取與日期時間解析
  - 時間序列重新取樣（1分鐘→1小時）
  - 滾動窗口統計（移動平均、標準差）
  - 缺失值填補（前向填補、線性插值）

### 2. 實驗數據整理與統計
- **目標**：整理多批次實驗數據，進行統計分析與比較
- **使用工具**：Pandas + NumPy
- **關鍵技術**：
  - 多個Excel檔案的讀取與合併
  - 分組聚合（groupby + agg）
  - 統計量計算（平均值、標準差、信賴區間）
  - 條件篩選（異常值排除）

### 3. 品質數據分析
- **目標**：分析產品品質數據，識別影響因子
- **使用工具**：Pandas
- **關鍵技術**：
  - 資料合併（製程參數+品質數據）
  - 相關性分析
  - 分組比較（不同批次、不同產品線）
  - Pivot table製作

### 4. 反應器溫度場分析
- **目標**：處理反應器內多點溫度測量數據（二維或三維空間）
- **使用工具**：NumPy
- **關鍵技術**：
  - 多維陣列建立與索引
  - 溫度梯度計算
  - 矩陣運算（溫度分布視覺化前處理）
  - 統計分析（溫度均勻性評估）

### 5. 批次資料向量化計算
- **目標**：對大量批次數據進行高效能數值計算
- **使用工具**：NumPy
- **關鍵技術**：
  - 向量化運算（取代Python迴圈）
  - 廣播機制（批次單位轉換）
  - 陣列方法（快速統計）
  - 布林索引（條件篩選）

---

## 📝 效能優化技巧總結

### NumPy效能優化
1. **使用向量化運算**：避免明確for迴圈
   ```python
   # 慢：Python迴圈
   result = [x * 2 for x in data]
   
   # 快：NumPy向量化
   result = data * 2
   ```

2. **善用廣播機制**：避免不必要的陣列複製
   ```python
   # NumPy會自動廣播
   matrix + vector  # 不需手動擴展vector維度
   ```

3. **選擇合適的資料型態**：節省記憶體
   ```python
   # float16適合精度要求不高的場合
   arr = np.array(data, dtype=np.float16)
   ```

### Pandas效能優化
1. **避免逐列處理**：使用向量化操作或apply()
   ```python
   # 慢：iterrows()
   for idx, row in df.iterrows():
       df.at[idx, 'new_col'] = row['A'] * 2
   
   # 快：向量化
   df['new_col'] = df['A'] * 2
   ```

2. **使用category型態**：處理重複值多的欄位
   ```python
   df['product_type'] = df['product_type'].astype('category')
   ```

3. **分塊讀取大檔案**：避免記憶體不足
   ```python
   for chunk in pd.read_csv('large_file.csv', chunksize=10000):
       process(chunk)
   ```

4. **使用query()方法**：條件篩選更快
   ```python
   # 稍慢
   df[(df['temp'] > 100) & (df['pressure'] < 5)]
   
   # 稍快
   df.query('temp > 100 and pressure < 5')
   ```

---

## 🚀 進階學習方向

完成本單元後，可以進一步探索：

1. **NumPy進階主題**：
   - 結構化陣列（Structured Arrays）
   - 記憶體佈局與效能優化
   - NumPy與C/Fortran的整合

2. **Pandas進階功能**：
   - 多層索引（MultiIndex）的高階操作
   - 自訂聚合函式
   - pipe()方法的鏈式資料處理
   - 效能分析與優化（profiling）

3. **整合其他套件**：
   - Polars（比Pandas更快的資料處理套件）
   - Dask（平行化處理大數據）
   - Vaex（超大型數據集處理）

4. **機器學習準備**：
   - 特徵工程技巧
   - 資料標準化與正規化
   - 訓練集與測試集分割

---

## 📚 參考資源

### 教科書
1. *Python for Data Analysis* by Wes McKinney（Pandas作者親著，經典必讀）
2. *NumPy Beginner's Guide* by Ivan Idris（NumPy入門）
3. *Python Data Science Handbook* by Jake VanderPlas（涵蓋NumPy、Pandas、Matplotlib）

### 線上資源
- [NumPy官方文件](https://numpy.org/doc/stable/)
- [Pandas官方文件](https://pandas.pydata.org/docs/)
- [Pandas速查表](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [NumPy速查表](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

### 視訊教學
- [DataCamp - NumPy Tutorial](https://www.datacamp.com/tutorial/python-numpy-tutorial)
- [Kaggle Learn - Pandas](https://www.kaggle.com/learn/pandas)

---

## ✍️ 課後習題提示

1. **NumPy向量化**：比較Python迴圈與NumPy向量化的效能差異
2. **陣列索引**：練習布林索引與花式索引的應用
3. **資料讀取**：讀取化工實驗CSV檔案並進行基本分析
4. **缺失值處理**：針對製程數據實作不同的缺失值填補策略
5. **分組聚合**：分析多批次實驗數據的統計特性
6. **時間序列**：處理時間序列製程數據的重新取樣與滾動統計

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- 課程討論區
- 指導教授 Office Hour
- 課程助教

---

**版本資訊**：Unit03 v1.0 | 最後更新：2026-01-27

**上一單元**：[Unit02 Python程式語言基礎](../Unit02/)  
**下一單元**：[Unit04 Matplotlib與Seaborn](../Unit04/)

---

**課程資訊**
- 課程名稱：電腦在化工上之應用
- 課程單元：Unit 03 Numpy與Pandas
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-02-23

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---