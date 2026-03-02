# Unit00 Google Colab環境設定教學

## 課程目標

本單元將帶領學生熟悉Google Colab平台，這是一個基於雲端的Jupyter Notebook環境，提供免費的GPU/TPU運算資源，非常適合進行機器學習與深度學習的開發與實驗。

### 學習目標
- 理解Google Colab的基本功能與操作介面
- 學會建立、管理與分享Colab筆記本
- 掌握在Colab環境中上傳與下載資料檔案的方法
- 學會安裝與使用Python套件與模組
- 了解如何使用GPU/TPU加速運算資源
- 實作基本的Python程式碼執行與資料分析流程

---

## 1. Google Colab簡介

### 1.1 什麼是Google Colab？

Google Colaboratory (簡稱Colab) 是Google提供的免費雲端Jupyter Notebook服務，讓使用者可以透過瀏覽器直接撰寫和執行Python程式碼，無需在本地電腦安裝任何軟體。

**主要特色：**
- **完全免費**：提供免費的運算資源，包括GPU和TPU
- **無需安裝**：透過瀏覽器即可使用，不需要安裝Python或其他套件
- **雲端儲存**：筆記本自動儲存在Google Drive中
- **協作功能**：可以像Google文件一樣與他人即時協作
- **預裝套件**：已預裝許多常用的數據科學和機器學習套件

### 1.2 為什麼選擇Colab？

對於化學工程領域的AI應用學習，Colab提供了以下優勢：

1. **降低入門門檻**：不需要購買高階硬體設備
2. **快速開始**：免去複雜的環境設定流程
3. **資源共享**：可以輕鬆與同學或同事分享程式碼
4. **版本控制**：自動儲存修改歷史記錄
5. **GPU加速**：免費使用GPU進行深度學習訓練

### 1.3 Colab的限制

使用Colab時也需要了解其限制：

- **執行時間限制**：免費版單次執行時間最長12小時
- **閒置中斷**：閒置90分鐘後會自動中斷連線
- **儲存空間**：使用Google Drive的儲存空間配額
- **運算資源**：免費版的GPU/TPU使用有配額限制
- **網路依賴**：需要穩定的網路連線

---

## 2. 開始使用Google Colab

### 2.1 建立Colab筆記本

#### 方法一：從Google Drive建立

1. 登入您的Google帳號
2. 前往 [Google Drive](https://drive.google.com)
3. 點選左上角的「新增」按鈕
4. 選擇「更多」→「Google Colaboratory」
5. 如果沒有看到此選項，需要先安裝Colab應用程式：
   - 選擇「更多」→「連結更多應用程式」
   - 搜尋「Colaboratory」並安裝

#### 方法二：直接訪問Colab網站

直接訪問 [https://colab.research.google.com](https://colab.research.google.com)，即可開始使用。

#### 方法三：從GitHub匯入

如果已有GitHub上的Jupyter Notebook，可以直接在Colab中開啟：
- 在Colab首頁選擇「GitHub」分頁
- 輸入GitHub使用者名稱或儲存庫URL
- 選擇要開啟的筆記本

### 2.2 Colab介面說明

Colab的介面主要由以下幾個部分組成：

#### 工具列
- **檔案**：新增、開啟、儲存筆記本
- **編輯**：剪下、複製、貼上等編輯功能
- **檢視**：調整介面顯示選項
- **插入**：新增程式碼或文字儲存格
- **執行階段**：管理執行環境和硬體加速器
- **工具**：設定、鍵盤快捷鍵等

#### 側邊欄
- **目錄**：顯示筆記本的章節結構
- **程式碼片段**：常用程式碼範例
- **檔案**：管理檔案系統
- **變數檢查器**：查看當前變數狀態

#### 儲存格 (Cells)
- **程式碼儲存格**：撰寫和執行Python程式碼
- **文字儲存格**：使用Markdown格式撰寫說明文字

### 2.3 基本操作

#### 新增儲存格
- 點選「+ 程式碼」按鈕新增程式碼儲存格
- 點選「+ 文字」按鈕新增文字儲存格
- 或使用快捷鍵：
  - `Ctrl + M B`：在下方新增儲存格
  - `Ctrl + M A`：在上方新增儲存格

#### 執行儲存格
- 點選儲存格左側的執行按鈕 (▶)
- 或使用快捷鍵：`Ctrl + Enter` (執行後停留) 或 `Shift + Enter` (執行後移至下一格)

#### 刪除儲存格
- 點選儲存格右上角的「⋮」圖示
- 選擇「刪除儲存格」
- 或使用快捷鍵：`Ctrl + M D`

#### 移動儲存格
- 點選儲存格右上角的「⋮」圖示
- 選擇「上移儲存格」或「下移儲存格」
- 或使用快捷鍵：`Ctrl + M K` (上移) 或 `Ctrl + M J` (下移)

---

## 3. 檔案管理與資料上傳

### 3.1 檔案系統架構

在Colab中，每次啟動執行階段都會建立一個暫時的虛擬機器環境，檔案系統架構如下：

- `/content/`：主要工作目錄，可以在此上傳檔案
- `/content/drive/`：掛載Google Drive後的路徑
- `/usr/local/lib/python3.x/`：Python套件安裝位置

**重要提醒**：每次執行階段結束後，`/content/` 目錄下的檔案會被清除，只有掛載到Google Drive的檔案才會永久保存。

### 3.2 上傳檔案的方法

#### 方法一：直接上傳（暫時性）

最簡單的方法是透過側邊欄直接上傳：

1. 點選左側邊欄的「檔案」圖示 (📁)
2. 點選「上傳」按鈕
3. 選擇要上傳的檔案

```python
# 使用Python程式碼上傳檔案
from google.colab import files

uploaded = files.upload()

# 查看上傳的檔案
for filename in uploaded.keys():
    print(f'Uploaded file: {filename}, Size: {len(uploaded[filename])} bytes')
```

**注意**：此方法上傳的檔案僅存在於當前執行階段，執行階段結束後檔案會消失。

#### 方法二：掛載Google Drive（永久性）

透過掛載Google Drive，可以直接存取雲端硬碟中的檔案：

```python
from google.colab import drive

# 掛載Google Drive
drive.mount('/content/drive')
```

執行後會出現授權連結，點選連結並授權後，即可在 `/content/drive/MyDrive/` 路徑下存取Google Drive的檔案。

```python
# 列出Google Drive根目錄的檔案
import os
os.listdir('/content/drive/MyDrive/')
```

#### 方法三：從URL下載

直接從網路下載檔案：

```python
# 使用wget下載檔案
!wget https://example.com/data.csv -O /content/data.csv

# 或使用Python的requests模組
import requests

url = 'https://example.com/data.csv'
response = requests.get(url)

with open('/content/data.csv', 'wb') as f:
    f.write(response.content)
```

### 3.3 下載檔案

#### 下載到本地電腦

```python
from google.colab import files

# 下載單個檔案
files.download('/content/result.csv')
```

#### 壓縮多個檔案後下載

```python
# 壓縮整個資料夾
!zip -r /content/results.zip /content/results/

# 下載壓縮檔
from google.colab import files
files.download('/content/results.zip')
```

### 3.4 最佳實踐建議

1. **使用Google Drive存放資料**：將資料集存放在Google Drive，透過掛載方式存取
2. **建立專案資料夾結構**：在Drive中建立清晰的資料夾結構
3. **定期備份重要檔案**：將重要的輸出結果儲存到Drive
4. **使用相對路徑**：程式碼中使用相對路徑，方便在不同環境中執行

---

## 4. Python套件管理

### 4.1 預裝套件

Colab已經預裝了許多常用的資料科學與機器學習套件，包括：

- **NumPy**：數值計算
- **Pandas**：資料分析
- **Matplotlib** / **Seaborn**：資料視覺化
- **Scikit-learn**：傳統機器學習
- **TensorFlow** / **Keras**：深度學習
- **PyTorch**：深度學習

### 4.2 查看已安裝套件

```python
# 列出所有已安裝的套件
!pip list

# 查看特定套件的版本
import numpy as np
import pandas as pd
import tensorflow as tf

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"TensorFlow version: {tf.__version__}")
```

### 4.3 安裝新套件

#### 使用pip安裝

```python
# 安裝單個套件
!pip install package_name

# 安裝特定版本
!pip install package_name==1.2.3

# 安裝多個套件
!pip install package1 package2 package3

# 靜默安裝（不顯示詳細訊息）
!pip install -q package_name
```

#### 使用conda安裝

Colab也支援conda套件管理器：

```python
# 安裝conda
!pip install -q condacolab
import condacolab
condacolab.install()

# 使用conda安裝套件
!conda install -c conda-forge package_name
```

#### 從GitHub安裝開發版本

```python
# 安裝GitHub上的套件
!pip install git+https://github.com/user/repo.git
```

### 4.4 升級套件

```python
# 升級單個套件
!pip install --upgrade package_name

# 升級pip本身
!pip install --upgrade pip
```

### 4.5 卸載套件

```python
# 卸載套件
!pip uninstall -y package_name
```

### 4.6 處理套件衝突

有時候安裝新套件可能會與現有套件版本衝突：

```python
# 檢查套件相依性
!pip check

# 強制重新安裝
!pip install --force-reinstall package_name
```

### 4.7 化工領域常用套件

以下是化工領域AI應用可能會用到的額外套件：

```python
# 化學資訊學
!pip install rdkit-pypi  # 分子結構處理
!pip install chempy      # 化學計算

# 數值計算與優化
!pip install scipy       # 科學計算（通常已預裝）
!pip install cvxpy       # 凸優化

# 時間序列分析
!pip install statsmodels # 統計模型
!pip install prophet     # Facebook時間序列預測

# 進階機器學習
!pip install xgboost     # 梯度提升樹
!pip install lightgbm    # 輕量級梯度提升
!pip install catboost    # CatBoost
```

---

## 5. GPU/TPU加速運算

### 5.1 什麼是GPU/TPU？

- **GPU (Graphics Processing Unit)**：圖形處理器，擅長平行運算，適合深度學習訓練
- **TPU (Tensor Processing Unit)**：Google開發的專用AI晶片，專為TensorFlow優化

### 5.2 啟用硬體加速器

#### 設定步驟

1. 點選選單列的「執行階段」
2. 選擇「變更執行階段類型」
3. 在「硬體加速器」下拉選單中選擇：
   - **None**：僅使用CPU
   - **GPU**：使用GPU加速（推薦用於PyTorch）
   - **TPU**：使用TPU加速（推薦用於TensorFlow）
4. 點選「儲存」

**注意**：更改硬體加速器會重新啟動執行階段，所有變數和未儲存的資料會遺失。

### 5.3 檢查GPU資訊

```python
# 檢查是否有GPU可用
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# 顯示GPU詳細資訊
!nvidia-smi
```

```python
# PyTorch檢查GPU
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

### 5.4 GPU記憶體管理

#### TensorFlow記憶體管理

```python
import tensorflow as tf

# 設定GPU記憶體動態分配
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 限制GPU記憶體使用量
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # 4GB
    except RuntimeError as e:
        print(e)
```

#### PyTorch記憶體管理

```python
import torch

# 清空GPU快取
torch.cuda.empty_cache()

# 查看GPU記憶體使用情況
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
```

### 5.5 TPU使用

#### TensorFlow with TPU

```python
import tensorflow as tf

# 檢查TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    print('Could not connect to TPU')
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
```

### 5.6 GPU/TPU使用注意事項

1. **配額限制**：免費版有使用時數限制，請合理使用
2. **自動斷線**：長時間無操作會自動斷線，建議定期執行程式碼或移動滑鼠
3. **資料傳輸**：將資料載入GPU記憶體需要時間，對小資料集可能不划算
4. **記憶體溢位**：注意batch size設定，避免GPU記憶體不足
5. **CPU vs GPU**：不是所有運算都適合GPU，簡單運算用CPU可能更快

---

## 6. 實作範例：化工數據分析流程

本節透過一個簡單的化工數據分析範例，展示如何在Colab中完整執行資料分析流程。

### 6.1 範例情境

假設我們有一組反應器操作數據，包含溫度、壓力、流量等參數，以及對應的產物轉化率。我們將進行以下步驟：

1. 產生模擬數據
2. 資料探索與視覺化
3. 簡單的統計分析
4. 基本的機器學習預測

### 6.2 產生模擬數據

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 設定隨機種子，確保結果可重現
np.random.seed(42)

# 產生模擬的反應器操作數據
n_samples = 200

# 操作參數
temperature = np.random.normal(350, 20, n_samples)  # Temperature (°C)
pressure = np.random.normal(5, 0.5, n_samples)      # Pressure (bar)
flow_rate = np.random.normal(100, 10, n_samples)    # Flow rate (L/min)

# 產物轉化率（模擬真實關係：受溫度和壓力影響）
conversion = (
    0.3 * (temperature - 300) / 50 + 
    0.4 * (pressure - 4) / 2 + 
    0.1 * (flow_rate - 90) / 20 +
    np.random.normal(0, 0.05, n_samples)
)
conversion = np.clip(conversion, 0, 1)  # 限制在0-1之間

# 建立DataFrame
data = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'Flow_Rate': flow_rate,
    'Conversion': conversion
})

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nBasic statistics:")
print(data.describe())
```

### 6.3 資料視覺化

```python
# 設定圖表樣式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# 建立子圖
fig, axes = plt.subplots(2, 2)

# 1. 溫度分布
axes[0, 0].hist(data['Temperature'], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Temperature Distribution')
axes[0, 0].set_xlabel('Temperature (°C)')
axes[0, 0].set_ylabel('Frequency')

# 2. 壓力分布
axes[0, 1].hist(data['Pressure'], bins=30, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Pressure Distribution')
axes[0, 1].set_xlabel('Pressure (bar)')
axes[0, 1].set_ylabel('Frequency')

# 3. 溫度 vs 轉化率
axes[1, 0].scatter(data['Temperature'], data['Conversion'], alpha=0.5)
axes[1, 0].set_title('Temperature vs Conversion')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Conversion')

# 4. 壓力 vs 轉化率
axes[1, 1].scatter(data['Pressure'], data['Conversion'], alpha=0.5, color='green')
axes[1, 1].set_title('Pressure vs Conversion')
axes[1, 1].set_xlabel('Pressure (bar)')
axes[1, 1].set_ylabel('Conversion')

plt.tight_layout()
plt.show()
```

### 6.4 相關性分析

```python
# 計算相關係數矩陣
correlation_matrix = data.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# 繪製熱力圖
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap')
plt.show()
```

### 6.5 簡單的機器學習預測

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 準備資料
X = data[['Temperature', 'Pressure', 'Flow_Rate']]
y = data['Conversion']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 建立並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 顯示係數
print(f"\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# 視覺化預測結果
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Conversion')
plt.ylabel('Predicted Conversion')
plt.title('Actual vs Predicted Conversion')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Colab進階技巧

### 7.1 Magic Commands

Colab支援IPython的magic commands，可以提升工作效率：

```python
# 顯示所有magic commands
%lsmagic

# 測量程式碼執行時間
%timeit sum(range(1000))

# 測量單次執行時間
%%time
result = sum(range(1000000))

# 顯示當前變數
%whos

# 執行外部Python檔案
%run script.py

# 查看命令執行歷史
%history
```

### 7.2 Shell Commands

使用 `!` 可以執行shell命令：

```python
# 查看當前目錄
!pwd

# 列出檔案
!ls -la

# 查看系統資訊
!cat /proc/cpuinfo | grep "model name" | head -1
!cat /proc/meminfo | grep "MemTotal"

# 安裝系統套件
!apt-get install -y package_name
```

### 7.3 表單功能

Colab提供互動式表單功能，可以讓使用者輸入參數：

```python
#@title 反應器參數設定
temperature = 350  #@param {type:"slider", min:300, max:400, step:5}
pressure = 5.0  #@param {type:"number"}
catalyst_type = "A"  #@param ["A", "B", "C"]
use_gpu = True  #@param {type:"boolean"}

print(f"Temperature: {temperature}°C")
print(f"Pressure: {pressure} bar")
print(f"Catalyst: {catalyst_type}")
print(f"Use GPU: {use_gpu}")
```

### 7.4 分享與協作

#### 分享筆記本

1. 點選右上角的「共用」按鈕
2. 設定存取權限：
   - **檢視者**：只能查看
   - **留言者**：可以留言但不能編輯
   - **編輯者**：可以編輯內容
3. 複製分享連結

#### 版本控制

Colab會自動儲存編輯歷史：
- 點選「檔案」→「修訂記錄」
- 可以查看和還原到先前的版本

#### 儲存到GitHub

1. 點選「檔案」→「在GitHub中儲存副本」
2. 授權GitHub存取
3. 選擇儲存庫和分支
4. 輸入commit訊息並儲存

### 7.5 快捷鍵

常用快捷鍵（Mac請將Ctrl改為Cmd）：

| 功能 | 快捷鍵 |
|------|--------|
| 執行儲存格 | `Ctrl + Enter` |
| 執行並移至下一格 | `Shift + Enter` |
| 新增程式碼儲存格 | `Ctrl + M B` |
| 新增文字儲存格 | `Ctrl + M M` |
| 刪除儲存格 | `Ctrl + M D` |
| 復原刪除 | `Ctrl + M Z` |
| 顯示快捷鍵列表 | `Ctrl + M H` |
| 註解/取消註解 | `Ctrl + /` |
| 尋找與取代 | `Ctrl + H` |

---

## 8. 常見問題與解決方案

### 8.1 執行階段相關

**Q: 執行階段自動中斷怎麼辦？**

A: Colab會在閒置90分鐘或執行超過12小時後自動中斷。建議：
- 定期移動滑鼠或執行程式碼
- 將重要資料儲存到Google Drive
- 使用Colab Pro可獲得更長的執行時間

**Q: 如何重新啟動執行階段？**

A: 點選「執行階段」→「重新啟動執行階段」，這會清除所有變數和記憶體。

**Q: 執行階段記憶體不足？**

A: 
- 刪除不需要的大型變數：`del variable_name`
- 使用 `gc.collect()` 強制垃圾回收
- 分批處理資料
- 考慮升級到Colab Pro

### 8.2 檔案相關

**Q: 上傳的檔案不見了？**

A: 每次執行階段重啟，`/content/` 目錄會被清空。請將檔案儲存到Google Drive。

**Q: 如何處理大型數據集？**

A: 
- 將資料集上傳到Google Drive
- 使用資料串流讀取，而非一次載入
- 考慮使用資料壓縮格式（如parquet）

### 8.3 套件相關

**Q: 安裝套件後仍無法import？**

A: 
- 確認安裝成功：`!pip show package_name`
- 重新啟動執行階段
- 檢查套件名稱是否正確

**Q: 套件版本衝突？**

A: 
- 使用虛擬環境隔離
- 強制重新安裝：`!pip install --force-reinstall package_name`
- 指定版本：`!pip install package_name==version`

### 8.4 GPU相關

**Q: GPU不可用？**

A: 
- 確認已啟用GPU：「執行階段」→「變更執行階段類型」→「GPU」
- 檢查配額是否用完
- 等待一段時間後再試

**Q: 如何查看GPU使用情況？**

A: 執行 `!nvidia-smi` 查看GPU狀態和記憶體使用。

---

## 9. 總結與學習資源

### 9.1 本單元重點回顧

- Google Colab是免費且強大的雲端Jupyter Notebook環境
- 支援GPU/TPU加速，適合機器學習與深度學習
- 透過Google Drive整合實現資料持久化
- 預裝常用套件，可輕鬆安裝額外套件
- 支援協作與分享功能

### 9.2 後續學習方向

完成本單元後，建議：

1. **熟練基本操作**：多練習建立筆記本和執行程式碼
2. **學習Python基礎**：如果Python基礎不夠扎實，建議加強
3. **探索範例**：參考Colab官方範例學習進階用法
4. **建立專案**：實際動手做一個小型資料分析專案

### 9.3 學習資源

**官方資源：**
- [Google Colab官網](https://colab.research.google.com)
- [Colab官方教學](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab常見問題](https://research.google.com/colaboratory/faq.html)

**進階學習：**
- [Colab Pro介紹](https://colab.research.google.com/signup)
- [TensorFlow教學](https://www.tensorflow.org/tutorials)
- [PyTorch教學](https://pytorch.org/tutorials/)

**社群資源：**
- [Stack Overflow - Colab標籤](https://stackoverflow.com/questions/tagged/google-colaboratory)
- [Reddit - r/GoogleColab](https://www.reddit.com/r/GoogleColab/)

---

## 附錄A：Colab與本地Jupyter的差異

| 特性 | Google Colab | 本地Jupyter |
|------|-------------|-------------|
| 安裝 | 無需安裝 | 需安裝Python和Jupyter |
| 運算資源 | 共享GPU/TPU（有限制） | 依本地硬體 |
| 儲存空間 | Google Drive | 本地硬碟 |
| 協作 | 支援即時協作 | 需額外工具 |
| 網路需求 | 需要穩定網路 | 離線可用 |
| 執行時間 | 有時間限制 | 無限制 |
| 套件管理 | 需重複安裝 | 持久化 |
| 資料安全 | 儲存在Google雲端 | 儲存在本地 |

---

## 附錄B：Colab快速參考表

### 常用程式碼片段

```python
# 掛載Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 上傳檔案
from google.colab import files
uploaded = files.upload()

# 下載檔案
files.download('/content/file.csv')

# 檢查GPU
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# 清除輸出
from IPython.display import clear_output
clear_output()

# 顯示影像
from IPython.display import Image, display
display(Image('/content/image.png'))
```

---

**課程結束**

恭喜您完成Unit00 Google Colab環境設定教學！現在您已經掌握了使用Colab進行Python程式開發和資料分析的基本技能。在接下來的課程中，我們將深入學習AI與機器學習的理論與實作。

---

**課程資訊**
- 課程名稱：電腦在化工上之應用
- 課程單元：Unit00 Google Colab環境設定
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-02-23

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---