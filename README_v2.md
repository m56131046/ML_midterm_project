# AI-Powered True Cost Flight Comparison System

AI 驅動的「真實機票費用」比較系統，結合機器學習價格預測與附加費用計算，幫助使用者找到最划算的航班。

---

## 系統架構

```
├── main.py                  # FastAPI 後端主程式
├── train_price_model.py     # 機器學習模型訓練腳本
├── price_model.pkl          # 訓練好的價格預測模型
├── 航空公司基本資料 airlines.json
├── 附加費用費率表 fees.json
├── 機場航區對應表 airport_zones.json
└── frontend/                # React 前端
```

---

## 環境需求

### Python（後端）

- **Python 版本**：3.10 以上

所需套件：

```
fastapi
uvicorn
pydantic
transformers
fast-flights
numpy
pandas
scikit-learn
```

### Node.js（前端）

- **Node.js 版本**：18 以上
- **套件管理器**：npm（隨 Node.js 一起安裝）

前往 [https://nodejs.org](https://nodejs.org) 下載 **LTS 版本**並安裝，安裝完成後確認版本：

```bash
node -v
npm -v
```

---

## 安裝步驟

### 1. 建立並啟動 Python 虛擬環境

```bash
# 建立虛擬環境（只需執行一次）
python -m venv .venv

# 啟動虛擬環境 — Windows
.venv\Scripts\activate

# 啟動虛擬環境 — macOS / Linux
source .venv/bin/activate
```

### 2. 安裝 Python 套件

```bash
pip install fastapi uvicorn pydantic transformers fast-flights numpy
```

### 3. 安裝前端套件

```bash
cd frontend
npm install
```

---

## 執行方式


### 步驟一：啟動後端 API

在**專案根目錄**（`main.py` 所在的資料夾）執行：

```bash
uvicorn main:app --reload
```

後端預設運行於 `http://127.0.0.1:8000`

> API 文件可在 `http://127.0.0.1:8000/docs` 查看

### 步驟二：啟動前端

開啟**新的終端機視窗**，切換到 `frontend` 資料夾後執行：

```bash
cd frontend
npm run dev
```

前端預設運行於 `http://localhost:5173`

---

## 使用流程

1. 開啟瀏覽器進入 `http://localhost:5173`
2. 輸入出發地、目的地、出發日期
3. 填入行李需求、座位偏好等附加需求
4. 系統將查詢航班並計算各航班的「真實總費用」
5. AI 推薦模組會給出最划算的選擇建議

---

## 注意事項

- 後端與前端需**同時執行**，系統才能正常運作
- 執行前請確認已啟動虛擬環境
- `fast-flights` 套件需要網路連線以查詢 Google Flights 資料
