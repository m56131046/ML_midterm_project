# AI 驅動的真實成本航班比較系統 — 專案架構說明

## 一、專案背景與動機

許多旅客在訂機票時，只看到比價平台顯示的「基礎票價」，卻不知道低成本航空（LCC）在加上托運行李、選位、機上餐點、刷卡手續費之後，實際費用可能遠超過表面價格，甚至比全服務航空（FSC）還貴。

本系統的核心目標是：**讓使用者輸入自己的需求，自動計算每個航班的「真實總費用」，並透過 AI 給出最佳建議。**

---

## 二、系統整體架構

本系統採用前後端分離的架構設計：

```
使用者瀏覽器
     │
     │  HTTP 請求 (JSON)
     ▼
┌──────────────┐
│   前端 (React)  │  ← Vite 開發伺服器，Port 5173
│   App.jsx      │
│   App.css      │
└──────┬───────┘
       │  POST /api/search
       ▼
┌──────────────────┐
│  後端 (FastAPI)    │  ← Uvicorn ASGI 伺服器，Port 8000
│  main.py          │
│                   │
│  ┌─────────────┐  │
│  │ 費用計算邏輯  │  │
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ TinyLlama AI │  │
│  └─────────────┘  │
└──────┬───────────┘
       │  讀取 JSON 資料
       ▼
┌───────────────────────┐
│        資料層           │
│  airlines.json         │  航空公司基本資料
│  fees.json             │  附加費用費率表
│  flights.json          │  航班搜尋結果
└───────────────────────┘
```

---

## 三、技術選型說明

用npm create vite@ 來建立一個 名叫 fronted的react專案(就會建立一個fronted的資料夾)，
後把APP.jsx and APP.css複製到 fronted\src

### 前端：Vite + React

選用 **Vite** 作為前端開發工具，原因如下：
- 啟動速度極快（毫秒級 HMR，Hot Module Replacement），比傳統 Create React App 快上數倍
- 設定簡單，`npm create vite@latest` 一行指令即可建立 React 專案
- npm 全名是 node package manager
- 適合中小型原型開發

選用 **React** 作為 UI 框架，原因如下：
- 以元件化（Component）方式組織 UI，方便後續擴充（例如新增更多篩選條件或航班卡片）
- `useState` Hook 管理使用者輸入的需求與 API 回傳的結果，介面會自動重新渲染
- 大量社群資源，容易找到解決方案

### 後端：FastAPI + Uvicorn

選用 **FastAPI** 作為後端 API 框架，原因如下：
- 基於 Python，可直接整合 HuggingFace `transformers` 模型，無需額外轉換
- 自動產生 Swagger 互動式 API 文件（`http://localhost:8000/docs`），便於測試
- Pydantic 資料驗證：使用者送來的 JSON 會自動被型別檢查，減少錯誤

選用 **Uvicorn** 作為 ASGI 伺服器，原因如下：
- FastAPI 的官方推薦執行方式
- 非同步（async）處理能力佳，未來可擴充為高並發場景

### AI 推薦：TinyLlama（HuggingFace Transformers）

選用 **TinyLlama/TinyLlama-1.1B-Chat-v1.0** 作為 AI 推薦引擎：
- 參數量僅 1.1B，可在一般筆電的 CPU 上執行，不需 GPU
- 採用 Chat 格式（`<|system|>`, `<|user|>`, `<|assistant|>` 標籤），方便撰寫 Prompt
- 完全免費，不需申請 API Key
- 注意：此模型以英文為主，中文輸出品質有限，未來可替換為 GPT-4o 或繁體中文模型

---

## 四、各檔案功能說明

### 後端核心

#### `main.py` — 後端主程式

系統最核心的檔案，負責：

1. **啟動 FastAPI 應用程式**，設定 CORS（跨來源資源共享）允許前端 localhost:5173 發送請求
2. **載入三個 JSON 資料檔**，使用 `BASE_DIR` 確保無論從哪個目錄執行，都能找到正確路徑
3. **載入 TinyLlama 模型**（`pipeline("text-generation", ...)`），程式啟動時即載入模型至記憶體
4. **定義 `SearchQuery` 資料模型**，接收前端傳來的三個欄位：`baggage_weight`、`need_meal`、`seat_preference`
5. **實作 `POST /api/search` 端點**，核心邏輯如下：
   - 遍歷所有航班，查找該航空公司的費率表
   - 依照使用者需求計算：行李費 + 選位費 + 餐點費 + 手續費
   - 加上基礎票價 → 得出「真實總成本」
   - 依真實成本由低到高排序
   - 呼叫 TinyLlama 產生 AI 推薦文字
   - 回傳完整結果（航班列表 + AI 推薦）

#### `calculator.py` — 獨立計算腳本（命令列版）

這是開發初期用來驗證費用計算邏輯的**命令列測試腳本**，功能與 `main.py` 中的計算邏輯相同，但不含 API 層和 AI 推薦。可直接執行 `python calculator.py` 在終端機看到計算結果，方便快速驗證資料是否正確。

---

### 資料層（JSON 檔案）

#### `航空公司基本資料 airlines.json` — 航空公司基本資料

儲存系統中所有航空公司的靜態資訊，包含：
- `id`：航空公司代號（例如 `"TNA"`、`"EVA"`），作為各檔案之間的關聯鍵值
- `name`：顯示用的中文名稱
- `type`：航空公司類型，`"LCC"`（低成本航空）或 `"FSC"`（全服務航空）
- `logo_url`：Logo 圖片路徑（預留欄位，供未來 UI 顯示）

目前收錄：台灣虎航（TNA）與長榮航空（EVA）。

#### `附加費用費率表 fees.json` — 附加費用費率表

**系統最關鍵的資料庫**，以航空公司代號為鍵值，儲存每家航空的各項附加費用：
- `baggage`：不同重量的行李費（以公斤級距為鍵）
- `carry_on_upgrade`：手提行李升級費
- `seat_selection`：不同座位類型的選位費（標準座、前排、大空間）
- `meal`：機上餐點費用（固定金額）
- `payment_fee`：信用卡刷卡手續費

全服務航空（EVA）的行李費、餐點費、手續費均為 0，體現了「含在票價內」的商業模式差異。

#### `航班搜尋結果 flights.json` — 航班搜尋結果

模擬來自航班搜尋引擎的回傳資料，每筆航班包含：
- `flight_id`：航班號碼（例如 `"IT200"`）
- `airline_id`：對應到費率表的航空公司代號
- `departure` / `arrival`：出發地與目的地（IATA 機場代碼）
- `departure_time` / `arrival_time`：起降時間（ISO 8601 格式）
- `base_fare`：基礎票價（新台幣）
- `currency`：幣別

目前資料：IT200（虎航，TPE→NRT，$3,500）及 BR198（長榮，TPE→NRT，$9,800）。

---

### 前端

#### `frontend/src/App.jsx` — React 主元件

前端唯一的核心元件，包含：

1. **狀態管理**（`useState`）：
   - `query`：儲存使用者在表單選擇的需求（行李、餐點、座位）
   - `result`：儲存後端 API 回傳的計算結果
   - `loading`：控制按鈕顯示「計算中...」的狀態

2. **`handleSearch` 函式**：點擊按鈕後觸發，透過 `fetch` 發送 POST 請求到後端，取得計算結果並更新 `result` 狀態

3. **UI 渲染**：
   - 搜尋表單（行李選單、餐點勾選框、座位選單）
   - AI 推薦文字框（藍色左框線區塊）
   - 航班卡片列表（依真實成本排序，最便宜的有藍色外框）

#### `frontend/src/App.css` — 前端樣式

定義整個應用程式的視覺風格，採用現代簡潔設計：
- 淺灰色背景（`#f0f4f8`）搭配白色卡片，製造層次感
- 圓角卡片（`border-radius: 12px`）配合陰影，呈現卡片式 UI
- 藍色主色調（`#3182ce`）用於按鈕、AI 推薦框左邊框、最佳航班外框
- 費用明細區塊使用淺灰底色（`#f7fafc`）與較小字體，視覺上區分「主要資訊」與「細節資訊」

---

### 文件

#### `CLAUDE.md` — AI 助理設定

供 Claude AI 讀取的專案設定檔，記錄：
- 使用哪個 Python 虛擬環境
- 偏好使用繁體中文溝通
- 要求每次工作後更新 `PROGRESS.md`
- 程式碼需加上注解

#### `PROGRESS.md` — 開發進度日誌

每日工作結束後記錄當天完成的事項、遇到的問題與解決方法、以及系統啟動方式，讓下一次工作（或其他人接手）時能快速了解現狀。

#### `PROJECT_OVERVIEW.md`（本檔案）— 專案架構說明

提供給其他人閱讀的技術說明文件，解釋整個系統如何設計與運作。

---

## 五、資料流程圖

```
使用者在網頁選擇需求
（行李 20kg、需要餐點、標準座位）
         │
         ▼
前端 App.jsx 將選項打包成 JSON
{ "baggage_weight": "20kg",
  "need_meal": true,
  "seat_preference": "standard" }
         │
         │  POST http://localhost:8000/api/search
         ▼
後端 main.py 接收請求
         │
         ├─── 查詢 fees.json（TNA）
         │    行李費: 1050, 選位: 200, 餐點: 350, 手續費: 250
         │    IT200 真實成本 = 3500 + 1050 + 200 + 350 + 250 = 5350
         │
         ├─── 查詢 fees.json（EVA）
         │    行李費: 0, 選位: 0, 餐點: 0, 手續費: 0
         │    BR198 真實成本 = 9800 + 0 + 0 + 0 + 0 = 9800
         │
         ├─── 排序：IT200 ($5,350) < BR198 ($9,800)
         │
         └─── 呼叫 TinyLlama 產生推薦文字
                   │
                   ▼
前端收到回應，顯示：
① AI 推薦文字
② 航班卡片（依真實成本排序）
   ┌─────────────────────────┐
   │ IT200（藍框）             │
   │ 表面票價: $3,500          │
   │ + 行李: $1,050            │
   │ 真實總成本: $5,350 ✓最低  │
   └─────────────────────────┘
   ┌─────────────────────────┐
   │ BR198                   │
   │ 表面票價: $9,800          │
   │ + 行李: $0（已含）         │
   │ 真實總成本: $9,800        │
   └─────────────────────────┘
```

---

## 六、系統啟動方式

**後端**（一般命令提示字元）：
```bash
cd C:\Users\user\oneclicklca_scraper
.venv\Scripts\activate
cd C:\Users\user\oneclicklca_scraper>cd ML_midterm_project
py main.py
# 或: python main.py
```

**前端**（一般命令提示字元）：
```bash
cd C:\Users\user\oneclicklca_scraper\ML_midterm_project\frontend
npm run dev
```

開啟瀏覽器：`http://localhost:5173`


ML模型：
加一個小 ML 模組，例如：

價格預測（最直接）：用 scikit-learn 訓練一個線性回歸或決策樹，根據航空公司、路線、日期、星期幾來預測基礎票價，顯示「預測票價 vs 實際票價」給使用者參考。

---

## 七、未來可擴充方向

- **增加更多航空公司**：在三個 JSON 檔案中新增資料即可，不需修改程式碼
- **加入出發日期與來回選項**：擴充 `SearchQuery` 模型與前端表單
- **升級 AI 推薦引擎**：替換 TinyLlama 為 OpenAI GPT-4o（更好的中文品質）或 Llama-3-Taiwan（免費繁中模型）
- **即時爬取票價**：整合航空公司或 OTA（線上旅行社）API，取代靜態 JSON 資料
- **部署上線**：後端部署至 Render / Railway，前端部署至 Vercel / Netlify
