# 進度紀錄

## 2026-05-26

### 完成項目

#### 1. 整合 fast-flights 真實航班 API
- 安裝 `fast-flights` 套件（虛擬環境：`C:\Users\user\oneclicklca_scraper\.venv`）
- 建立 `test_flights_api.py` 驗證查詢功能
- 確認可查詢 TPE->NRT、TSA->HND 等路線的真實 Google Flights 資料

#### 2. 台灣籍航空公司篩選
- 限定只顯示 4 家台灣籍航空：Tigerair Taiwan、EVA Air、China Airlines、STARLUX Airlines
- 對應內部 airline_id：TNA、EVA、CAL、SJX（資料來源：航空公司基本資料 airlines.json）
- 以 (airline_name, departure) 為唯一鍵去除重複班次

#### 3. 重試機制
- fetch_real_flights() 解析失敗（name 全空）時最多自動重試 3 次
- 經測試成功率約 60%，重試後可大幅提升穩定性

#### 4. main.py 整合真實 API
- 移除靜態 flights.json，改由 fetch_real_flights() 即時抓取
- SearchQuery 新增 date（出發日期，必填）、from_airport（預設 TPE）
- parse_price() 將 "NT,699" 字串解析為整數
- 回傳資料新增 airline_name、duration、stops

#### 5. 虎航 Bundle 優惠邏輯
- Bundle A：行李 1~20kg + 標準選位，(行李費+選位費) > NT 時提示，套裝價 NT
- Bundle B：行李 1~20kg + 大空間座位，(行李費+選位費) > NT 時提示，套裝價 NT，含免餐費
- 後端計算 bundle_true_cost 與 savings，前端顯示黃色提示框

#### 6. 前端 App.jsx 更新
- 新增出發機場下拉選單（TPE / TSA）
- 新增出發日期選擇器（預設 7 天後，不可選過去日期）
- 航班卡片顯示 airline_name、飛行時間、經停次數
- Bundle B 含餐時顯示「此 Bundle 已含餐費」綠色提示

#### 7. AI 推薦調整
- Prompt 加入 Reply in English within 100 words.
- max_new_tokens 從 100 調整為 200，避免輸出被截斷

---

### 待處理 / 注意事項
- fast-flights 套件偶爾因 Google 回傳結構不同導致解析失敗，已加重試機制
- test_flights_api.py 篩選邏輯已註解，可清理
- 前端根目錄 App.jsx 與 frontend/src/App.jsx 內容需保持同步

---

## 2026-05-29

### 完成項目

#### 1. Demo PPT（6 頁）
- 生成 `TrueCostFlight_Demo.pptx`，共 6 頁：標題、問題背景、解決方案、系統架構、技術選型、結語
- 配色：Ocean Gradient（深藍 #065A82 + 薄荷綠 #02C39A）
- 生成腳本：`make_ppt.py`（使用 python-pptx）

#### 2. 票價預測 ML 模組（建立中）
- 建立 `train_price_model.py`：模擬 2000 筆訓練資料，特徵包含 airline、zone、day_of_week、days_ahead、from_airport
- 使用 RandomForestRegressor 訓練票價預測模型，儲存為 `price_model.pkl`
- 訓練腳本已寫好，需在 venv 環境下執行（需 scikit-learn）

### 待處理（下次繼續）
- **Step 1**：在本機 venv 執行 `python train_price_model.py` 產生 `price_model.pkl`
  - 指令：`C:\Users\user\oneclicklca_scraper\.venv\Scripts\python train_price_model.py`
- **Step 2**：更新 `main.py` — 啟動時載入 price_model.pkl，/api/search 回傳 `predicted_price` 與 `price_diff_pct`
- **Step 3**：更新 `frontend/src/App.jsx` — 航班卡片顯示「AI 預測票價 vs 實際票價」，標示特價或偏高
