"""
票價預測 ML 模組 v2
===================
資料來源：Kaggle Clean_Dataset.csv（印度航班，30 萬筆）
模型    ：RandomForestRegressor（scikit-learn）
目標    ：預測 price_ratio（相對倍率），消除幣值差異後套用到台灣航班

與 v1（economy.csv）的主要差異：
  1. 資料集已包含 days_left（距出發天數），無需資料增強
  2. duration 已為 float（小時），stops / departure_time 為類別字串
  3. 無 date 欄位，以 month=0, day_of_week=0 填充（RF 低權重特徵）
  4. 輸出 pkl 格式與 v1 相同，main.py 可直接載入

執行方式（需在 venv 中）：
    python train_price_model1.py
產出：
    price_model.pkl（覆蓋舊版）
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ── 路徑 ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "data_kaggle", "Clean_Dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "price_model.pkl")

# ── 航空業定價曲線（保留供 main.py _curve_ratio 使用，訓練不再依賴它）────────
PRICING_CURVE = {
    90: 0.82, 75: 0.86, 60: 0.90, 45: 0.95,
    30: 1.00, 21: 1.08, 14: 1.20,
     7: 1.45,  3: 1.75,  1: 2.10,
}

# ── 時段標籤 → 代表小時（供 dep_hour 特徵使用）──────────────────────────────
DEPARTURE_TIME_MAP = {
    "Early_Morning": 5,
    "Morning":       9,
    "Afternoon":    14,
    "Evening":      18,
    "Night":        21,
    "Late_Night":   23,
}

# ── stops 字串 → 整數 ────────────────────────────────────────────────────────
STOPS_MAP = {
    "zero":         0,
    "one":          1,
    "two_or_more":  2,
}

# ── Step 1：讀取並清洗 ────────────────────────────────────────────────────────
print("=" * 55)
print("Step 1：讀取並清洗 Clean_Dataset.csv")
print("=" * 55)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"原始筆數：{len(df)}")

# 只保留經濟艙（Business 票價結構不同，不適合混訓）
df = df[df["class"] == "Economy"].copy()
print(f"過濾剩 Economy 後：{len(df)}")

# 欄位轉換
# 先把 departure_time 的類別字串轉換成對應小時
# .fillna如果某一筆沒有對應到任何時段，就用 12 當預設值
# .astype 最後把整個欄位變成整數型態，便於模型使用
df["stops_int"]  = df["stops"].map(STOPS_MAP).fillna(1).astype(int)
df["dep_hour"]   = df["departure_time"].map(DEPARTURE_TIME_MAP).fillna(12).astype(int)
df["duration"]   = df["duration"].astype(float)
df["days_left"]  = df["days_left"].astype(int)
df["price"]      = df["price"].astype(float)

# 去除 price <= 0 或缺值
df = df.dropna(subset=["duration", "price"])
df = df[df["price"] > 0]

# 計算路線中位數（作為分母，轉換為 price_ratio）
# 這行是把三個欄位拼接成一個「路線識別字串」，例如 "AirlineA|CityX|CityY"
df["route"]      = df["airline"] + "|" + df["source_city"] + "|" + df["destination_city"]
# 用來計算同一航空公司、同一條路線的票價中位數。
route_median     = df.groupby("route")["price"].median()
# 這行是把 route_median 的結果（每條路線的中位數）映射回原始 df，讓每一筆資料都有對應的 route_median 欄位。
df["route_median"] = df["route"].map(route_median)

# 過濾票價倍率極端值
df["price_ratio"] = df["price"] / df["route_median"]
df = df[df["price_ratio"].between(0.3, 3.0)]
print(f"清洗後筆數：{len(df)}")

# ── Step 2：準備訓練資料 ──────────────────────────────────────────────────────
# 移除 airline_enc / from_enc / to_enc：
#   台灣航空公司不在訓練資料的 LabelEncoder 內，預測時只能填 0 當 fallback，
#   等於沒有實質資訊，留著反而讓模型學到無用的 noise。
#   移除後模型完全依賴「飛行特性 + 距出發天數」預測漲跌趨勢，更通用。
print("\nStep 2：準備訓練資料")

# month 與 day_of_week 在此資料集無法取得，填 0
# RandomForest 對這兩個特徵的 importance 將趨近 0，不影響預測品質
df["month"]       = 0
df["day_of_week"] = 0

# ── Step 3：準備訓練資料 ──────────────────────────────────────────────────────
print("\nStep 3：準備訓練資料")

# 特徵順序需與 main.py predict_price() 的 feat 向量完全一致
FEATURES = [
    "duration",              # 飛行時間（小時）
    "stops_int",             # 經停次數
    "dep_hour",              # 出發時段（轉為小時整數）
    "month",                 # 月份（此資料集填 0）
    "day_of_week",           # 星期幾（此資料集填 0）
    "days_left",             # ⭐ 距出發天數（最重要特徵）
]

X = df[FEATURES].values
y = df["price_ratio"].values

print(f"特徵向量數：{len(X)}")
print(f"price_ratio 統計：\n{df['price_ratio'].describe().round(3)}")
print(f"days_left 分布：\n{df['days_left'].value_counts().sort_index().head(10)}")

# ── Step 4：訓練 RandomForestRegressor ───────────────────────────────────────
print("\nStep 4：訓練 RandomForestRegressor")

# X 是輸入特徵矩陣（features）
# y 是輸出標籤（output），也就是 price_ratio
# test_size=0.2 意味著 20% 的資料會被劃分為測試集，剩下 80% 則是訓練集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")

# 設定模型參數
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
# 訓練模型，讓它學習從 X_train 預測 y_train 的關係
model.fit(X_train, y_train)

# ── Step 5：評估 ──────────────────────────────────────────────────────────────
print("\nStep 5：模型評估")

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
print(f"  MAE（price_ratio 誤差）：+-{mae:.4f}")
print(f"  R2                     ：{r2:.4f}")

importances = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
print("\n  Feature Importance:")
for feat, imp in importances:
    bar = "█" * int(imp * 50)
    print(f"    {feat:<25} {imp:.4f}  {bar}")

# ── Step 6：儲存模型 ──────────────────────────────────────────────────────────
print("\nStep 6：儲存模型")

payload = {
    "model":         model,
    "features":      FEATURES,
    "pricing_curve": PRICING_CURVE,           # 保留供 main.py _curve_ratio 使用
    "route_medians": route_median.to_dict(),
}
with open(MODEL_OUT, "wb") as f:
    pickle.dump(payload, f)

print(f"  [OK] 模型儲存至：{MODEL_OUT}")

# ── 驗證：模擬台灣航班預測流程 ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("驗證：模擬台灣航班預測流程")
print("=" * 55)

# 假設：台灣虎航 TPE→NRT，目前票價 NT$8,500，距出發 14 天
# 預測 7 天後（距出發 7 天時）的票價
# features: [duration, stops_int, dep_hour, month, day_of_week, days_left]
sample_feat_14d = [3.0, 0, 18, 0, 0, 14]
sample_feat_7d  = [3.0, 0, 18, 0, 0,  7]

#如果沒有 [0]，model.predict([sample_feat_7d]) 會回傳一個長度為 1 的陣列，例如：

# array([0.95])，有 [0] 才會取出第一個值變成純量：0.95
ratio_14d = model.predict([sample_feat_14d])[0]
ratio_7d  = model.predict([sample_feat_7d])[0]

current_fare = 8500
predicted_7d = current_fare * (ratio_7d / ratio_14d)

print(f"  當前票價（距出發14天）：NT${current_fare:,}")
print(f"  模型預測 ratio@14天  ：{ratio_14d:.4f}")
print(f"  模型預測 ratio@7天   ：{ratio_7d:.4f}")
print(f"  推算 7 天後票價      ：NT${predicted_7d:,.0f}")
print(f"  漲跌幅              ：{(predicted_7d - current_fare) / current_fare * 100:+.1f}%")
if predicted_7d > current_fare:
    print("  → 建議：現在購買（預測票價將上漲）")
else:
    print("  → 建議：等待購買（預測票價將下降）")
