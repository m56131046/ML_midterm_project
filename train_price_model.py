"""
票價預測 ML 模組
================
資料來源：Kaggle Clean_Dataset.csv（印度航班歷史票價）
模型    ：RandomForestRegressor（scikit-learn）
目標    ：預測 price_ratio（相對倍率），消除幣值差異後套用到台灣航班

核心方法：
  1. 用 price_ratio 取代絕對票價，解決「印度 INR → 台灣 TWD」幣值差異問題
  2. 模型學習「days_left → 票價倍率變化規律」
  3. 預測時：
       predicted_tw_price = current_tw_price × (ratio_future / ratio_now)

執行方式（在 venv 中）：
    python train_price_model.py
產出：
    price_model.pkl
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "data_kaggle", "Clean_Dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "price_model.pkl")

# ── Step 1：讀取資料 ──────────────────────────────────────────────────────────
print("=" * 55)
print("Step 1：讀取 Clean_Dataset.csv")
print("=" * 55)

df = pd.read_csv(CSV_PATH)
print(f"原始筆數：{len(df)}")
print(f"欄位    ：{list(df.columns)}")

# 只取 Economy 艙（Business 票價結構不同）
df = df[df["class"] == "Economy"].copy()
print(f"Economy 艙筆數：{len(df)}")

# ── Step 2：特徵工程 ──────────────────────────────────────────────────────────
print("\nStep 2：特徵工程")

# stops：文字 → 整數（zero=0, one=1, two_or_more=2）
stops_map = {"zero": 0, "one": 1, "two_or_more": 2}
df["stops_int"] = df["stops"].map(stops_map).fillna(0).astype(int)

# departure_time：文字 → 整數（出發時段編碼）
dep_time_map = {
    "Early_Morning": 0,   # 00:00–06:00
    "Morning":       1,   # 06:00–12:00
    "Afternoon":     2,   # 12:00–16:00
    "Evening":       3,   # 16:00–20:00
    "Night":         4,   # 20:00–24:00
    "Late_Night":    5,   # 深夜
}
df["dep_period"] = df["departure_time"].map(dep_time_map).fillna(2).astype(int)

# duration：已是 float（小時），直接使用
# days_left：已是 int，直接使用
# price：已是 int，直接使用

# 過濾異常值
df = df[(df["duration"] > 0) & (df["price"] > 0) & (df["days_left"] > 0)]
print(f"清洗後筆數：{len(df)}")
print(f"days_left 範圍：{df['days_left'].min()} ~ {df['days_left'].max()} 天")

# ── Step 3：計算 price_ratio（消除幣值差異的核心）────────────────────────────
print("\nStep 3：計算 price_ratio")

# 每條「airline + source_city + destination_city」路線的中位數基準票價
df["route"] = df["airline"] + "|" + df["source_city"] + "|" + df["destination_city"]
route_median = df.groupby("route")["price"].median()
df["route_median"] = df["route"].map(route_median)
df["price_ratio"]  = df["price"] / df["route_median"]

# 過濾極端倍率（< 0.3 或 > 3.0 視為異常）
before = len(df)
df = df[(df["price_ratio"] >= 0.3) & (df["price_ratio"] <= 3.0)]
print(f"過濾極端值：{before - len(df)} 筆移除，剩餘 {len(df)} 筆")
print(f"price_ratio 統計：\n{df['price_ratio'].describe().round(3)}")

# ── Step 4：只保留重要特徵（依 Feature Importance 結果刪除無效欄位）────────
# 刪除：month（0.0000）、day_of_week（0.0000）
# 保留：days_left（0.8093）、duration（0.1542）、stops_int（0.0191）、dep_period（0.0174）
FEATURES = [
    "days_left",    # ⭐ 距出發天數（最重要，importance ≈ 0.81）
    "duration",     # 飛行時間（importance ≈ 0.15）
    "stops_int",    # 經停次數（importance ≈ 0.02）
    "dep_period",   # 出發時段（importance ≈ 0.02）
]

X = df[FEATURES].values
y = df["price_ratio"].values

print(f"\n使用特徵：{FEATURES}")

# ── Step 5：訓練 / 測試切分 ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nStep 5：訓練集 {len(X_train)} 筆 | 測試集 {len(X_test)} 筆")

# ── Step 6：訓練 RandomForestRegressor ───────────────────────────────────────
print("\nStep 6：訓練 RandomForestRegressor ...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Step 7：評估 ──────────────────────────────────────────────────────────────
print("\nStep 7：模型評估")
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
print(f"  MAE（price_ratio 誤差）：±{mae:.4f}")
print(f"  R²                     ：{r2:.4f}")

# Feature Importance
importances = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
print("\n  Feature Importance:")
for feat, imp in importances:
    bar = "█" * int(imp * 50)
    print(f"    {feat:<12} {imp:.4f}  {bar}")

# ── Step 8：儲存模型 ──────────────────────────────────────────────────────────
payload = {
    "model":         model,
    "features":      FEATURES,
    "route_medians": route_median.to_dict(),
}
with open(MODEL_OUT, "wb") as f:
    pickle.dump(payload, f)
print(f"\n✅ 模型已儲存 → {MODEL_OUT}")

# ── 驗證：模擬台灣航班預測邏輯 ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("驗證：模擬台灣航班預測流程")
print("=" * 55)

# 情境：虎航 TPE→NRT，飛行時間 3h，non-stop，傍晚出發
# 當前票價 NT$8,500，距出發 14 天
# 預測 7 天後的票價

f_14d = [14, 3.0, 0, 3]   # days_left=14
f_7d  = [ 7, 3.0, 0, 3]   # days_left=7

ratio_14d = model.predict([f_14d])[0]
ratio_7d  = model.predict([f_7d])[0]

current_fare   = 8500
predicted_fare = round(current_fare * (ratio_7d / ratio_14d) / 100) * 100
change_pct     = (predicted_fare - current_fare) / current_fare * 100
saving         = current_fare - predicted_fare

print(f"  當前票價（距出發 14 天）：NT${current_fare:,}")
print(f"  7 天後預測票價          ：NT${predicted_fare:,}")
print(f"  漲跌幅                  ：{change_pct:+.1f}%")
if predicted_fare > current_fare:
    print(f"  → 建議：現在購買，可省 NT${abs(saving):,}")
else:
    print(f"  → 建議：等待購買，預計省 NT${abs(saving):,}")
