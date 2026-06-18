from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
from fast_flights import FlightData, Passengers, create_filter, get_flights_from_filter
import json
import os
import pickle
import re
import numpy as np
from datetime import date as date_type, datetime

app = FastAPI()

# 允許前端跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 載入靜態資料 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

airlines_data = load_json("航空公司基本資料 airlines.json")
fees_data     = load_json("附加費用費率表 fees.json")
zones_data    = load_json("機場航區對應表 airport_zones.json")

# --- IATA 機場代碼 → 中文顯示名稱 ---
AIRPORT_NAMES = {
    "NRT": "東京成田 (NRT)",
    "HND": "東京羽田 (HND)",
    "KIX": "大阪關西 (KIX)",
    "FUK": "福岡 (FUK)",
    "CTS": "札幌新千歲 (CTS)",
    "OKA": "沖繩那霸 (OKA)",
    "NGO": "名古屋 (NGO)",
    "ICN": "首爾仁川 (ICN)",
    "GMP": "首爾金浦 (GMP)",
    "PUS": "釜山 (PUS)",
    "BKK": "曼谷素萬那普 (BKK)",
    "DMK": "曼谷廊曼 (DMK)",
    "SIN": "新加坡 (SIN)",
    "KUL": "吉隆坡 (KUL)",
    "MNL": "馬尼拉 (MNL)",
    "CEB": "宿霧 (CEB)",
    "SGN": "胡志明市 (SGN)",
    "HAN": "河內 (HAN)",
    "DPS": "峇里島 (DPS)",
    "SYD": "雪梨 (SYD)",
    "MEL": "墨爾本 (MEL)",
    "LAX": "洛杉磯 (LAX)",
    "SFO": "舊金山 (SFO)",
    "JFK": "紐約 (JFK)",
    "LHR": "倫敦希斯洛 (LHR)",
    "AMS": "阿姆斯特丹 (AMS)",
    "VIE": "維也納 (VIE)",
    "KHH": "高雄小港 (KHH)",
}

# 航區中文說明
ZONE_LABELS = {
    "zone_1": "Zone 1・短程",
    "zone_2": "Zone 2・中程",
    "zone_3": "Zone 3・長程",
    "zone_4": "Zone 4・長程",
}

# 台灣虎航 Bundle 優惠組合定義
# Bundle A：20kg 行李 + 標準選位 = NT$950
TNA_BUNDLE_A = {
    "price":        950,
    "max_kg":       20,
    "seat":         "standard",
    "include_meal": False,  # Bundle A 不包含餐費
    "description":  "虎航 Bundle A：20kg 行李 + 標準選位 = NT$950",
}
# Bundle B：20kg 行李 + 大空間選位 + 免餐費 = NT$1600
TNA_BUNDLE_B = {
    "price":        1600,
    "max_kg":       20,
    "seat":         "extra_legroom",
    "include_meal": True,   # Bundle B 涵蓋餐費，不另外加收
    "description":  "虎航 Bundle B：20kg 行李 + 大空間座位 + 免餐費 = NT$1600",
}

# 台灣籍航空公司：Google Flights 名稱 → 內部 airline_id
TAIWAN_AIRLINES = {
    "Tigerair Taiwan":  "TNA",
    "EVA Air":          "EVA",
    "China Airlines":   "CAL",
    "STARLUX Airlines": "SJX",
}

# --- 載入 TinyLlama 模型 ---
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# --- 載入票價預測模型 ---
_MODEL_PATH = os.path.join(BASE_DIR, "price_model.pkl")
with open(_MODEL_PATH, "rb") as _f:
    _price_payload = pickle.load(_f)

_price_model    = _price_payload["model"]
_pricing_curve  = _price_payload["pricing_curve"]   # {days: ratio, ...}



def _parse_duration_hours(s: str) -> float:
    """'3h 55m' → 3.917"""
    m = re.match(r"(\d+)h\s*(\d+)m", str(s).strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60
    m2 = re.match(r"(\d+)h", str(s).strip())
    if m2:
        return float(m2.group(1))
    return 3.0   # fallback：短程平均


def _parse_dep_hour(s: str) -> int:
    """'18:55' or '6:30 PM' → 整數小時"""
    try:
        return int(str(s).strip().split(":")[0])
    except Exception:
        return 12


def predict_price(
    base_fare: int,
    duration_str: str,
    departure_str: str,
    stops_val: int,
    departure_date: str,   # YYYY-MM-DD
) -> dict:
    """
    預測 7 天後的票價，並給出購買建議。
        - 由於模型訓練時的 target 是 price_ratio（相對倍率），因此預測結果也是 ratio。
    模型仍能透過 days_before_departure 給出有意義的漲跌趨勢。
    """
    today       = date_type.today()
    dep_date    = datetime.strptime(departure_date, "%Y-%m-%d").date()
    days_now    = max(1, (dep_date - today).days)          # 距今天數（至少 1）
    days_7d     = max(1, days_now - 7)                     # 7 天後距出發天數
    dep_month   = dep_date.month
    dep_weekday = dep_date.weekday()
    dur_hours   = _parse_duration_hours(duration_str)
    dep_hour    = _parse_dep_hour(departure_str)
    stops_int   = 0 if str(stops_val).strip() in ("0", "non-stop", "Nonstop") else 1

    # 組合特徵向量，順序需與 train_price_model1.py 的 FEATURES 完全一致：
    # [duration, stops_int, dep_hour, month, day_of_week, days_left]
    feat_now = [dur_hours, stops_int, dep_hour, dep_month, dep_weekday, days_now]
    feat_7d  = [dur_hours, stops_int, dep_hour, dep_month, dep_weekday, days_7d]

    ratio_now = _price_model.predict([feat_now])[0]
    ratio_7d  = _price_model.predict([feat_7d])[0]

    # 避免除以零
    predicted_fare = int(base_fare * (ratio_7d / ratio_now)) if ratio_now > 0 else base_fare
    price_change   = round((predicted_fare - base_fare) / base_fare * 100, 1)

    # 購買建議
    if price_change >= 5:
        recommendation = "立即購買"
        reason = f"預測 7 天後漲 {price_change}%，建議現在訂票"
    elif price_change <= -5:
        recommendation = "等待觀望"
        reason = f"預測 7 天後降 {abs(price_change)}%，可等待更低票價"
    else:
        recommendation = "可購買"
        reason = f"預測 7 天後票價變動 {price_change:+}%，波動不大"

    return {
        "days_before_departure": days_now,
        "predicted_fare":        predicted_fare,
        "price_change_pct":      price_change,
        "buy_recommendation":    recommendation,
        "recommendation_reason": reason,
    }


def fetch_real_flights(from_airport: str, to_airport: str, date: str):
    """
    透過 fast-flights 向 Google Flights 查詢真實航班。
    只回傳台灣籍航空公司的班次。
    最多重試 5 次，每次確認有台灣航空班次才停止。
    """
    query_filter = create_filter(
        flight_data=[FlightData(date=date, from_airport=from_airport, to_airport=to_airport)],
        trip="one-way",
        seat="economy",
        passengers=Passengers(adults=1),
    )

    result = None
    MAX_ATTEMPTS = 5
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[fetch] 第 {attempt} 次查詢 {from_airport}→{to_airport} {date}")
            result = get_flights_from_filter(query_filter)

            # 確認有抓到資料，且其中包含台灣航空班次
            if result and result.flights:
                has_taiwan = any(f.name in TAIWAN_AIRLINES for f in result.flights)
                if has_taiwan:
                    print(f"[fetch] 第 {attempt} 次成功，找到台灣航班")
                    break
                else:
                    print(f"[fetch] 第 {attempt} 次：有資料但無台灣航班，繼續重試")
            else:
                print(f"[fetch] 第 {attempt} 次：回傳空資料，繼續重試")

        except Exception as e:
            print(f"[fetch] 第 {attempt} 次發生例外：{e}，繼續重試")

    if not result or not result.flights:
        return []

    # 只取台灣籍航空公司，並以 (airline_name, departure) 為唯一鍵去除重複班次
    seen = set()
    unique_flights = []
    for f in result.flights:
        if f.name not in TAIWAN_AIRLINES:
            continue
        key = (f.name, f.departure)   # 同航空公司 + 同出發時間 = 重複
        if key not in seen:
            seen.add(key)
            unique_flights.append(f)
    return unique_flights


def parse_price(price_str: str) -> int:
    """將 'NT$5,699' 或 'NT$5699' 格式的票價字串轉為整數。"""
    cleaned = price_str.replace("NT$", "").replace(",", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return 0


def resolve_baggage_key(fees: dict, kg: int) -> str:
    """
    將使用者輸入的行李公斤數映射到 fees.json 中的費率鍵值。
    kg == 0 代表無托運行李，回傳 "none"。
    """
    if kg == 0:
        return "none"

    tiers = fees.get("baggage_tiers", [])
    for tier in tiers:
        if kg <= tier["max_kg"]:
            return tier["key"]

    # 超出所有級距，套用最後一個（防呆）
    if tiers:
        return tiers[-1]["key"]
    return "none"


class SearchQuery(BaseModel):
    arrival_airport: str
    date:            str                      # 出發日期，格式 YYYY-MM-DD
    from_airport:    str = "TPE"              # 出發機場，預設桃園
    baggage_kg:      int = Field(ge=0, le=32) # 0 = 無托運行李
    need_meal:       bool
    seat_preference: str                      # "standard", "extra_legroom", "none"


@app.get("/api/airports")
def get_airports():
    """
    從 zones_data 動態蒐集所有目的地機場，
    回傳 [{ code, name, zone_label }] 供前端下拉選單使用。
    """
    seen = set()
    airports = []

    for airline_id, airline_zones in zones_data.items():
        if not isinstance(airline_zones, dict):
            continue
        for airport_code, zone in airline_zones.items():
            if airport_code.startswith("_"):
                continue
            if airport_code not in seen:
                seen.add(airport_code)
                airports.append({
                    "code":       airport_code,
                    "name":       AIRPORT_NAMES.get(airport_code, airport_code),
                    "zone_label": ZONE_LABELS.get(zone, zone),
                })

    airports.sort(key=lambda x: x["name"])
    return airports


@app.post("/api/search")
def search_flights(query: SearchQuery):
    # --- Step 1: 向 Google Flights 抓真實班次 ---
    raw_flights = fetch_real_flights(query.from_airport, query.arrival_airport, query.date)

    if not raw_flights:
        return {"flights": [], "ai_recommendation": "目前查無台灣航空公司飛往此目的地的班次，請稍後再試。"}

    results = []

    for flight in raw_flights:
        airline_id = TAIWAN_AIRLINES[flight.name]
        base_fare  = parse_price(flight.price)

        # --- Step 2: 查出航區 ---
        zone = zones_data.get(airline_id, {}).get(query.arrival_airport, "zone_1")

        # --- Step 3: 計算附加費用 ---
        fees = fees_data.get(airline_id, {})

        baggage_key       = resolve_baggage_key(fees, query.baggage_kg)
        baggage_zone_fees = fees.get("baggage", {}).get(zone, {})
        baggage_fee       = baggage_zone_fees.get(baggage_key, 0)

        meal_fee    = fees.get("meal", 0) if query.need_meal else 0
        seat_fee    = fees.get("seat_selection", {}).get(query.seat_preference, 0)
        payment_fee = fees.get("payment_fee", 0)

        true_cost = base_fare + baggage_fee + meal_fee + seat_fee + payment_fee

        # --- Step 4: 檢查台灣虎航 Bundle 優惠是否划算 ---
        # 依座位偏好選擇對應的 Bundle 方案，再判斷是否比單買便宜
        bundle_info = None
        if airline_id == "TNA" and 0 < query.baggage_kg <= 20:
            # 根據使用者選的座位，挑選對應 Bundle 方案
            if query.seat_preference == TNA_BUNDLE_A["seat"]:
                bundle = TNA_BUNDLE_A
            elif query.seat_preference == TNA_BUNDLE_B["seat"]:
                bundle = TNA_BUNDLE_B
            else:
                bundle = None

            # 只有在 Bundle 行李上限內、且 Bundle 比單買便宜時才提示
            if bundle and (baggage_fee + seat_fee) > bundle["price"]:
                # Bundle B 包含餐費，計算時餐費不重複加
                bundled_meal_fee = 0 if bundle["include_meal"] else meal_fee
                bundle_true_cost = base_fare + bundle["price"] + bundled_meal_fee + payment_fee
                bundle_info = {
                    "description":      bundle["description"],
                    "bundle_addon":     bundle["price"],
                    "include_meal":     bundle["include_meal"],  # 告知前端是否含餐
                    "bundle_true_cost": bundle_true_cost,
                    "savings":          true_cost - bundle_true_cost,  # 可省下的金額
                }

        # --- Step 5: ML 票價預測 ---
        price_pred = predict_price(
            base_fare      = base_fare,
            duration_str   = flight.duration,
            departure_str  = flight.departure,
            stops_val      = flight.stops,
            departure_date = query.date,
        )

        results.append({
            "flight_id":    f"{airline_id}-{flight.departure.replace(' ', '').replace(',', '').replace(':', '')}",
            "airline_id":   airline_id,
            "airline_name": flight.name,
            "departure":    flight.departure,
            "arrival":      flight.arrival,
            "duration":     flight.duration,
            "stops":        flight.stops,
            "zone":         zone,
            "zone_label":   ZONE_LABELS.get(zone, zone),
            "base_fare":    base_fare,
            "baggage_tier": baggage_key,
            "add_on_fees": {
                "baggage": baggage_fee,
                "meal":    meal_fee,
                "seat":    seat_fee,
                "payment": payment_fee,
            },
            "true_cost":       true_cost,
            "bundle_info":     bundle_info,    # None 表示無適用優惠
            "price_prediction": price_pred,    # ML 預測結果
        })

    # 根據 true_cost 由低到高排序
    results.sort(key=lambda x: x["true_cost"])

    # --- Step 4: 產生 AI 推薦 ---
    best = results[0]
    prompt = (
        f"<|system|>\nYou are a helpful travel assistant. "
        f"Reply in 50 words.</s>\n"
        f"<|user|>\nWhy should I choose {best['airline_name']} flight "
        f"because it costs {best['true_cost']} NTD including add-ons.</s>\n"
        f"<|assistant|>"
    )

    ai_response = pipe(prompt, max_new_tokens=120, do_sample=True)[0]["generated_text"]
    raw_text    = ai_response.split("<|assistant|>")[-1].strip()

    # 裁切到最後一個完整句子（以 . ! ? 結尾），去除尾端不完整的編號行（如 "3." 或 "3.\n"）
    last_end = max(raw_text.rfind("."), raw_text.rfind("!"), raw_text.rfind("?"))
    if last_end != -1:
        trimmed = raw_text[:last_end + 1].strip()
        # 若最後一行只有編號（如 "2." "3."），代表句子不完整，再往前裁一次
        lines = trimmed.splitlines()
        while lines and lines[-1].strip().rstrip(".").isdigit():
            lines.pop()
        recommendation = "\n".join(lines).strip()
    else:
        recommendation = raw_text

    return {
        "flights":           results,
        "ai_recommendation": recommendation,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
