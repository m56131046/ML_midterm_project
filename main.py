"""
AI-Powered True Cost Flight Comparison — FastAPI 後端
修改記錄：
  - 移除 TinyLlama（記憶體需求 ~2GB），改用 OpenAI GPT-4o-mini + rule-based fallback
  - 修正 predict_price 特徵順序，符合 train_price_model.py 的新格式：
      [days_left, duration, stops_int, dep_period]
  - 修正 price_model.pkl 載入邏輯（移除不存在的 pricing_curve key）
  - fast_flights 優雅降級：雲端環境缺 primp wheel 時改用模擬航班
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json, os, pickle, numpy as np
from datetime import date as date_type, datetime

from fast_flights import FlightQuery, Passengers, create_query, get_flights
from typing import List, Optional
from agent import agent_chat   # AI 對話代理（agent.py）

# ── OpenAI 客戶端（選用）────────────────────────────────────────────────────────
# 設定環境變數 OPENAI_API_KEY 即可啟用；未設定時自動改用 rule-based 文字
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
    _oai_client = _OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
except Exception:
    _oai_client = None

# ── FastAPI 初始化 ─────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允許所有來源（前端可從任何網域呼叫）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 載入靜態資料 ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(filename: str) -> dict:
    with open(os.path.join(BASE_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)

airlines_data = load_json("航空公司基本資料 airlines.json")
fees_data     = load_json("附加費用費率表 fees.json")
zones_data    = load_json("機場航區對應表 airport_zones.json")

# ── 常數 ───────────────────────────────────────────────────────────────────────
AIRPORT_NAMES = {
    "NRT": "東京成田 (NRT)",   "HND": "東京羽田 (HND)",
    "KIX": "大阪關西 (KIX)",   "FUK": "福岡 (FUK)",
    "CTS": "札幌新千歲 (CTS)", "OKA": "沖繩那霸 (OKA)",
    "NGO": "名古屋 (NGO)",     "ICN": "首爾仁川 (ICN)",
    "GMP": "首爾金浦 (GMP)",   "PUS": "釜山 (PUS)",
    "BKK": "曼谷素萬那普 (BKK)","DMK": "曼谷廊曼 (DMK)",
    "SIN": "新加坡 (SIN)",     "KUL": "吉隆坡 (KUL)",
    "MNL": "馬尼拉 (MNL)",     "CEB": "宿霧 (CEB)",
    "SGN": "胡志明市 (SGN)",   "HAN": "河內 (HAN)",
    "DPS": "峇里島 (DPS)",     "SYD": "雪梨 (SYD)",
    "MEL": "墨爾本 (MEL)",     "LAX": "洛杉磯 (LAX)",
    "SFO": "舊金山 (SFO)",     "JFK": "紐約 (JFK)",
    "LHR": "倫敦希斯洛 (LHR)", "AMS": "阿姆斯特丹 (AMS)",
    "VIE": "維也納 (VIE)",     "KHH": "高雄小港 (KHH)",
}

ZONE_LABELS = {
    "zone_1": "Zone 1・短程",
    "zone_2": "Zone 2・中程",
    "zone_3": "Zone 3・長程",
    "zone_4": "Zone 4・長程",
}

TAIWAN_AIRLINES = {
    "Tigerair Taiwan":  "TNA",
    "EVA Air":          "EVA",
    "China Airlines":   "CAL",
    "STARLUX Airlines": "SJX",
}

TNA_BUNDLE_A = {
    "price": 950,  "max_kg": 20, "seat": "standard",
    "include_meal": False,
    "description": "虎航 Bundle A：20kg 行李 + 標準選位 = NT$950",
}
TNA_BUNDLE_B = {
    "price": 1600, "max_kg": 20, "seat": "extra_legroom",
    "include_meal": True,
    "description": "虎航 Bundle B：20kg 行李 + 大空間座位 + 免餐費 = NT$1600",
}

# ── 定價曲線（ML 模型不可用時的 fallback）──────────────────────────────────────
PRICING_CURVE = {
    90: 0.82, 60: 0.90, 30: 1.00, 21: 1.08,
    14: 1.20,  7: 1.45,  3: 1.75,  1: 2.10,
}

def _curve_ratio(days: int) -> float:
    """依定價曲線插值，回傳 days 對應的票價倍率。"""
    keys = sorted(PRICING_CURVE.keys())
    if days <= keys[0]:  return PRICING_CURVE[keys[0]]
    if days >= keys[-1]: return PRICING_CURVE[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= days <= hi:
            t = (days - lo) / (hi - lo)
            return PRICING_CURVE[lo] * (1 - t) + PRICING_CURVE[hi] * t
    return 1.0

# ── 載入 ML 票價預測模型 ───────────────────────────────────────────────────────
# 模型由 train_price_model.py 產生，特徵順序：
#   [days_left, duration, stops_int, dep_period]
_price_model = None
_MODEL_PATH  = os.path.join(BASE_DIR, "price_model.pkl")
if os.path.exists(_MODEL_PATH):
    try:
        with open(_MODEL_PATH, "rb") as _f:
            _payload     = pickle.load(_f)
        _price_model = _payload["model"]
        print(f"[ML] price_model.pkl 載入成功，特徵：{_payload.get('features')}")
    except Exception as e:
        print(f"[ML] price_model.pkl 載入失敗：{e}，改用定價曲線")
else:
    print("[ML] price_model.pkl 不存在，改用定價曲線（請先執行 train_price_model.py）")

# ── 輔助函式 ───────────────────────────────────────────────────────────────────
def _simpledatetime_hour(sdt) -> int:
    """v3.0 SimpleDatetime → 出發小時（int）"""
    t = sdt.time if sdt.time else []
    return t[0] if t else 12


def _simpledatetime_to_str(sdt) -> str:
    """v3.0 SimpleDatetime → 'HH:MM' 顯示字串"""
    t = sdt.time if sdt.time else []
    h = t[0] if len(t) > 0 else 0
    m = t[1] if len(t) > 1 else 0
    return f"{h:02d}:{m:02d}"


def _minutes_to_hours(minutes: int) -> float:
    """飛行分鐘數 → 小時（浮點），供 ML 特徵使用"""
    return minutes / 60


def _minutes_to_str(minutes: int) -> str:
    """飛行分鐘數 → 'Xh Ym' 顯示字串"""
    return f"{minutes // 60}h {minutes % 60}m"


def _hour_to_period(h: int) -> int:
    """
    出發小時 → dep_period 編碼（與 train_price_model.py 一致）
    0=Early_Morning(0-5), 1=Morning(6-11), 2=Afternoon(12-15),
    3=Evening(16-19), 4=Night(20-23)
    """
    if h < 6:  return 0
    if h < 12: return 1
    if h < 16: return 2
    if h < 20: return 3
    return 4


def parse_price(price) -> int:
    """v3.0 price 已是 int；保留字串解析作為 fallback"""
    if isinstance(price, int):
        return price
    cleaned = str(price).replace("NT$", "").replace(",", "").strip()
    try:    return int(cleaned)
    except: return 0


def resolve_baggage_key(fees: dict, kg: int) -> str:
    """行李公斤數 → fees.json 費率鍵值"""
    if kg == 0:
        return "none"
    for tier in fees.get("baggage_tiers", []):
        if kg <= tier["max_kg"]:
            return tier["key"]
    tiers = fees.get("baggage_tiers", [])
    return tiers[-1]["key"] if tiers else "none"


# ── ML 票價預測 ────────────────────────────────────────────────────────────────
def predict_price(
    base_fare: int,
    total_minutes: int,   # v3.0：各腳段 duration（分鐘）加總
    dep_hour: int,        # v3.0：第一腳段出發小時
    stops_int: int,       # v3.0：len(flight.flights) - 1
    departure_date: str,  # YYYY-MM-DD
) -> dict:
    """
    預測 7 天後的票價。
    - 若 ML 模型可用：特徵 [duration, stops_int, dep_hour, month, day_of_week, days_left]
    - 否則：使用 PRICING_CURVE 插值
    """
    today       = date_type.today()
    dep_date    = datetime.strptime(departure_date, "%Y-%m-%d").date()
    days_now    = max(1, (dep_date - today).days)
    days_7d     = max(1, days_now - 7)
    dur_hours   = _minutes_to_hours(total_minutes)
    if _price_model:
        try:
            # 特徵順序需與 price_model.pkl 訓練時一致：
            # ['duration', 'stops_int', 'dep_hour', 'month', 'day_of_week', 'days_left']
            # 訓練時 month/day_of_week 固定為 0，預測時同樣傳 0
            feat_now = [[float(dur_hours), int(stops_int), int(dep_hour), 0, 0, int(days_now)]]
            feat_7d  = [[float(dur_hours), int(stops_int), int(dep_hour), 0, 0, int(days_7d)]]
            ratio_now = _price_model.predict(feat_now)[0]
            ratio_7d  = _price_model.predict(feat_7d)[0]
            predicted_fare = int(base_fare * (ratio_7d / ratio_now)) if ratio_now > 0 else base_fare
        except Exception as e:
            print(f"[ML] 預測失敗：{e}，改用定價曲線")
            ratio_now = _curve_ratio(days_now)
            ratio_7d  = _curve_ratio(days_7d)
            predicted_fare = int(base_fare * (ratio_7d / ratio_now))
    else:
        # Fallback：定價曲線
        ratio_now = _curve_ratio(days_now)
        ratio_7d  = _curve_ratio(days_7d)
        predicted_fare = int(base_fare * (ratio_7d / ratio_now))

    price_change = round((predicted_fare - base_fare) / base_fare * 100, 1)

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


# ── AI 推薦文字 ────────────────────────────────────────────────────────────────
def _generate_recommendation(best: dict) -> str:
    """
    優先用 OpenAI GPT-4o-mini 生成繁體中文推薦；
    API Key 未設定或呼叫失敗時，改用 rule-based 文字。
    """
    pred = best["price_prediction"]
    pct  = pred["price_change_pct"]

    if _oai_client:
        prompt = (
            f"你是專業旅遊顧問，請用繁體中文在 50 字以內給出購票建議：\n"
            f"推薦航空：{best['airline_name']}\n"
            f"真實總費用：NT${best['true_cost']:,}（基礎票價 NT${best['base_fare']:,}"
            f" + 附加費 NT${best['true_cost'] - best['base_fare']:,}）\n"
            f"7 天後預測費用：NT${pred['predicted_fare']:,}（{pct:+.1f}%）\n"
            f"請說明是否建議立即購買。"
        )
        try:
            resp = _oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI] 呼叫失敗：{e}，改用 rule-based")

    # Rule-based fallback
    if pct >= 5:
        return (f"✅ 建議立即購買 {best['airline_name']}（NT${best['true_cost']:,}）。"
                f"ML 模型預測 7 天後票價將上漲 {pct:.1f}%，"
                f"現在購買可省約 NT${pred['predicted_fare'] - best['true_cost']:,}。")
    elif pct <= -5:
        return (f"⏳ 可考慮等待。ML 預測 {best['airline_name']} 7 天後票價將下降 {abs(pct):.1f}%，"
                f"等待購買預計可省約 NT${best['true_cost'] - pred['predicted_fare']:,}。")
    else:
        return (f"💡 推薦 {best['airline_name']}（NT${best['true_cost']:,}），整體費用最划算。"
                f"票價預測波動不大（{pct:+.1f}%），可依個人需求決定購票時機。")


# ── 查詢真實航班 ───────────────────────────────────────────────────────────────
def fetch_real_flights(
    from_airport: str,
    to_airport: str,
    date: str,
    adults: int = 1,
    children: int = 0,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
):
    """
    透過 fast_flights 查詢 Google Flights 真實航班。
    只保留台灣籍航空班次，去除重複，最多重試 5 次。
    """
    query = create_query(
        flights=[FlightQuery(date=date, from_airport=from_airport, to_airport=to_airport)],
        trip="one-way",
        seat="economy",
        passengers=Passengers(
            adults=adults,
            children=children,
            infants_in_seat=infants_in_seat,
            infants_on_lap=infants_on_lap,
        ),
        currency="TWD",    # 強制使用台幣，避免 Render 美國伺服器回傳 USD 價格
        language="zh-TW",  # 語系設為繁體中文
    )

    result = None
    for attempt in range(1, 6):
        try:
            print(f"[fetch] 第 {attempt} 次查詢 {from_airport}→{to_airport} {date}")
            result = get_flights(query)
            # v3.0 ResultList 本身就是 list；airlines 為 list，取第一個元素比對
            if result and any(f.airlines[0] in TAIWAN_AIRLINES for f in result if f.airlines):
                break
        except Exception as e:
            print(f"[fetch] 第 {attempt} 次例外：{e}")

    if not result:
        return []

    seen, unique = set(), []
    for f in result:
        if not f.airlines or f.airlines[0] not in TAIWAN_AIRLINES:
            continue
        # 以（航空公司, 第一腳段出發時間）為去重鍵
        dep_str = _simpledatetime_to_str(f.flights[0].departure) if f.flights else "00:00"
        key = (f.airlines[0], dep_str)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


# ── API Endpoints ──────────────────────────────────────────────────────────────
class SearchQuery(BaseModel):
    to_airport:      str
    date:            str
    from_airport:    str = "TPE"
    baggage_kg:      int = Field(ge=0, le=32)
    need_meal:       bool
    seat_preference: str  # "standard" | "extra_legroom" | "none"
    airline_filter:  Optional[str] = None   # "TNA"/"EVA"/"CAL"/"SJX" 或 null（不限）
    # 乘客人數
    adults:          int = Field(default=1, ge=1, le=9)   # 成人（12歲以上）
    children:        int = Field(default=0, ge=0, le=9)   # 兒童（2-11歲）
    infants_in_seat: int = Field(default=0, ge=0, le=9)   # 占位嬰兒
    infants_on_lap:  int = Field(default=0, ge=0, le=9)   # 不占位嬰兒

    @property
    def total_passengers(self) -> int:
        """總搭乘人數：成人 + 兒童 + 占位嬰兒（不占位嬰兒不佔座位不計入）"""
        return self.adults + self.children + self.infants_in_seat


@app.get("/api/airports")
def get_airports():
    """回傳所有目的地機場清單，供前端下拉選單使用。"""
    seen, airports = set(), []
    for airline_id, airline_zones in zones_data.items():
        if not isinstance(airline_zones, dict):
            continue
        for code, zone in airline_zones.items():
            if code.startswith("_") or code in seen:
                continue
            seen.add(code)
            airports.append({
                "code":       code,
                "name":       AIRPORT_NAMES.get(code, code),
                "zone_label": ZONE_LABELS.get(zone, zone),
            })
    airports.sort(key=lambda x: x["name"])
    return airports


@app.post("/api/search")
def search_flights(query: SearchQuery):
    """
    主要搜尋 API。
    1. 查詢 Google Flights（或 Demo 資料）
    2. 計算各航班真實總費用（基礎票價 + 附加費用）
    3. ML 預測 7 天後票價
    4. 檢查虎航 Bundle 優惠
    5. 用 OpenAI 或 rule-based 產生 AI 推薦文字
    """
    raw_flights = fetch_real_flights(
        query.from_airport,
        query.to_airport,
        query.date,
        adults          = query.adults,
        children        = query.children,
        infants_in_seat = query.infants_in_seat,
        infants_on_lap  = query.infants_on_lap,
    )
    if not raw_flights:
        return {"flights": [], "ai_recommendation": "目前查無台灣航空公司飛往此目的地的班次，請稍後再試。"}

    results = []
    for flight in raw_flights:
        airline_name = flight.airlines[0]                          # v3.0：airlines 為 list
        airline_id   = TAIWAN_AIRLINES[airline_name]
        base_fare    = parse_price(flight.price)                   # v3.0：price 已是 int
        zone         = zones_data.get(airline_id, {}).get(query.to_airport, "zone_1")
        fees         = fees_data.get(airline_id, {})

        # v3.0：從腳段列表取出出發／抵達時間與飛行時間
        first_leg     = flight.flights[0] if flight.flights else None
        last_leg      = flight.flights[-1] if flight.flights else None
        dep_str       = _simpledatetime_to_str(first_leg.departure) if first_leg else "00:00"
        arr_str       = _simpledatetime_to_str(last_leg.arrival)    if last_leg  else "00:00"
        total_minutes = sum(leg.duration for leg in flight.flights)  # 各腳段分鐘數加總
        stops_int     = len(flight.flights) - 1                      # 腳段數 - 1 = 轉機次數

        baggage_key = resolve_baggage_key(fees, query.baggage_kg)
        baggage_fee = fees.get("baggage", {}).get(zone, {}).get(baggage_key, 0)
        meal_fee    = fees.get("meal", 0) * query.total_passengers if query.need_meal else 0
        seat_fee    = fees.get("seat_selection", {}).get(query.seat_preference, 0)
        payment_fee = fees.get("payment_fee", 0)
        true_cost   = base_fare + baggage_fee + meal_fee + seat_fee + payment_fee

        # 虎航 Bundle 優惠
        bundle_info = None
        if airline_id == "TNA" and 0 < query.baggage_kg <= 20:
            bundle = (TNA_BUNDLE_A if query.seat_preference == TNA_BUNDLE_A["seat"]
                      else TNA_BUNDLE_B if query.seat_preference == TNA_BUNDLE_B["seat"]
                      else None)
            if bundle and (baggage_fee + seat_fee) > bundle["price"]:
                bm_fee           = 0 if bundle["include_meal"] else meal_fee
                bundle_true_cost = base_fare + bundle["price"] + bm_fee + payment_fee
                bundle_info = {
                    "description":      bundle["description"],
                    "bundle_addon":     bundle["price"],
                    "include_meal":     bundle["include_meal"],
                    "bundle_true_cost": bundle_true_cost,
                    "savings":          true_cost - bundle_true_cost,
                }

        # ML 票價預測（傳入 v3.0 已解析的數值）
        price_pred = predict_price(
            base_fare      = base_fare,
            total_minutes  = total_minutes,
            dep_hour       = _simpledatetime_hour(first_leg.departure) if first_leg else 12,
            stops_int      = stops_int,
            departure_date = query.date,
        )

        results.append({
            "flight_id":        f"{airline_id}-{dep_str.replace(':','')}",
            "airline_id":       airline_id,
            "airline_name":     airline_name,
            "departure":        dep_str,
            "arrival":          arr_str,
            "duration":         _minutes_to_str(total_minutes),
            "stops":            stops_int,
            "zone":             zone,
            "zone_label":       ZONE_LABELS.get(zone, zone),
            "base_fare":        base_fare,
            "baggage_tier":     baggage_key,
            "add_on_fees":      {"baggage": baggage_fee, "meal": meal_fee,
                                 "seat": seat_fee, "payment": payment_fee},
            "true_cost":        true_cost,
            "bundle_info":      bundle_info,
            "price_prediction": price_pred,
        })

    # 航空公司篩選（由 AI Agent 解析使用者偏好後傳入）
    if query.airline_filter:
        results = [r for r in results if r["airline_id"] == query.airline_filter]
        if not results:
            return {
                "flights": [],
                "ai_recommendation": f"查無 {query.airline_filter} 飛往此目的地的班次，請換其他航空或移除篩選條件。"
            }

    results.sort(key=lambda x: x["true_cost"])

    # AI 推薦
    recommendation = _generate_recommendation(results[0])

    return {
        "flights":           results,
        "ai_recommendation": recommendation,
    }


# ── AI 對話代理 Endpoint ───────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]   # 完整對話歷史

@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    AI 對話代理：將使用者的自然語言轉為結構化搜尋條件。
    前端傳入完整對話歷史，Agent 回傳：
      - {"ready": false, "message": "反問文字"}   ← 繼續對話
      - {"ready": true, "arrival_airport": "KIX", "date": "...", ...}  ← 觸發搜尋
    """
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    return agent_chat(messages)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
