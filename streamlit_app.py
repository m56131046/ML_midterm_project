"""
AI-Powered True Cost Flight Comparison System
Streamlit 版本（取代 React + FastAPI，可部署至 Streamlit Cloud）
"""

import os
import json
import pickle
import datetime
import numpy as np
import streamlit as st
from openai import OpenAI

# fast_flights 使用 Rust 套件 primp；某些雲端環境無對應 wheel 時會 ImportError
# → 優雅降級：顯示 Demo 模擬資料，不影響 ML 預測與費用計算功能
try:
    from fast_flights import FlightData, Passengers, create_filter, get_flights_from_filter
    FAST_FLIGHTS_OK = True
except Exception:
    FAST_FLIGHTS_OK = False

# ── 頁面設定 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI True Cost Flight Comparison",
    page_icon="✈️",
    layout="wide",
)

# ── 載入靜態資料 ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_json(filename):
    with open(os.path.join(BASE_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)

airlines_data = load_json("航空公司基本資料 airlines.json")
fees_data     = load_json("附加費用費率表 fees.json")
zones_data    = load_json("機場航區對應表 airport_zones.json")

# ── 載入 ML 模型 ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "price_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

ml_payload = load_model()

# ── OpenAI 客戶端 ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_openai_client():
    # 優先從 Streamlit Secrets 取得 API Key（部署用）
    # 本機開發時可設定環境變數 OPENAI_API_KEY
    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if api_key:
        return OpenAI(api_key=api_key)
    return None

openai_client = get_openai_client()

# ── 常數 ──────────────────────────────────────────────────────────────────────
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
    "VIE": "維也納 (VIE)",
}
ZONE_LABELS = {
    "zone_1": "Zone 1・短程",
    "zone_2": "Zone 2・中程",
    "zone_3": "Zone 3・長程",
}
TAIWAN_AIRLINES = {
    "Tigerair Taiwan":  "TNA",
    "EVA Air":          "EVA",
    "China Airlines":   "CAL",
    "STARLUX Airlines": "SJX",
}
TNA_BUNDLE_A = {"price": 950,  "seat": "standard",     "include_meal": False,
                "description": "虎航 Bundle A：20kg 行李 + 標準選位 = NT$950"}
TNA_BUNDLE_B = {"price": 1600, "seat": "extra_legroom", "include_meal": True,
                "description": "虎航 Bundle B：20kg 行李 + 大空間座位 + 免餐費 = NT$1600"}

FROM_AIRPORTS = [
    ("TPE", "台北桃園 (TPE)"),
    ("TSA", "台北松山 (TSA)"),
    ("KHH", "高雄小港 (KHH)"),
]

# ── 定價曲線（用於 ML 模型不存在時的 fallback）────────────────────────────────
PRICING_CURVE = {
    90: 0.82, 60: 0.90, 30: 1.00, 21: 1.08,
    14: 1.20,  7: 1.45,  3: 1.75,  1: 2.10,
}

def curve_ratio(days: int) -> float:
    keys = sorted(PRICING_CURVE.keys())
    if days <= keys[0]:  return PRICING_CURVE[keys[0]]
    if days >= keys[-1]: return PRICING_CURVE[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i+1]
        if lo <= days <= hi:
            t = (days - lo) / (hi - lo)
            return PRICING_CURVE[lo]*(1-t) + PRICING_CURVE[hi]*t
    return 1.0

# ── 輔助函式 ──────────────────────────────────────────────────────────────────
def get_airports():
    """從 zones_data 動態取得所有目的地機場清單。"""
    seen, airports = set(), []
    for airline_zones in zones_data.values():
        if not isinstance(airline_zones, dict):
            continue
        for code, zone in airline_zones.items():
            if not code.startswith("_") and code not in seen:
                seen.add(code)
                airports.append({
                    "code": code,
                    "name": AIRPORT_NAMES.get(code, code),
                    "zone_label": ZONE_LABELS.get(zone, zone),
                })
    return sorted(airports, key=lambda x: x["name"])

def parse_price(price_str: str) -> int:
    cleaned = str(price_str).replace("NT$", "").replace(",", "").strip()
    try:    return int(cleaned)
    except: return 0

def resolve_baggage_key(fees: dict, kg: int) -> str:
    if kg == 0:
        return "none"
    for tier in fees.get("baggage_tiers", []):
        if kg <= tier["max_kg"]:
            return tier["key"]
    tiers = fees.get("baggage_tiers", [])
    return tiers[-1]["key"] if tiers else "none"

def predict_price(current_fare: int, days_now: int, days_future: int) -> int:
    """
    用 ML 模型（或定價曲線）預測 days_future 天前的票價。
    回傳預測票價（NT$）。
    """
    if ml_payload:
        model = ml_payload["model"]
        # 用通用特徵（飛行時間 3h、non-stop、傍晚出發）代入
        feat_now    = [[days_now,    3.0, 0, 3]]
        feat_future = [[days_future, 3.0, 0, 3]]
        ratio_now    = model.predict(feat_now)[0]
        ratio_future = model.predict(feat_future)[0]
        if ratio_now > 0:
            predicted = current_fare * (ratio_future / ratio_now)
        else:
            predicted = current_fare
    else:
        # Fallback：用定價曲線
        ratio_now    = curve_ratio(days_now)
        ratio_future = curve_ratio(days_future)
        predicted = current_fare * (ratio_future / ratio_now)

    return max(0, round(predicted / 100) * 100)

def _demo_flights(to_airport: str):
    """
    fast_flights 不可用時的模擬航班資料。
    用 dataclass 模擬 fast_flights Flight 物件介面。
    """
    from dataclasses import dataclass

    @dataclass
    class MockFlight:
        name: str
        price: str
        departure: str
        arrival: str
        duration: str
        stops: int

    # 依目的地返回代表性航班
    japan  = ["NRT", "HND", "KIX", "FUK", "CTS", "OKA", "NGO"]
    korea  = ["ICN", "GMP", "PUS"]
    sea    = ["BKK", "DMK", "SIN", "KUL", "MNL", "CEB", "SGN", "HAN", "DPS"]
    long   = ["SYD", "MEL", "LAX", "SFO", "JFK", "LHR", "AMS", "VIE"]

    if to_airport in japan:
        return [
            MockFlight("Tigerair Taiwan", "NT$3,500", "08:00", "12:10", "3h 10m", 0),
            MockFlight("EVA Air",          "NT$5,200", "10:00", "14:20", "4h 00m", 0),
            MockFlight("China Airlines",   "NT$4,800", "09:30", "13:40", "3h 50m", 0),
        ]
    elif to_airport in korea:
        return [
            MockFlight("Tigerair Taiwan", "NT$3,200", "09:00", "12:30", "2h 45m", 0),
            MockFlight("EVA Air",          "NT$4,900", "11:00", "14:30", "2h 45m", 0),
            MockFlight("China Airlines",   "NT$4,500", "13:00", "16:30", "2h 45m", 0),
        ]
    elif to_airport in sea:
        return [
            MockFlight("Tigerair Taiwan", "NT$4,500", "00:30", "03:50", "4h 20m", 0),
            MockFlight("EVA Air",          "NT$6,800", "08:00", "11:20", "4h 20m", 0),
            MockFlight("China Airlines",   "NT$6,200", "09:00", "12:20", "4h 20m", 0),
        ]
    else:  # long-haul
        return [
            MockFlight("EVA Air",        "NT$18,500", "10:00", "16:00", "12h 00m", 1),
            MockFlight("China Airlines", "NT$17,200", "09:00", "15:00", "13h 00m", 1),
        ]


def fetch_flights(from_airport: str, to_airport: str, date: str):
    """
    向 Google Flights 查詢台灣籍航空班次。
    若 fast_flights 不可用（雲端環境缺少 wheel），回傳 Demo 模擬資料。
    """
    if not FAST_FLIGHTS_OK:
        # Demo 模式：直接回傳模擬航班（ML 預測與費用計算仍正常運作）
        return _demo_flights(to_airport)

    query_filter = create_filter(
        flight_data=[FlightData(date=date, from_airport=from_airport, to_airport=to_airport)],
        trip="one-way",
        seat="economy",
        passengers=Passengers(adults=1),
    )
    result = None
    for _ in range(3):
        try:
            result = get_flights_from_filter(query_filter)
            if result and result.flights and any(f.name for f in result.flights):
                break
        except Exception:
            pass
    if not result or not result.flights:
        return []
    seen, unique = set(), []
    for f in result.flights:
        if f.name not in TAIWAN_AIRLINES:
            continue
        key = (f.name, f.departure)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique

def generate_recommendation(best: dict, predicted: dict) -> str:
    """用 OpenAI GPT 生成購買建議文字。"""
    if not openai_client:
        # Fallback：rule-based 文字
        return _rule_based_recommendation(best, predicted)

    prompt = f"""你是一位專業旅遊顧問，請根據以下資訊給出簡潔的購票建議（繁體中文，50字以內）：

航空公司：{best['airline_name']}
目前真實總費用：NT${best['true_cost']:,}（基礎票價 NT${best['base_fare']:,} + 附加費用 NT${best['true_cost'] - best['base_fare']:,}）
{days_label(predicted['days_now'])}後購買預測費用：NT${predicted['true_cost_future']:,}
漲跌幅：{predicted['change_pct']:+.1f}%

請說明是否建議現在購買，以及預期能省多少錢。"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _rule_based_recommendation(best, predicted)

def _rule_based_recommendation(best: dict, predicted: dict) -> str:
    saving = predicted["true_cost_future"] - best["true_cost"]
    days   = predicted["days_future"]
    if saving > 0:
        return (f"✅ 建議現在購買。根據歷史定價模型，{days} 天後票價預計上漲 "
                f"NT${saving:,}（{predicted['change_pct']:+.1f}%），"
                f"現在購買可省下約 NT${saving:,}。")
    else:
        return (f"⏳ 可考慮等待。根據歷史定價模型，{days} 天後票價預計下降 "
                f"NT${abs(saving):,}（{predicted['change_pct']:+.1f}%），"
                f"等待購買預計可省 NT${abs(saving):,}。")

def days_label(days: int) -> str:
    return f"{days} 天"

# ── UI：頁面標題 ──────────────────────────────────────────────────────────────
st.title("✈️ AI-Powered True Cost Flight Comparison")
st.caption("透過 AI 計算「真實總費用」並預測未來票價，幫助你決定現在買還是等待")

# Demo 模式提示（fast_flights 在雲端環境無法取得即時 Google Flights 資料時顯示）
if not FAST_FLIGHTS_OK:
    st.info(
        "ℹ️ **Demo 模式**：目前使用模擬航班資料（雲端環境限制，無法直接查詢 Google Flights）。"
        "票價預測（ML）與附加費用計算功能完全正常。",
        icon="ℹ️",
    )

st.divider()

# ── UI：搜尋表單 ──────────────────────────────────────────────────────────────
airports = get_airports()
airport_options = {a["code"]: f"{a['name']}  ({a['zone_label']})" for a in airports}

col1, col2, col3 = st.columns(3)

with col1:
    from_airport = st.selectbox(
        "出發機場",
        options=[c for c, _ in FROM_AIRPORTS],
        format_func=lambda c: dict(FROM_AIRPORTS)[c],
    )
    arrival_airport = st.selectbox(
        "目的地機場",
        options=list(airport_options.keys()),
        format_func=lambda c: airport_options[c],
        index=list(airport_options.keys()).index("NRT") if "NRT" in airport_options else 0,
    )

with col2:
    default_date = datetime.date.today() + datetime.timedelta(days=7)
    travel_date  = st.date_input(
        "出發日期",
        value=default_date,
        min_value=datetime.date.today(),
    )
    baggage_kg = st.number_input(
        "托運行李（kg）",
        min_value=0, max_value=32, value=0, step=1,
        help="0 = 無托運行李。廉航 23kg 以下套用 20kg 費率",
    )

with col3:
    need_meal = st.checkbox("需要機上餐點")
    seat_pref = st.selectbox(
        "預選座位",
        options=["none", "standard", "extra_legroom"],
        format_func=lambda x: {"none": "不預選", "standard": "標準座位",
                                "extra_legroom": "大空間座位"}[x],
    )
    predict_days = st.selectbox(
        "預測幾天後的票價",
        options=[7, 14, 3],
        format_func=lambda x: f"{x} 天後",
    )

search_clicked = st.button("🔍 計算真實成本並預測票價", type="primary", use_container_width=True)

# ── UI：搜尋結果 ──────────────────────────────────────────────────────────────
if search_clicked:
    date_str = travel_date.strftime("%Y-%m-%d")
    days_now = (travel_date - datetime.date.today()).days
    days_future = max(1, days_now - predict_days)

    with st.spinner("查詢航班並計算費用中..."):
        raw_flights = fetch_flights(from_airport, arrival_airport, date_str)

    if not raw_flights:
        st.warning("目前查無台灣航空公司飛往此目的地的班次，請換日期或目的地再試。")
    else:
        # 計算每個航班的真實費用
        results = []
        for flight in raw_flights:
            airline_id = TAIWAN_AIRLINES[flight.name]
            base_fare  = parse_price(flight.price)
            zone       = zones_data.get(airline_id, {}).get(arrival_airport, "zone_1")
            fees       = fees_data.get(airline_id, {})

            baggage_key  = resolve_baggage_key(fees, baggage_kg)
            baggage_fee  = fees.get("baggage", {}).get(zone, {}).get(baggage_key, 0)
            meal_fee     = fees.get("meal", 0) if need_meal else 0
            seat_fee     = fees.get("seat_selection", {}).get(seat_pref, 0)
            payment_fee  = fees.get("payment_fee", 0)
            addons       = baggage_fee + meal_fee + seat_fee + payment_fee
            true_cost    = base_fare + addons

            # ML 票價預測
            pred_base        = predict_price(base_fare, days_now, days_future)
            pred_true_cost   = pred_base + addons
            change_pct       = (pred_true_cost - true_cost) / true_cost * 100 if true_cost > 0 else 0
            buy_now          = pred_true_cost >= true_cost

            # Bundle 優惠（虎航）
            bundle_info = None
            if airline_id == "TNA" and 0 < baggage_kg <= 20:
                bundle = TNA_BUNDLE_A if seat_pref == "standard" else (
                         TNA_BUNDLE_B if seat_pref == "extra_legroom" else None)
                if bundle and (baggage_fee + seat_fee) > bundle["price"]:
                    bm_fee = 0 if bundle["include_meal"] else meal_fee
                    b_cost = base_fare + bundle["price"] + bm_fee + payment_fee
                    bundle_info = {**bundle, "bundle_true_cost": b_cost,
                                   "savings": true_cost - b_cost}

            results.append({
                "airline_id": airline_id, "airline_name": flight.name,
                "departure": flight.departure, "arrival": flight.arrival,
                "duration": flight.duration, "stops": flight.stops,
                "zone_label": ZONE_LABELS.get(zone, zone),
                "base_fare": base_fare, "baggage_tier": baggage_key,
                "add_on_fees": {"baggage": baggage_fee, "meal": meal_fee,
                                "seat": seat_fee, "payment": payment_fee},
                "true_cost": true_cost,
                "predicted": {
                    "days_now": days_now, "days_future": days_future,
                    "base_fare_future": pred_base,
                    "true_cost_future": pred_true_cost,
                    "change_pct": change_pct,
                    "buy_now": buy_now,
                },
                "bundle_info": bundle_info,
            })

        results.sort(key=lambda x: x["true_cost"])
        best = results[0]

        # AI 推薦
        with st.spinner("AI 分析中..."):
            ai_text = generate_recommendation(best, best["predicted"])

        # AI 推薦區塊
        st.subheader("🤖 AI 推薦分析")
        st.info(ai_text)

        st.subheader("✈️ 航班選項（依真實成本排序）")

        for i, flight in enumerate(results):
            pred = flight["predicted"]
            is_best = (i == 0)

            with st.container(border=True):
                # 標題列
                header_col, badge_col = st.columns([4, 1])
                with header_col:
                    medal = "🥇 " if is_best else ""
                    st.markdown(f"### {medal}{flight['airline_name']}（{flight['airline_id']}）")
                with badge_col:
                    st.caption(flight["zone_label"])

                # 基本資訊
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.write(f"🛫 出發：{flight['departure']}")
                    st.write(f"🛬 抵達：{flight['arrival']}")
                with info_col2:
                    st.write(f"⏱ 飛行時間：{flight['duration']}")
                    st.write(f"🔁 經停：{flight['stops']} 次")

                # 費用明細
                st.markdown("**費用明細**")
                fee_cols = st.columns(5)
                fee_cols[0].metric("基礎票價", f"NT${flight['base_fare']:,}")
                fee_cols[1].metric("行李費",   f"NT${flight['add_on_fees']['baggage']:,}")
                fee_cols[2].metric("選位費",   f"NT${flight['add_on_fees']['seat']:,}")
                fee_cols[3].metric("餐點費",   f"NT${flight['add_on_fees']['meal']:,}")
                fee_cols[4].metric("手續費",   f"NT${flight['add_on_fees']['payment']:,}")

                # 真實成本
                st.markdown(f"## 💰 真實總成本：NT${flight['true_cost']:,}")

                # ML 票價預測區塊
                st.divider()
                st.markdown("**📊 AI 票價預測（Random Forest）**")
                pred_cols = st.columns(3)
                change_str = f"{pred['change_pct']:+.1f}%"
                delta_color = "inverse" if pred["buy_now"] else "normal"

                pred_cols[0].metric("現在真實總費用",
                                    f"NT${flight['true_cost']:,}")
                pred_cols[1].metric(f"{predict_days} 天後預測費用",
                                    f"NT${pred['true_cost_future']:,}",
                                    delta=change_str,
                                    delta_color=delta_color)
                with pred_cols[2]:
                    if pred["buy_now"]:
                        st.success(f"✅ 建議現在購買\n預計漲 NT${pred['true_cost_future'] - flight['true_cost']:,}")
                    else:
                        st.warning(f"⏳ 可考慮等待\n預計省 NT${flight['true_cost'] - pred['true_cost_future']:,}")

                # Bundle 優惠提示（虎航）
                if flight["bundle_info"]:
                    b = flight["bundle_info"]
                    st.warning(
                        f"💡 **優惠提示**：{b['description']}\n\n"
                        f"Bundle 後真實總成本：**NT${b['bundle_true_cost']:,}**　"
                        f"比單買便宜 NT${b['savings']:,}！"
                    )
