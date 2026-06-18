import json
import os
import sys

# 強制將 Windows 終端機輸出設為 utf-8，避免印出 emoji 時報錯
sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "data", "airlines.json"), "r", encoding="utf-8") as f:
        airlines = {item["id"]: item for item in json.load(f)}
    with open(os.path.join(base_dir, "data", "fees.json"), "r", encoding="utf-8") as f:
        fees = json.load(f)
    with open(os.path.join(base_dir, "data", "flights.json"), "r", encoding="utf-8") as f:
        flights = json.load(f)
    return airlines, fees, flights

def calculate_true_cost(baggage, seat, need_meal):
    airlines, fees, flights = load_data()
    results = []

    for flight in flights:
        aid = flight["airline_id"]
        base = flight["base_fare"]
        airline_fees = fees.get(aid, {})

        baggage_fee = airline_fees.get("baggage", {}).get(baggage, 0)
        seat_fee = airline_fees.get("seat_selection", {}).get(seat, 0)
        meal_fee = airline_fees.get("meal", 0) if need_meal else 0
        payment_fee = airline_fees.get("payment_fee", 0)

        total_addon = baggage_fee + seat_fee + meal_fee + payment_fee
        true_cost = base + total_addon

        results.append({
            "flight_id": flight["flight_id"],
            "airline_name": airlines[aid]["name"],
            "base_fare": base,
            "addons": {
                "baggage": baggage_fee,
                "seat": seat_fee,
                "meal": meal_fee,
                "payment": payment_fee
            },
            "true_cost": true_cost
        })
    
    # 依真實成本排序
    results.sort(key=lambda x: x["true_cost"])
    return results

if __name__ == "__main__":
    # 模擬使用者剛才透過 UI 選擇的需求
    user_baggage = "15 kg"
    user_seat = "標準座位"
    user_need_meal = True

    results = calculate_true_cost(user_baggage, user_seat, user_need_meal)
    
    print("=== ✈️ 航班真實成本比較分析 ===")
    for r in results:
        print(f"\n{r['airline_name']} ({r['flight_id']})")
        print(f"表面基礎票價: ${r['base_fare']}")
        print(f"附加費用: 行李(${r['addons']['baggage']}) + 選位(${r['addons']['seat']}) + 餐點(${r['addons']['meal']}) + 手續費(${r['addons']['payment']})")
        print(f"✨ 真實總成本: ${r['true_cost']}")
