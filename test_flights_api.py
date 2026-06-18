"""
測試 fast-flights 套件能否查詢真實航班資料
查詢路線：台北（TPE）→ 東京成田（NRT）
"""

from fast_flights import FlightData, Passengers, create_filter, get_flights_from_filter

# 台灣籍航空公司名稱，需與 Google Flights 回傳的 flight.name 完全一致
TAIWAN_AIRLINES = {"Tigerair Taiwan", "EVA Air", "China Airlines", "STARLUX Airlines"}


def test_query(label: str, from_airport: str, to_airport: str, date: str):
    print(f"\n{'='*55}")
    print(f"測試：{label}  ({from_airport} -> {to_airport})  日期：{date}")
    print('='*55)

    # Step 1: 建立查詢篩選器
    query_filter = create_filter(
        flight_data=[
            FlightData(
                date=date,
                from_airport=from_airport,
                to_airport=to_airport,
            ),
        ],
        trip="one-way",
        seat="economy",
        passengers=Passengers(adults=1),
    )

    # Step 2: 送出查詢，遇到解析失敗（name 全空）最多重試 3 次
    result = None
    for attempt in range(1, 4):
        try:
            result = get_flights_from_filter(query_filter)
            # 判斷是否解析成功：至少有一筆 name 不為空
            if result and result.flights and any(f.name for f in result.flights):
                break
            print(f"  [第 {attempt} 次] 解析失敗（name 全空），重試中...")
        except Exception as e:
            print(f"  [第 {attempt} 次] 錯誤：{e}，重試中...")

    if not result or not result.flights:
        print("[WARN] 沒有查到任何航班")
        return
    result_flights = result.flights
    #列出所有航班
    print(f"  [OK] 原始共 {len(result_flights)} 班，解析成功！")
    print(f"  [INFO] 航班資訊：")
    for i, flight in enumerate(result_flights, 1):
        print(f"    [{i}] {flight}")
   
    # Step 3: 只保留台灣籍航空公司、且 is_best=True 的航班（排除重複）
    #tw_flights = [
    #    f for f in result.flights
    #    if f.name in TAIWAN_AIRLINES and f.is_best
    #]

    #print(f"[OK] 原始共 {len(result.flights)} 班，台灣航空公司篩選後剩 {len(tw_flights)} 班")
    #print(f"     目前票價狀態：{result.current_price}\n")

    #if not tw_flights:
    #    print("  [WARN] 此路線無台灣航空公司班次")
    #    return

   # for i, flight in enumerate(tw_flights, 1):
    #    print(f"  [{i}] {flight}")


if __name__ == "__main__":
    test_query(
        label="台北桃園 -> 東京成田",
        from_airport="TPE",
        to_airport="NRT",
        date="2026-06-10",
    )

    #test_query(
    #    label="台北松山 -> 東京羽田",
    #    from_airport="TSA",
    #    to_airport="HND",
    #       date="2026-06-10",
    #)
