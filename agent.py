"""
Flight Search AI Agent (agent.py)
===================================
多輪對話代理，負責：
  1. 從自然語言萃取搜尋參數（目的地、日期、行李、航空公司…）
  2. 當資訊模糊時主動反問（例如「日本」→ 列出可選城市）
  3. 將城市名稱對應到 IATA 機場代碼
  4. 所有必要資訊齊全後，回傳結構化搜尋條件給主系統

使用方式（在 main.py 中）：
    from agent import agent_chat
    result = agent_chat(messages)   # messages 為完整對話歷史
    # result 範例：
    #   {"ready": false, "message": "你想去日本哪個城市？"}
    #   {"ready": true, "arrival_airport": "KIX", "date": "2025-07-10", ...}
"""

import json
import os
from datetime import date
from openai import OpenAI

# ── OpenAI 客戶端 ──────────────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        _client = OpenAI(api_key=key) if key else None
    return _client


# ── 城市／地名 → IATA 機場代碼 ─────────────────────────────────────────────────
# 多機場城市：列表第一個為預設（台灣出發最常用）
CITY_AIRPORT_MAP = {
    # 日本
    "東京": ["NRT", "HND"], "tokyo": ["NRT", "HND"],
    "成田": ["NRT"],         "羽田": ["HND"],
    "大阪": ["KIX"],         "關西": ["KIX"],  "osaka": ["KIX"],
    "福岡": ["FUK"],         "fukuoka": ["FUK"],
    "沖繩": ["OKA"],         "那霸": ["OKA"],  "okinawa": ["OKA"],
    "札幌": ["CTS"],         "北海道": ["CTS"], "sapporo": ["CTS"],
    "名古屋": ["NGO"],       "nagoya": ["NGO"],
    # 韓國
    "首爾": ["ICN", "GMP"],  "서울": ["ICN", "GMP"], "seoul": ["ICN", "GMP"],
    "仁川": ["ICN"],         "金浦": ["GMP"],
    "釜山": ["PUS"],         "부산": ["PUS"],  "busan": ["PUS"],
    # 東南亞
    "曼谷": ["BKK", "DMK"],  "bangkok": ["BKK", "DMK"],
    "素萬那普": ["BKK"],     "廊曼": ["DMK"],
    "新加坡": ["SIN"],       "singapore": ["SIN"],
    "吉隆坡": ["KUL"],       "kuala lumpur": ["KUL"],
    "馬尼拉": ["MNL"],       "manila": ["MNL"],
    "宿霧": ["CEB"],         "cebu": ["CEB"],
    "胡志明市": ["SGN"],     "ho chi minh": ["SGN"], "西貢": ["SGN"],
    "河內": ["HAN"],         "hanoi": ["HAN"],
    "峇里島": ["DPS"],       "bali": ["DPS"],
    # 澳洲
    "雪梨": ["SYD"],         "sydney": ["SYD"],
    "墨爾本": ["MEL"],       "melbourne": ["MEL"],
    # 美國
    "洛杉磯": ["LAX"],       "los angeles": ["LAX"],
    "舊金山": ["SFO"],       "san francisco": ["SFO"],
    "紐約": ["JFK"],         "new york": ["JFK"],
    # 歐洲
    "倫敦": ["LHR"],         "london": ["LHR"],
    "荷蘭": ["AMS"], "阿姆斯特丹": ["AMS"],   "amsterdam": ["AMS"],
    "維也納": ["VIE"],       "vienna": ["VIE"],
}

# 地區 → 可選城市（讓 Agent 在地區模糊時列出選項）
REGION_CITIES = {
    "日本":   ["東京", "大阪", "福岡", "沖繩", "札幌", "名古屋"],
    "韓國":   ["首爾", "釜山"],
    "東南亞": ["曼谷", "新加坡", "吉隆坡", "馬尼拉", "胡志明市", "河內", "峇里島"],
    "泰國":   ["曼谷"],
    "越南":   ["胡志明市", "河內"],
    "澳洲":   ["雪梨", "墨爾本"],
    "美國":   ["洛杉磯", "舊金山", "紐約"],
    "歐洲":   ["倫敦", "阿姆斯特丹", "維也納"],
}

# 航空公司關鍵字 → 內部代號
AIRLINE_KEYWORDS = {
    "虎航": "TNA", "台灣虎航": "TNA", "tigerair": "TNA", "tiger": "TNA",
    "長榮": "EVA", "eva": "EVA",
    "華航": "CAL", "中華航空": "CAL", "china airlines": "CAL",
    "星宇": "SJX", "starlux": "SJX",
}

# ── System Prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = f"""你是一個親切的機票搜尋助手，幫助從台灣出發的旅客找到合適的航班。
今天是 {date.today().strftime("%Y-%m-%d")}。

你的工作是透過自然對話，理解使用者想去哪裡、什麼時候出發。
語氣要像朋友一樣自然，不需要制式化地一題一題問。
遇到模糊的地方（例如只說「日本」但沒說城市）就自然地追問，善用你對地理和航空的知識幫助使用者。

你需要在確認以下資訊後，才可以回傳 ready: true：
- 目的地機場代碼（arrival_airport）：必須是具體城市對應的機場，不能只有國家或地區
- 出發日期（date）：格式 YYYY-MM-DD

其他資訊若使用者沒提到，用預設值即可：
- adults：成人人數（12歲以上），預設 1
- children：兒童人數（2–11歲），預設 0
- infants_in_seat：占位嬰兒（2歲以下，需要座位），預設 0
- infants_on_lap：不占位嬰兒（2歲以下，坐大人腿上），預設 0
- baggage_kg：托運行李公斤數，預設 0（不托運）
- need_meal：是否需要機上餐點，預設 false
- seat_preference：座位偏好，預設 "none"（也可以是 "standard" 或 "extra_legroom"）
- airline_filter：預設 null，若使用者指定航空公司則填入代碼

乘客說明：使用者說「一個大人帶一個小孩」→ adults=1, children=1；「帶嬰兒」通常是 infants_on_lap=1（除非說要買嬰兒座位）。
航空公司代碼對應：虎航→TNA、長榮→EVA、華航→CAL、星宇→SJX

你知道的城市與機場對應：
{json.dumps({k: v for k, v in CITY_AIRPORT_MAP.items() if not k.isascii()}, ensure_ascii=False)}

若城市有多個機場（如東京有成田 NRT 和羽田 HND），可以根據情境自行判斷或詢問使用者偏好。

回覆格式必須是合法 JSON，兩種情況：

還不夠確定時（繼續對話）：
{{"ready": false, "message": "你想說的話"}}

資訊齊全，可以搜尋時：
{{"ready": true, "arrival_airport": "KIX", "date": "2026-07-10",
  "adults": 1, "children": 0, "infants_in_seat": 0, "infants_on_lap": 0,
  "baggage_kg": 0, "need_meal": false, "seat_preference": "none",
  "airline_filter": null,
  "summary": "一句話說明即將搜尋的內容"}}
"""


# ── 主要對外函式 ───────────────────────────────────────────────────────────────
def agent_chat(messages: list) -> dict:
    """
    傳入完整對話歷史（不含 system prompt），回傳 Agent 的下一步回應。

    Args:
        messages: [{"role": "user"|"assistant", "content": "..."}, ...]

    Returns:
        dict:
          {"ready": False, "message": "..."}      ← 繼續對話
          {"ready": True,  "arrival_airport": "KIX", "date": "...", ...}  ← 可搜尋
    """
    client = _get_client()
    if not client:
        return {"ready": False, "message": "⚠️ OpenAI API Key 尚未設定，無法使用 AI 對話功能。"}

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
            max_tokens=400,
            temperature=0.2,                       # 低溫度：讓輸出更穩定、結構化
            response_format={"type": "json_object"},  # 強制輸出合法 JSON
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)

        # 確保 ready 是 bool
        result["ready"] = bool(result.get("ready", False))
        return result

    except json.JSONDecodeError as e:
        return {"ready": False, "message": "抱歉，我理解出了一點問題，可以再說一次嗎？"}
    except Exception as e:
        return {"ready": False, "message": f"服務暫時無法使用，請稍後再試。（{e}）"}
