import React, { useState, useEffect } from 'react';
import './App.css';

// 各航空公司 Logo（存放於 src/assets/logos/）
import logoTNA from './assets/logos/TNA.png';
import logoEVA from './assets/logos/EVA.png';
import logoCAL from './assets/logos/CAL.png';
import logoSJX from './assets/logos/SJX.png';

const AIRLINE_LOGOS = {
  TNA: logoTNA,
  EVA: logoEVA,
  CAL: logoCAL,
  SJX: logoSJX,
};

// 出發機場選項（目前支援桃園與松山）
const FROM_AIRPORTS = [
  { code: 'TPE', name: '台北桃園 (TPE)' },
  { code: 'TSA', name: '台北松山 (TSA)' },
  { code: 'KHH', name: '高雄小港 (KHH)' },
];

// 取得今天之後 7 天的預設日期（避免查詢過近班次）
function getDefaultDate() {
  const d = new Date();
  d.setDate(d.getDate() + 7);
  return d.toISOString().split('T')[0]; // YYYY-MM-DD
}

// 後端 API 位址：本機開發用 localhost，部署到 Netlify 時改為 Render URL
// 在 Netlify 環境變數設定 VITE_API_BASE=https://your-app.onrender.com
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  // 目的地機場清單：從後端 /api/airports 動態載入
  const [airports, setAirports] = useState([]);

  // 使用者的搜尋條件
  const [query, setQuery] = useState({
    from_airport:    'TPE',
    to_airport:      'NRT',           // 預設東京成田
    date:            getDefaultDate(), // 預設 7 天後
    baggage_kg:      '',              // 空字串 = 無托運行李（送出時轉為 0）
    need_meal:       false,
    seat_preference: 'none',
    airline_filter:  null,            // AI Agent 解析使用者偏好後填入
    // 乘客人數
    adults:          1,
    children:        0,
    infants_in_seat: 0,
    infants_on_lap:  0,
  });

  // 行李輸入驗證錯誤訊息
  const [baggageError, setBaggageError] = useState('');

  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  // ── AI 對話框狀態 ──────────────────────────────────────────────────────────
  const [chatOpen,     setChatOpen]     = useState(false);
  const [chatMessages, setChatMessages] = useState([]);  // [{role, content}]
  const [chatInput,    setChatInput]    = useState('');
  const [chatLoading,  setChatLoading]  = useState(false);

  // 送出對話訊息給 AI Agent
  const handleChatSend = async () => {
    const text = chatInput.trim();
    if (!text || chatLoading) return;

    const newMessages = [...chatMessages, { role: 'user', content: text }];
    setChatMessages(newMessages);
    setChatInput('');
    setChatLoading(true);

    try {
      const res  = await fetch(`${API_BASE}/api/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ messages: newMessages }),
      });
      const data = await res.json();

      if (data.ready) {
        // Agent 已收集齊全 → 自動填入表單並觸發搜尋
        const summary = data.summary || '已為你設定搜尋條件，開始查詢中…';
        setChatMessages(prev => [...prev, { role: 'assistant', content: `✅ ${summary}` }]);

        // 更新表單欄位（含出發機場、乘客人數）
        setQuery(q => ({
          ...q,
          from_airport:    data.from_airport ?? q.from_airport,
          to_airport:      data.arrival_airport   ?? q.to_airport,
          date:            data.date              ?? q.date,
          baggage_kg:      data.baggage_kg != null ? String(data.baggage_kg) : q.baggage_kg,
          need_meal:       data.need_meal         ?? q.need_meal,
          seat_preference: data.seat_preference   ?? q.seat_preference,
          airline_filter:  data.airline_filter    ?? null,
          adults:          data.adults            ?? q.adults,
          children:        data.children          ?? q.children,
          infants_in_seat: data.infants_in_seat   ?? q.infants_in_seat,
          infants_on_lap:  data.infants_on_lap    ?? q.infants_on_lap,
        }));

        // 短暫延遲後自動搜尋（讓 state 更新完成）
        setTimeout(() => handleSearchWithParams({
          from_airport:    data.from_airport,
          to_airport:      data.arrival_airport,
          date:            data.date,
          baggage_kg:      data.baggage_kg      ?? 0,
          need_meal:       data.need_meal       ?? false,
          seat_preference: data.seat_preference ?? 'none',
          airline_filter:  data.airline_filter  ?? null,
          adults:          data.adults          ?? 1,
          children:        data.children        ?? 0,
          infants_in_seat: data.infants_in_seat ?? 0,
          infants_on_lap:  data.infants_on_lap  ?? 0,
        }), 300);

      } else {
        // Agent 繼續反問
        setChatMessages(prev => [...prev, { role: 'assistant', content: data.message }]);
      }
    } catch (e) {
      setChatMessages(prev => [...prev,
        { role: 'assistant', content: '抱歉，連線發生問題，請稍後再試。' }
      ]);
    }
    setChatLoading(false);
  };

  // 直接帶入參數觸發搜尋（供 Agent 自動觸發用）
  const handleSearchWithParams = async (params) => {
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/search`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          // 優先用 params 帶來的值（agent 解析的），否則用表單 state
          from_airport:    params.from_airport    ?? query.from_airport,
          to_airport:      params.to_airport      ?? query.to_airport,
          date:            params.date            ?? query.date,
          baggage_kg:      params.baggage_kg      ?? query.baggage_kg,
          need_meal:       params.need_meal       ?? query.need_meal,
          seat_preference: params.seat_preference ?? query.seat_preference,
          airline_filter:  params.airline_filter  ?? null,
          adults:          params.adults          ?? query.adults,
          children:        params.children        ?? query.children,
          infants_in_seat: params.infants_in_seat ?? query.infants_in_seat,
          infants_on_lap:  params.infants_on_lap  ?? query.infants_on_lap,
        }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('搜尋時發生錯誤:', error);
    }
    setLoading(false);
  };

  // 元件載入時，向後端取得目的地機場清單
  useEffect(() => {
    fetch(`${API_BASE}/api/airports`)
      .then(res => res.json())
      .then(data => {
        setAirports(data);
        if (data.length > 0 && !data.find(a => a.code === 'NRT')) {
          setQuery(q => ({ ...q, to_airport: data[0].code }));
        }
      })
      .catch(err => console.error('無法取得機場清單:', err));
  }, []);

  // 處理行李輸入變更，並即時驗證範圍
  const handleBaggageChange = (e) => {
    const val = e.target.value;
    setQuery({ ...query, baggage_kg: val });

    if (val === '') {
      setBaggageError('');
    } else {
      const num = parseInt(val, 10);
      if (isNaN(num) || num < 1 || num > 32) {
        setBaggageError('請輸入 1–32 之間的整數，或留空表示無托運行李');
      } else {
        setBaggageError('');
      }
    }
  };

  // 判斷是否可以送出搜尋
  const canSearch = airports.length > 0 && !loading && baggageError === '' && query.date !== '';

  const handleSearch = async () => {
    setLoading(true);
    setResult(null);
    try {
      const baggageKg = query.baggage_kg === '' ? 0 : parseInt(query.baggage_kg, 10);

      const response = await fetch(`${API_BASE}/api/search`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          from_airport:    query.from_airport,
          to_airport:      query.to_airport,
          date:            query.date,
          baggage_kg:      baggageKg,
          need_meal:       query.need_meal,
          seat_preference: query.seat_preference,
          airline_filter:  query.airline_filter,
          adults:          query.adults,
          children:        query.children,
          infants_in_seat: query.infants_in_seat,
          infants_on_lap:  query.infants_on_lap,
        }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('搜尋時發生錯誤:', error);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>AI-Powered True Cost Flight Comparison System</h1>

      {/* ── AI 對話框 ─────────────────────────────────────────────────── */}
      <div className="chat-panel">
        <button className="chat-toggle" onClick={() => setChatOpen(o => !o)}>
          💬 用自然語言描述需求 {chatOpen ? '▲' : '▼'}
        </button>

        {chatOpen && (
          <div className="chat-body">
            {/* 對話歷史 */}
            <div className="chat-history">
              {chatMessages.length === 0 && (
                <p className="chat-hint">
                  試試說：「我想 7/10 去大阪，帶 20kg 行李，只看星宇航班」
                </p>
              )}
              {chatMessages.map((m, i) => (
                <div key={i} className={`chat-msg ${m.role}`}>
                  <span className="chat-bubble">{m.content}</span>
                </div>
              ))}
              {chatLoading && (
                <div className="chat-msg assistant">
                  <span className="chat-bubble chat-typing">AI 思考中…</span>
                </div>
              )}
            </div>

            {/* 輸入列 */}
            <div className="chat-input-row">
              <input
                className="chat-input"
                type="text"
                placeholder="描述你的出行需求…"
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleChatSend()}
                disabled={chatLoading}
              />
              <button
                className="chat-send"
                onClick={handleChatSend}
                disabled={chatLoading || !chatInput.trim()}
              >
                送出
              </button>
              <button
                className="chat-clear"
                onClick={() => setChatMessages([])}
                title="清除對話"
              >
                🗑
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="search-box">

        {/* 出發機場 */}
        <label>
          出發機場_Departure airport：
          <select
            value={query.from_airport}
            onChange={e => setQuery({ ...query, from_airport: e.target.value })}
          >
            {FROM_AIRPORTS.map(a => (
              <option key={a.code} value={a.code}>{a.name}</option>
            ))}
          </select>
        </label>

        {/* 目的地機場 */}
        <label>
          目的地機場_Arrival airport：
          <select
            value={query.to_airport}
            onChange={e => setQuery({ ...query, to_airport: e.target.value })}
          >
            {airports.length === 0 ? (
              <option value="">載入中...</option>
            ) : (
              airports.map(airport => (
                <option key={airport.code} value={airport.code}>
                  {airport.name}
                </option>
              ))
            )}
          </select>
        </label>

        {/* 出發日期 */}
        <label>
          出發日期_Departure date：
          <input
            type="date"
            value={query.date}
            min={new Date().toISOString().split('T')[0]}
            onChange={e => setQuery({ ...query, date: e.target.value })}
          />
        </label>

        {/* 托運行李：自由輸入公斤數 */}
        <div className="baggage-input-group">
          <label>
            托運行李（kg）_Baggage (kg)：
            <input
              type="number"
              min={1}
              max={32}
              step={1}
              value={query.baggage_kg}
              onChange={handleBaggageChange}
              placeholder="輸入托運行李重量，範圍：1-32kg"
              className={baggageError ? 'input-error' : ''}
            />
          </label>
          <p className="baggage-hint">
            留空 = 無托運行李（$0）｜廉價航空：23kg 以下套用 20kg 費率
          </p>
          {baggageError && (
            <p className="baggage-error">{baggageError}</p>
          )}
        </div>

        {/* 是否需要機上餐點 */}
        <label>
          需要機上餐點_Need meal:
          <input
            type="checkbox"
            checked={query.need_meal}
            onChange={e => setQuery({ ...query, need_meal: e.target.checked })}
          />
        </label>

        {/* 座位偏好 */}
        <label>
          預選座位_Seat preference:
          <select
            value={query.seat_preference}
            onChange={e => setQuery({ ...query, seat_preference: e.target.value })}
          >
            <option value="none">不預選</option>
            <option value="standard">標準座位</option>
            <option value="extra_legroom">大空間座位</option>
          </select>
        </label>

        {/* 乘客人數 */}
        <div className="passengers-group">
          <label>
            成人（12歲以上）_Adults：
            <input
              type="number" min={1} max={9} step={1}
              value={query.adults}
              onChange={e => setQuery({ ...query, adults: parseInt(e.target.value) || 1 })}
            />
          </label>
          <label>
            兒童（2-11歲）_Children：
            <input
              type="number" min={0} max={9} step={1}
              value={query.children}
              onChange={e => setQuery({ ...query, children: parseInt(e.target.value) || 0 })}
            />
          </label>
          <label>
            占位嬰兒（2歲以下）_Infants in seat：
            <input
              type="number" min={0} max={9} step={1}
              value={query.infants_in_seat}
              onChange={e => setQuery({ ...query, infants_in_seat: parseInt(e.target.value) || 0 })}
            />
          </label>
          <label>
            不占位嬰兒（2歲以下）_Infants on lap：
            <input
              type="number" min={0} max={9} step={1}
              value={query.infants_on_lap}
              onChange={e => setQuery({ ...query, infants_on_lap: parseInt(e.target.value) || 0 })}
            />
          </label>
        </div>

        <button onClick={handleSearch} disabled={!canSearch}>
          {loading ? '查詢與分析中...' : '計算真實成本'}
        </button>
      </div>

      {/* 搜尋結果 */}
      {result && (
        <div className="results-section">

          <h2>AI 推薦分析</h2>
          <div className="ai-box">
            <p>{result.ai_recommendation}</p>
          </div>

          <h2>航班選項（依真實成本排序）</h2>

          {result.flights.length === 0 ? (
            <p>目前查無台灣航空公司飛往此目的地的班次，請換日期或目的地再試。</p>
          ) : (
            <div className="flight-list">
              {result.flights.map(flight => (
                <div key={flight.flight_id} className="flight-card">

                  <div className="flight-header">
                       {/* 航空公司 Logo */}
                    {AIRLINE_LOGOS[flight.airline_id] && (
                      <img
                        src={AIRLINE_LOGOS[flight.airline_id]}
                        alt={flight.airline_name}
                        className="airline-logo"
                      />
                    )}
                    <h3>{flight.airline_name}（{flight.airline_id}）</h3>
                    <span className="zone-badge">{flight.zone_label}</span>
                  </div>

                  <p>出發：{flight.departure}</p>
                  <p>抵達：{flight.arrival}　飛行時間：{flight.duration}　經停：{flight.stops} 次</p>
                  <p>表面基礎票價：NT${flight.base_fare}</p>

                  <div className="breakdown">
                    <p>
                      + 行李費：NT${flight.add_on_fees.baggage}
                      {flight.baggage_tier !== 'none' && (
                        <span className="tier-note">（套用 {flight.baggage_tier} 費率）</span>
                      )}
                    </p>
                    <p>+ 選位費：NT${flight.add_on_fees.seat}</p>
                    <p>+ 餐點費：NT${flight.add_on_fees.meal}</p>
                    <p>+ 手續費：NT${flight.add_on_fees.payment}</p>
                  </div>

                  <h3 className="true-cost">真實總成本：NT${flight.true_cost}</h3>

                  {/* 虎航 Bundle 優惠提示：只有符合條件時才顯示 */}
                  {flight.bundle_info && (
                    <div className="bundle-box">
                      <p className="bundle-title">💡 優惠提示：{flight.bundle_info.description}</p>
                      {flight.bundle_info.include_meal && (
                        <p className="bundle-meal">✅ 此 Bundle 已含餐費，不另外加收</p>
                      )}
                      <p>使用 Bundle 後真實總成本：<strong>NT${flight.bundle_info.bundle_true_cost}</strong></p>
                      <p className="bundle-saving">比單買便宜 NT${flight.bundle_info.savings}！</p>
                    </div>
                  )}

                  {/* ML 票價預測區塊 */}
                  {flight.price_prediction && (
                    <div className={`prediction-box ${
                      flight.price_prediction.buy_recommendation === '立即購買' ? 'pred-buy' :
                      flight.price_prediction.buy_recommendation === '等待觀望' ? 'pred-wait' :
                      'pred-neutral'
                    }`}>
                      <p className="pred-title">📈 ML 票價預測</p>
                      <p>
                        如果你等 7 天再買：<strong>NT${flight.price_prediction.predicted_fare.toLocaleString()}</strong>
                        <span className={`pred-change ${flight.price_prediction.price_change_pct >= 0 ? 'pred-up' : 'pred-down'}`}>
                          {' '}（{flight.price_prediction.price_change_pct >= 0 ? '+' : ''}{flight.price_prediction.price_change_pct}%）
                        </span>
                      </p>
                      <p className="pred-recommend">
                        建議：<strong>{flight.price_prediction.buy_recommendation}</strong>
                        　{flight.price_prediction.recommendation_reason}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
