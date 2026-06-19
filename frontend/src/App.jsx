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
    arrival_airport: 'NRT',           // 預設東京成田
    date:            getDefaultDate(), // 預設 7 天後
    baggage_kg:      '',              // 空字串 = 無托運行李（送出時轉為 0）
    need_meal:       false,
    seat_preference: 'none',
  });

  // 行李輸入驗證錯誤訊息
  const [baggageError, setBaggageError] = useState('');

  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  // 元件載入時，向後端取得目的地機場清單
  useEffect(() => {
    fetch(`${API_BASE}/api/airports`)
      .then(res => res.json())
      .then(data => {
        setAirports(data);
        if (data.length > 0 && !data.find(a => a.code === 'NRT')) {
          setQuery(q => ({ ...q, arrival_airport: data[0].code }));
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
          arrival_airport: query.arrival_airport,
          date:            query.date,
          baggage_kg:      baggageKg,
          need_meal:       query.need_meal,
          seat_preference: query.seat_preference,
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
            value={query.arrival_airport}
            onChange={e => setQuery({ ...query, arrival_airport: e.target.value })}
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
