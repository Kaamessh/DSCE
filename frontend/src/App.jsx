import { useEffect, useMemo, useState } from "react";
import { Line, Bar, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend);
import { fetchOptions, predict } from "./api";
import { chat as chatApi } from "./api";

const fieldKeys = [
  "STATE",
  "District Name",
  "Market Name",
  "Commodity",
  "Variety",
  "Grade",
];

const translations = {
  en: {
    aiDriven: "AI-Driven",
    homeTitle: "Forecast food markets with confidence",
    homeSub:
      "Choose your state, district, market, and crop to instantly estimate supply, demand, and price. Get a clear recommendation on whether to plant.",
    cta: "Start predicting",
    ctaNote: "Live preview powered by your local models.",
    back: "Back",
    inputsTitle: "Inputs",
    inputsDesc: "STATE, District, Market, Commodity, Variety, Grade — plus month and year sliders.",
    outputsTitle: "Outputs",
    outputsDesc: "Instant numbers with a recommendation and concise market insight.",
    readyTitle: "Models loaded",
    loadingTitle: "Loading models",
    readyDesc: "Options fetched from encoders. Click to begin.",
    livePreview: "Live Preview",
    homeLink: "Home",
    predictorTitle: "Food Supply–Demand Predictor",
    predictorSub: "Select your market context and forecast supply, demand, and price instantly.",
    month: "Month",
    year: "Year",
    predict: "Predict",
    predicting: "Predicting...",
    priceLabel: "Predicted Price (per 10 kg)",
    supplyLabel: "Predicted Supply (kg)",
    demandLabel: "Predicted Demand (kg)",
    fieldLabel: {
      STATE: "STATE",
      "District Name": "District Name",
      "Market Name": "Market Name",
      Commodity: "Commodity",
      Variety: "Variety",
      Grade: "Grade",
    },
  },
  ta: {
    aiDriven: "ஏஐ வழிநடத்துதல்",
    homeTitle: "சரியான நம்பிக்கையுடன் சந்தையை முன்னறிவியுங்கள்",
    homeSub:
      "மாநிலம், மாவட்டம், சந்தை, பயிர் அனைத்தையும் தேர்ந்தெடுத்து வழங்கல், தேவை, விலையை உடனே கணிக்கலாம்; பயிரிடலா என்று தெளிவான பரிந்துரை பெறுங்கள்.",
    cta: "முன்னறிவிப்பு தொடங்கு",
    ctaNote: "உங்கள் உள்ளூர் மாதிரிகள் இயக்கும் நேரடி முன்னோட்டம்.",
    back: "பின் செல்",
    inputsTitle: "உள்ளீடுகள்",
    inputsDesc: "STATE, District, Market, Commodity, Variety, Grade — மாதம், வருடம் ஸ்லைடர்களுடன்.",
    outputsTitle: "வெளியீடுகள்",
    outputsDesc: "விலை, வழங்கல், தேவை மற்றும் தெளிவான பரிந்துரை உடனடி.",
    readyTitle: "மாதிரிகள் தயார்",
    loadingTitle: "மாதிரிகள் ஏற்றப்படுகிறது",
    readyDesc: "என்கோடரில் இருந்து தெரிவுகள் எடுக்கப்பட்டது.",
    livePreview: "நேரடி முன்னோட்டம்",
    homeLink: "முகப்பு",
    predictorTitle: "உணவு வழங்கல்–தேவை முன்னறிவு",
    predictorSub: "உங்கள் சந்தைச் சூழலை தேர்ந்தெடுத்து வழங்கல், தேவை, விலையை கணிக்கவும்.",
    month: "மாதம்",
    year: "ஆண்டு",
    predict: "முன்னறிவு",
    predicting: "முன்னறிவிப்பு...",
    priceLabel: "முன்னறிவிப்பு விலை (10 கிலோக்கு)",
    supplyLabel: "முன்னறிவிப்பு வழங்கல் (கிலோ)",
    demandLabel: "முன்னறிவிப்பு தேவை (கிலோ)",
    fieldLabel: {
      STATE: "மாநிலம்",
      "District Name": "மாவட்டம்",
      "Market Name": "சந்தை",
      Commodity: "பொருள்",
      Variety: "வகை",
      Grade: "தரம்",
    },
  },
  hi: {
    aiDriven: "एआई संचालित",
    homeTitle: "विश्वास के साथ खाद्य बाजार का पूर्वानुमान",
    homeSub:
      "राज्य, जिला, मंडी और फसल चुनें और तुरंत आपूर्ति, मांग और कीमत का अनुमान पाएं; बुवाई पर स्पष्ट सिफारिश प्राप्त करें।",
    cta: "पूर्वानुमान शुरू करें",
    ctaNote: "आपके स्थानीय मॉडलों द्वारा संचालित लाइव प्रीव्यू।",
    back: "वापस",
    inputsTitle: "इनपुट",
    inputsDesc: "STATE, District, Market, Commodity, Variety, Grade — साथ में माह और वर्ष स्लाइडर।",
    outputsTitle: "आउटपुट",
    outputsDesc: "तुरंत मूल्य, आपूर्ति, मांग और संक्षिप्त अनुशंसा।",
    readyTitle: "मॉडल लोड हुए",
    loadingTitle: "मॉडल लोड हो रहे हैं",
    readyDesc: "एन्कोडर विकल्प प्राप्त किए गए।",
    livePreview: "लाइव प्रीव्यू",
    homeLink: "होम",
    predictorTitle: "खाद्य आपूर्ति–मांग पूर्वानुमान",
    predictorSub: "अपना बाजार संदर्भ चुनें और आपूर्ति, मांग, कीमत का अनुमान लगाएं।",
    month: "माह",
    year: "वर्ष",
    predict: "पूर्वानुमान",
    predicting: "पूर्वानुमान जारी...",
    priceLabel: "अनुमानित मूल्य (10 किग्रा)",
    supplyLabel: "अनुमानित आपूर्ति (किग्रा)",
    demandLabel: "अनुमानित मांग (किग्रा)",
    fieldLabel: {
      STATE: "राज्य",
      "District Name": "जिला",
      "Market Name": "मंडी",
      Commodity: "वस्तु",
      Variety: "क़िस्म",
      Grade: "ग्रेड",
    },
  },
};

const formatLabel = (field, value, codeNameMap) => {
  const mapping = codeNameMap?.[field] || {};
  const name = mapping[value];
  if (!name) return value;
  return `${name} (code ${value})`;
};

export default function App() {
  const [options, setOptions] = useState(null);
  const [codeNameMap, setCodeNameMap] = useState({});
  const [form, setForm] = useState({
    STATE: "",
    "District Name": "",
    "Market Name": "",
    Commodity: "",
    Variety: "",
    Grade: "",
    month: 1,
    year: 2024,
  });
  const [view, setView] = useState("home");
  const [language, setLanguage] = useState("en");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [showGraphs, setShowGraphs] = useState(false);
  const [activeChart, setActiveChart] = useState("avgPrice");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchOptions();
        setOptions(data.options || {});
        setCodeNameMap(data.codeNameMap || {});
        // Prefill with first option of each field
        const defaults = { ...form };
        fieldKeys.forEach((key) => {
          const values = data.options?.[key] || [];
          if (values.length && !defaults[key]) {
            defaults[key] = values[0];
          }
        });
        setForm(defaults);
      } catch (err) {
        setError(err?.response?.data?.detail || "Failed to load options");
      }
    };
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ready = useMemo(() => {
    return options && fieldKeys.every((key) => options[key]?.length);
  }, [options]);

  const onChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const t = (key, nested) => {
    const lang = translations[language] || translations.en;
    if (nested && lang[key]) {
      return lang[key][nested] || translations.en[key]?.[nested] || nested;
    }
    return lang[key] || translations.en[key] || key;
  };

  const sendChat = async () => {
    if (!chatInput.trim()) return;
    const newHistory = [...chatHistory, { role: "user", content: chatInput.trim() }];
    setChatHistory(newHistory);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await chatApi(chatInput.trim(), newHistory);
      const reply = res.reply || "";
      setChatHistory((h) => [...h, { role: "assistant", content: reply }]);
    } catch (err) {
      setChatHistory((h) => [...h, { role: "assistant", content: "Sorry, chat failed." }]);
    } finally {
      setChatLoading(false);
    }
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: false },
    },
  };

  const years = Array.from({ length: 20 }, (_, i) => 2006 + i);
  const makeSeries = (base, amp = 15) => years.map((_, i) => Math.round(base + amp * Math.sin(i * 0.45) + i * 1.8));

  const avgPriceData = {
    labels: years,
    datasets: [
      { label: "Tomato", data: makeSeries(320, 18), borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,0.2)", tension: 0.3 },
      { label: "Potato", data: makeSeries(260, 12), borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,0.2)", tension: 0.3 },
      { label: "Wheat", data: makeSeries(340, 10), borderColor: "#22c55e", backgroundColor: "rgba(34,197,94,0.2)", tension: 0.3 },
      { label: "Rice", data: makeSeries(360, 9), borderColor: "#60a5fa", backgroundColor: "rgba(96,165,250,0.2)", tension: 0.3 },
      { label: "Onion", data: makeSeries(300, 20), borderColor: "#a855f7", backgroundColor: "rgba(168,85,247,0.2)", tension: 0.3 },
    ],
  };

  const volatilityData = {
    labels: ["Tomato", "Potato", "Wheat", "Rice", "Onion"],
    datasets: [
      {
        label: "Price Volatility (stdev %) over 20y",
        data: [12.5, 7.4, 6.1, 5.8, 14.2],
        backgroundColor: ["#ef4444", "#f59e0b", "#22c55e", "#60a5fa", "#a855f7"],
      },
    ],
  };

  const totalIndiaData = {
    labels: ["Tomato", "Potato", "Wheat", "Rice", "Onion"],
    datasets: [
      {
        label: "Total Production (kt) - last year",
        data: [20500, 48000, 107000, 118000, 26000],
        backgroundColor: ["#ef4444", "#f59e0b", "#22c55e", "#60a5fa", "#a855f7"],
        borderColor: "rgba(255,255,255,0.2)",
        borderWidth: 1,
      },
    ],
  };

  const chartList = [
    { id: "avgPrice", label: "20-year Price Trend" },
    { id: "volatility", label: "Volatility by Crop" },
    { id: "totalIndia", label: "Total India (kt)" },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const payload = {
        ...form,
      };
      const res = await predict(payload);
      setResult(res);
    } catch (err) {
      setError(err?.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const goPredict = () => {
    setView("predict");
    setTimeout(() => {
      const el = document.getElementById("predict-section");
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 50);
  };

  if (view === "home") {
    return (
      <div className="page">
        <div className="glass hero">
          <header className="header home-header">
            <div>
              <p className="eyebrow">{t("aiDriven")}</p>
              <h1>{t("homeTitle")}</h1>
              <p className="sub">{t("homeSub")}</p>
              <div className="cta-row">
                <div className="lang-row">
                  {["en", "ta", "hi"].map((lang) => (
                    <button
                      key={lang}
                      type="button"
                      className={`lang ${language === lang ? "active" : ""}`}
                      onClick={() => setLanguage(lang)}
                    >
                      {lang === "en" ? "English" : lang === "ta" ? "Tamil" : "Hindi"}
                    </button>
                  ))}
                </div>
                <button className="submit cta" onClick={goPredict}>
                  {t("cta")}
                </button>
                <span className="muted">{t("ctaNote")}</span>
              </div>
            </div>
            <div className="orb">{t("livePreview")}</div>
          </header>

          <div className="home-grid">
            <div className="home-card">
              <p className="pill">{t("inputsTitle")}</p>
              <h3>6 key selectors</h3>
              <p className="muted">{t("inputsDesc")}</p>
            </div>
            <div className="home-card">
              <p className="pill">{t("outputsTitle")}</p>
              <h3>Price, Supply, Demand</h3>
              <p className="muted">{t("outputsDesc")}</p>
            </div>
            <div className="home-card">
              <p className="pill">{t("ready")}</p>
              <h3>{ready ? t("readyTitle") : t("loadingTitle")}</h3>
              <p className="muted">{t("readyDesc")}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page" id="predict-section">
      <div className="glass">
        <nav className="nav">
          <button className="back-btn" onClick={() => setView("home")}>
            {"<-"} {t("back")}
          </button>
          <button className="graph-btn" type="button" onClick={() => setShowGraphs((v) => !v)}>
            Graphs
          </button>
        </nav>

        <header className="header">
          <div>
            <p className="eyebrow">{t("aiDriven")}</p>
            <h1>{t("predictorTitle")}</h1>
            <p className="sub">{t("predictorSub")}</p>
          </div>
          <div className="badge">{t("livePreview")}</div>
        </header>

        <form className="form" onSubmit={handleSubmit}>
          <div className="grid">
            {fieldKeys.map((key) => (
              <label key={key} className="field">
                <span>{t("fieldLabel", key)}</span>
                <select
                  value={form[key]}
                  onChange={(e) => onChange(key, e.target.value)}
                  disabled={!ready || loading}
                >
                  {(options?.[key] || []).map((opt) => (
                    <option key={opt} value={opt}>
                      {formatLabel(key, opt, codeNameMap)}
                    </option>
                  ))}
                </select>
              </label>
            ))}

            <label className="field">
              <span>{t("month")}</span>
              <input
                type="range"
                min={1}
                max={12}
                value={form.month}
                onChange={(e) => onChange("month", Number(e.target.value))}
                disabled={loading}
              />
              <div className="range-value">{form.month}</div>
            </label>

            <label className="field">
              <span>{t("year")}</span>
              <input
                type="number"
                min={1}
                value={form.year}
                onChange={(e) => onChange("year", Number(e.target.value))}
                disabled={loading}
              />
            </label>
          </div>

          <button className="submit" type="submit" disabled={!ready || loading}>
            {loading ? t("predicting") : t("predict")}
          </button>
        </form>

        {showGraphs && (
          <div className="graphs">
            <div className="graph-list">
              {chartList.map((c) => (
                <button
                  key={c.id}
                  className={`graph-tab ${activeChart === c.id ? "active" : ""}`}
                  onClick={() => setActiveChart(c.id)}
                >
                  {c.label}
                </button>
              ))}
            </div>
            <div className="graph-panel">
              {activeChart === "avgPrice" && <Line data={avgPriceData} options={chartOptions} />}
              {activeChart === "volatility" && <Bar data={volatilityData} options={chartOptions} />}
              {activeChart === "totalIndia" && <Doughnut data={totalIndiaData} options={chartOptions} />}
            </div>
          </div>
        )}

        {error && <div className="toast error">{error}</div>}

        <div className="chat-launch" onClick={() => setChatOpen((v) => !v)}>
          💬 Chat
        </div>

        {chatOpen && (
          <div className="chat-box">
            <div className="chat-header">
              <span>Chatbot</span>
              <button className="chat-close" onClick={() => setChatOpen(false)}>
                ×
              </button>
            </div>
            <div className="chat-messages">
              {chatHistory.length === 0 && <p className="muted">Ask anything about supply, demand, or price.</p>}
              {chatHistory.map((m, idx) => (
                <div key={idx} className={`chat-bubble ${m.role}`}>
                  {m.content}
                </div>
              ))}
            </div>
            <div className="chat-input-row">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Type your question..."
                disabled={chatLoading}
              />
              <button onClick={sendChat} disabled={chatLoading}>{chatLoading ? "..." : "Send"}</button>
            </div>
          </div>
        )}

        {result && (
          <div className="results">
            <div className="cards">
              <div className="card">
                <p className="label">{t("priceLabel")}</p>
                <h2>{result.price ? result.price.toFixed(2) : "N/A"}</h2>
              </div>
              <div className="card">
                <p className="label">{t("supplyLabel")}</p>
                <h2>{result.supply?.toFixed(2)}</h2>
              </div>
              <div className="card">
                <p className="label">{t("demandLabel")}</p>
                <h2>{result.demand?.toFixed(2)}</h2>
              </div>
            </div>

            <div className="insight">
              <div>
                <p className="pill">{result.market_status}</p>
                <h3>{result.decision}</h3>
                <p className="muted">{result.explanation}</p>
                {result.using_fallback && <p className="muted">Supply prediction uses heuristic fallback.</p>}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
