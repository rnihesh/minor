/**
 * History.jsx — Reference theme: white cards, pill badges, clean list
 */
import { useState, useEffect } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { format } from "date-fns";
import { useAuth } from "../context/AuthContext";

const API_BASE = "http://localhost:8000";

const EMOTION_STYLES = {
  neutral:   { bg:"#f1f5f9", text:"#475569", dot:"#94a3b8" },
  calm:      { bg:"#cffafe", text:"#0e7490", dot:"#06b6d4" },
  happy:     { bg:"#fef9c3", text:"#b45309", dot:"#f59e0b" },
  sad:       { bg:"#dbeafe", text:"#1d4ed8", dot:"#3b82f6" },
  angry:     { bg:"#ffe4e6", text:"#be123c", dot:"#f43f5e" },
  fearful:   { bg:"#ede9fe", text:"#6d28d9", dot:"#8b5cf6" },
  disgust:   { bg:"#dcfce7", text:"#047857", dot:"#10b981" },
  surprised: { bg:"#ffedd5", text:"#c2410c", dot:"#f97316" },
};

const EMOJIS = { neutral:"😐",calm:"😌",happy:"😄",sad:"😢",angry:"😠",fearful:"😨",disgust:"🤢",surprised:"😲" };

const Card = ({ children, className="" }) => (
  <div className={`rounded-3xl ${className}`}
    style={{ background:"rgba(255,255,255,0.88)", backdropFilter:"blur(16px)", boxShadow:"0 2px 20px rgba(139,92,246,0.07)" }}>
    {children}
  </div>
);

export default function History() {
  const { user } = useAuth();
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  const fetchHistory = () => {
    setLoading(true); setError(null);
    const headers = user?.token ? { Authorization: `Bearer ${user.token}` } : {};
    fetch(`${API_BASE}/history`, { headers })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(data => setRecords(data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchHistory(); }, [user]);

  return (
    <div className="p-6 h-full overflow-y-auto">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-extrabold text-gray-900">Emotion History</h1>
          <p className="text-sm text-gray-400 font-medium mt-1">Last 50 predictions from MongoDB</p>
        </div>
        <button onClick={fetchHistory}
          className="flex items-center gap-2 px-5 py-2.5 rounded-2xl text-sm font-bold transition-all duration-200"
          style={{ background:"#111827", color:"white", boxShadow:"0 4px 12px rgba(17,24,39,0.2)" }}
          onMouseEnter={e => e.currentTarget.style.opacity = "0.85"}
          onMouseLeave={e => e.currentTarget.style.opacity = "1"}>
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <div className="w-16 h-16 rounded-full flex items-center justify-center"
            style={{ background:"rgba(255,255,255,0.8)", boxShadow:"0 4px 20px rgba(139,92,246,0.15)" }}>
            <Loader2 className="w-8 h-8 animate-spin" style={{ color:"#8b5cf6" }} />
          </div>
          <p className="text-sm font-semibold text-gray-400">Loading history…</p>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <Card className="p-5 text-center">
          <p className="text-sm font-semibold" style={{ color:"#e11d48" }}>⚠️ {error}</p>
          <p className="text-xs text-gray-400 mt-1">Make sure the backend is running.</p>
        </Card>
      )}

      {/* Empty */}
      {!loading && !error && records.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 gap-5">
          <div className="w-20 h-20 rounded-full flex items-center justify-center text-4xl"
            style={{ background:"rgba(255,255,255,0.8)", boxShadow:"0 4px 20px rgba(139,92,246,0.1)" }}>🕒</div>
          <div className="text-center">
            <p className="text-base font-bold text-gray-700 mb-1">No predictions yet</p>
            <p className="text-sm text-gray-400">Upload or record audio on the Dashboard to get started.</p>
          </div>
        </div>
      )}

      {/* Record list */}
      {!loading && records.length > 0 && (
        <div className="space-y-3">
          {records.map((rec, idx) => {
            const s = EMOTION_STYLES[rec.emotion] || EMOTION_STYLES.neutral;
            return (
              <Card key={rec._id} className="px-5 py-4 flex items-center gap-4 transition-all duration-200"
                style={{ animationDelay:`${idx * 25}ms` }}
                onMouseEnter={e => e.currentTarget.style.transform = "translateX(4px)"}
                onMouseLeave={e => e.currentTarget.style.transform = ""}>

                {/* Emoji circle */}
                <div className="w-11 h-11 rounded-2xl flex items-center justify-center text-2xl flex-shrink-0"
                  style={{ background: s.bg }}>
                  {EMOJIS[rec.emotion] || "🎤"}
                </div>

                {/* Emotion + bar */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-sm font-extrabold capitalize text-gray-900">{rec.emotion}</span>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide"
                      style={{ background: s.bg, color: s.text }}>
                      {(rec.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {/* Mini confidence bar */}
                  <div className="w-full h-1.5 rounded-full" style={{ background:"#f3f4f6" }}>
                    <div className="h-full rounded-full transition-all duration-700"
                      style={{ width:`${rec.confidence * 100}%`, background: s.dot }} />
                  </div>
                </div>

                {/* Timestamp */}
                <div className="text-right flex-shrink-0">
                  <p className="text-xs font-bold text-gray-700">
                    {format(new Date(rec.timestamp), "MMM d, yyyy")}
                  </p>
                  <p className="text-xs font-mono text-gray-400">
                    {format(new Date(rec.timestamp), "HH:mm:ss")}
                  </p>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
