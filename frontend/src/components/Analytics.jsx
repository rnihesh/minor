/**
 * Analytics.jsx — Reference theme: white cards, purple accents, sky-pink gradient bg
 */
import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  PieChart, Pie, Cell,
} from "recharts";
import { TrendingUp, TrendingDown, Minus, Loader2 } from "lucide-react";
import { format } from "date-fns";
import { useAuth } from "../context/AuthContext";

const API_BASE = "http://localhost:8000";

const EMOTION_COLORS = {
  neutral:"#94a3b8", calm:"#06b6d4", happy:"#f59e0b",
  sad:"#3b82f6", angry:"#f43f5e", fearful:"#8b5cf6",
  disgust:"#10b981", surprised:"#f97316",
};

/* White frosted card */
const Card = ({ children, className = "" }) => (
  <div className={`rounded-3xl ${className}`}
    style={{ background:"rgba(255,255,255,0.88)", backdropFilter:"blur(16px)", boxShadow:"0 2px 20px rgba(139,92,246,0.08)" }}>
    {children}
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-2xl p-3 text-xs shadow-xl"
      style={{ background:"white", border:"1px solid #ede9fe" }}>
      <p className="font-bold text-gray-500 mb-2">{label}</p>
      {payload.map(p => (
        <div key={p.name} className="flex items-center gap-2 mb-1">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="capitalize font-semibold text-gray-700">{p.name}</span>
          <span className="ml-auto pl-4 font-mono font-bold" style={{ color: p.color }}>{(p.value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
};

export default function Analytics() {
  const { user } = useAuth();
  const [weekly,  setWeekly]  = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    const headers = user?.token ? { Authorization: `Bearer ${user.token}` } : {};
    Promise.all([
      fetch(`${API_BASE}/weekly-analysis`, { headers }).then(r => { if (!r.ok) throw new Error("Failed to load"); return r.json(); }),
      fetch(`${API_BASE}/history`, { headers }).then(r => { if (!r.ok) throw new Error("Failed to load"); return r.json(); }),
    ])
      .then(([w, h]) => { setWeekly(w); setHistory(h); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [user]);

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="flex flex-col items-center gap-4">
        <div className="w-16 h-16 rounded-full flex items-center justify-center"
          style={{ background:"rgba(255,255,255,0.8)", boxShadow:"0 4px 20px rgba(139,92,246,0.15)" }}>
          <Loader2 className="w-8 h-8 animate-spin" style={{ color:"#8b5cf6" }} />
        </div>
        <p className="text-sm font-semibold text-gray-500">Loading your insights…</p>
      </div>
    </div>
  );

  if (error) return (
    <div className="flex items-center justify-center h-full">
      <p className="text-sm font-semibold text-red-500">⚠️ {error}</p>
    </div>
  );

  const lineData = [...history].reverse().slice(0, 20).map(d => ({
    time: format(new Date(d.timestamp), "HH:mm"),
    ...d.all_scores,
  }));

  const pieData = weekly?.emotion_counts
    ? Object.entries(weekly.emotion_counts).map(([name, value]) => ({ name, value }))
    : [];

  const TrendIcon = weekly?.trend === "improving" ? TrendingUp
    : weekly?.trend === "worsening" ? TrendingDown : Minus;
  const trendColor = weekly?.trend === "improving" ? "#059669"
    : weekly?.trend === "worsening" ? "#e11d48" : "#6b7280";

  return (
    <div className="p-6 overflow-y-auto h-full space-y-6">

      {/* Page heading */}
      <div>
        <h1 className="text-2xl font-extrabold text-gray-900">Find Your Inner Peace</h1>
        <p className="text-sm text-gray-400 mt-1 font-medium">Mood insights from the last 7 days</p>
      </div>

      {/* Summary cards row */}
      <div className="grid grid-cols-3 gap-4">

        <Card className="p-5">
          <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Dominant Emotion</p>
          <p className="text-2xl font-extrabold text-gray-900 capitalize">{weekly?.dominant_emotion || "—"}</p>
          <span className="inline-block mt-2 px-3 py-1 rounded-full text-xs font-bold"
            style={{ background:"#ede9fe", color:"#7c3aed" }}>
            This week
          </span>
        </Card>

        <Card className="p-5">
          <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Weekly Trend</p>
          <div className="flex items-center gap-2">
            <TrendIcon className="w-6 h-6" style={{ color: trendColor }} />
            <span className="text-2xl font-extrabold capitalize" style={{ color: trendColor }}>
              {weekly?.trend || "—"}
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-2 font-medium">vs. previous period</p>
        </Card>

        <Card className="p-5">
          <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Total Sessions</p>
          <p className="text-2xl font-extrabold text-gray-900">{history.length}</p>
          <span className="inline-block mt-2 px-3 py-1 rounded-full text-xs font-bold"
            style={{ background:"#dcfce7", color:"#059669" }}>
            All time
          </span>
        </Card>
      </div>

      {/* Smart suggestion box */}
      {weekly?.suggestion && (
        <Card className="p-5 flex items-start gap-4">
          <div className="w-10 h-10 rounded-2xl flex items-center justify-center text-xl flex-shrink-0"
            style={{ background:"#fffbeb" }}>
            💡
          </div>
          <div>
            <p className="text-sm font-bold text-gray-900 mb-1">Smart Suggestion</p>
            <p className="text-sm text-gray-600 leading-relaxed">{weekly.suggestion}</p>
          </div>
          <span className="ml-auto px-3 py-1 rounded-full text-xs font-bold flex-shrink-0"
            style={{ background:"#fef3c7", color:"#d97706" }}>AI</span>
        </Card>
      )}

      {/* Charts */}
      <div className="grid grid-cols-2 gap-5">

        {/* Line chart */}
        <Card className="p-5">
          <p className="text-sm font-bold text-gray-900 mb-1">Mood Analysis</p>
          <p className="text-xs text-gray-400 font-medium mb-4">Emotion confidence over time</p>
          {lineData.length > 0 ? (
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={lineData}>
                <XAxis dataKey="time" stroke="transparent" tick={{ fill:"#9ca3af", fontSize:11, fontWeight:600 }} />
                <YAxis stroke="transparent" tick={{ fill:"#9ca3af", fontSize:11 }} domain={[0,1]} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize:"11px", fontWeight:600 }} />
                {["happy","sad","angry","neutral"].map(e => (
                  <Line key={e} type="monotone" dataKey={e} stroke={EMOTION_COLORS[e]}
                    strokeWidth={2.5} dot={false} isAnimationActive />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-40">
              <p className="text-sm text-gray-400 font-medium">No history data yet</p>
            </div>
          )}
        </Card>

        {/* Pie chart */}
        <Card className="p-5">
          <p className="text-sm font-bold text-gray-900 mb-1">Emotion Distribution</p>
          <p className="text-xs text-gray-400 font-medium mb-4">Share of each emotion this week</p>
          {pieData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" outerRadius={80} innerRadius={48} paddingAngle={4}>
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={EMOTION_COLORS[entry.name] || "#8b5cf6"} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background:"white", border:"1px solid #ede9fe", borderRadius:"16px", fontSize:"12px", fontWeight:600 }}
                  />
                </PieChart>
              </ResponsiveContainer>
              {/* Legend pills */}
              <div className="flex flex-wrap gap-2 mt-3 justify-center">
                {pieData.map(d => (
                  <span key={d.name} className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold"
                    style={{ background:`${EMOTION_COLORS[d.name]}18`, color: EMOTION_COLORS[d.name] }}>
                    <div className="w-2 h-2 rounded-full" style={{ background: EMOTION_COLORS[d.name] }} />
                    {d.name}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-40">
              <p className="text-sm text-gray-400 font-medium">No weekly data yet</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
