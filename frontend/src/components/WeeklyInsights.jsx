/**
 * WeeklyInsights Component
 * ========================
 * Fetches and displays trend analysis for the past 7 days from MongoDB.
 */

import { useState, useEffect } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, Legend } from "recharts";
import "./Dashboard.css";

const API_BASE = "http://localhost:8000";

const COLORS = {
  neutral: "#64748b", calm: "#06b6d4", happy: "#f59e0b", sad: "#3b82f6",
  angry: "#ef4444", fearful: "#8b5cf6", disgust: "#10b981", surprised: "#f97316",
};

export default function WeeklyInsights() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/weekly-analysis`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch weekly analysis");
        return res.json();
      })
      .then((data) => {
        setData(data);
        setIsLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setIsLoading(false);
      });
  }, []);

  if (isLoading) return <div className="loading-text text-center">Loading insights...</div>;
  if (error) return <div className="error-text text-center">{error}</div>;
  if (!data || !data.emotion_counts) return <div className="text-center">Not enough data for weekly insights.</div>;

  const pieData = Object.entries(data.emotion_counts).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <div className="weekly-insights animate-fade-in-up">
      <h2 className="dashboard-title">📅 7-Day Insights</h2>
      
      <div className="insights-summary glass">
        <div className="insight-item">
          <span className="insight-label">Dominant Emotion</span>
          <span className="insight-value" style={{ color: COLORS[data.dominant_emotion] || "#fff" }}>
            {data.dominant_emotion}
          </span>
        </div>
        <div className="insight-item">
          <span className="insight-label">Trend</span>
          <span className={`insight-value trend-${data.trend}`}>
            {data.trend === "improving" ? "📈 Improving" : data.trend === "worsening" ? "📉 Worsening" : "➖ Stable"}
          </span>
        </div>
      </div>

      <div className="suggestion-box glass">
        <h4>💡 Smart Suggestion</h4>
        <p>{data.suggestion}</p>
      </div>

      <div className="chart-container glass">
        <h4>Emotion Distribution</h4>
        <div style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                innerRadius={60}
                paddingAngle={5}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[entry.name] || "#8b5cf6"} />
                ))}
              </Pie>
              <RechartsTooltip 
                contentStyle={{ backgroundColor: "#12122a", borderColor: "rgba(255,255,255,0.1)", borderRadius: "8px" }}
                itemStyle={{ color: "#f0f0f5" }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
