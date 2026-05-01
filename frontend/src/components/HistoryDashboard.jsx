/**
 * HistoryDashboard Component
 * ==========================
 * Fetches and displays the last 50 emotion predictions from MongoDB.
 */

import { useState, useEffect } from "react";
import { format } from "date-fns";
import "./Dashboard.css";

const API_BASE = "http://localhost:8000";

const EMOTION_EMOJIS = {
  neutral: "😐", calm: "😌", happy: "😄", sad: "😢",
  angry: "😠", fearful: "😨", disgust: "🤢", surprised: "😲",
};

export default function HistoryDashboard() {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/history`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch history");
        return res.json();
      })
      .then((data) => {
        setHistory(data);
        setIsLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setIsLoading(false);
      });
  }, []);

  if (isLoading) return <div className="loading-text text-center">Loading history...</div>;
  if (error) return <div className="error-text text-center">{error}</div>;

  return (
    <div className="history-dashboard animate-fade-in-up">
      <h2 className="dashboard-title">🕒 Emotion History</h2>
      
      {history.length === 0 ? (
        <p className="empty-state">No predictions yet. Try uploading or streaming some audio!</p>
      ) : (
        <div className="history-list">
          {history.map((record) => (
            <div key={record._id} className="history-card glass">
              <div className="history-icon">
                {EMOTION_EMOJIS[record.emotion] || "🎤"}
              </div>
              <div className="history-details">
                <span className="history-emotion">{record.emotion}</span>
                <span className="history-confidence">{(record.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="history-time">
                {format(new Date(record.timestamp), "MMM d, yyyy • HH:mm:ss")}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
