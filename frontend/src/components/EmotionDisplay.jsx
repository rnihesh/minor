/**
 * EmotionDisplay Component
 * ========================
 * Shows the detected emotion in a visually striking way:
 *   - Large emoji animation
 *   - Emotion label with gradient text
 *   - Animated confidence bar
 *   - Per-emotion probability breakdown
 */

import { useEffect, useState } from "react";
import "./EmotionDisplay.css";

/** Map emotion names to colour CSS variables */
const EMOTION_COLORS = {
  neutral: "var(--emotion-neutral)",
  calm: "var(--emotion-calm)",
  happy: "var(--emotion-happy)",
  sad: "var(--emotion-sad)",
  angry: "var(--emotion-angry)",
  fearful: "var(--emotion-fearful)",
  disgust: "var(--emotion-disgust)",
  surprised: "var(--emotion-surprised)",
};

/** Map emotion names to emojis */
const EMOTION_EMOJIS = {
  neutral: "😐",
  calm: "😌",
  happy: "😄",
  sad: "😢",
  angry: "😠",
  fearful: "😨",
  disgust: "🤢",
  surprised: "😲",
};

export default function EmotionDisplay({ result }) {
  const [showBars, setShowBars] = useState(false);

  // Delay bar animations so they stagger nicely after the card fades in
  useEffect(() => {
    const timer = setTimeout(() => setShowBars(true), 400);
    return () => clearTimeout(timer);
  }, [result]);

  if (!result) return null;

  const { emotion, confidence, all_scores, models_used } = result;
  const color = EMOTION_COLORS[emotion] || "var(--accent-purple)";
  const emoji = result.emoji || EMOTION_EMOJIS[emotion] || "🎤";

  // Sort scores descending for the breakdown chart
  const sortedScores = Object.entries(all_scores || {}).sort(
    ([, a], [, b]) => b - a
  );

  return (
    <div className="emotion-display animate-fade-in-up">
      {/* Hero emotion */}
      <div className="emotion-hero">
        <div className="emotion-emoji animate-bounce-in">{emoji}</div>
        <h2
          className="emotion-label"
          style={{ "--emotion-color": color }}
        >
          {emotion}
        </h2>
        <div className="confidence-badge" style={{ "--emotion-color": color }}>
          {(confidence * 100).toFixed(1)}% confident
        </div>
      </div>

      {/* Confidence bar */}
      <div className="confidence-bar-container">
        <div className="confidence-bar-track">
          <div
            className="confidence-bar-fill"
            style={{
              width: showBars ? `${confidence * 100}%` : "0%",
              background: `linear-gradient(90deg, ${color}, ${color}88)`,
            }}
          />
        </div>
      </div>

      {/* Score breakdown */}
      <div className="scores-section">
        <h3 className="scores-title">Emotion Breakdown</h3>
        <div className="scores-grid">
          {sortedScores.map(([emo, score], idx) => (
            <div
              key={emo}
              className={`score-row ${emo === emotion ? "highlight" : ""}`}
              style={{ animationDelay: `${idx * 0.06}s` }}
            >
              <div className="score-left">
                <span className="score-emoji">
                  {EMOTION_EMOJIS[emo] || "🎤"}
                </span>
                <span className="score-name">{emo}</span>
              </div>
              <div className="score-right">
                <div className="score-bar-track">
                  <div
                    className="score-bar-fill"
                    style={{
                      width: showBars ? `${score * 100}%` : "0%",
                      background: EMOTION_COLORS[emo] || "var(--accent-purple)",
                      transitionDelay: `${idx * 0.06}s`,
                    }}
                  />
                </div>
                <span className="score-value">
                  {(score * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model details (collapsible) */}
      {models_used && (
        <details className="model-details">
          <summary className="model-summary">
            🔬 Model Breakdown
          </summary>
          <div className="model-grid">
            {Object.entries(models_used).map(([name, info]) => (
              <div key={name} className="model-card glass">
                <span className="model-name">{name}</span>
                <span className="model-pred">
                  {EMOTION_EMOJIS[info.predicted] || "🎤"} {info.predicted}
                </span>
                <span className="model-conf">
                  {(info.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
