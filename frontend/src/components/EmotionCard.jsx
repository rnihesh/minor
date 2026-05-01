/**
 * EmotionCard.jsx — Reference theme: white cards, purple/pastel accents
 */

const EMOTION_STYLES = {
  neutral:   { light: "#f8fafc", accent: "#64748b", bar: "#94a3b8", text: "#475569", pill: "#f1f5f9" },
  calm:      { light: "#ecfeff", accent: "#0891b2", bar: "#06b6d4", text: "#0e7490", pill: "#cffafe" },
  happy:     { light: "#fffbeb", accent: "#d97706", bar: "#f59e0b", text: "#b45309", pill: "#fef3c7" },
  sad:       { light: "#eff6ff", accent: "#2563eb", bar: "#3b82f6", text: "#1d4ed8", pill: "#dbeafe" },
  angry:     { light: "#fff1f2", accent: "#e11d48", bar: "#f43f5e", text: "#be123c", pill: "#ffe4e6" },
  fearful:   { light: "#faf5ff", accent: "#7c3aed", bar: "#8b5cf6", text: "#6d28d9", pill: "#ede9fe" },
  disgust:   { light: "#f0fdf4", accent: "#059669", bar: "#10b981", text: "#047857", pill: "#dcfce7" },
  surprised: { light: "#fff7ed", accent: "#ea580c", bar: "#f97316", text: "#c2410c", pill: "#ffedd5" },
};

const EMOJIS = { neutral:"😐",calm:"😌",happy:"😄",sad:"😢",angry:"😠",fearful:"😨",disgust:"🤢",surprised:"😲" };
const AI_MESSAGES = {
  neutral:"You seem balanced right now.", calm:"Calm and composed — perfect for focused work.",
  happy:"Great vibes! Keep spreading the joy 🎉", sad:"You seem a bit low. Try some gentle breathing.",
  angry:"Take a slow breath. You'll feel better soon.", fearful:"You're safe. Try grounding yourself.",
  disgust:"Step back, breathe, redirect your energy.", surprised:"Embrace this moment of surprise! 🌟",
};

const Card = ({ children, style = {} }) => (
  <div className="rounded-3xl" style={{ background: "rgba(255,255,255,0.9)", backdropFilter: "blur(16px)", boxShadow: "0 2px 24px rgba(139,92,246,0.08)", ...style }}>
    {children}
  </div>
);

export default function EmotionCard({ result }) {
  if (!result) return null;
  const { emotion, confidence, all_scores } = result;
  const s = EMOTION_STYLES[emotion] || EMOTION_STYLES.neutral;
  const emoji = result.emoji || EMOJIS[emotion] || "🎤";
  const sorted = Object.entries(all_scores || {}).sort(([,a],[,b]) => b - a);

  return (
    <div className="animate-slide-up space-y-4">

      {/* Hero card */}
      <Card style={{ background: `rgba(255,255,255,0.92)` }}>
        <div className="p-8 text-center">
          {/* Emoji circle */}
          <div className="w-24 h-24 rounded-full mx-auto flex items-center justify-center text-5xl mb-5 animate-bounce-in"
            style={{ background: s.light, boxShadow: `0 8px 28px ${s.accent}22` }}>
            {emoji}
          </div>

          {/* Emotion label */}
          <h2 className="text-3xl font-extrabold capitalize mb-1 text-gray-900">{emotion}</h2>

          {/* Message pill */}
          <span className="inline-block px-4 py-1.5 rounded-full text-xs font-semibold mb-5"
            style={{ background: s.pill, color: s.accent }}>
            {AI_MESSAGES[emotion]}
          </span>

          {/* Confidence bar */}
          <div className="mb-3">
            <div className="flex justify-between text-xs font-semibold mb-2" style={{ color: s.accent }}>
              <span>Confidence</span>
              <span>{(confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full h-2.5 rounded-full" style={{ background: s.pill }}>
              <div className="h-full rounded-full transition-all duration-1000"
                style={{ width: `${confidence * 100}%`, background: s.bar }} />
            </div>
          </div>
        </div>
      </Card>

      {/* Score breakdown */}
      <Card>
        <div className="p-5">
          <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-4">Emotion Breakdown</p>
          <div className="space-y-3">
            {sorted.map(([emo, score]) => {
              const es = EMOTION_STYLES[emo] || EMOTION_STYLES.neutral;
              return (
                <div key={emo} className="flex items-center gap-3">
                  <span className="text-base w-6">{EMOJIS[emo] || "🎤"}</span>
                  <span className="text-sm w-20 capitalize font-semibold text-gray-700">{emo}</span>
                  <div className="flex-1 h-2 rounded-full" style={{ background: "#f3f4f6" }}>
                    <div className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${score * 100}%`, background: es.bar }} />
                  </div>
                  <span className="text-xs font-mono font-bold w-12 text-right" style={{ color: es.accent }}>
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </Card>

    </div>
  );
}
