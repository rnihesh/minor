/**
 * SuggestionCard.jsx — Reference theme: white cards, purple pill tags
 */

const CARD_COLORS = [
  { bg: "#faf5ff", tag: "#ede9fe", tagText: "#7c3aed", icon: "🧘" },
  { bg: "#fce7f3", tag: "#fce7f3", tagText: "#db2777", icon: "🎵" },
  { bg: "#eff6ff", tag: "#dbeafe", tagText: "#2563eb", icon: "📝" },
  { bg: "#f0fdf4", tag: "#dcfce7", tagText: "#059669", icon: "📞" },
  { bg: "#fff7ed", tag: "#ffedd5", tagText: "#ea580c", icon: "🌳" },
  { bg: "#fefce8", tag: "#fef9c3", tagText: "#ca8a04", icon: "💡" },
];

export default function SuggestionCard({ suggestions }) {
  if (!suggestions || suggestions.length === 0) return null;
  return (
    <div className="animate-slide-up">
      <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-4">Suggestions for You</p>
      <div className="space-y-3">
        {suggestions.map((text, i) => {
          const c = CARD_COLORS[i % CARD_COLORS.length];
          return (
            <div key={i}
              className="p-4 rounded-2xl transition-all duration-300 cursor-default"
              style={{ background: c.bg, boxShadow: "0 1px 8px rgba(0,0,0,0.04)" }}
              onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 6px 20px rgba(139,92,246,0.12)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "0 1px 8px rgba(0,0,0,0.04)"; }}
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-xl">{c.icon}</span>
                <span className="inline-block px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide"
                  style={{ background: c.tag, color: c.tagText }}>
                  Tip
                </span>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed font-medium">{text}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
