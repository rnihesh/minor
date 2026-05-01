/**
 * SuggestionCards Component
 * =========================
 * Renders a list of wellness / coping suggestions as animated cards.
 * Each card fades in with a stagger delay for a polished feel.
 */

import "./SuggestionCards.css";

/** Colour accents per suggestion index to keep things vibrant */
const CARD_ACCENTS = [
  "var(--accent-purple)",
  "var(--accent-cyan)",
  "var(--accent-pink)",
  "var(--accent-green)",
  "var(--accent-yellow)",
  "var(--accent-orange)",
];

export default function SuggestionCards({ suggestions, emotion }) {
  if (!suggestions || suggestions.length === 0) return null;

  return (
    <div className="suggestion-cards animate-fade-in-up">
      <h3 className="suggestions-title">
        <span className="suggestions-icon">💡</span>
        Wellness Suggestions
      </h3>
      <div className="cards-list">
        {suggestions.map((text, idx) => (
          <div
            key={idx}
            className="suggestion-card glass"
            style={{
              "--card-accent": CARD_ACCENTS[idx % CARD_ACCENTS.length],
              animationDelay: `${idx * 0.1}s`,
            }}
          >
            <div className="card-accent-bar" />
            <p className="card-text">{text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
