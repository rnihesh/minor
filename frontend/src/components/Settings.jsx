import { useState, useEffect } from "react";
import { Brain, Server, Database, CheckCircle2, Loader2 } from "lucide-react";

const API_BASE = "http://localhost:8000";

const PRESET_ICONS = { current: "🔀", consistent: "⚖️", tess: "🎯" };
const PRESET_COLORS = {
  current:    { bg: "#ede9fe", border: "#c4b5fd", accent: "#7c3aed", pill: "#ddd6fe" },
  consistent: { bg: "#dbeafe", border: "#93c5fd", accent: "#2563eb", pill: "#bfdbfe" },
  tess:       { bg: "#dcfce7", border: "#86efac", accent: "#059669", pill: "#bbf7d0" },
};

const Card = ({ children, className = "", style = {} }) => (
  <div className={`rounded-3xl ${className}`}
    style={{ background: "rgba(255,255,255,0.88)", backdropFilter: "blur(16px)", boxShadow: "0 2px 20px rgba(139,92,246,0.07)", ...style }}>
    {children}
  </div>
);

export default function Settings() {
  const [config,   setConfig]   = useState(null);
  const [saving,   setSaving]   = useState(false);
  const [saved,    setSaved]    = useState(false);
  const [pending,  setPending]  = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/config/model`)
      .then(r => r.json())
      .then(d => { setConfig(d); setPending(d.active); })
      .catch(() => {});
  }, []);

  const apply = async () => {
    if (!pending || pending === config?.active) return;
    setSaving(true);
    try {
      const res = await fetch(`${API_BASE}/config/model/${pending}`, { method: "POST" });
      const d = await res.json();
      setConfig(prev => ({ ...prev, active: d.active }));
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {}
    setSaving(false);
  };

  const presets = config?.presets ? Object.entries(config.presets) : [];
  const changed = pending && pending !== config?.active;

  return (
    <div className="p-6 h-full overflow-y-auto max-w-2xl">

      <div className="mb-7">
        <h1 className="text-2xl font-extrabold text-gray-900">Settings</h1>
        <p className="text-sm text-gray-400 font-medium mt-1">Model configuration &amp; system info</p>
      </div>

      {/* Model preset selector */}
      <Card className="p-5 mb-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm font-bold text-gray-900">Model Preset</p>
            <p className="text-xs text-gray-400 mt-0.5">Select which models to use for inference</p>
          </div>
          {saved && (
            <div className="flex items-center gap-1.5 text-emerald-600">
              <CheckCircle2 className="w-4 h-4" />
              <span className="text-xs font-bold">Applied</span>
            </div>
          )}
        </div>

        {config === null ? (
          <div className="flex items-center gap-2 text-gray-400 py-4">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Loading config…</span>
          </div>
        ) : (
          <div className="space-y-3">
            {presets.map(([key, preset]) => {
              const c = PRESET_COLORS[key] || PRESET_COLORS.current;
              const isActive = config.active === key;
              const isSelected = pending === key;
              return (
                <button key={key} onClick={() => setPending(key)} className="w-full text-left"
                  style={{
                    padding: "14px 16px",
                    borderRadius: "16px",
                    border: `2px solid ${isSelected ? c.border : "#e5e7eb"}`,
                    background: isSelected ? c.bg : "white",
                    transition: "all 0.2s",
                    cursor: "pointer",
                  }}>
                  <div className="flex items-start gap-3">
                    <span className="text-xl mt-0.5">{PRESET_ICONS[key] || "🤖"}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-bold text-gray-900">{preset.label}</span>
                        {isActive && (
                          <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                            style={{ background: c.pill, color: c.accent }}>
                            Active
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 mt-1 leading-relaxed">{preset.description}</p>
                      <div className="flex items-center gap-3 mt-2 flex-wrap">
                        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
                          {preset.accuracy}
                        </span>
                        {preset.lw_weight > 0 && (
                          <span className="text-xs text-gray-400">
                            LW {Math.round(preset.lw_weight * 100)}% · AT {Math.round(preset.at_weight * 100)}%
                          </span>
                        )}
                        {preset.lw_weight === 0 && (
                          <span className="text-xs text-gray-400">Attention only (100%)</span>
                        )}
                      </div>
                    </div>
                    <div className="w-5 h-5 rounded-full border-2 flex-shrink-0 mt-0.5 flex items-center justify-center"
                      style={{ borderColor: isSelected ? c.accent : "#d1d5db", background: isSelected ? c.accent : "white" }}>
                      {isSelected && <div className="w-2 h-2 rounded-full bg-white" />}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {changed && (
          <button onClick={apply} disabled={saving}
            className="mt-4 w-full py-3 rounded-2xl text-sm font-bold text-white transition-all disabled:opacity-50"
            style={{ background: "linear-gradient(135deg,#8b5cf6,#a78bfa)", boxShadow: "0 4px 14px rgba(139,92,246,0.3)" }}>
            {saving ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" /> Applying…
              </span>
            ) : "Apply Preset"}
          </button>
        )}
      </Card>

      {/* System info */}
      <Card className="p-5 mb-4">
        <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">System</p>
        <div className="space-y-2.5">
          {[
            { icon: Server,   label: "Backend API", value: "http://localhost:8000",              bg:"#dcfce7", color:"#059669" },
            { icon: Database, label: "Database",     value: "MongoDB → emotion_db → emotion_logs", bg:"#fef9c3", color:"#d97706" },
            { icon: Brain,    label: "Datasets",     value: "RAVDESS · CREMA-D · TESS · SAVEE",   bg:"#dbeafe", color:"#2563eb" },
          ].map(({ icon: Icon, label, value, bg, color }) => (
            <div key={label} className="flex items-center gap-3 p-3 rounded-2xl" style={{ background: bg + "55" }}>
              <div className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: bg }}>
                <Icon className="w-4 h-4" style={{ color }} />
              </div>
              <div className="min-w-0">
                <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">{label}</p>
                <p className="text-xs font-semibold text-gray-700 font-mono truncate">{value}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="p-5 rounded-3xl text-center"
        style={{ background:"linear-gradient(135deg,rgba(196,181,253,0.3),rgba(251,207,232,0.3))", border:"1px solid rgba(196,181,253,0.4)" }}>
        <p className="text-sm font-semibold text-gray-700 leading-relaxed italic">
          "An intelligent system that detects emotions from speech, tracks mood over time, and provides personalised suggestions."
        </p>
      </div>
    </div>
  );
}
