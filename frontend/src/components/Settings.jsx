/**
 * Settings.jsx — Reference theme
 */
import { Brain, Server, Database, Cpu, Info } from "lucide-react";

const INFO_ROWS = [
  { label:"Lightweight Model",  value:"ser_lightweight_20260425_235238_random_stratified_best.keras", icon:Cpu,      bg:"#ede9fe", color:"#7c3aed" },
  { label:"Attention Model",    value:"ser_attention_20260425_235453_speaker_independent_best.keras",  icon:Brain,    bg:"#dbeafe", color:"#2563eb" },
  { label:"Backend API",        value:"http://localhost:8000",                                          icon:Server,   bg:"#dcfce7", color:"#059669" },
  { label:"Database",           value:"MongoDB → emotion_db → emotion_logs",                           icon:Database, bg:"#fef9c3", color:"#d97706" },
];

const Card = ({ children, className="" }) => (
  <div className={`rounded-3xl ${className}`}
    style={{ background:"rgba(255,255,255,0.88)", backdropFilter:"blur(16px)", boxShadow:"0 2px 20px rgba(139,92,246,0.07)" }}>
    {children}
  </div>
);

export default function Settings() {
  return (
    <div className="p-6 h-full overflow-y-auto max-w-2xl">

      <div className="mb-7">
        <h1 className="text-2xl font-extrabold text-gray-900">Settings</h1>
        <p className="text-sm text-gray-400 font-medium mt-1">System information &amp; configuration</p>
      </div>

      <div className="space-y-3">
        {INFO_ROWS.map(({ label, value, icon: Icon, bg, color }) => (
          <Card key={label} className="p-4 flex items-start gap-4">
            <div className="w-10 h-10 rounded-2xl flex items-center justify-center flex-shrink-0"
              style={{ background: bg }}>
              <Icon className="w-5 h-5" style={{ color }} />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-1">{label}</p>
              <p className="text-sm font-semibold text-gray-800 break-all font-mono">{value}</p>
            </div>
          </Card>
        ))}
      </div>

      {/* Weights info */}
      <Card className="p-5 mt-4">
        <div className="flex items-center gap-2 mb-3">
          <Info className="w-4 h-4" style={{ color:"#8b5cf6" }} />
          <p className="text-sm font-bold text-gray-900">Ensemble Weights</p>
        </div>
        <div className="space-y-2">
          {[["Lightweight Model", "35%", "#0891b2", "#cffafe"], ["Attention Model", "65%", "#7c3aed", "#ede9fe"]].map(([label, val, color, bg]) => (
            <div key={label} className="flex items-center justify-between p-3 rounded-2xl" style={{ background: bg }}>
              <span className="text-sm font-semibold" style={{ color }}>{label}</span>
              <span className="text-sm font-extrabold" style={{ color }}>{val}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Tagline */}
      <div className="mt-5 p-5 rounded-3xl text-center"
        style={{ background:"linear-gradient(135deg,rgba(196,181,253,0.3),rgba(251,207,232,0.3))", border:"1px solid rgba(196,181,253,0.4)" }}>
        <p className="text-sm font-semibold text-gray-700 leading-relaxed italic">
          "An intelligent system that detects emotions from speech, tracks mood over time, and provides personalized suggestions."
        </p>
      </div>
    </div>
  );
}
