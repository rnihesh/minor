/**
 * Dashboard.jsx — Reference theme: white cards, purple accents, sky-pink gradient bg
 */
import { useState, useRef, useCallback } from "react";
import { Mic, Square, Upload, Loader2, Wifi, Radio } from "lucide-react";
import EmotionCard    from "./EmotionCard";
import SuggestionCard from "./SuggestionCard";
import { useAuth } from "../context/AuthContext";

const API_BASE = "http://localhost:8000";
const WS_URL   = "ws://localhost:8000/ws/emotion-stream";

const EMOJIS = { neutral:"😐",calm:"😌",happy:"😄",sad:"😢",angry:"😠",fearful:"😨",disgust:"🤢",surprised:"😲" };

const BANNER_MESSAGES = {
  neutral:"You seem balanced today.", calm:"You're calm — a great state for deep work.",
  happy:"You're feeling happy! Spread the positivity 🎉", sad:"You seem a bit low. Try some gentle breathing.",
  angry:"Take a deep breath — give yourself a moment.", fearful:"It's okay. Ground yourself, you're safe.",
  disgust:"Something seems off. Step back and breathe.", surprised:"Whoa! Embrace this moment 🌟",
};

const BANNER_COLORS = {
  neutral:"#64748b", calm:"#0891b2", happy:"#d97706",
  sad:"#2563eb", angry:"#e11d48", fearful:"#7c3aed",
  disgust:"#059669", surprised:"#ea580c",
};

/* White frosted card wrapper */
const Card = ({ children, className = "", style = {} }) => (
  <div className={`rounded-3xl ${className}`}
    style={{ background: "rgba(255,255,255,0.85)", backdropFilter: "blur(16px)", boxShadow: "0 2px 20px rgba(139,92,246,0.08)", ...style }}>
    {children}
  </div>
);

/* Animated waveform */
function WaveformBars({ active }) {
  return (
    <div className="flex items-center justify-center gap-1 h-12">
      {Array.from({ length: 20 }).map((_, i) => (
        <div key={i} className="w-1 rounded-full transition-all duration-200"
          style={{
            background: active ? `hsl(${260 + i * 4},80%,65%)` : "#e5e7eb",
            height: active ? `${8 + Math.sin(i * 0.6) * 18 + 6}px` : "5px",
            animation: active ? `wave ${0.35 + (i % 5) * 0.07}s ease-in-out infinite alternate` : "none",
            animationDelay: `${i * 0.04}s`,
          }}
        />
      ))}
    </div>
  );
}

/* Pill button */
const PillBtn = ({ onClick, disabled, children, variant = "primary" }) => {
  const base = "flex items-center justify-center gap-2 w-full py-3.5 rounded-2xl text-sm font-semibold transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed";
  const styles = {
    primary: { background: "#111827", color: "white", boxShadow: "0 4px 14px rgba(17,24,39,0.2)" },
    outline: { background: "white", color: "#374151", border: "1.5px solid #e5e7eb" },
    purple:  { background: "linear-gradient(135deg,#8b5cf6,#a78bfa)", color: "white", boxShadow: "0 4px 14px rgba(139,92,246,0.3)" },
    danger:  { background: "#fff1f2", color: "#e11d48", border: "1.5px solid #fecdd3" },
    green:   { background: "#f0fdf4", color: "#15803d", border: "1.5px solid #bbf7d0" },
  };
  return (
    <button onClick={onClick} disabled={disabled} className={base} style={styles[variant]}
      onMouseEnter={e => { if (!disabled && variant === "outline") e.currentTarget.style.borderColor = "#c4b5fd"; }}
      onMouseLeave={e => { if (!disabled && variant === "outline") e.currentTarget.style.borderColor = "#e5e7eb"; }}>
      {children}
    </button>
  );
};

export default function Dashboard() {
  const { user } = useAuth();
  const [result,        setResult]        = useState(null);
  const [isLoading,     setIsLoading]     = useState(false);
  const [error,         setError]         = useState(null);
  const [isRecording,   setIsRecording]   = useState(false);
  const [recTime,       setRecTime]       = useState(0);
  const [isStreaming,   setIsStreaming]    = useState(false);
  const [streamEmotion, setStreamEmotion] = useState(null);

  const mediaRecRef  = useRef(null);
  const streamRef    = useRef(null);
  const timerRef     = useRef(null);
  const wsRef        = useRef(null);
  const fileInputRef = useRef(null);
  const fmt = s => `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;

  const handleFile = useCallback(async (file) => {
    setIsLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData(); fd.append("file", file);
      const headers = user?.token ? { Authorization: `Bearer ${user.token}` } : {};
      const res = await fetch(`${API_BASE}/predict-emotion`, { 
        method: "POST", 
        body: fd,
        headers
      });
      if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.detail || `Error ${res.status}`); }
      setResult(await res.json());
    } catch(e) { setError(e.message); }
    finally    { setIsLoading(false); }
  }, [user]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const chunks = [];
      const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecRef.current = rec;
      rec.ondataavailable = e => e.data.size > 0 && chunks.push(e.data);
      rec.onstop = () => {
        handleFile(new File([new Blob(chunks,{type:"audio/webm"})],"rec.webm",{type:"audio/webm"}));
        stream.getTracks().forEach(t=>t.stop());
      };
      rec.start(); setIsRecording(true); setRecTime(0);
      timerRef.current = setInterval(()=>setRecTime(t=>t+1),1000);
    } catch { setError("Microphone access denied."); }
  };

  const stopRecording = () => { mediaRecRef.current?.stop(); setIsRecording(false); clearInterval(timerRef.current); };

  const startStream = async () => {
    setError(null);
    try {
      const tokenQuery = user?.token ? `?token=${user.token}` : "";
      const ws = new WebSocket(WS_URL + tokenQuery); wsRef.current = ws;
      ws.onmessage = e => setStreamEmotion(JSON.parse(e.data));
      ws.onerror = () => setError("WebSocket error — is backend running?");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const rec = new MediaRecorder(stream,{mimeType:"audio/webm"});
      rec.ondataavailable = e => { if(e.data.size>0 && ws.readyState===1) ws.send(e.data); };
      rec.start(3000); mediaRecRef.current = rec; setIsStreaming(true);
    } catch { setError("Could not start stream."); }
  };

  const stopStream = () => {
    mediaRecRef.current?.stop(); streamRef.current?.getTracks().forEach(t=>t.stop());
    wsRef.current?.close(); setIsStreaming(false);
  };

  const activeResult = isStreaming && streamEmotion
    ? { emotion:streamEmotion.emotion, confidence:streamEmotion.confidence, all_scores:streamEmotion.all_scores||{}, suggestions:[] }
    : result;

  const emotion    = activeResult?.emotion;
  const accentColor = emotion ? (BANNER_COLORS[emotion] || "#8b5cf6") : "#8b5cf6";
  const bannerMsg  = emotion ? BANNER_MESSAGES[emotion] : "How's your mood right now?";

  return (
    <div className="flex flex-col h-screen overflow-hidden">

      {/* TOP BANNER */}
      <div className="px-8 py-4 flex items-center gap-4 transition-all duration-700"
        style={{ background: "rgba(255,255,255,0.65)", backdropFilter: "blur(12px)", borderBottom: "1px solid rgba(196,181,253,0.25)" }}>
        <div className="w-10 h-10 rounded-full flex items-center justify-center text-2xl shadow-sm"
          style={{ background: "rgba(255,255,255,0.9)" }}>
          {emotion ? EMOJIS[emotion] : "🧠"}
        </div>
        <div className="flex-1">
          <p className="text-base font-bold text-gray-900">{bannerMsg}</p>
          {isStreaming && (
            <div className="flex items-center gap-1.5 mt-0.5">
              <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
              <span className="text-xs font-medium text-gray-500">Live stream active</span>
            </div>
          )}
        </div>
        {emotion && (
          <span className="px-4 py-1.5 rounded-full text-xs font-bold text-white"
            style={{ background: accentColor }}>
            {(activeResult?.confidence * 100).toFixed(0)}% confidence
          </span>
        )}
      </div>

      {/* THREE-PANEL BODY */}
      <div className="flex flex-1 overflow-hidden gap-0">

        {/* ═══ LEFT — Audio Input ═══ */}
        <div className="w-72 flex-shrink-0 p-5 overflow-y-auto space-y-4"
          style={{ background: "rgba(255,255,255,0.4)", borderRight: "1px solid rgba(196,181,253,0.25)" }}>

          <p className="text-xs font-bold uppercase tracking-widest text-gray-400 px-1">Audio Input</p>

          {/* Waveform */}
          <Card className="p-4">
            <WaveformBars active={isRecording || isStreaming} />
            {isRecording && (
              <div className="flex items-center justify-center gap-2 mt-2">
                <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: "#8b5cf6" }} />
                <span className="text-sm font-mono font-bold" style={{ color: "#7c3aed" }}>{fmt(recTime)}</span>
              </div>
            )}
          </Card>

          <PillBtn onClick={isRecording ? stopRecording : startRecording}
            disabled={isLoading || isStreaming}
            variant={isRecording ? "danger" : "primary"}>
            {isRecording ? <Square className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            {isRecording ? `Stop · ${fmt(recTime)}` : "Record Audio"}
          </PillBtn>

          <PillBtn onClick={() => fileInputRef.current?.click()}
            disabled={isLoading || isRecording || isStreaming}
            variant="outline">
            <Upload className="w-4 h-4" />
            Upload Audio File
          </PillBtn>
          <input ref={fileInputRef} type="file" accept=".wav,.mp3,.flac,.ogg,.m4a,.webm" hidden
            onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />

          {/* Live stream section */}
          <div className="pt-2">
            <div className="flex items-center gap-2 mb-3">
              <Radio className="w-3.5 h-3.5" style={{ color: "#8b5cf6" }} />
              <p className="text-xs font-bold uppercase tracking-widest text-gray-400">Live Stream</p>
            </div>
            <PillBtn onClick={isStreaming ? stopStream : startStream}
              disabled={isLoading || isRecording}
              variant={isStreaming ? "danger" : "purple"}>
              <div className={`w-2.5 h-2.5 rounded-full ${isStreaming ? "bg-red-400 animate-pulse" : "bg-white"}`} />
              {isStreaming ? "Stop Stream" : "Start Real-time"}
            </PillBtn>

            {streamEmotion && (
              <Card className="mt-3 p-3 text-center">
                <p className="text-xs text-gray-400 mb-1">Latest result</p>
                <p className="text-lg font-bold capitalize text-gray-900">{streamEmotion.emotion}</p>
                <p className="text-xs font-semibold" style={{ color: "#8b5cf6" }}>
                  {(streamEmotion.confidence * 100).toFixed(1)}%
                </p>
              </Card>
            )}
          </div>

          {error && (
            <div className="p-3 rounded-2xl text-sm font-medium"
              style={{ background: "#fff1f2", border: "1px solid #fecdd3", color: "#e11d48" }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* ═══ CENTER — Emotion Result ═══ */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-full gap-5">
              <div className="w-16 h-16 rounded-full flex items-center justify-center"
                style={{ background: "rgba(255,255,255,0.8)", boxShadow: "0 4px 20px rgba(139,92,246,0.2)" }}>
                <Loader2 className="w-8 h-8 animate-spin" style={{ color: "#8b5cf6" }} />
              </div>
              <p className="text-base font-bold text-gray-800">Analysing your audio…</p>
              <p className="text-sm text-gray-500">Running dual-model ensemble inference</p>
            </div>
          ) : activeResult ? (
            <EmotionCard result={activeResult} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
              <div className="w-28 h-28 rounded-full flex items-center justify-center text-5xl animate-float"
                style={{ background: "rgba(255,255,255,0.8)", boxShadow: "0 8px 32px rgba(139,92,246,0.15)" }}>
                🎙️
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">How's your mood right now?</h3>
                <p className="text-sm text-gray-500 max-w-xs leading-relaxed">
                  Please take some time to reflect — record your voice or upload audio to assess your emotion.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* ═══ RIGHT — Suggestions ═══ */}
        <div className="w-72 flex-shrink-0 p-5 overflow-y-auto"
          style={{ background: "rgba(255,255,255,0.4)", borderLeft: "1px solid rgba(196,181,253,0.25)" }}>
          {activeResult?.suggestions?.length > 0 ? (
            <SuggestionCard suggestions={activeResult.suggestions} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center gap-4">
              <div className="w-16 h-16 rounded-full flex items-center justify-center text-3xl"
                style={{ background: "rgba(255,255,255,0.8)" }}>💡</div>
              <p className="text-sm text-gray-400 font-medium leading-relaxed">
                Personalised wellness suggestions will appear here after detection.
              </p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
