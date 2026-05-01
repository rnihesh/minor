/**
 * WaveformVisualizer Component
 * ============================
 * Uses wavesurfer.js to render a real-time waveform from an audio Blob URL.
 * Includes play/pause controls and a current-time display.
 */

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import "./WaveformVisualizer.css";

export default function WaveformVisualizer({ audioUrl, emotionColor }) {
  const containerRef = useRef(null);
  const wavesurferRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!audioUrl || !containerRef.current) return;

    // Destroy previous instance
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
    }

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(139, 92, 246, 0.35)",
      progressColor: emotionColor || "#8b5cf6",
      cursorColor: "rgba(255,255,255,0.5)",
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      height: 70,
      responsive: true,
      normalize: true,
      backend: "WebAudio",
    });

    ws.load(audioUrl);

    ws.on("ready", () => setDuration(ws.getDuration()));
    ws.on("audioprocess", () => setCurrentTime(ws.getCurrentTime()));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));

    wavesurferRef.current = ws;

    return () => {
      ws.destroy();
    };
  }, [audioUrl, emotionColor]);

  const togglePlay = () => {
    wavesurferRef.current?.playPause();
  };

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  if (!audioUrl) return null;

  return (
    <div className="waveform-visualizer glass animate-fade-in-up">
      <div className="waveform-header">
        <button className="waveform-play-btn" onClick={togglePlay}>
          {isPlaying ? "⏸️" : "▶️"}
        </button>
        <span className="waveform-time">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
      <div ref={containerRef} className="waveform-container" />
    </div>
  );
}
