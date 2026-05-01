/**
 * EmotionStreamHandler Component
 * ==============================
 * Handles real-time audio recording via MediaRecorder, sending 3-second chunks
 * to the FastAPI WebSocket endpoint, and rendering the live emotion graph.
 */

import { useState, useRef, useEffect, useCallback } from "react";
import LiveEmotionGraph from "./LiveEmotionGraph";
import "./EmotionStreamHandler.css";

const WS_URL = "ws://localhost:8000/ws/emotion-stream";

export default function EmotionStreamHandler() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [emotionHistory, setEmotionHistory] = useState([]);
  const [latestEmotion, setLatestEmotion] = useState(null);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  const startStream = async () => {
    setError(null);
    try {
      // Connect to WebSocket
      wsRef.current = new WebSocket(WS_URL);

      wsRef.current.onopen = () => {
        console.log("WebSocket connected");
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setLatestEmotion(data);
        setEmotionHistory((prev) => [...prev, data]);
      };

      wsRef.current.onerror = (err) => {
        console.error("WebSocket error:", err);
        setError("WebSocket connection failed.");
      };

      wsRef.current.onclose = () => {
        console.log("WebSocket disconnected");
      };

      // Get audio stream
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(e.data);
        }
      };

      // Start recording and emit chunks every 3 seconds
      recorder.start(3000);
      setIsStreaming(true);

    } catch (err) {
      console.error("Error starting stream:", err);
      setError("Failed to access microphone or connect to server.");
    }
  };

  const stopStream = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsStreaming(false);
  }, []);

  useEffect(() => {
    return () => stopStream();
  }, [stopStream]);

  return (
    <div className="emotion-stream-handler glass animate-fade-in-up">
      <div className="stream-header">
        <h3 className="stream-title">🎙️ Live Emotion Stream</h3>
        <p className="stream-subtitle">Record and analyze emotions in real-time</p>
      </div>

      <div className="stream-controls">
        <button
          className={`stream-btn ${isStreaming ? "streaming" : ""}`}
          onClick={isStreaming ? stopStream : startStream}
        >
          {isStreaming ? "⏹️ Stop Stream" : "⏺️ Start Real-time Stream"}
        </button>
      </div>

      {error && <p className="stream-error">{error}</p>}

      {latestEmotion && (
        <div className="latest-emotion">
          <span className="live-badge">LIVE</span>
          Detected: <strong>{latestEmotion.emotion}</strong> ({(latestEmotion.confidence * 100).toFixed(1)}%)
        </div>
      )}

      {emotionHistory.length > 0 && (
        <div className="stream-graph">
          <LiveEmotionGraph data={emotionHistory} />
        </div>
      )}
    </div>
  );
}
