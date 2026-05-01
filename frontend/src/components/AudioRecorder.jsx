/**
 * AudioRecorder Component
 * =======================
 * Provides two ways to supply audio:
 *   1. Drag-and-drop / click-to-upload a file (.wav, .mp3, etc.)
 *   2. Record directly from the microphone via the MediaRecorder API
 *
 * When audio is ready, it calls `onAudioReady(file)` so the parent
 * can send it to the backend.
 */

import { useState, useRef, useCallback, useEffect } from "react";
import "./AudioRecorder.css";

// Allowed file types for upload
const ALLOWED_TYPES = [
  "audio/wav",
  "audio/x-wav",
  "audio/mpeg",
  "audio/mp3",
  "audio/flac",
  "audio/ogg",
  "audio/mp4",
  "audio/m4a",
  "audio/x-m4a",
];

const ALLOWED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"];

export default function AudioRecorder({ onAudioReady, isLoading }) {
  // ── State ──────────────────────────────────────────────────────────────
  const [isDragging, setIsDragging] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [activeTab, setActiveTab] = useState("upload"); // "upload" | "record"

  // ── Refs ───────────────────────────────────────────────────────────────
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const streamRef = useRef(null);

  // ── Helpers ────────────────────────────────────────────────────────────

  /** Clean up object URLs to avoid memory leaks */
  const revokePreview = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  /** Validate file type */
  const isValidFile = (file) => {
    if (ALLOWED_TYPES.includes(file.type)) return true;
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    return ALLOWED_EXTENSIONS.includes(ext);
  };

  /** Process a picked / dropped file */
  const handleFile = useCallback(
    (file) => {
      if (!isValidFile(file)) {
        alert("Unsupported file format. Please upload a .wav, .mp3, .flac, .ogg, or .m4a file.");
        return;
      }
      revokePreview();
      setUploadedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      onAudioReady(file);
    },
    [onAudioReady, revokePreview]
  );

  // ── Drag-and-drop handlers ────────────────────────────────────────────
  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  // ── File input handler ────────────────────────────────────────────────
  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  // ── Recording handlers ────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        revokePreview();
        setUploadedFile(file);
        setPreviewUrl(URL.createObjectURL(blob));
        onAudioReady(file);
        // Stop all tracks
        streamRef.current?.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Timer for display
      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Microphone access is required for recording. Please allow it in your browser settings.");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
    clearInterval(timerRef.current);
  };

  // ── Cleanup on unmount ────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      revokePreview();
      clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ── Format seconds → mm:ss ────────────────────────────────────────────
  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="audio-recorder animate-fade-in-up">
      {/* Tab bar */}
      <div className="recorder-tabs">
        <button
          className={`tab-btn ${activeTab === "upload" ? "active" : ""}`}
          onClick={() => setActiveTab("upload")}
          disabled={isLoading || isRecording}
        >
          <span className="tab-icon">📁</span>
          Upload Audio
        </button>
        <button
          className={`tab-btn ${activeTab === "record" ? "active" : ""}`}
          onClick={() => setActiveTab("record")}
          disabled={isLoading}
        >
          <span className="tab-icon">🎙️</span>
          Record Audio
        </button>
      </div>

      {/* Upload panel */}
      {activeTab === "upload" && (
        <div
          className={`drop-zone ${isDragging ? "dragging" : ""} ${isLoading ? "disabled" : ""}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => !isLoading && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a"
            onChange={onFileChange}
            hidden
          />

          <div className="drop-zone-content">
            <div className="upload-icon">
              {isDragging ? "📥" : "🎵"}
            </div>
            <p className="drop-zone-title">
              {isDragging ? "Drop your audio file here" : "Drag & drop audio file"}
            </p>
            <p className="drop-zone-subtitle">
              or <span className="browse-link">browse files</span>
            </p>
            <p className="drop-zone-formats">
              Supported: WAV, MP3, FLAC, OGG, M4A
            </p>
          </div>
        </div>
      )}

      {/* Record panel */}
      {activeTab === "record" && (
        <div className="record-zone">
          <div className="record-visualizer">
            {isRecording && (
              <div className="wave-bars">
                {[...Array(20)].map((_, i) => (
                  <div
                    key={i}
                    className="wave-bar"
                    style={{ animationDelay: `${i * 0.05}s` }}
                  />
                ))}
              </div>
            )}
            {!isRecording && (
              <div className="mic-idle-icon">🎙️</div>
            )}
          </div>

          {isRecording && (
            <div className="recording-timer">
              <span className="rec-dot" />
              {formatTime(recordingTime)}
            </div>
          )}

          <button
            className={`record-btn ${isRecording ? "recording" : ""}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isLoading}
          >
            {isRecording ? (
              <>
                <span className="btn-icon">⏹️</span>
                Stop Recording
              </>
            ) : (
              <>
                <span className="btn-icon">⏺️</span>
                Start Recording
              </>
            )}
          </button>
        </div>
      )}

      {/* Audio preview */}
      {previewUrl && (
        <div className="audio-preview animate-fade-in-up">
          <div className="preview-header">
            <span className="preview-label">📎 {uploadedFile?.name || "Recording"}</span>
            <span className="preview-size">
              {uploadedFile && `${(uploadedFile.size / 1024).toFixed(1)} KB`}
            </span>
          </div>
          <audio controls src={previewUrl} className="audio-player" />
        </div>
      )}
    </div>
  );
}
