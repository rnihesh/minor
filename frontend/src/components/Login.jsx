/**
 * Login.jsx — Combined Login / Register page
 * Matches the app's glassmorphism + purple accent design system
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Brain, Eye, EyeOff, Loader2, AlertCircle } from "lucide-react";

export default function Login() {
  const [tab, setTab] = useState("login"); // "login" | "register"
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handle = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (tab === "login") {
        await login(form.email, form.password);
      } else {
        if (!form.name.trim()) throw new Error("Name is required");
        await register(form.name, form.email, form.password);
      }
      navigate("/");
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const switchTab = (t) => {
    setTab(t);
    setError("");
    setForm({ name: "", email: "", password: "" });
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center px-4"
      style={{
        background:
          "linear-gradient(135deg, #c8eaf5 0%, #ddd6fe 40%, #f9c4d4 80%, #fde8f0 100%)",
      }}
    >
      {/* Decorative blobs */}
      <div
        style={{
          position: "fixed", top: "-80px", left: "-80px",
          width: "340px", height: "340px", borderRadius: "50%",
          background: "rgba(139,92,246,0.18)", filter: "blur(60px)", zIndex: 0,
        }}
      />
      <div
        style={{
          position: "fixed", bottom: "-80px", right: "-60px",
          width: "280px", height: "280px", borderRadius: "50%",
          background: "rgba(249,196,212,0.35)", filter: "blur(50px)", zIndex: 0,
        }}
      />

      {/* Card */}
      <div
        className="relative w-full max-w-md animate-slide-up"
        style={{
          background: "rgba(255,255,255,0.78)",
          backdropFilter: "blur(28px)",
          borderRadius: "28px",
          border: "1px solid rgba(196,181,253,0.35)",
          boxShadow: "0 24px 64px rgba(139,92,246,0.14)",
          padding: "40px 36px",
          zIndex: 1,
        }}
      >
        {/* Brand */}
        <div className="flex flex-col items-center mb-8">
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg mb-4"
            style={{ background: "linear-gradient(135deg,#8b5cf6,#a78bfa)" }}
          >
            <Brain className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">Emotion AI</h1>
          <p className="text-sm font-medium mt-0.5" style={{ color: "#8b5cf6" }}>
            Speech Emotion Recognition
          </p>
        </div>

        {/* Tab Toggle */}
        <div
          className="flex rounded-2xl p-1 mb-8"
          style={{ background: "rgba(139,92,246,0.08)" }}
        >
          {["login", "register"].map((t) => (
            <button
              key={t}
              onClick={() => switchTab(t)}
              className="flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all duration-300"
              style={
                tab === t
                  ? {
                      background: "#111827",
                      color: "white",
                      boxShadow: "0 4px 14px rgba(17,24,39,0.22)",
                    }
                  : { color: "#6b7280" }
              }
            >
              {t === "login" ? "Sign In" : "Sign Up"}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={submit} className="space-y-4">
          {tab === "register" && (
            <div>
              <label className="block text-xs font-semibold text-gray-600 mb-1.5 ml-1">
                Full Name
              </label>
              <input
                type="text"
                name="name"
                value={form.name}
                onChange={handle}
                placeholder="John Doe"
                required
                className="w-full px-4 py-3 text-sm font-medium text-gray-900 outline-none transition-all duration-200"
                style={{
                  background: "rgba(139,92,246,0.06)",
                  border: "1.5px solid rgba(196,181,253,0.5)",
                  borderRadius: "14px",
                }}
                onFocus={(e) => (e.target.style.borderColor = "#8b5cf6")}
                onBlur={(e) => (e.target.style.borderColor = "rgba(196,181,253,0.5)")}
              />
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold text-gray-600 mb-1.5 ml-1">
              Email Address
            </label>
            <input
              type="email"
              name="email"
              value={form.email}
              onChange={handle}
              placeholder="you@example.com"
              required
              className="w-full px-4 py-3 text-sm font-medium text-gray-900 outline-none transition-all duration-200"
              style={{
                background: "rgba(139,92,246,0.06)",
                border: "1.5px solid rgba(196,181,253,0.5)",
                borderRadius: "14px",
              }}
              onFocus={(e) => (e.target.style.borderColor = "#8b5cf6")}
              onBlur={(e) => (e.target.style.borderColor = "rgba(196,181,253,0.5)")}
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-600 mb-1.5 ml-1">
              Password
            </label>
            <div className="relative">
              <input
                type={showPass ? "text" : "password"}
                name="password"
                value={form.password}
                onChange={handle}
                placeholder="••••••••"
                required
                className="w-full px-4 py-3 pr-12 text-sm font-medium text-gray-900 outline-none transition-all duration-200"
                style={{
                  background: "rgba(139,92,246,0.06)",
                  border: "1.5px solid rgba(196,181,253,0.5)",
                  borderRadius: "14px",
                }}
                onFocus={(e) => (e.target.style.borderColor = "#8b5cf6")}
                onBlur={(e) => (e.target.style.borderColor = "rgba(196,181,253,0.5)")}
              />
              <button
                type="button"
                onClick={() => setShowPass(!showPass)}
                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
              >
                {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div
              className="flex items-center gap-2 px-4 py-3 rounded-2xl text-sm font-medium animate-fade-in"
              style={{
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.25)",
                color: "#dc2626",
              }}
            >
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-3.5 text-sm font-bold text-white rounded-2xl transition-all duration-300 flex items-center justify-center gap-2 mt-2"
            style={{
              background: loading
                ? "rgba(139,92,246,0.6)"
                : "linear-gradient(135deg,#8b5cf6,#7c3aed)",
              boxShadow: loading ? "none" : "0 8px 24px rgba(139,92,246,0.35)",
            }}
            onMouseEnter={(e) => {
              if (!loading) e.currentTarget.style.transform = "translateY(-1px)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
            }}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                {tab === "login" ? "Signing in…" : "Creating account…"}
              </>
            ) : tab === "login" ? (
              "Sign In"
            ) : (
              "Create Account"
            )}
          </button>
        </form>

        {/* Footer switch */}
        <p className="text-center text-xs text-gray-500 mt-6">
          {tab === "login" ? (
            <>
              Don't have an account?{" "}
              <button
                onClick={() => switchTab("register")}
                className="font-semibold"
                style={{ color: "#8b5cf6" }}
              >
                Sign up free
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                onClick={() => switchTab("login")}
                className="font-semibold"
                style={{ color: "#8b5cf6" }}
              >
                Sign in
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
