/**
 * Sidebar — Matches reference: white glass pill nav, purple accents
 */
import { Link, useLocation, useNavigate } from "react-router-dom";
import { LayoutDashboard, BarChart3, Clock, Settings, Brain, LogOut } from "lucide-react";
import { useAuth } from "../context/AuthContext";

const NAV_ITEMS = [
  { path: "/",          label: "Dashboard", icon: LayoutDashboard },
  { path: "/analytics", label: "Analytics", icon: BarChart3 },
  { path: "/history",   label: "History",   icon: Clock },
  { path: "/settings",  label: "Settings",  icon: Settings },
];

export default function Sidebar() {
  const { pathname } = useLocation();
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 flex flex-col z-50"
      style={{ background: "rgba(255,255,255,0.72)", backdropFilter: "blur(24px)", borderRight: "1px solid rgba(196,181,253,0.3)" }}>

      {/* Brand */}
      <div className="flex items-center gap-3 px-6 py-7">
        <div className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-md"
          style={{ background: "linear-gradient(135deg,#8b5cf6,#a78bfa)" }}>
          <Brain className="w-5 h-5 text-white" />
        </div>
        <div>
          <p className="text-base font-bold text-gray-900 leading-tight">Emotion AI</p>
          <p className="text-xs font-medium" style={{ color: "#8b5cf6" }}>Speech Recognition</p>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-5 h-px" style={{ background: "rgba(196,181,253,0.3)" }} />

      {/* Nav */}
      <nav className="flex-1 px-4 py-6 space-y-1.5">
        {NAV_ITEMS.map(({ path, label, icon: Icon }) => {
          const active = pathname === path;
          return (
            <Link key={path} to={path}
              className="flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-300"
              style={active
                ? { background: "#111827", color: "white", boxShadow: "0 4px 14px rgba(17,24,39,0.25)" }
                : { color: "#6b7280" }
              }
              onMouseEnter={e => { if (!active) { e.currentTarget.style.background = "rgba(139,92,246,0.08)"; e.currentTarget.style.color = "#7c3aed"; }}}
              onMouseLeave={e => { if (!active) { e.currentTarget.style.background = ""; e.currentTarget.style.color = "#6b7280"; }}}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* User + Logout */}
      <div className="px-4 pb-6 space-y-3">
        {user && (
          <div className="flex items-center gap-3 px-4 py-3 rounded-2xl"
            style={{ background: "rgba(139,92,246,0.06)", border: "1px solid rgba(196,181,253,0.3)" }}>
            {/* Avatar */}
            <div className="w-8 h-8 rounded-xl flex items-center justify-center text-white text-xs font-bold flex-shrink-0"
              style={{ background: "linear-gradient(135deg,#8b5cf6,#a78bfa)" }}>
              {user.name?.[0]?.toUpperCase() || "U"}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-gray-900 truncate">{user.name}</p>
              <p className="text-xs text-gray-400 truncate">{user.email}</p>
            </div>
          </div>
        )}

        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-200"
          style={{ color: "#ef4444" }}
          onMouseEnter={e => { e.currentTarget.style.background = "rgba(239,68,68,0.08)"; }}
          onMouseLeave={e => { e.currentTarget.style.background = ""; }}
        >
          <LogOut className="w-4 h-4" />
          Sign Out
        </button>
      </div>
    </aside>
  );
}

