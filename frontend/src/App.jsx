import { Routes, Route, useLocation } from "react-router-dom";
import Sidebar      from "./components/Sidebar";
import Dashboard    from "./components/Dashboard";
import Analytics    from "./components/Analytics";
import History      from "./components/History";
import Settings     from "./components/Settings";
import Login        from "./components/Login";
import ProtectedRoute from "./components/ProtectedRoute";

export default function App() {
  const { pathname } = useLocation();
  const isAuth = pathname === "/login";

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {!isAuth && <Sidebar />}
      <main className={!isAuth ? "ml-64 flex-1 overflow-hidden" : "flex-1 overflow-hidden"}>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<Login />} />

          {/* Protected */}
          <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/analytics" element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
          <Route path="/history" element={<ProtectedRoute><History /></ProtectedRoute>} />
          <Route path="/settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
        </Routes>
      </main>
    </div>
  );
}

