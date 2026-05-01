import { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext(null);

const API = "http://localhost:8000";

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);        // { name, email, token }
  const [loading, setLoading] = useState(true);

  // Restore session from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem("ser_user");
    if (stored) {
      try { setUser(JSON.parse(stored)); }
      catch { localStorage.removeItem("ser_user"); }
    }
    setLoading(false);
  }, []);

  async function register(name, email, password) {
    const res = await fetch(`${API}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Registration failed");
    const userObj = { name: data.name, email: data.email, token: data.access_token };
    setUser(userObj);
    localStorage.setItem("ser_user", JSON.stringify(userObj));
    return userObj;
  }

  async function login(email, password) {
    const res = await fetch(`${API}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed");
    const userObj = { name: data.name, email: data.email, token: data.access_token };
    setUser(userObj);
    localStorage.setItem("ser_user", JSON.stringify(userObj));
    return userObj;
  }

  function logout() {
    setUser(null);
    localStorage.removeItem("ser_user");
  }

  return (
    <AuthContext.Provider value={{ user, loading, register, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
