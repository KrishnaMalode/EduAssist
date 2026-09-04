import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { ToastProvider, useToast } from "./hooks/useToast";
import Navbar from "./components/Navbar";
import ProtectedRoute from "./components/ProtectedRoute";
import Toast from "./components/Toast";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import NCERT from "./pages/NCERT";
import TestConfig from "./pages/TestConfig";
import TestTake from "./pages/TestTake";
import TestResults from "./pages/TestResults";
import Chat from "./pages/Chat";
import Leaderboard from "./pages/Leaderboard";
import Admin from "./pages/Admin";

function Shell({ children }) {
  const location = useLocation();
  const hideNav = location.pathname === "/login" || location.pathname === "/register";
  return (
    <div className="min-h-screen">
      {!hideNav ? <Navbar /> : null}
      {children}
    </div>
  );
}

function ToastHost() {
  const { toasts, removeToast } = useToast();
  return <Toast toasts={toasts} onClose={removeToast} />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/ncert"
        element={
          <ProtectedRoute>
            <NCERT />
          </ProtectedRoute>
        }
      />
      <Route
        path="/test"
        element={
          <ProtectedRoute>
            <TestConfig />
          </ProtectedRoute>
        }
      />
      <Route
        path="/test/:test_id"
        element={
          <ProtectedRoute>
            <TestTake />
          </ProtectedRoute>
        }
      />
      <Route
        path="/test-results"
        element={
          <ProtectedRoute>
            <TestResults />
          </ProtectedRoute>
        }
      />
      <Route
        path="/chat"
        element={
          <ProtectedRoute>
            <Chat />
          </ProtectedRoute>
        }
      />
      <Route
        path="/leaderboard"
        element={
          <ProtectedRoute>
            <Leaderboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin"
        element={
          <ProtectedRoute adminOnly>
            <Admin />
          </ProtectedRoute>
        }
      />
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route
        path="*"
        element={
          <div className="flex min-h-screen flex-col items-center justify-center gap-4 p-6 text-center">
            <div className="text-4xl font-semibold">Lost in the void 🌌</div>
            <button
              onClick={() => (window.location.href = "/dashboard")}
              className="rounded-full bg-slate-900 px-6 py-2 text-sm font-semibold text-white"
            >
              Back to Dashboard
            </button>
          </div>
        }
      />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <BrowserRouter>
          <Shell>
            <AppRoutes />
          </Shell>
        </BrowserRouter>
        <ToastHost />
      </ToastProvider>
    </AuthProvider>
  );
}
