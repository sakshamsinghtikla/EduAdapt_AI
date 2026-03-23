import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Session from "./pages/Session";
import Dashboard from "./pages/Dashboard";

export default function App() {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="page-container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/session/:studentId" element={<Session />} />
          <Route path="/dashboard/:studentId" element={<Dashboard />} />
        </Routes>
      </main>
    </div>
  );
}
