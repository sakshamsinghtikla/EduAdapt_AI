import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const [studentId, setStudentId] = useState("s000");
  const navigate = useNavigate();

  const handleStart = () => {
    if (!studentId.trim()) return;
    navigate(`/session/${studentId.trim()}`);
  };

  return (
    <div className="card">
      <h2>Start Adaptive Session</h2>
      <p>
        Enter a seeded student ID to begin an adaptive assessment session.
      </p>

      <div className="input-row">
        <input
          type="text"
          value={studentId}
          onChange={(e) => setStudentId(e.target.value)}
          placeholder="e.g. s000"
        />
        <button onClick={handleStart}>Start Session</button>
      </div>

      <p className="small-text" style={{ marginTop: 16 }}>
        Example student IDs: s000, s001, s002
      </p>
    </div>
  );
}
