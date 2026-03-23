import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import {
  getStudentState,
  getMetrics,
  getRecommendationMetrics,
  getModelComparison,
} from "../services/api";

export default function Dashboard() {
  const { studentId } = useParams();

  const [studentState, setStudentState] = useState(null);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [recommendationMetrics, setRecommendationMetrics] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadDashboard() {
      try {
        const [student, metrics, recMetrics, comparison] = await Promise.all([
          getStudentState(studentId),
          getMetrics(),
          getRecommendationMetrics().catch(() => ({})),
          getModelComparison().catch(() => ({})),
        ]);

        if (!active) return;
        setStudentState(student);
        setSystemMetrics(metrics);
        setRecommendationMetrics(recMetrics);
        setModelComparison(comparison);
      } catch (err) {
        if (!active) return;
        setError(err.message);
      }
    }

    loadDashboard();
    return () => {
      active = false;
    };
  }, [studentId]);

  return (
    <>
      {error && <div className="card error">{error}</div>}

      <div className="card">
        <h2>Dashboard: {studentId}</h2>
      </div>

      {studentState && (
        <div className="card">
          <h3>Student Mastery</h3>
          <div className="metric-grid">
            {Object.entries(studentState.mastery || {}).map(([concept, value]) => (
              <div className="metric-box" key={concept}>
                <strong>{concept}</strong>
                <div>{Number(value).toFixed(4)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {systemMetrics && (
        <div className="card">
          <h3>System Metrics</h3>
          <pre>{JSON.stringify(systemMetrics, null, 2)}</pre>
        </div>
      )}

      {recommendationMetrics && Object.keys(recommendationMetrics).length > 0 && (
        <div className="card">
          <h3>Recommendation Metrics</h3>
          <pre>{JSON.stringify(recommendationMetrics, null, 2)}</pre>
        </div>
      )}

      {modelComparison && Object.keys(modelComparison).length > 0 && (
        <div className="card">
          <h3>Model Comparison</h3>
          <pre>{JSON.stringify(modelComparison, null, 2)}</pre>
        </div>
      )}
    </>
  );
}
