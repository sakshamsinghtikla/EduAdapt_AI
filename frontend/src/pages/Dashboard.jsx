import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import {
  getStudentState,
  getMetrics,
  getRecommendationMetrics,
  getModelComparison,
  getBenchmarkSummary,
} from "../services/api";
import MetricCard from "../components/MetricCard";
import BarMeter from "../components/BarMeter";

export default function Dashboard() {
  const { studentId } = useParams();

  const [studentState, setStudentState] = useState(null);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [recommendationMetrics, setRecommendationMetrics] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [benchmarkSummary, setBenchmarkSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadDashboard() {
      try {
        const [student, metrics, recMetrics, comparison, benchmark] = await Promise.all([
          getStudentState(studentId),
          getMetrics(),
          getRecommendationMetrics().catch(() => ({})),
          getModelComparison().catch(() => ({})),
          getBenchmarkSummary().catch(() => ({})),
        ]);

        if (!active) return;
        setStudentState(student);
        setSystemMetrics(metrics);
        setRecommendationMetrics(recMetrics);
        setModelComparison(comparison);
        setBenchmarkSummary(benchmark);
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
        <p className="small-text">
          Student state, model evaluation, recommendation quality, and system performance.
        </p>
      </div>

      {studentState && (
        <div className="card">
          <h3>Student Mastery</h3>
          <div className="metric-grid">
            {Object.entries(studentState.mastery || {}).map(([concept, value]) => (
              <MetricCard
                key={concept}
                title={concept}
                value={Number(value).toFixed(4)}
              />
            ))}
          </div>
        </div>
      )}

      {systemMetrics && (
        <div className="card">
          <h3>System Metrics</h3>
          <div className="metric-grid">
            {Object.entries(systemMetrics).map(([key, value]) => (
              <MetricCard key={key} title={key} value={String(value)} />
            ))}
          </div>
        </div>
      )}

      {recommendationMetrics && Object.keys(recommendationMetrics).length > 0 && (
        <div className="card">
          <h3>Recommendation Quality</h3>
          <div className="metric-grid">
            <MetricCard
              title="Target Zone Rate"
              value={recommendationMetrics.target_zone_rate}
            />
            <MetricCard
              title="Repetition Rate"
              value={recommendationMetrics.repetition_rate}
            />
            <MetricCard
              title="Avg Predicted Correctness"
              value={recommendationMetrics.avg_predicted_correctness}
            />
            <MetricCard
              title="Concept Diversity"
              value={recommendationMetrics.concept_diversity_normalized}
            />
          </div>

          <div style={{ marginTop: 20 }}>
            <BarMeter label="Target Zone Rate" value={recommendationMetrics.target_zone_rate} />
            <BarMeter label="Concept Diversity" value={recommendationMetrics.concept_diversity_normalized} />
            <BarMeter label="Avg Predicted Correctness" value={recommendationMetrics.avg_predicted_correctness} />
          </div>

          <p className="small-text" style={{ marginTop: 12 }}>
            {recommendationMetrics.interpretation}
          </p>
        </div>
      )}

      {modelComparison && Object.keys(modelComparison).length > 0 && (
        <div className="card">
          <h3>Model Comparison</h3>
          <div className="metric-grid">
            <MetricCard
              title="Baseline Accuracy"
              value={modelComparison.baseline?.accuracy ?? "N/A"}
            />
            <MetricCard
              title="Baseline F1"
              value={modelComparison.baseline?.f1 ?? "N/A"}
            />
            <MetricCard
              title="Baseline ROC-AUC"
              value={modelComparison.baseline?.roc_auc ?? "N/A"}
            />
            <MetricCard
              title="GNN Val Accuracy"
              value={modelComparison.dynamic_gnn?.val_accuracy ?? "N/A"}
            />
            <MetricCard
              title="Accuracy Gap"
              value={modelComparison.summary?.baseline_vs_gnn_accuracy_gap ?? "N/A"}
            />
            <MetricCard
              title="Winner"
              value={modelComparison.summary?.winner_by_accuracy ?? "N/A"}
            />
          </div>

          <p className="small-text" style={{ marginTop: 12 }}>
            {modelComparison.summary?.interpretation}
          </p>
        </div>
      )}

      {benchmarkSummary && Object.keys(benchmarkSummary).length > 0 && (
        <div className="card">
          <h3>Benchmark Summary</h3>
          <div className="metric-grid">
            <MetricCard title="Total Requests" value={benchmarkSummary.total_requests ?? "N/A"} />
            <MetricCard title="Error Rate" value={benchmarkSummary.error_rate ?? "N/A"} />
            <MetricCard title="Avg Latency (ms)" value={benchmarkSummary.weighted_avg_latency_ms ?? "N/A"} />
            <MetricCard title="Max P95 (ms)" value={benchmarkSummary.max_p95_latency_ms ?? "N/A"} />
            <MetricCard title="Max Req/s" value={benchmarkSummary.max_requests_per_sec ?? "N/A"} />
          </div>

          <p className="small-text" style={{ marginTop: 12 }}>
            {benchmarkSummary.interpretation}
          </p>
        </div>
      )}
    </>
  );
}
