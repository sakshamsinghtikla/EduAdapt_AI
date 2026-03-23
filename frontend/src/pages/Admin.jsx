import { useState } from "react";
import {
  buildDataset,
  trainBaseline,
  trainDynamicGnn,
  compareModels,
  evaluateRecommendations,
  getModelComparison,
  getRecommendationMetrics,
  getBenchmarkSummary,
} from "../services/api";
import MetricCard from "../components/MetricCard";

export default function Admin() {
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [comparison, setComparison] = useState(null);
  const [recommendationMetrics, setRecommendationMetrics] = useState(null);
  const [benchmarkSummary, setBenchmarkSummary] = useState(null);

  async function runAction(actionName, actionFn) {
    try {
      setLoading(actionName);
      setError("");
      setMessage("");
      const result = await actionFn();
      setMessage(`${actionName} completed successfully`);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading("");
    }
  }

  async function handleBuildDataset() {
    await runAction("Build dataset", buildDataset);
  }

  async function handleTrainBaseline() {
    await runAction("Train baseline", trainBaseline);
  }

  async function handleTrainGnn() {
    await runAction("Train dynamic GNN", trainDynamicGnn);
  }

  async function handleCompareModels() {
    const result = await runAction("Compare models", compareModels);
    if (result) setComparison(result);
  }

  async function handleEvaluateRecommendations() {
    const result = await runAction("Evaluate recommendations", evaluateRecommendations);
    if (result) setRecommendationMetrics(result);
  }

  async function handleRefreshReports() {
    try {
      setLoading("Refresh reports");
      setError("");
      const [comparisonData, recommendationData, benchmarkData] = await Promise.all([
        getModelComparison().catch(() => ({})),
        getRecommendationMetrics().catch(() => ({})),
        getBenchmarkSummary().catch(() => ({})),
      ]);
      setComparison(comparisonData);
      setRecommendationMetrics(recommendationData);
      setBenchmarkSummary(benchmarkData);
      setMessage("Reports refreshed");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  }

  return (
    <>
      <div className="card">
        <h2>Admin Panel</h2>
        <p className="small-text">
          Run training and evaluation workflows, then inspect the latest reports.
        </p>

        <div className="admin-actions">
          <button onClick={handleBuildDataset} disabled={!!loading}>
            Build Dataset
          </button>
          <button onClick={handleTrainBaseline} disabled={!!loading}>
            Train Baseline
          </button>
          <button onClick={handleTrainGnn} disabled={!!loading}>
            Train Dynamic GNN
          </button>
          <button onClick={handleCompareModels} disabled={!!loading}>
            Compare Models
          </button>
          <button onClick={handleEvaluateRecommendations} disabled={!!loading}>
            Evaluate Recommendations
          </button>
          <button className="secondary" onClick={handleRefreshReports} disabled={!!loading}>
            Refresh Reports
          </button>
        </div>

        {loading && <p className="loading">Running: {loading}</p>}
        {message && <p className="success-text">{message}</p>}
        {error && <p className="error">{error}</p>}
      </div>

      {comparison && Object.keys(comparison).length > 0 && (
        <div className="card">
          <h3>Latest Model Comparison</h3>
          <div className="metric-grid">
            <MetricCard title="Baseline Accuracy" value={comparison.baseline?.accuracy ?? "N/A"} />
            <MetricCard title="GNN Accuracy" value={comparison.dynamic_gnn?.val_accuracy ?? "N/A"} />
            <MetricCard title="Accuracy Gap" value={comparison.summary?.baseline_vs_gnn_accuracy_gap ?? "N/A"} />
            <MetricCard title="Winner" value={comparison.summary?.winner_by_accuracy ?? "N/A"} />
          </div>
          <p className="small-text" style={{ marginTop: 12 }}>
            {comparison.summary?.interpretation}
          </p>
        </div>
      )}

      {recommendationMetrics && Object.keys(recommendationMetrics).length > 0 && (
        <div className="card">
          <h3>Latest Recommendation Metrics</h3>
          <div className="metric-grid">
            <MetricCard title="Target Zone Rate" value={recommendationMetrics.target_zone_rate ?? "N/A"} />
            <MetricCard title="Repetition Rate" value={recommendationMetrics.repetition_rate ?? "N/A"} />
            <MetricCard title="Concept Diversity" value={recommendationMetrics.concept_diversity_normalized ?? "N/A"} />
            <MetricCard title="Avg Predicted Correctness" value={recommendationMetrics.avg_predicted_correctness ?? "N/A"} />
          </div>
        </div>
      )}

      {benchmarkSummary && Object.keys(benchmarkSummary).length > 0 && (
        <div className="card">
          <h3>Latest Benchmark Summary</h3>
          <div className="metric-grid">
            <MetricCard title="Total Requests" value={benchmarkSummary.total_requests ?? "N/A"} />
            <MetricCard title="Avg Latency (ms)" value={benchmarkSummary.weighted_avg_latency_ms ?? "N/A"} />
            <MetricCard title="Max P95 (ms)" value={benchmarkSummary.max_p95_latency_ms ?? "N/A"} />
            <MetricCard title="Error Rate" value={benchmarkSummary.error_rate ?? "N/A"} />
          </div>
          <p className="small-text" style={{ marginTop: 12 }}>
            {benchmarkSummary.interpretation}
          </p>
        </div>
      )}
    </>
  );
}
