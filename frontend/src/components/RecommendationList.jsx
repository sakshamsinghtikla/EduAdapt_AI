export default function RecommendationList({ recommendations }) {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <h3>Next Recommendations</h3>
      <ul className="list">
        {recommendations.map((rec) => (
          <li key={rec.question_id} style={{ marginBottom: 10 }}>
            <strong>{rec.question_id}</strong> — {rec.text}
            <br />
            <span className="small-text">
              Concept: {rec.concept_id} | Predicted correctness: {rec.predicted_correctness} | Score: {rec.score}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
