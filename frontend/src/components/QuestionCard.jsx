import { useEffect, useState } from "react";

export default function QuestionCard({ question, onSubmit, isSubmitting }) {
  const [selectedOption, setSelectedOption] = useState("");
  const [responseTime, setResponseTime] = useState(10);

  useEffect(() => {
    setSelectedOption("");
  }, [question?.question_id]);

  if (!question) {
    return (
      <div className="card">
        <h2>No question available</h2>
      </div>
    );
  }

  const options = Array.isArray(question.options) ? question.options : [];

  const handleSubmit = () => {
    if (!selectedOption) return;
    onSubmit(selectedOption, Number(responseTime));
  };

  return (
    <div className="card">
      <h2>Current Question</h2>
      <p><strong>{question.text}</strong></p>

      <div className="small-text" style={{ marginBottom: 12 }}>
        <span className="badge">Question ID: {question.question_id}</span>
        <span className="badge">Concept: {question.concept_id}</span>
        <span className="badge">
          Predicted Correctness: {question.predicted_correctness}
        </span>
      </div>

      {options.length > 0 ? (
        <div className="option-grid">
          {options.map((option, index) => (
            <button
              key={`${question.question_id}-${index}-${option}`}
              type="button"
              className={`option-button ${selectedOption === option ? "active" : ""}`}
              onClick={() => setSelectedOption(option)}
            >
              {option}
            </button>
          ))}
        </div>
      ) : (
        <p className="error">No options found for this question.</p>
      )}

      <div style={{ marginTop: 16 }}>
        <label>
          Response Time (seconds)
          <br />
          <input
            type="number"
            min="1"
            step="0.1"
            value={responseTime}
            onChange={(e) => setResponseTime(e.target.value)}
          />
        </label>
      </div>

      <div style={{ marginTop: 16 }}>
        <button onClick={handleSubmit} disabled={!selectedOption || isSubmitting}>
          {isSubmitting ? "Submitting..." : "Submit Answer"}
        </button>
      </div>
    </div>
  );
}
