export default function FeedbackPanel({ result }) {
  if (!result) return null;

  return (
    <div className="card">
      <h3>Answer Feedback</h3>
      <p className={result.is_correct ? "feedback-correct" : "feedback-wrong"}>
        {result.is_correct ? "Correct answer" : "Incorrect answer"}
      </p>
      <p>
        Correct option: <strong>{result.correct_option}</strong>
      </p>
    </div>
  );
}
