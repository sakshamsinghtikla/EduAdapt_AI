export default function FeedbackPanel({ result }) {
  if (!result) return null;

  return (
    <div className="card">
      <h3>Feedback for Last Submitted Question</h3>

      <p className={result.is_correct ? "feedback-correct" : "feedback-wrong"}>
        {result.is_correct ? "Correct answer" : "Incorrect answer"}
      </p>

      <p>
        Question ID: <strong>{result.question_id}</strong>
      </p>

      {result.event?.selected_option && (
        <p>
          Your answer: <strong>{result.event.selected_option}</strong>
        </p>
      )}

      <p>
        Correct option: <strong>{result.correct_option}</strong>
      </p>
    </div>
  );
}
