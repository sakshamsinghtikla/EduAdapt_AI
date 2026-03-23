import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import QuestionCard from "../components/QuestionCard";
import FeedbackPanel from "../components/FeedbackPanel";
import RecommendationList from "../components/RecommendationList";
import { startSession, submitAnswer } from "../services/api";

export default function Session() {
  const { studentId } = useParams();
  const navigate = useNavigate();

  const [question, setQuestion] = useState(null);
  const [result, setResult] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadSession() {
      try {
        setLoading(true);
        setError("");
        const data = await startSession(studentId);
        if (!active) return;
        setQuestion(data.first_question);
      } catch (err) {
        if (!active) return;
        setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    }

    loadSession();
    return () => {
      active = false;
    };
  }, [studentId]);

  const handleSubmit = async (selectedOption, responseTime) => {
    try {
      setSubmitting(true);
      setError("");
      const data = await submitAnswer(
        studentId,
        question.question_id,
        selectedOption,
        responseTime
      );
      setResult(data);
      setRecommendations(data.next_recommendations || []);
      if (data.next_recommendations && data.next_recommendations.length > 0) {
        setQuestion(data.next_recommendations[0]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <div className="card loading">Loading session...</div>;
  }

  return (
    <>
      {error && <div className="card error">{error}</div>}

      <div className="card">
        <h2>Student Session: {studentId}</h2>
        <div className="input-row">
          <button className="secondary" onClick={() => navigate(`/dashboard/${studentId}`)}>
            View Dashboard
          </button>
        </div>
      </div>

      <QuestionCard
        question={question}
        onSubmit={handleSubmit}
        isSubmitting={submitting}
      />

      <FeedbackPanel result={result} />
      <RecommendationList recommendations={recommendations} />
    </>
  );
}
