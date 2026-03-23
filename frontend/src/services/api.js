const API_BASE = "http://127.0.0.1:8000";

async function handleResponse(response) {
  if (!response.ok) {
    let detail = "Request failed";
    try {
      const data = await response.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function startSession(studentId) {
  const response = await fetch(`${API_BASE}/start_session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ student_id: studentId }),
  });
  return handleResponse(response);
}

export async function submitAnswer(studentId, questionId, selectedOption, responseTime) {
  const response = await fetch(`${API_BASE}/submit_answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      student_id: studentId,
      question_id: questionId,
      selected_option: selectedOption,
      response_time: responseTime,
    }),
  });
  return handleResponse(response);
}

export async function getStudentState(studentId) {
  const response = await fetch(`${API_BASE}/student_state/${studentId}`);
  return handleResponse(response);
}

export async function getNextQuestion(studentId) {
  const response = await fetch(`${API_BASE}/next_question/${studentId}`);
  return handleResponse(response);
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE}/metrics`);
  return handleResponse(response);
}

export async function getRecommendationMetrics() {
  const response = await fetch(`${API_BASE}/debug/recommendation_metrics`);
  return handleResponse(response);
}

export async function getModelComparison() {
  const response = await fetch(`${API_BASE}/debug/model_comparison`);
  return handleResponse(response);
}

export async function getBenchmarkSummary() {
  const response = await fetch(`${API_BASE}/debug/benchmark_summary`);
  return handleResponse(response);
}

export async function buildDataset() {
  const response = await fetch(`${API_BASE}/admin/build_dataset?min_history=0`, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function trainBaseline() {
  const response = await fetch(`${API_BASE}/admin/train_baseline`, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function trainDynamicGnn() {
  const response = await fetch(`${API_BASE}/admin/train_dynamic_gnn`, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function compareModels() {
  const response = await fetch(`${API_BASE}/admin/compare_models`, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function evaluateRecommendations() {
  const response = await fetch(`${API_BASE}/admin/evaluate_recommendations`, {
    method: "POST",
  });
  return handleResponse(response);
}