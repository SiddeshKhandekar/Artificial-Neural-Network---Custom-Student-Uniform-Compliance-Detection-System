/**
 * API & WebSocket utility helpers
 */

const API_BASE = '/api';
const WS_BASE = `ws://${window.location.hostname}:8000`;

/**
 * POST an image file for analysis
 */
export async function analyzeImage(file, manualStudentId = null) {
  const formData = new FormData();
  formData.append('file', file);
  if (manualStudentId) {
    formData.append('manual_student_id', manualStudentId);
  }

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Analysis failed: ${res.statusText}`);
  }

  return res.json();
}

/**
 * Manually save analysis to history/DB
 */
export async function saveAnalysis(data) {
  const res = await fetch(`${API_BASE}/save-analysis`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });

  if (!res.ok) {
    throw new Error(`Failed to save analysis: ${res.statusText}`);
  }

  return res.json();
}

/**
 * Enroll an unknown face with a manual ID (Real-time Learning)
 */
export async function enrollUnknown(studentId, embedding) {
  const res = await fetch(`${API_BASE}/enroll-unknown`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ student_id: studentId, embedding })
  });

  if (!res.ok) {
    throw new Error(`Failed to enroll: ${res.statusText}`);
  }

  return res.json();
}

/**
 * Create a WebSocket connection for camera streaming
 */
export function createStreamSocket(onMessage, onClose) {
  const ws = new WebSocket(`${WS_BASE}/ws/stream`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket closed');
    if (onClose) onClose();
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };

  return ws;
}

/**
 * Send a video frame via WebSocket
 */
export function sendFrame(ws, canvas) {
  if (ws.readyState !== WebSocket.OPEN) return;
  const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
  ws.send(dataUrl);
}

/**
 * Fetch all students
 */
export async function fetchStudents() {
  const res = await fetch(`${API_BASE}/students`);
  return res.json();
}

/**
 * Fetch violations
 */
export async function fetchViolations(params = {}) {
  const query = new URLSearchParams(params).toString();
  const res = await fetch(`${API_BASE}/violations${query ? '?' + query : ''}`);
  return res.json();
}

/**
 * Health check
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Fetch history
 */
export async function fetchHistory(params = {}) {
  const query = new URLSearchParams(params).toString();
  const res = await fetch(`${API_BASE}/history${query ? '?' + query : ''}`);
  return res.json();
}
