import { useState } from 'react'

export default function ManualOverride({ onOverride }) {
  const [studentId, setStudentId] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = studentId.trim().toUpperCase()
    if (trimmed) {
      onOverride(trimmed)
      setStudentId('')
    }
  }

  return (
    <div className="professional-card fade-in" style={{ borderColor: 'var(--college-gold)', background: '#fffef0' }}>
      <div className="card-header" style={{ background: 'rgba(212, 175, 55, 0.05)' }}>
        <span style={{ fontSize: '1.2rem' }}>⚠️</span>
        <h2>Administrative Correction</h2>
      </div>
      <div className="card-body">
        <p style={{ fontSize: '0.85rem', color: '#555', marginBottom: '15px', fontWeight: 500 }}>
          Identification pending. Please provide the Student ID to finalize compliance:
        </p>
        <form onSubmit={handleSubmit}>
          <input
            id="manual-student-id"
            className="override-input"
            type="text"
            placeholder="Enter ID (e.g. STU001)"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
            autoFocus
          />
          <button
            type="submit"
            className="btn-academic"
            style={{ width: '100%' }}
            disabled={!studentId.trim()}
          >
            Assign ID & Finalize
          </button>
        </form>
      </div>
    </div>
  )
}
