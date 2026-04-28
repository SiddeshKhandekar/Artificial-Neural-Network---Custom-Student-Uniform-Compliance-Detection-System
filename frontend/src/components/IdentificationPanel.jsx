export default function IdentificationPanel({ student }) {
  const studentId = student?.student_id ?? 'N/A'
  const name = student?.name ?? 'Waiting for analysis...'
  const confidence = student?.confidence ?? 0
  const method = student?.method ?? 'None'
  const isIdentified = studentId !== 'UNKNOWN' && studentId !== 'N/A'

  const initials = name
    .split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)

  return (
    <div className="id-panel professional-card">
      <div className="card-header">
        <span style={{ fontSize: '1.2rem' }}>🎓</span>
        <h2>Student Identification</h2>
      </div>

      <div className="card-body">
        <div className="id-card-view">
          <div className="id-avatar" style={{ background: isIdentified ? 'var(--college-maroon)' : 'var(--college-red)' }}>
            {isIdentified ? initials : '?'}
          </div>
          <div className="id-details">
            <h3>{isIdentified ? name : 'Unknown Person'}</h3>
            <p>{isIdentified ? `ID: ${studentId}` : 'Face not recognized'}</p>
          </div>
        </div>

        <div className="confidence-meter">
          <div className="bar">
            <div
              className="fill"
              style={{
                width: `${confidence * 100}%`,
                background: confidence >= 0.7 ? 'var(--college-maroon)' : 'var(--college-gold)'
              }}
            />
          </div>
          <span style={{ fontWeight: 800, color: 'var(--college-maroon)', fontSize: '0.9rem' }}>
            {(confidence * 100).toFixed(0)}% Match
          </span>
        </div>

      </div>
    </div>
  )
}
