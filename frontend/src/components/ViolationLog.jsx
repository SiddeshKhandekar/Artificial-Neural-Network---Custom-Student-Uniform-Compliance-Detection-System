import { useState, useEffect } from 'react'
import { fetchViolations } from '../utils/api'

export default function ViolationLog({ violations: newViolations, currentAnalysisViolations }) {
  const [allViolations, setAllViolations] = useState([])

  // Load recent violations on mount
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchViolations({ limit: 20 })
        setAllViolations(data.violations || [])
      } catch {
        // Backend might not be running yet
      }
    }
    load()
  }, [])

  // Append newly logged violations
  useEffect(() => {
    if (newViolations?.length > 0) {
      setAllViolations(prev => {
        // Simple deduplication based on ID if available
        const newItems = newViolations.filter(nv => !prev.some(pv => pv.id === nv.id))
        return [...newItems, ...prev].slice(0, 20)
      })
    }
  }, [newViolations])

  return (
    <div className="violations-card professional-card">
      <div className="card-header">
        <span style={{ fontSize: '1.2rem' }}>🚩</span>
        <h2>Compliance Issues Log</h2>
      </div>

      <div className="card-body">
        {/* Issues found in the CURRENT analysis */}
        {currentAnalysisViolations && currentAnalysisViolations.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <div className="violations-list" style={{ maxHeight: 'none' }}>
              {currentAnalysisViolations.map((v, i) => (
                <div key={`current-${i}`} className="violation-row" style={{ borderLeft: '4px solid var(--college-red)', background: '#fffcfc' }}>
                  <span className="v-icon" style={{ background: 'var(--college-red)' }}></span>
                  <span style={{ fontWeight: 700, color: 'var(--college-red)' }}>{v}</span>
                  <span style={{ fontSize: '0.7rem', fontWeight: 800, marginLeft: 'auto', color: 'var(--college-maroon)', opacity: 0.6 }}>CURRENT ANALYSIS</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Database Violation Log */}
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 800, marginBottom: '12px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Database Record ({allViolations.length})
        </div>
        
        <div className="violation-log-scroll" style={{ maxHeight: '300px', overflowY: 'auto' }}>
          {allViolations.length === 0 && !currentAnalysisViolations?.length ? (
            <div className="empty-state">
              <div className="icon">🏛️</div>
              <p>Full Compliance Maintained</p>
              <span style={{ fontSize: '0.85rem', color: '#999' }}>No institutional violations on record</span>
            </div>
          ) : (
            allViolations.map((v, i) => (
              <div key={v.id || i} className="violation-row">
                <span className="v-icon" style={{ background: '#ddd' }}></span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#333' }}>{v.violation_type}</div>
                  <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '2px' }}>ID: {v.student_id}</div>
                </div>
                <span style={{ fontSize: '0.7rem', fontWeight: 600, color: '#bbb' }}>{v.date || 'RECENT'}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
