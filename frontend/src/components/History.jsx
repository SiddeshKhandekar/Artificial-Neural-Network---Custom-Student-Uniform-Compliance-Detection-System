import { useState, useEffect } from 'react'
import { fetchHistory } from '../utils/api'

export default function History() {
  const [historyRecords, setHistoryRecords] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadHistory = async () => {
      try {
        setLoading(true)
        // Fetch up to 100 recent analyses for the history page
        const data = await fetchHistory({ limit: 100 })
        setHistoryRecords(data.history || [])
      } catch (err) {
        console.error("Failed to load history", err)
      } finally {
        setLoading(false)
      }
    }
    loadHistory()
  }, [])

  return (
    <main className="dashboard" style={{ display: 'block', padding: '20px' }}>
      <div className="glass-card" style={{ width: '100%', minHeight: '80vh', padding: '30px' }}>
        <div className="section-header" style={{ marginBottom: '20px' }}>
          <h2>📚 Complete Analysis History</h2>
          <div className="divider" />
          <p style={{ color: 'var(--text-muted)', marginTop: '10px' }}>
            A complete record of all image analyses performed by the system, including fully compliant uniforms.
          </p>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '50px', color: 'var(--text-muted)' }}>
            Loading history...
          </div>
        ) : historyRecords.length === 0 ? (
          <div className="empty-state" style={{ padding: '60px' }}>
            <div className="icon" style={{ fontSize: '3rem' }}>✅</div>
            <p style={{ fontSize: '1.2rem', marginTop: '15px' }}>No analysis history found.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', marginTop: '20px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <th style={{ padding: '12px 15px', color: 'var(--text-muted)', fontWeight: 600 }}>Date/Time</th>
                  <th style={{ padding: '12px 15px', color: 'var(--text-muted)', fontWeight: 600 }}>Student Info</th>
                  <th style={{ padding: '12px 15px', color: 'var(--text-muted)', fontWeight: 600 }}>Department</th>
                  <th style={{ padding: '12px 15px', color: 'var(--text-muted)', fontWeight: 600 }}>Uniform Status</th>
                  <th style={{ padding: '12px 15px', color: 'var(--text-muted)', fontWeight: 600 }}>Violations</th>
                </tr>
              </thead>
              <tbody>
                {historyRecords.map((record, i) => {
                  const isUnknown = record.student_id === 'UNKNOWN'
                  const issues = Array.isArray(record.issues_found) ? record.issues_found : []
                  const isCompliant = issues.length === 0

                  return (
                    <tr key={record.id || i} style={{ 
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      background: isUnknown ? 'rgba(255, 255, 255, 0.02)' : 'transparent'
                    }}>
                      <td style={{ padding: '15px' }}>{new Date(record.created_at || record.date).toLocaleString()}</td>
                      <td style={{ padding: '15px' }}>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                          <span style={{ 
                            color: isUnknown ? 'var(--text-muted)' : 'var(--accent-blue)',
                            fontWeight: isUnknown ? 'normal' : 'bold'
                          }}>
                            {record.student_id}
                          </span>
                          {!isUnknown && record.name && (
                            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{record.name}</span>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '15px', color: 'var(--text-muted)' }}>
                        {isUnknown ? '-' : (record.department || 'N/A')}
                      </td>
                      <td style={{ padding: '15px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontWeight: 'bold', color: record.total_score >= 80 ? 'var(--accent-emerald)' : (record.total_score >= 60 ? 'var(--accent-amber)' : 'var(--accent-rose)') }}>
                            {record.total_score}%
                          </span>
                          <span style={{ 
                            padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold',
                            background: isCompliant ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                            color: isCompliant ? 'var(--accent-emerald)' : 'var(--accent-rose)'
                          }}>
                            {isCompliant ? 'DETECTED' : 'NOT DETECTED'}
                          </span>
                        </div>
                      </td>
                      <td style={{ padding: '15px' }}>
                        {issues.length > 0 ? (
                          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                            {issues.map((issue, idx) => (
                              <span key={idx} style={{ background: 'rgba(239, 68, 68, 0.15)', color: 'var(--accent-rose)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>
                                {issue}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span style={{ color: 'var(--accent-emerald)' }}>✅ None</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  )
}
