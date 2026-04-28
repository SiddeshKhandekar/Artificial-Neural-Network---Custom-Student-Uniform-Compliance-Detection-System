export default function AnalysisAudit({ logs }) {
  if (!logs || logs.length === 0) return null

  return (
    <div className="professional-card fade-in" style={{ marginTop: '20px' }}>
      <div className="card-header" style={{ background: '#f8f9fa' }}>
        <span style={{ fontSize: '1.2rem' }}>🔍</span>
        <h2>Technical Analysis Audit</h2>
      </div>
      <div className="card-body" style={{ padding: '0' }}>
        <div className="audit-table">
          <div className="audit-header" style={{ 
            display: 'grid', 
            gridTemplateColumns: '150px 200px 1fr', 
            padding: '12px 20px', 
            background: '#eee', 
            fontSize: '0.75rem', 
            fontWeight: 800, 
            textTransform: 'uppercase',
            color: '#666'
          }}>
            <span>Step</span>
            <span>Algorithm</span>
            <span>Methodology & Rational</span>
          </div>
          <div className="audit-list">
            {logs.map((log, i) => (
              <div key={i} className="audit-row" style={{ 
                display: 'grid', 
                gridTemplateColumns: '150px 200px 1fr', 
                padding: '15px 20px', 
                borderBottom: '1px solid #eee',
                fontSize: '0.85rem',
                alignItems: 'center',
                gap: '15px'
              }}>
                <span style={{ fontWeight: 700, color: 'var(--college-maroon)' }}>{log.step}</span>
                <span style={{ 
                  background: '#f1f3f5', 
                  padding: '4px 8px', 
                  borderRadius: '4px', 
                  fontSize: '0.75rem', 
                  fontWeight: 700,
                  color: 'var(--college-navy)',
                  width: 'fit-content'
                }}>
                  {log.algorithm}
                </span>
                <span style={{ color: '#555', lineHeight: '1.4' }}>{log.reason}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
