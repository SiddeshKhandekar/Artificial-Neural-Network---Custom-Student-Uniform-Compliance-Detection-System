import { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard'
import { checkHealth } from './utils/api'

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false)

  useEffect(() => {
    const check = async () => {
      const ok = await checkHealth()
      setBackendOnline(ok)
    }
    check()
    const interval = setInterval(check, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────── */}
      <header className="app-header">
        <div className="header-brand">
          <img 
            src="/logo.png" 
            alt="College Logo" 
            className="college-logo"
            onError={(e) => e.target.src = '🏛️'} 
          />
          <div className="college-names">
            <h1>Shri. Chhatrapati Shivaji Maharaj</h1>
            <div className="sub-title">College of Engineering, Nepti, A.Nagar</div>
          </div>
        </div>
        
        <div className="header-meta">
          <div className="system-status">
            <div className="status-pulse" style={{ background: backendOnline ? '#10b981' : '#ef4444' }} />
            <span>{backendOnline ? 'System Online' : 'Backend Offline'}</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: '#888', fontWeight: 600 }}>
            UNIFORM COMPLIANCE MONITOR v2.1
          </div>
        </div>
      </header>

      {/* ── Main Content ──────────────────────────── */}
      <Dashboard backendOnline={backendOnline} />
    </div>
  )
}
