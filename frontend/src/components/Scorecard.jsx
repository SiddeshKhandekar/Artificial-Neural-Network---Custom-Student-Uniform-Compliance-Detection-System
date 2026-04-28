import { useMemo, useEffect } from 'react'
import confetti from 'canvas-confetti'

export default function Scorecard({ score }) {
  const totalScore = score?.total_score ?? 0
  const label = score?.label ?? 'Pending'
  const components = score?.components ?? {}

  // Congrats Rockets Animation
  useEffect(() => {
    if (label === 'BEST PROFESSIONAL ATTIRE' || (totalScore >= 90 && label !== 'Pending')) {
      const duration = 5 * 1000
      const animationEnd = Date.now() + duration
      const defaults = { startVelocity: 45, spread: 360, ticks: 100, zIndex: 9999 }

      const randomInRange = (min, max) => Math.random() * (max - min) + min

      const interval = setInterval(() => {
        const timeLeft = animationEnd - Date.now()

        if (timeLeft <= 0) {
          return clearInterval(interval)
        }

        const particleCount = 50 * (timeLeft / duration)
        
        // Burst from left
        confetti({ 
          ...defaults, 
          particleCount, 
          origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 } 
        })
        
        // Burst from right
        confetti({ 
          ...defaults, 
          particleCount, 
          origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 } 
        })
      }, 250)
      
      return () => clearInterval(interval)
    }
  }, [label])

  const statusClass = useMemo(() => {
    if (totalScore >= 90) return 'excellent'
    if (totalScore >= 70) return 'warning'
    return 'danger'
  }, [totalScore])

  const dashArray = (totalScore / 100) * 264

  const componentList = [
    { key: 'shirt', label: 'Shirt', icon: '👔' },
    { key: 'pant', label: 'Pant', icon: '👖' },
    { key: 'tucked', label: 'Tucked', icon: '📏' },
    { key: 'id_card', label: 'ID Card', icon: '🪪' },
  ]

  return (
    <div className="scorecard professional-card">
      <div className="card-header">
        <span style={{ fontSize: '1.2rem' }}>📋</span>
        <h2>Attire Compliance Score</h2>
      </div>

      <div className="card-body score-card">
        <div className="score-gauge">
          <svg className="gauge-svg" viewBox="0 0 100 100" style={{ width: '100%', height: '100%' }}>
            <circle className="gauge-bg" cx="50" cy="50" r="42" />
            <circle
              className="gauge-fill"
              cx="50"
              cy="50"
              r="42"
              strokeDasharray={`${dashArray} 264`}
            />
          </svg>
          <div className="gauge-text">{Math.round(totalScore)}%</div>
        </div>

        <div className={`compliance-status ${statusClass}`}>
          {label}
        </div>

        <div style={{ marginTop: '32px', textAlign: 'left' }}>
          {componentList.map(({ key, label, icon }) => {
            const comp = components[key] || { score: 0, max_score: 100 }
            const pct = comp.max_score ? (comp.score / comp.max_score) * 100 : 0
            return (
              <div key={key} className="comp-row">
                <span className="comp-name">{icon} {label}</span>
                <div className="comp-track">
                  <div
                    className="comp-fill"
                    style={{ width: `${pct}%`, background: pct >= 80 ? 'var(--college-maroon)' : 'var(--college-gold)' }}
                  />
                </div>
                <span style={{ fontSize: '0.85rem', fontWeight: 800, width: '55px', textAlign: 'right', color: 'var(--college-maroon)' }}>
                  {Math.round(pct)}%
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
