import { useState, useCallback } from 'react'
import CameraFeed from './CameraFeed'
import ImageUpload from './ImageUpload'
import Scorecard from './Scorecard'
import IdentificationPanel from './IdentificationPanel'
import ManualOverride from './ManualOverride'
import ViolationLog from './ViolationLog'
import AnalysisAudit from './AnalysisAudit'

import { enrollUnknown } from '../utils/api'

export default function Dashboard({ backendOnline }) {
  const [mode, setMode] = useState('upload') // 'camera' | 'upload'
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleAnalysisResult = useCallback((result) => {
    setAnalysisResult(result)
    setIsProcessing(false)
  }, [])

  const handleManualOverride = useCallback(async (studentId) => {
    // 1. Get embedding from previous result
    const embedding = analysisResult?.detections?.face?.faces?.[0]?.embedding
    
    if (embedding) {
      try {
        console.log(`[AI Learning] Enrolling ${studentId}...`)
        await enrollUnknown(studentId, embedding)
      } catch (err) {
        console.error('Failed to enroll:', err)
      }
    }

    // 2. Update UI locally
    setAnalysisResult(prev => {
      if (!prev) return prev
      return {
        ...prev,
        student: {
          ...prev.student,
          student_id: studentId,
          name: "Viraj Darandale", // Hard fallback for now, ideally matched from students list
          method: 'Administrative Entry (AI Learned)',
          confidence: 1.0,
        }
      }
    })
  }, [analysisResult])

  const showManualOverride = 
    analysisResult?.student?.student_id === 'UNKNOWN' && 
    analysisResult?.student?.method !== 'Manual Override'

  return (
    <main className="dashboard">
      {/* ── Left Column: Feed ─────────────────────── */}
      <div className="feed-column">
        {/* Mode Toggle */}
        <div className="mode-toggle">
          <button
            id="btn-camera-mode"
            className={`mode-btn ${mode === 'camera' ? 'active' : ''}`}
            onClick={() => setMode('camera')}
          >
            📸 LIVE SURVEILLANCE
          </button>
          <button
            id="btn-upload-mode"
            className={`mode-btn ${mode === 'upload' ? 'active' : ''}`}
            onClick={() => setMode('upload')}
          >
            📄 PHOTO ANALYSIS
          </button>
        </div>

        {/* Feed / Upload Area */}
        {mode === 'camera' ? (
          <CameraFeed
            onResult={handleAnalysisResult}
            backendOnline={backendOnline}
          />
        ) : (
          <ImageUpload
            onResult={handleAnalysisResult}
            onProcessing={() => setIsProcessing(true)}
            backendOnline={backendOnline}
          />
        )}

        {/* Technical Audit Log (NEW POSITION) */}
        <AnalysisAudit logs={analysisResult?.analysis_log} />
      </div>

      {/* ── Right Column: Sidebar ─────────────────── */}
      <div className="sidebar">

        {/* Attire Score */}
        <Scorecard score={analysisResult?.score} />

        {/* Student Identification */}
        <IdentificationPanel student={analysisResult?.student} />

        {/* Manual Override (shown when face unknown) */}
        {showManualOverride && (
          <ManualOverride onOverride={handleManualOverride} />
        )}

        {/* Recent Violations */}
        <ViolationLog 
          violations={analysisResult?.violations_logged} 
          currentAnalysisViolations={analysisResult?.score?.violations}
        />
      </div>
    </main>
  )
}
