import { useState, useRef, useCallback } from 'react'
import { analyzeImage } from '../utils/api'

export default function ImageUpload({ onResult, onProcessing, backendOnline }) {
  const [preview, setPreview] = useState(null)
  const [annotatedImage, setAnnotatedImage] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)
  const selectedFileRef = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return

    selectedFileRef.current = file
    setAnnotatedImage(null)

    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFileRef.current || !backendOnline) return

    setIsLoading(true)
    onProcessing?.()

    try {
      const result = await analyzeImage(selectedFileRef.current)
      setAnnotatedImage(result.annotated_image)
      onResult(result)
    } catch (err) {
      console.error('Analysis error:', err)
      alert('Analysis failed. Is the backend running?')
    } finally {
      setIsLoading(false)
    }
  }, [backendOnline, onResult, onProcessing])

  const clearImage = useCallback(() => {
    setPreview(null)
    setAnnotatedImage(null)
    selectedFileRef.current = null
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [])

  return (
    <div className="fade-in">
      {!preview ? (
        /* ── Academic Drop Zone ────────────────────── */
        <div
          className={`upload-dropzone ${dragOver ? 'dragover' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div style={{ fontSize: '3.5rem', marginBottom: '15px' }}>🏛️</div>
          <h2 style={{ fontSize: '1.2rem', marginBottom: '8px' }}>Student Photo Submission</h2>
          <p style={{ color: '#666' }}>
            Drag student photo here, or{' '}
            <span style={{ color: 'var(--college-maroon)', fontWeight: 700 }}>browse files</span>
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => handleFile(e.target.files[0])}
            style={{ display: 'none' }}
            id="image-upload-input"
          />
        </div>
      ) : (
        /* ── Image Preview / Result ────────────────── */
        <div>
          <div className="feed-container professional-card" style={{ background: '#000', border: 'none' }}>
            <img
              src={annotatedImage || preview}
              alt="Analysis preview"
            />
            {isLoading && (
              <div className="loading-overlay" style={{ background: 'rgba(255,255,255,0.8)' }}>
                <div className="spinner" style={{ borderColor: 'rgba(128,0,0,0.1)', borderTopColor: 'var(--college-maroon)' }} />
                <span style={{ color: 'var(--college-maroon)', fontWeight: 700 }}>AI PROCESSING...</span>
              </div>
            )}
          </div>

          <div style={{ marginTop: 12, display: 'flex', gap: 10 }}>
            <button
              id="btn-analyze"
              className="btn-academic"
              onClick={handleAnalyze}
              disabled={isLoading || !backendOnline}
            >
              {isLoading ? '⏳ Processing...' : '📋 Analyze Attire'}
            </button>
            <button
              id="btn-clear"
              className="btn-outline"
              onClick={clearImage}
            >
              ✕ Clear
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
