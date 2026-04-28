import { useState, useRef, useEffect, useCallback } from 'react'
import { createStreamSocket, sendFrame } from '../utils/api'

export default function CameraFeed({ onResult, backendOnline }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [streaming, setStreaming] = useState(false)
  const [annotatedFrame, setAnnotatedFrame] = useState(null)
  const streamingRef = useRef(false)
  const streamRef = useRef(null)

  // Attach stream to video element when it becomes available in the DOM
  useEffect(() => {
    if (streaming && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(e => console.error("Playback failed:", e))
    }
  }, [streaming])

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      })
      
      streamRef.current = stream
      setStreaming(true)
      streamingRef.current = true

      // Connect WebSocket
      wsRef.current = createStreamSocket(
        (data) => {
          if (data.annotated_frame) {
            setAnnotatedFrame(data.annotated_frame)
          }
          onResult(data)
        },
        () => {
          setStreaming(false)
          streamingRef.current = false
        }
      )

      // Start sending frames
      const sendLoop = () => {
        if (!streamingRef.current) return
        
        // Ensure video is playing and ready
        if (videoRef.current && canvasRef.current && wsRef.current && videoRef.current.readyState === 4) {
          const ctx = canvasRef.current.getContext('2d')
          canvasRef.current.width = 640
          canvasRef.current.height = 480
          ctx.drawImage(videoRef.current, 0, 0, 640, 480)
          sendFrame(wsRef.current, canvasRef.current)
        }
        setTimeout(sendLoop, 200) // ~5 FPS
      }

      // Wait for WS to open before sending
      wsRef.current.onopen = () => {
        sendLoop()
      }
    } catch (err) {
      console.error('Camera error:', err)
      alert('Could not access camera. Please allow camera permissions.')
    }
  }, [onResult])

  const stopCamera = useCallback(() => {
    streamingRef.current = false
    setStreaming(false)
    setAnnotatedFrame(null)

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      streamingRef.current = false
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return (
    <div className="fade-in">
      <div className="feed-container professional-card" style={{ background: '#000', border: 'none' }}>
        {/* Placeholder: Only show when NOT streaming */}
        {!streaming && (
          <div className="feed-placeholder">
            <div style={{ fontSize: '3.5rem', marginBottom: '15px' }}>🏛️</div>
            <h2 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '8px' }}>Surveillance System Standby</h2>
            <p style={{ color: '#aaa' }}>Connect to the central AI gateway to begin analysis</p>
          </div>
        )}

        {/* Video Element: Always present if streaming is true */}
        {streaming && (
          <video 
            ref={videoRef} 
            muted 
            playsInline 
            autoPlay
            style={{ 
              width: '100%', 
              height: '100%', 
              objectFit: 'contain',
              display: annotatedFrame ? 'none' : 'block' 
            }} 
          />
        )}

        {/* Annotated Frame Overlay */}
        {streaming && annotatedFrame && (
          <img
            src={annotatedFrame}
            alt="Annotated camera feed"
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        )}

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      <div style={{ marginTop: 12, display: 'flex', gap: 10 }}>
        {!streaming ? (
          <button
            id="btn-start-camera"
            className="btn-academic"
            onClick={startCamera}
            disabled={!backendOnline}
          >
            ▶ INITIALIZE SURVEILLANCE
          </button>
        ) : (
          <button
            id="btn-stop-camera"
            className="btn btn-danger"
            onClick={stopCamera}
          >
            ⏹ TERMINATE FEED
          </button>
        )}
        {!backendOnline && (
          <span style={{ fontSize: '0.8rem', color: 'var(--college-red)', alignSelf: 'center', fontWeight: 700 }}>
            ! BACKEND OFFLINE
          </span>
        )}
      </div>
    </div>
  )
}
