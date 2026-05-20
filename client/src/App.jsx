import { useMemo, useState } from 'react'
import './App.css'

function formatClassName(name) {
  return name.replaceAll('_', ' ')
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const sortedProbabilities = useMemo(() => {
    if (!result?.probabilities) {
      return []
    }

    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
  }, [result])

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0]
    if (!selectedFile) {
      return
    }

    setFile(selectedFile)
    setResult(null)
    setError('')
    setPreviewUrl(URL.createObjectURL(selectedFile))
  }

  async function handleSubmit(event) {
    event.preventDefault()

    if (!file) {
      setError('Please choose an image first.')
      return
    }

    const formData = new FormData()
    formData.append('image', file)
    setIsLoading(true)
    setError('')

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed.')
      }

      setResult(data)
    } catch (requestError) {
      setResult(null)
      setError(requestError.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="intro">
          <p className="eyebrow">AI oral screening</p>
          <h1>Oral Disease Classifier</h1>
          <p className="summary">
            Upload an oral image to classify healthy tissue, mouth ulcer, tooth
            decay, or gingivitis.
          </p>
        </div>

        <form className="upload-form" onSubmit={handleSubmit}>
          <label className="drop-zone">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={handleFileChange}
            />
            <span className="drop-title">Choose image</span>
            <span className="drop-meta">{file?.name || 'JPG, PNG, or WEBP'}</span>
          </label>

          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Predicting' : 'Predict'}
          </button>
        </form>

        <div className="preview-frame">
          {previewUrl ? (
            <img src={previewUrl} alt="Selected oral scan preview" />
          ) : (
            <span>Image preview</span>
          )}
        </div>
      </section>

      <section className="result-panel" aria-live="polite">
        {error && <div className="message error">{error}</div>}

        {!error && !result && (
          <div className="message">Prediction result will appear here.</div>
        )}

        {result && (
          <div className="result-card">
            <p className="eyebrow">Prediction</p>
            <h2>{formatClassName(result.class)}</h2>
            <p className="confidence">
              Confidence {formatPercent(result.confidence)}
            </p>

            <div className="probabilities">
              {sortedProbabilities.map(([name, value]) => (
                <div className="probability-row" key={name}>
                  <span className="probability-name">{formatClassName(name)}</span>
                  <span className="probability-bar">
                    <span
                      className="probability-fill"
                      style={{ width: `${value * 100}%` }}
                    />
                  </span>
                  <span className="probability-value">{formatPercent(value)}</span>
                </div>
              ))}
            </div>

            <p className="note">
              This project is for AI/ML demonstration only, not medical
              diagnosis.
            </p>
          </div>
        )}
      </section>
    </main>
  )
}

export default App
