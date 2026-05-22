import { useEffect, useMemo, useState } from 'react'
import './App.css'

function formatClassName(name) {
  return name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function formatPercent(value) {
  if (typeof value !== 'number') {
    return '0.0%'
  }

  return `${(value * 100).toFixed(1)}%`
}

function formatFileSize(size) {
  if (!size) {
    return '0 KB'
  }

  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }

  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const sortedDiseaseProbabilities = useMemo(() => {
    if (!result?.probabilities?.diseases) {
      return []
    }

    return Object.entries(result.probabilities.diseases).sort((a, b) => b[1] - a[1])
  }, [result])

  const sortedRegionProbabilities = useMemo(() => {
    if (!result?.probabilities?.regions) {
      return []
    }

    return Object.entries(result.probabilities.regions).sort((a, b) => b[1] - a[1])
  }, [result])

  const fileDetails = useMemo(() => {
    if (!file) {
      return null
    }

    return [
      { label: 'Name', value: file.name },
      { label: 'Type', value: file.type || 'Image file' },
      { label: 'Size', value: formatFileSize(file.size) },
    ]
  }, [file])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

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
            Upload an oral image to detect the oral region and predict the
            possible disease from the same CNN.
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

        <div className="preview-block">
          <div className="preview-heading">
            <span>Selected image preview</span>
            {file && <strong>{formatFileSize(file.size)}</strong>}
          </div>

          <div className="preview-frame">
            {previewUrl ? (
              <img src={previewUrl} alt="Selected oral scan preview" />
            ) : (
              <div className="preview-empty">
                <span>Preview will show here</span>
                <small>Choose one JPG, PNG, or WEBP image.</small>
              </div>
            )}
          </div>

          {fileDetails && (
            <dl className="file-details">
              {fileDetails.map((detail) => (
                <div className="file-detail" key={detail.label}>
                  <dt>{detail.label}</dt>
                  <dd>{detail.value}</dd>
                </div>
              ))}
            </dl>
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
            <div className="prediction-lines">
              <div className="prediction-line">
                <span>Detected Region:</span>
                <strong>{result.detected_region || result.region?.name}</strong>
              </div>
              <div className="prediction-line">
                <span>Possible Disease:</span>
                <strong>{result.possible_disease || result.disease?.name}</strong>
              </div>
              <div className="prediction-line">
                <span>Prediction Value:</span>
                <strong>
                  {result.prediction_percent ||
                    formatPercent(result.prediction_value ?? result.confidence)}
                </strong>
              </div>
            </div>

            <div className="probability-section">
              <h3>Disease probabilities</h3>
              <div className="probabilities">
                {sortedDiseaseProbabilities.map(([name, value]) => (
                  <div className="probability-row" key={name}>
                    <span className="probability-name">{formatClassName(name)}</span>
                    <span className="probability-bar">
                      <span
                        className="probability-fill disease-fill"
                        style={{ width: `${value * 100}%` }}
                      />
                    </span>
                    <span className="probability-value">{formatPercent(value)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="probability-section">
              <h3>Region probabilities</h3>
              <div className="probabilities">
                {sortedRegionProbabilities.map(([name, value]) => (
                  <div className="probability-row" key={name}>
                    <span className="probability-name">{formatClassName(name)}</span>
                    <span className="probability-bar">
                      <span
                        className="probability-fill region-fill"
                        style={{ width: `${value * 100}%` }}
                      />
                    </span>
                    <span className="probability-value">{formatPercent(value)}</span>
                  </div>
                ))}
              </div>
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
