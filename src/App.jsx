import { useState } from 'react'

function App() {
  const [text, setText] = useState('')
  const [sentiment, setSentiment] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handlePredict = async () => {
    if (!text.trim()) return
    
    setLoading(true)
    setError('')
    setSentiment(null)
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })
      
      const data = await response.json()
      
      if (response.ok) {
        setSentiment(data.sentiment)
      } else {
        setError(data.error || 'Terjadi kesalahan saat memprediksi.')
      }
    } catch (err) {
      setError('Gagal terhubung ke server.')
    } finally {
      setLoading(false)
    }
  }

  const handleRandom = async () => {
    setLoading(true)
    setError('')
    setSentiment(null)
    setText('')
    
    try {
      const response = await fetch('/api/random')
      const data = await response.json()
      
      if (response.ok) {
        setText(data.text)
      } else {
        setError(data.error || 'Gagal mengambil data acak.')
      }
    } catch (err) {
      setError('Gagal terhubung ke server.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>
          <span role="img" aria-label="search">🔍</span>
          Sentiment Analysis App
        </h1>
        <p>Masukkan kalimat dan lihat prediksi sentimennya!</p>
      </header>

      <main>
        <div className="input-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <label htmlFor="review-text" style={{ marginBottom: 0 }}>Masukkan teks ulasan di sini:</label>
            <button 
              className="btn-secondary" 
              onClick={handleRandom}
              disabled={loading}
            >
              <span role="img" aria-label="dice">🎲</span> Ambil Data Amazon
            </button>
          </div>
          <textarea
            id="review-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Contoh: I like this product"
            disabled={loading}
          />
        </div>

        <button 
          className="btn-primary" 
          onClick={handlePredict}
          disabled={loading || !text.trim()}
        >
          {loading ? (
            <>
              <div className="spinner"></div>
              Memproses...
            </>
          ) : (
            'Prediksi Sentimen'
          )}
        </button>

        {error && (
          <div className="result-card negative">
            <span role="img" aria-label="error">❌</span>
            Error: {error}
          </div>
        )}

        {sentiment && !error && (
          <div className={`result-card ${sentiment === 'POSITIF' ? 'positive' : 'negative'}`}>
            <span role="img" aria-label="result">
              {sentiment === 'POSITIF' ? '✅' : '⛔'}
            </span>
            Sentimen: {sentiment}
            <span role="img" aria-label="emoji" style={{marginLeft: '4px'}}>
              {sentiment === 'POSITIF' ? '😊' : '😞'}
            </span>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
