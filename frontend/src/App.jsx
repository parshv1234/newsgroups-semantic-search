import { useState, useEffect } from 'react'
import { search, getCacheStats } from './api'
import HealthBar from './components/HealthBar'
import SearchBar from './components/SearchBar'
import CacheBadge from './components/CacheBadge'
import ResultsList from './components/ResultsList'
import StatsPanel from './components/StatsPanel'

export default function App() {
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getCacheStats().then(setStats).catch(() => { })
  }, [])

  async function handleSearch(query) {
    setLoading(true)
    setError(null)
    try {
      const data = await search(query)
      setResult(data)
      const fresh = await getCacheStats()
      setStats(fresh)
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header__icon">🔍</div>
        <h1 className="header__title">20 Newsgroups Semantic Search</h1>
        <p className="header__subtitle">
          NMF fuzzy clustering · Sentence-Transformers · Cluster-bucketed semantic cache
        </p>
      </header>

      <HealthBar />
      <SearchBar onSearch={handleSearch} loading={loading} />

      <div className="dashboard">
        <main className="results-area">
          {error && <div className="error-msg">⚠️ {error}</div>}
          <CacheBadge data={result} />
          <ResultsList data={result} />
        </main>
        <StatsPanel stats={stats} onStatsUpdate={setStats} />
      </div>

      <footer className="footer">
        Built for{' '}
        <a href="https://www.trademarkia.com" target="_blank" rel="noreferrer">
          Trademarkia
        </a>{' '}
        AI/ML Engineer Assignment ·{' '}
        <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
          API Docs ↗
        </a>
      </footer>
    </div>
  )
}
