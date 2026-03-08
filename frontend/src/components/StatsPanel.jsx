import { useState } from 'react'
import { getCacheStats, flushCache } from '../api'

export default function StatsPanel({ stats, onStatsUpdate }) {
    const [flushing, setFlushing] = useState(false)
    const [toast, setToast] = useState(null)

    async function handleFlush() {
        if (!confirm('Flush the entire semantic cache? This cannot be undone.')) return
        setFlushing(true)
        try {
            const data = await flushCache()
            showToast(`✓ Cache flushed — ${data.entries_cleared} entries cleared`, 'success')
            const fresh = await getCacheStats()
            onStatsUpdate(fresh)
        } catch {
            showToast('Failed to flush cache', 'error')
        } finally {
            setFlushing(false)
        }
    }

    function showToast(msg, type) {
        setToast({ msg, type })
        setTimeout(() => setToast(null), 3200)
    }

    const hitRate = stats ? (stats.hit_rate * 100).toFixed(1) : '0.0'

    return (
        <aside className="sidebar">
            <div className="stats-card">
                <div className="stats-card__title">📊 Cache Statistics</div>

                <div className="hit-rate-section">
                    <div className="hit-rate-label">
                        <span>Hit Rate</span>
                        <span id="hitRateValue">{hitRate}%</span>
                    </div>
                    <div className="hit-rate-bar">
                        <div
                            className="hit-rate-bar__fill"
                            id="hitRateBar"
                            style={{ width: `${hitRate}%` }}
                        />
                    </div>
                </div>

                <div className="stat-row">
                    <span className="stat-row__label">Total Entries</span>
                    <span className="stat-row__value" id="statEntries">
                        {stats?.total_entries ?? 0}
                    </span>
                </div>
                <div className="stat-row">
                    <span className="stat-row__label">Hits</span>
                    <span className="stat-row__value stat-hits" id="statHits">
                        {stats?.hit_count ?? 0}
                    </span>
                </div>
                <div className="stat-row">
                    <span className="stat-row__label">Misses</span>
                    <span className="stat-row__value stat-misses" id="statMisses">
                        {stats?.miss_count ?? 0}
                    </span>
                </div>

                <button
                    id="flush-btn"
                    className="flush-btn"
                    type="button"
                    onClick={handleFlush}
                    disabled={flushing}
                >
                    🗑️ {flushing ? 'Flushing…' : 'Flush Cache'}
                </button>
            </div>

            {toast && (
                <div className={`toast toast--${toast.type}`}>{toast.msg}</div>
            )}
        </aside>
    )
}
