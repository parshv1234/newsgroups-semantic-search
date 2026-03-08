import { useState } from 'react'

export default function SearchBar({ onSearch, loading }) {
    const [query, setQuery] = useState('')

    function handleSubmit(e) {
        e.preventDefault()
        const q = query.trim()
        if (!q || loading) return
        onSearch(q)
    }

    return (
        <div className="search-card">
            <form className="search-form" onSubmit={handleSubmit}>
                <input
                    id="search-input"
                    type="text"
                    className="search-input"
                    placeholder='Ask anything — e.g. "How do space shuttles work?"'
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    autoComplete="off"
                    autoFocus
                />
                <button
                    id="search-btn"
                    type="submit"
                    className={`search-btn${loading ? ' loading' : ''}`}
                    disabled={loading}
                >
                    <span className="btn-text">Search</span>
                    <span className="spinner" />
                </button>
            </form>
        </div>
    )
}
