export default function CacheBadge({ data }) {
    if (!data) return null

    return (
        <div className="cache-badge-area">
            {data.cache_hit ? (
                <>
                    <div className="cache-badge hit" id="cache-hit-badge">
                        ⚡ Cache Hit
                        <span className="cache-badge__score">
                            {(data.similarity_score * 100).toFixed(1)}% match
                        </span>
                    </div>
                    {data.matched_query && (
                        <div className="matched-query">
                            <strong>Matched:</strong> &quot;{data.matched_query}&quot;
                        </div>
                    )}
                </>
            ) : (
                <div className="cache-badge miss" id="cache-miss-badge">
                    🔄 Cache Miss — Queried ChromaDB
                </div>
            )}

            <div className="cluster-label">
                Dominant cluster:{' '}
                <span className="tag tag--cluster">C{data.dominant_cluster}</span>
            </div>
        </div>
    )
}
