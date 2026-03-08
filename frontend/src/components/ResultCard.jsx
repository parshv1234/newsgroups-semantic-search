export default function ResultCard({ block, index }) {
    const lines = block.split('\n')
    const headerLine = lines[0] || ''
    const snippet = lines.slice(1).join(' ').trim()

    const simMatch = headerLine.match(/similarity=([\d.]+)/)
    const ngMatch = headerLine.match(/newsgroup=([\w.]+)/)
    const clMatch = headerLine.match(/cluster=(\d+)/)

    const sim = simMatch ? simMatch[1] : '—'
    const newsgroup = ngMatch ? ngMatch[1] : 'unknown'
    const cluster = clMatch ? clMatch[1] : '?'

    return (
        <div
            className="result-card"
            style={{ animationDelay: `${index * 0.06}s` }}
        >
            <div className="result-card__header">
                <span className="result-card__rank">#{index + 1}</span>
                <span className="result-card__sim">
                    {(parseFloat(sim) * 100).toFixed(1)}%
                </span>
                <span className="tag tag--newsgroup">{newsgroup}</span>
                <span className="tag tag--cluster">C{cluster}</span>
            </div>
            <div className="result-card__snippet">{snippet}</div>
        </div>
    )
}
