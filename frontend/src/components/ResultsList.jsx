import ResultCard from './ResultCard'

export default function ResultsList({ data }) {
    if (!data) {
        return (
            <div className="no-results" id="emptyState">
                <div className="no-results__icon">✨</div>
                <div className="no-results__text">
                    Enter a query to search across 20 newsgroup categories
                </div>
            </div>
        )
    }

    const blocks = data.result.split('\n\n').filter(Boolean)

    return (
        <div>
            <div className="results-title">
                Results <span className="count">{blocks.length}</span>
            </div>
            {blocks.map((block, i) => (
                <ResultCard key={i} block={block} index={i} />
            ))}
        </div>
    )
}
