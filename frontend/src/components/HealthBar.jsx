import { useEffect, useState } from 'react'
import { getHealth } from '../api'

export default function HealthBar() {
    const [health, setHealth] = useState(null)
    const [error, setError] = useState(false)

    useEffect(() => {
        getHealth()
            .then((data) => { setHealth(data); setError(false) })
            .catch(() => setError(true))
    }, [])

    return (
        <div className="health-bar" id="healthBar">
            <div className={`health-chip ${error ? 'err' : health ? 'ok' : ''}`}>
                <span className="dot" />
                <span>{error ? 'API Offline' : health ? 'API Online' : 'Connecting…'}</span>
            </div>

            {health && (
                <>
                    <div className="health-chip ok" id="healthDocs">
                        📄 {health.vector_store_count.toLocaleString()} documents indexed
                    </div>
                    <div className="health-chip ok" id="healthModel">
                        🧠 {health.model}
                    </div>
                </>
            )}
        </div>
    )
}
