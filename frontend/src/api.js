/**
 * API client for the FastAPI backend.
 *
 * In development, Vite's proxy rewrites /api/* → http://localhost:8000/*
 * In production, set VITE_API_URL to the deployed backend URL.
 */
const BASE = import.meta.env.VITE_API_URL || '/api';

async function request(path, options = {}) {
    const res = await fetch(`${BASE}${path}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

export async function search(query) {
    return request('/query', {
        method: 'POST',
        body: JSON.stringify({ query }),
    });
}

export async function getCacheStats() {
    return request('/cache/stats');
}

export async function flushCache() {
    return request('/cache', { method: 'DELETE' });
}

export async function getHealth() {
    return request('/health');
}
