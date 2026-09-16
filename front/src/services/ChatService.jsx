const baseURL = '/api'

export async function sendMessage(question, history = []) {
    const response = await fetch(`${baseURL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history })
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function ingestAll() {
    const response = await fetch(`${baseURL}/documents/ingest-all`, { method: 'POST' });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function ingestSingle(filepath) {
    const response = await fetch(`${baseURL}/documents/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath })
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function getDocuments() {
    const response = await fetch(`${baseURL}/documents/`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function getChunksCount() {
    const response = await fetch(`${baseURL}/chunks/count`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}
