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

/**
 * Streams tokens from /chat/stream, calling onToken for each token received.
 * Returns the final { history } payload from the done event.
 */
export async function streamMessage(question, history = [], onToken) {
    const response = await fetch(`${baseURL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history }),
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete last line

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));
            if (data.type === 'token') onToken(data.content);
            if (data.type === 'done') return data;
        }
    }
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
