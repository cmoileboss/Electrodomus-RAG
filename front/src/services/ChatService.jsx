const baseURL = '/api'

export async function sendMessage(question, history = [], modelId = null, errorCode = null) {
    const response = await fetch(`${baseURL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history, model_id: modelId, error_code: errorCode })
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function getModels() {
    const response = await fetch(`${baseURL}/models/`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function getErrorCodes(modelId = null) {
    const url = modelId ? `${baseURL}/error-codes/?model_id=${modelId}` : `${baseURL}/error-codes/`;
    const response = await fetch(url);
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

export async function ingestUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${baseURL}/documents/ingest-upload`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function getDocuments() {
    const response = await fetch(`${baseURL}/documents/`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function deleteDocument(documentId) {
    const response = await fetch(`${baseURL}/documents/${documentId}`, { method: 'DELETE' });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
}

export async function deleteAllDocuments() {
    const response = await fetch(`${baseURL}/documents/all`, { method: 'DELETE' });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

export async function getChunksCount() {
    const response = await fetch(`${baseURL}/chunks/count`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}
