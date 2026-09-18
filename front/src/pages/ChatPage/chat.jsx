import "./chat.css";
import { useRef, useState } from "react";
import ChatBot from "../../components/chatbot/chatbot";
import { deleteAllDocuments, deleteDocument, getChunksCount, getDocuments, ingestAll, ingestUpload } from "../../services/ChatService";

export default function ChatPage() {
    const [ingesting, setIngesting] = useState(false);
    const [ingestMessage, setIngestMessage] = useState(null);
    const [showDocuments, setShowDocuments] = useState(false);
    const [documents, setDocuments] = useState([]);
    const [loadingDocuments, setLoadingDocuments] = useState(false);
    const [documentsError, setDocumentsError] = useState(null);
    const [chunksCount, setChunksCount] = useState(null);
    const [deletingDocumentId, setDeletingDocumentId] = useState(null);
    const [deletingAll, setDeletingAll] = useState(false);
    const fileInputRef = useRef(null);

    const handleIngestAll = () => {
        setIngesting(true);
        setIngestMessage(null);
        ingestAll()
            .then(() => setIngestMessage({ ok: true, text: "Ingestion complète terminée." }))
            .catch(() => setIngestMessage({ ok: false, text: "Erreur lors de l'ingestion." }))
            .finally(() => setIngesting(false));
    };

    const handleFileSelected = (e) => {
        const file = e.target.files[0];
        e.target.value = "";
        if (!file) return;
        setIngesting(true);
        setIngestMessage(null);
        ingestUpload(file)
            .then(() => setIngestMessage({ ok: true, text: `"${file.name}" ingéré avec succès.` }))
            .catch((e) => setIngestMessage({ ok: false, text: e.message }))
            .finally(() => setIngesting(false));
    };

    const handleToggleDocuments = () => {
        const next = !showDocuments;
        setShowDocuments(next);
        if (next) {
            refreshDocuments();
        }
    };

    const refreshDocuments = () => {
        setLoadingDocuments(true);
        setDocumentsError(null);
        getDocuments()
            .then(setDocuments)
            .catch((e) => setDocumentsError(e.message))
            .finally(() => setLoadingDocuments(false));
        getChunksCount()
            .then((res) => setChunksCount(res.count))
            .catch(() => setChunksCount(null));
    };

    const handleDeleteDocument = (doc) => {
        if (!window.confirm(`Supprimer "${doc.title}" et tous ses chunks ?`)) return;
        setDeletingDocumentId(doc.id);
        deleteDocument(doc.id)
            .then(() => refreshDocuments())
            .catch((e) => setDocumentsError(e.message))
            .finally(() => setDeletingDocumentId(null));
    };

    const handleDeleteAllDocuments = () => {
        if (!window.confirm("Supprimer tous les documents et tous leurs chunks ?")) return;
        setDeletingAll(true);
        deleteAllDocuments()
            .then(() => refreshDocuments())
            .catch((e) => setDocumentsError(e.message))
            .finally(() => setDeletingAll(false));
    };

    return (
        <div className="chat-component">
            <div className="ingest-toolbar">
                <button className="ingest-button" onClick={handleIngestAll} disabled={ingesting}>
                    Ingérer tous les documents
                </button>
                <button className="ingest-button" onClick={() => fileInputRef.current?.click()} disabled={ingesting}>
                    Ingérer un fichier…
                </button>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.docx,.html"
                    onChange={handleFileSelected}
                    style={{ display: "none" }}
                />
                <button className="ingest-button" onClick={handleToggleDocuments}>
                    {showDocuments ? "Masquer les fichiers" : "Voir les fichiers du RAG"}
                </button>
                {ingestMessage && (
                    <span className={`ingest-message ${ingestMessage.ok ? "ok" : "error"}`}>
                        {ingestMessage.text}
                    </span>
                )}
            </div>
            {showDocuments && (
                <div className="documents-panel">
                    {chunksCount !== null && (
                        <span className="documents-chunks-count">Total de chunks : {chunksCount}</span>
                    )}
                    {documents.length > 0 && (
                        <button
                            className="documents-delete-all-button"
                            onClick={handleDeleteAllDocuments}
                            disabled={deletingAll}
                        >
                            {deletingAll ? "Suppression…" : "Supprimer tous les documents"}
                        </button>
                    )}
                    {loadingDocuments && <span className="documents-status">Chargement…</span>}
                    {documentsError && <span className="documents-status error">{documentsError}</span>}
                    {!loadingDocuments && !documentsError && (
                        documents.length === 0 ? (
                            <span className="documents-status">Aucun document ingéré.</span>
                        ) : (
                            <ul className="documents-list">
                                {documents.map((doc) => (
                                    <li key={doc.id} className="documents-list-item">
                                        <span className="documents-list-title">{doc.title}</span>
                                        <span className="documents-list-path">{doc.filepath}</span>
                                        <button
                                            className="documents-delete-button"
                                            onClick={() => handleDeleteDocument(doc)}
                                            disabled={deletingDocumentId === doc.id}
                                        >
                                            {deletingDocumentId === doc.id ? "Suppression…" : "Supprimer"}
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        )
                    )}
                </div>
            )}
            <ChatBot />
        </div>
    );
}