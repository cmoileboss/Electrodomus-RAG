import "./chat.css";
import { useState } from "react";
import ChatBot from "../../components/chatbot/chatbot";
import { getChunksCount, getDocuments, ingestAll, ingestSingle } from "../../services/ChatService";

export default function ChatPage() {
    const [ingesting, setIngesting] = useState(false);
    const [ingestMessage, setIngestMessage] = useState(null);
    const [showPathInput, setShowPathInput] = useState(false);
    const [filepath, setFilepath] = useState("");
    const [showDocuments, setShowDocuments] = useState(false);
    const [documents, setDocuments] = useState([]);
    const [loadingDocuments, setLoadingDocuments] = useState(false);
    const [documentsError, setDocumentsError] = useState(null);
    const [chunksCount, setChunksCount] = useState(null);

    const handleIngestAll = () => {
        setIngesting(true);
        setIngestMessage(null);
        ingestAll()
            .then(() => setIngestMessage({ ok: true, text: "Ingestion complète terminée." }))
            .catch(() => setIngestMessage({ ok: false, text: "Erreur lors de l'ingestion." }))
            .finally(() => setIngesting(false));
    };

    const handleIngestSingle = () => {
        if (!filepath.trim()) return;
        setIngesting(true);
        setIngestMessage(null);
        ingestSingle(filepath.trim())
            .then(() => {
                setIngestMessage({ ok: true, text: `"${filepath.trim()}" ingéré avec succès.` });
                setFilepath("");
                setShowPathInput(false);
            })
            .catch((e) => setIngestMessage({ ok: false, text: e.message }))
            .finally(() => setIngesting(false));
    };

    const handleToggleDocuments = () => {
        const next = !showDocuments;
        setShowDocuments(next);
        if (next) {
            setLoadingDocuments(true);
            setDocumentsError(null);
            getDocuments()
                .then(setDocuments)
                .catch((e) => setDocumentsError(e.message))
                .finally(() => setLoadingDocuments(false));
            getChunksCount()
                .then((res) => setChunksCount(res.count))
                .catch(() => setChunksCount(null));
        }
    };

    return (
        <div className="chat-component">
            <div className="ingest-toolbar">
                <button className="ingest-button" onClick={handleIngestAll} disabled={ingesting}>
                    Ingérer tous les documents
                </button>
                <button className="ingest-button" onClick={() => { setShowPathInput(p => !p); setIngestMessage(null); }} disabled={ingesting}>
                    Ingérer un fichier…
                </button>
                <button className="ingest-button" onClick={handleToggleDocuments}>
                    {showDocuments ? "Masquer les fichiers" : "Voir les fichiers du RAG"}
                </button>
                {showPathInput && (
                    <>
                        <input
                            className="ingest-path-input"
                            type="text"
                            placeholder="Chemin du fichier sur le serveur"
                            value={filepath}
                            onChange={e => setFilepath(e.target.value)}
                            onKeyUp={e => e.key === "Enter" && handleIngestSingle()}
                        />
                        <button className="ingest-button ingest-confirm" onClick={handleIngestSingle} disabled={ingesting || !filepath.trim()}>
                            Lancer
                        </button>
                    </>
                )}
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