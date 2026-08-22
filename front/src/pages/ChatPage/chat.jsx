import "./chat.css";
import { useState } from "react";
import ChatBot from "../../components/chatbot/chatbot";
import { ingestAll, ingestSingle } from "../../services/ChatService";

export default function ChatPage() {
    const [ingesting, setIngesting] = useState(false);
    const [ingestMessage, setIngestMessage] = useState(null);
    const [showPathInput, setShowPathInput] = useState(false);
    const [filepath, setFilepath] = useState("");

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

    return (
        <div className="chat-component">
            <div className="ingest-toolbar">
                <button className="ingest-button" onClick={handleIngestAll} disabled={ingesting}>
                    Ingérer tous les documents
                </button>
                <button className="ingest-button" onClick={() => { setShowPathInput(p => !p); setIngestMessage(null); }} disabled={ingesting}>
                    Ingérer un fichier…
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
            <ChatBot />
        </div>
    );
}