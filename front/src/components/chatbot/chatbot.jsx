import "./chatbot.css"
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getErrorCodes, getModels, getRagConfig, sendMessage } from "../../services/ChatService";

const CONFIG_STORAGE_KEY = "ragConfig";

const CHUNK_FIELDS = [
    { key: "vector_limit", label: "Recherche vectorielle" },
    { key: "bm25_limit", label: "Recherche BM25" },
    { key: "rrf_limit", label: "Après fusion (RRF)" },
    { key: "rerank_top_k", label: "Envoyés au LLM (après reranking)" },
];

function loadSavedConfig() {
    try {
        return JSON.parse(localStorage.getItem(CONFIG_STORAGE_KEY)) ?? {};
    } catch {
        return {};
    }
}

// Ramène chaque nombre de chunks dans [1, maxLimit] et vérifie que le modèle est autorisé.
function sanitizeConfig(config, defaults, serverConfig) {
    const result = { model: serverConfig.models.includes(config.model) ? config.model : defaults.model };
    for (const { key } of CHUNK_FIELDS) {
        const value = parseInt(config[key], 10);
        result[key] = Number.isNaN(value) ? defaults[key] : Math.min(Math.max(value, 1), serverConfig.max_limit);
    }
    return result;
}

function remarkBrTags() {
    return (tree) => {
        const convertBreakTags = (node) => {
            if (!Array.isArray(node.children)) return;
            node.children = node.children.flatMap((child) => {
                if (child.type === "html" && /^<br\s*\/?>$/i.test(child.value.trim())) {
                    return [{ type: "break" }];
                }
                convertBreakTags(child);
                return [child];
            });
        };
        convertBreakTags(tree);
    };
}

const markdownPlugins = [remarkGfm, remarkBrTags];

// Le conteneur gère le défilement horizontal : le <table> peut ainsi occuper toute la largeur de la bulle.
const markdownComponents = {
    table: ({ node, ...props }) => (
        <div className="table-wrapper">
            <table {...props} />
        </div>
    ),
};

export default function ChatBot() {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState("");
    const [loadingAIResponse, setLoadingAIResponse] = useState(false);
    const [streamingContent, setStreamingContent] = useState("");
    const [models, setModels] = useState([]);
    const [errorCodes, setErrorCodes] = useState([]);
    const [selectedModelId, setSelectedModelId] = useState("");
    const [selectedErrorCode, setSelectedErrorCode] = useState("");
    const [serverConfig, setServerConfig] = useState(null);
    const [ragConfig, setRagConfig] = useState(null);
    const [showSettings, setShowSettings] = useState(false);

    useEffect(() => {
        getModels().then(setModels).catch(() => setModels([]));
        getRagConfig()
            .then((server) => {
                setServerConfig(server);
                setRagConfig(sanitizeConfig({ ...server.defaults, ...loadSavedConfig() }, server.defaults, server));
            })
            .catch(() => setServerConfig(null));
    }, []);

    const updateRagConfig = (key, value) => {
        const next = { ...ragConfig, [key]: value };
        setRagConfig(next);
        localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(next));
    };

    const resetRagConfig = () => {
        setRagConfig({ ...serverConfig.defaults });
        localStorage.removeItem(CONFIG_STORAGE_KEY);
    };

    useEffect(() => {
        setSelectedErrorCode("");
        getErrorCodes(selectedModelId || null)
            .then(setErrorCodes)
            .catch(() => setErrorCodes([]));
    }, [selectedModelId]);

    const handleSendMessage = async () => {
        if (loadingAIResponse || !inputText.trim()) return;
        const question = inputText.trim();
        const history = messages.filter(m => !m.isError).map(m => ({ role: m.role, content: m.content }));

        setInputText("");
        setLoadingAIResponse(true);
        setMessages(prev => [...prev, { role: "user", content: question }]);

        try {
            const { answer } = await sendMessage(
                question,
                history,
                selectedModelId || null,
                selectedErrorCode || null,
                ragConfig && serverConfig ? sanitizeConfig(ragConfig, serverConfig.defaults, serverConfig) : null
            );
            setMessages(prev => [...prev, { role: "assistant", content: answer }]);
        } catch (error) {
            console.error("Error sending message:", error);
            setMessages(prev => [...prev, { role: "assistant", content: error.message, isError: true }]);
        } finally {
            setStreamingContent("");
            setLoadingAIResponse(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === "Enter") handleSendMessage();
    };

    return (
        <section className="chat">
            <div className="chat-filters">
                <select
                    className="chat-filter-select"
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                >
                    <option value="">Tous les modèles</option>
                    {models.map((m) => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                </select>
                <select
                    className="chat-filter-select"
                    value={selectedErrorCode}
                    onChange={(e) => setSelectedErrorCode(e.target.value)}
                >
                    <option value="">Tous les codes erreur</option>
                    {errorCodes.map((e) => (
                        <option key={e.code} value={e.code}>{e.code}</option>
                    ))}
                </select>
                {serverConfig && ragConfig && (
                    <button
                        type="button"
                        className="chat-settings-toggle"
                        onClick={() => setShowSettings((v) => !v)}
                    >
                        {showSettings ? "Masquer les paramètres" : "Paramètres du RAG"}
                    </button>
                )}
            </div>
            {showSettings && serverConfig && ragConfig && (
                <div className="chat-settings">
                    <label className="chat-settings-field">
                        <span>Modèle de langage</span>
                        <select
                            className="chat-filter-select"
                            value={ragConfig.model}
                            onChange={(e) => updateRagConfig("model", e.target.value)}
                        >
                            {serverConfig.models.map((name) => (
                                <option key={name} value={name}>{name}</option>
                            ))}
                        </select>
                    </label>
                    {CHUNK_FIELDS.map(({ key, label }) => (
                        <label key={key} className="chat-settings-field">
                            <span>Chunks : {label}</span>
                            <input
                                type="number"
                                className="chat-settings-number"
                                min="1"
                                max={serverConfig.max_limit}
                                value={ragConfig[key]}
                                onChange={(e) => updateRagConfig(key, e.target.value)}
                                onBlur={() => setRagConfig(sanitizeConfig(ragConfig, serverConfig.defaults, serverConfig))}
                            />
                        </label>
                    ))}
                    <button type="button" className="chat-settings-reset" onClick={resetRagConfig}>
                        Valeurs par défaut
                    </button>
                </div>
            )}
            <p className="chat-filter-hint">
                L'utilisation des filtres n'est pas obligatoire mais permet de réaliser une recherche plus optimale.
            </p>
            <div className="messages-container">
                {messages.length === 0 && !streamingContent && (
                    <div className="no-messages">Aucun message pour le moment. Commencez la conversation !</div>
                )}
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.role}${msg.isError ? " error" : ""}`}>
                        <ReactMarkdown remarkPlugins={markdownPlugins} components={markdownComponents}>{msg.content}</ReactMarkdown>
                    </div>
                ))}
                {streamingContent && (
                    <div className="message assistant">
                        <ReactMarkdown remarkPlugins={markdownPlugins} components={markdownComponents}>{streamingContent}</ReactMarkdown>
                    </div>
                )}
                {loadingAIResponse && !streamingContent && (
                    <div className="message assistant loading-indicator"><span>...</span></div>
                )}
            </div>
            <div className="input-container">
                <input
                    type="text"
                    className="prompt-input"
                    placeholder="Écrivez votre message..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyUp={handleKeyPress}
                    maxLength="40000"
                />
                <button className="send-button" onClick={handleSendMessage} disabled={loadingAIResponse}>
                    Envoyer
                </button>
            </div>
        </section>
    );
}