import "./chatbot.css"
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getErrorCodes, getModels, sendMessage } from "../../services/ChatService";

export default function ChatBot() {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState("");
    const [loadingAIResponse, setLoadingAIResponse] = useState(false);
    const [streamingContent, setStreamingContent] = useState("");
    const [models, setModels] = useState([]);
    const [errorCodes, setErrorCodes] = useState([]);
    const [selectedModelId, setSelectedModelId] = useState("");
    const [selectedErrorCode, setSelectedErrorCode] = useState("");

    useEffect(() => {
        getModels().then(setModels).catch(() => setModels([]));
    }, []);

    useEffect(() => {
        setSelectedErrorCode("");
        getErrorCodes(selectedModelId || null)
            .then(setErrorCodes)
            .catch(() => setErrorCodes([]));
    }, [selectedModelId]);

    const handleSendMessage = async () => {
        if (loadingAIResponse || !inputText.trim()) return;
        const question = inputText.trim();
        const history = messages.map(m => ({ role: m.role, content: m.content }));

        setInputText("");
        setLoadingAIResponse(true);
        setMessages(prev => [...prev, { role: "user", content: question }]);

        try {
            const { answer } = await sendMessage(
                question,
                history,
                selectedModelId || null,
                selectedErrorCode || null
            );
            setMessages(prev => [...prev, { role: "assistant", content: answer }]);
        } catch (error) {
            console.error("Error sending message:", error);
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
            </div>
            <div className="messages-container">
                {messages.length === 0 && !streamingContent && (
                    <div className="no-messages">Aucun message pour le moment. Commencez la conversation !</div>
                )}
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.role}`}>
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                ))}
                {streamingContent && (
                    <div className="message assistant">
                        <ReactMarkdown>{streamingContent}</ReactMarkdown>
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