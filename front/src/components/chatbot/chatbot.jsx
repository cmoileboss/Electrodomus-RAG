import "./chatbot.css"
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamMessage } from "../../services/ChatService";

export default function ChatBot() {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState("");
    const [loadingAIResponse, setLoadingAIResponse] = useState(false);
    const [streamingContent, setStreamingContent] = useState("");

    const handleSendMessage = async () => {
        if (loadingAIResponse || !inputText.trim()) return;
        const question = inputText.trim();
        const history = messages.map(m => ({ role: m.role, content: m.content }));

        setInputText("");
        setLoadingAIResponse(true);
        setMessages(prev => [...prev, { role: "user", content: question }]);

        let accumulated = "";
        try {
            await streamMessage(question, history, (token) => {
                accumulated += token;
                setStreamingContent(accumulated);
            });
            setMessages(prev => [...prev, { role: "assistant", content: accumulated }]);
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