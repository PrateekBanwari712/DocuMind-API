import { Bot, Loader2, Send, User } from "lucide-react";
import { useState, type FormEvent } from "react";

type ChatMessage = {
  sender: "user" | "bot";
  text: string;
  sources?: string[];
};

type ChatResponse = {
  answer: string;
  sources: string[];
};

const AssistantSection = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Handle Chat / Query Submission
  const handleSend = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input;
    setInput("");
    setMessages((prev: any) => [
      ...prev,
      { sender: "user", text: userMessage },
    ]);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/v1/chat/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });
      if (!response.ok) throw new Error("Chat request failed");

      const data: ChatResponse = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error connecting to RAG backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="h-full w-full flex-1 flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <h2 className="text-lg font-semibold">Query Assistant</h2>
      </header>

      {/* Message Log */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <Bot className="mx-auto h-12 w-12 mb-2 opacity-50" />
            <p>Upload a document and ask questions about its content.</p>
          </div>
        )}

        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex gap-3 ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
          >
            {msg.sender === "bot" && (
              <div className="p-2 bg-blue-600 text-white rounded-full h-8 w-8 flex items-center justify-center">
                <Bot size={16} />
              </div>
            )}
            <div
              className={`max-w-xl p-4 rounded-xl ${msg.sender === "user" ? "bg-blue-600 text-white" : "bg-white border border-gray-200 shadow-sm"}`}
            >
              <p className="whitespace-pre-wrap">{msg.text}</p>

              {/* Optional Source Citations Rendering */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-500">
                  <p className="font-semibold mb-1">Sources:</p>
                  {msg.sources.map((src, i) => (
                    <div
                      key={i}
                      className="bg-gray-50 p-1.5 rounded mb-1 border border-gray-200"
                    >
                      {src}
                    </div>
                  ))}
                </div>
              )}
            </div>
            {msg.sender === "user" && (
              <div className="p-2 bg-gray-700 text-white rounded-full h-8 w-8 flex items-center justify-center">
                <User size={16} />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-gray-400 text-sm">
            <Loader2 className="animate-spin" size={16} /> Analyzing
            documents...
          </div>
        )}
      </div>

      {/* Input Bar */}
      <form
        onSubmit={handleSend}
        className="bg-white border-t border-gray-200 p-4 flex gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your documents..."
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-5 py-2 rounded-lg font-medium hover:bg-blue-700 transition flex items-center gap-2 disabled:opacity-50"
        >
          <Send size={16} /> Send
        </button>
      </form>
    </main>
  );
};

export default AssistantSection;
