import AssistantSection from "./components/AssistantSection";
import Sidebar from "./components/Sidebar";
// import { uploadDocument, queryRAG } from './services/api';

export default function App() {
  return (
    <div className="flex h-screen bg-gray-50 text-gray-800">
      {/* Sidebar: Document Upload */}
      <Sidebar />
      {/* Main Chat / Query Area */}
      <AssistantSection />
    </div>
  );
}
