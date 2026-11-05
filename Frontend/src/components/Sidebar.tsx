import { FileText } from "lucide-react";
import FileUploader from "./FileUploader";

const Sidebar = () => {
  return (
    <aside className="w-80 bg-white border-r border-gray-200 p-6 flex flex-col justify-between">
      <div>
        <h1 className="text-xl font-bold mb-6 flex items-center gap-2">
          <FileText className="text-blue-600" /> DocuMind API
        </h1>

        <FileUploader />
      </div>

      {/* <div className="border-b border-2 border-b-black"></div> */}

      <div className="text-xs text-gray-400 text-center">
        Powered by React & Python RAG
      </div>
    </aside>
  );
};

export default Sidebar;
