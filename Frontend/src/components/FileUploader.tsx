import axios from "axios";
import { Upload } from "lucide-react";
import { useState } from "react";

const FileUploader = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
  };

  // Handle File Upload
  const handleFileUpload = async () => {
    if (!file || uploading) return;

    setUploading(true);
    setUploadStatus("Uploading & vectorizing...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        "http://localhost:8000/api/v1/files/upload",
        formData,
        {
          withCredentials: true,
        },
      );

      setUploadStatus(response.data.message);
    } catch (error: unknown) {
      const detail = axios.isAxiosError<{ detail?: string }>(error)
        ? error.response?.data?.detail
        : undefined;
      setUploadStatus(detail ?? "Upload failed. Check backend logs.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      {uploading ? (
        <div>uploading</div>
      ) : (
        <div>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition cursor-pointer relative">
            <input
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
            <Upload className="mx-auto h-8 w-8 text-gray-400 mb-2" />
            <p className="text-sm font-medium">
              Click or drag document to upload
            </p>
            <p className="text-xs text-gray-500 mt-1">
              PDF files only
            </p>
            {file && (
              <div>
                <div className="mt-4 p-3 bg-blue-50 rounded-md text-sm">
                  <p className="font-semibold text-blue-900 truncate ">
                    {file.name}
                  </p>
                  <p className="text-xs text-blue-700 mt-1">{uploadStatus}</p>
                </div>
              
              </div>
            )}
          </div>
            <button
              type="button"
              onClick={handleFileUpload}
              disabled={!file || uploading}
              className="border border-blue-400 bg-blue-400 text-white mt-4 disabled:opacity-50"
            >
              Upload
            </button>
        </div>
      )}
    </>
  );
};

export default FileUploader;
