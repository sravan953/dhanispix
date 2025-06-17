import React, { useState, useCallback } from "react";
import {
  Upload,
  Search,
  ImageIcon,
  Loader2,
  FolderOpen,
  Play,
} from "lucide-react";
import { useEffect } from "react";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [images, setImages] = useState([]);
  const [filteredImages, setFilteredImages] = useState([]);
  const [query, setQuery] = useState("");
  const [loadingImages, setLoadingImages] = useState(false);
  const [processingImages, setProcessingImages] = useState(false);
  const [searching, setSearching] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [processed, setProcessed] = useState(false);

  useEffect(() => {
    // Clear server variables when app loads
    fetch(`${API_BASE}/clear-variables/`, {
      method: "POST",
    });
  }, []);

  const handleFileUpload = useCallback(async (files) => {
    if (!files.length) return;

    const fileList = Array.from(files).filter((file) =>
      file.type.startsWith("image/")
    );
    setUploadedFiles(fileList);
    setProcessed(false);
  }, []);

  const handleProcessImages = useCallback(async () => {
    if (!uploadedFiles.length) return;

    setLoadingImages(true);
    setProcessingImages(true);

    const formData = new FormData();

    uploadedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      // Load images
      var response = await fetch(`${API_BASE}/upload-images/`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setImages(data.images);
      setFilteredImages(data.images);
      setLoadingImages(false);

      // Generate embeddings
      response = await fetch(`${API_BASE}/generate-embeddings/`, {
        method: "POST",
      });
      setProcessingImages(false);
    } catch (error) {
      console.error("Upload error:", error);
      alert(
        "Failed to upload images. Make sure the FastAPI server is running."
      );
    } finally {
      setLoadingImages(false);
      setProcessingImages(false);
      setProcessed(true);
    }
  }, [uploadedFiles]);

  const handleSearch = useCallback(async () => {
    if (!query.trim() || !images.length) return;

    setSearching(true);
    try {
      const response = await fetch(`${API_BASE}/search/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query.trim(),
        }),
      });

      const data = await response.json();
      setFilteredImages(data.results);
    } catch (error) {
      console.error("Search error:", error);
      alert("Search failed. Check the server connection.");
    } finally {
      setSearching(false);
    }
  }, [query, images]);

  const clearSearch = () => {
    setQuery("");
    setFilteredImages(images);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    handleFileUpload(files);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="min-h-screen w-full bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="w-full bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-800">dhanispix</h1>
      </div>

      <div className="flex flex-1">
        {/* Left Sidebar - 30% */}
        <div className="w-[30%] bg-white border-r border-gray-200 p-6 flex flex-col">
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <FolderOpen className="w-5 h-5" />
              File Browser
            </h2>

            {/* File Upload Area */}
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors cursor-pointer"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={(e) => handleFileUpload(e.target.files)}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer block">
                <Upload className="w-8 h-8 mx-auto text-gray-400 mb-3" />
                <p className="text-sm font-medium text-gray-700 mb-1">
                  Click or drag files here
                </p>
                <p className="text-xs text-gray-500">
                  Select multiple image files
                </p>
              </label>
            </div>
          </div>

          {/* File List */}
          {uploadedFiles.length > 0 && (
            <div className="flex-1 mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-3">
                Files ({uploadedFiles.length})
              </h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {uploadedFiles.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-2 bg-gray-50 rounded text-xs"
                  >
                    <div className="flex-1 min-w-0">
                      <p
                        className="truncate font-medium text-gray-800"
                        title={file.name}
                      >
                        {file.name}
                      </p>
                      <p className="text-gray-500">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                    <ImageIcon className="w-4 h-4 text-gray-400 ml-2 flex-shrink-0" />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Process Button */}
          <button
            onClick={handleProcessImages}
            disabled={
              loadingImages || processingImages || uploadedFiles.length === 0
            }
            className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
          >
            {loadingImages ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading images...
              </>
            ) : processingImages ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Process Images
              </>
            )}
          </button>

          {!loadingImages && images.length > 0 && (
            <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-700">
              ✓ {images.length} images processed successfully
            </div>
          )}
        </div>

        {/* Main Content - 70% */}
        <div className="w-[70%] p-6 flex flex-col">
          {!processed ? (
            /* Placeholder Content */
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-md">
                <ImageIcon className="w-16 h-16 mx-auto text-gray-300 mb-6" />
                <h2 className="text-2xl font-semibold text-gray-600 mb-3">
                  Load and Process Images First
                </h2>
                <p className="text-gray-500 leading-relaxed">
                  Select image files from the file browser on the left, then
                  click "Process Images" to analyze them with the SigLIP model.
                  Once processed, you'll be able to search through your images
                  using natural language.
                </p>
              </div>
            </div>
          ) : (
            /* Processed Content */
            <>
              <div className="mb-6">
                <h1 className="text-3xl font-bold text-gray-800 mb-2">
                  SigLIP Image Search
                </h1>
                <p className="text-gray-600">
                  Search your images using natural language
                </p>
              </div>

              {/* Search Section */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
                <div className="flex gap-4 items-center">
                  <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search images (e.g., 'a dog', 'red car', 'sunset')"
                      className="w-full text-black placeholder-text-gray-600 pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                      disabled={searching}
                    />
                  </div>
                  <button
                    onClick={handleSearch}
                    disabled={searching || !query.trim()}
                    className="px-6 py-3 bg-blue-600 text-white flex items-center gap-2"
                  >
                    {searching ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Search className="w-4 h-4" />
                    )}
                    Search
                  </button>
                  {query && (
                    <button
                      onClick={clearSearch}
                      className="px-4 py-3 text-white"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>

              {/* Results Section */}
              {filteredImages.length > 0 ? (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex-1">
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-semibold text-gray-800">
                      {query
                        ? `Search Results (${filteredImages.length})`
                        : `All Images (${filteredImages.length})`}
                    </h2>
                    {query && (
                      <span className="text-sm text-gray-500">
                        Searching for: "{query}"
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 overflow-y-auto">
                    {filteredImages.map((image) => (
                      <div key={image.id} className="group relative">
                        <div className="aspect-square overflow-hidden rounded-lg bg-gray-100 border hover:shadow-lg transition-shadow">
                          <img
                            src={image.base64_data}
                            alt={image.filename}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                          />
                        </div>
                        <div className="mt-2">
                          <p
                            className="text-xs text-gray-600 truncate"
                            title={image.filename}
                          >
                            {image.filename}
                          </p>
                          {image.similarity !== undefined && (
                            <p className="text-xs text-blue-600 font-medium">
                              Match: {(image.similarity * 100).toFixed(1)}%
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center flex-1 flex items-center justify-center">
                  <div>
                    <ImageIcon className="w-16 h-16 mx-auto text-gray-300 mb-4" />
                    <p className="text-gray-500 text-lg">
                      No images match your search
                    </p>
                    <p className="text-gray-400">Try a different search term</p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
