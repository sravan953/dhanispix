import {
  norm as tfNorm,
  tensor as tfTensor,
  matMul as tfMatMul,
  scalar as tfScalar,
  exp as tfExp,
  mul as tfMul,
  add as tfAdd,
  topk as tfTopk,
  sigmoid as tfSigmoid,
} from "@tensorflow/tfjs";
import { AutoTokenizer, SiglipTextModel } from "@huggingface/transformers";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  Upload,
  Search,
  ImageIcon,
  Loader2,
  FolderOpen,
  Link2,
  Play,
} from "lucide-react";
import * as ort from "onnxruntime-web";
import { AutoProcessor, RawImage } from "@huggingface/transformers";

ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

// Cosine similarity constants
const biasScalar = tfScalar(-12.9296875);
const expScale = tfExp(tfScalar(4.765625));

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [images, setImages] = useState([]);
  const [filteredImages, setFilteredImages] = useState([]);
  const [query, setQuery] = useState("");
  const [imagesLoaded, setImagesLoaded] = useState(false);
  const [processingImages, setProcessingImages] = useState(false);
  const [searching, setSearching] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [ngrokURL, setNgrokURL] = useState("");
  const [processor, setProcessor] = useState(null);
  const [embeddingModel, setEmbeddingModel] = useState(null);
  const [tokenizer, setTokenizer] = useState(null);
  const [textModel, setTextModel] = useState(null);
  const [imageEmbeddings, setImageEmbeddings] = useState([]);

  const modelLoaded = useRef(false);
  const tokenizerLoaded = useRef(false);
  const textModelLoaded = useRef(false);
  const processorLoaded = useRef(false);

  const handleNgrokURLChange = (e) => {
    setNgrokURL(e.target.value);
    console.log("Ngrok URL: ", e.target.value);
  };

  const handleFileUpload = useCallback(async (files) => {
    if (!files.length) return;

    const fileList = Array.from(files).filter((file) =>
      file.type.startsWith("image/")
    );
    setUploadedFiles(fileList);

    const imageUrls = fileList.map((file) => ({
      file: file,
      url: URL.createObjectURL(file),
      name: file.name,
      id: file.name,
    }));

    console.log(`Loading ${fileList.length} images...`);
    setImagesLoaded(true);
    setImages(imageUrls);
    setFilteredImages(imageUrls);
    console.log(`${imageUrls.length} images loaded.`);
  }, []);

  const handleProcessImages = useCallback(async () => {
    if (!uploadedFiles.length) return;

    setProcessingImages(true);
    console.log(`Reading ${uploadedFiles.length} images...`);

    const loadedImages = await Promise.all(
      uploadedFiles.map(async (file) => {
        return await RawImage.read(file);
      })
    );
    console.log(`Read ${loadedImages.length} images.`);

    console.log("Generating embeddings...");
    const processedImages = await processor(loadedImages);
    let embeddings = [];
    for (const input of processedImages.pixel_values) {
      const result = await embeddingModel.run({
        input: input.unsqueeze(0),
      });
      embeddings.push(result.output.cpuData);
    }
    embeddings = embeddings.map((arr) => Array.from(arr));
    console.log("Done.");

    console.log("Running inference on cloud...");
    const response = await fetch(`${ngrokURL}/generate-embeddings/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        pixel_values: embeddings,
      }),
    });
    const data = await response.json();
    let imageEmbeddings = data.image_embeddings;
    imageEmbeddings = Array.from(imageEmbeddings);
    imageEmbeddings = tfTensor(imageEmbeddings);
    setImageEmbeddings(imageEmbeddings);

    setProcessingImages(false);
    console.log(
      `Cloud inference completed. Received image embeddings of shape: ${imageEmbeddings.shape}`
    );
  }, [uploadedFiles, processor, ngrokURL, embeddingModel]);

  const l2Normalize = (tensor) => {
    let norm = tfNorm(tensor, 2, -1, true); // keepDims=true
    norm = tensor.div(norm);
    norm = norm.expandDims(0);
    return norm;
  };

  const handleSearch = useCallback(async () => {
    if (!query.trim() || !images.length) return;

    setSearching(true);
    const texts = [`This is a photo of ${query}`];
    const text_inputs = tokenizer(texts, {
      padding: "max_length",
    });
    let textEmbeddings = await textModel(text_inputs);
    textEmbeddings = Array.from(textEmbeddings.pooler_output[0]);
    textEmbeddings = tfTensor(textEmbeddings);
    textEmbeddings = l2Normalize(textEmbeddings);

    // Cosine similarity
    let logitsPerText = tfMatMul(imageEmbeddings, textEmbeddings, false, true);
    logitsPerText = tfMul(logitsPerText, expScale);
    logitsPerText = tfAdd(logitsPerText, biasScalar);
    logitsPerText = logitsPerText.squeeze();

    // Sort
    const size = logitsPerText.shape[0];
    const k = Math.min(10, size); // Get top 10 or all available
    const topk = tfTopk(logitsPerText, k, true);
    const topkLogitIndices = await topk.indices.array();
    const topkLogitValues = await topk.values.array();
    const topkLogitSigmoid = await tfSigmoid(topkLogitValues).array();
    console.log(topkLogitIndices, topkLogitValues, topkLogitSigmoid);

    // Reorder images based on topk indices
    const reorderedImages = topkLogitIndices.map((index) => images[index]);
    setFilteredImages(reorderedImages);
    setSearching(false);
  }, [query, images, textModel, tokenizer, imageEmbeddings]);

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

  const getReady = async () => {
    try {
      // if (!modelLoaded.current) {
      const loadedModel = await ort.InferenceSession.create(
        ".\\embeddings_onnx.onnx"
      );
      setEmbeddingModel(loadedModel);
      // }

      // if (!processorLoaded.current) {
      const loadedProcessor = await AutoProcessor.from_pretrained(
        "google/siglip-base-patch16-224"
      );
      setProcessor(() => loadedProcessor);
      // }

      // if (!tokenizerLoaded.current) {
      const loadedTokenizer = await AutoTokenizer.from_pretrained(
        "Xenova/siglip-base-patch16-224"
      );
      setTokenizer(() => loadedTokenizer);
      // }

      // if (!textModelLoaded.current) {
      const loadedTextModel = await SiglipTextModel.from_pretrained(
        "Xenova/siglip-base-patch16-224"
      );
      setTextModel(() => loadedTextModel);
      // }
    } catch (error) {
      console.error("Failed to load model:", error);
    }
  };

  useEffect(() => {
    if (textModel) {
      textModelLoaded.current = true;
      console.log("Text model loaded.");
    }
    if (processor) {
      processorLoaded.current = true;
      console.log("Processor loaded.");
    }
    if (embeddingModel) {
      modelLoaded.current = true;
      console.log("Embeddings model loaded.");
    }
    if (tokenizer) {
      tokenizerLoaded.current = true;
      console.log("Tokenizer loaded.");
    }
  }, [textModel, processor, tokenizer, embeddingModel]);

  useEffect(() => {
    if (isReady) return;
    setIsReady(true);
    getReady();
  }, [isReady]);

  return (
    <div className="min-h-screen w-full bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="w-full bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-800">dhanispix</h1>
      </div>

      <div className="flex flex-1">
        {/* Left Sidebar - 30% */}
        {isReady && (
          <div className="w-[30%] bg-white border-r border-gray-200 p-6 flex flex-col">
            <div className="flex items-center space-x-2 mb-6">
              <Link2 className="w-5 h-5 text-gray-500" />
              <input
                placeholder="ngrok public URL"
                value={ngrokURL}
                onChange={handleNgrokURLChange}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

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
                !modelLoaded ||
                !processorLoaded ||
                processingImages ||
                uploadedFiles.length === 0
              }
              className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
            >
              {processingImages ? (
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

            {imageEmbeddings.length > 0 && (
              <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-700">
                ✓ {images.length} images processed
              </div>
            )}
          </div>
        )}

        {/* Main Content - 70% */}
        {isReady && (
          <div className="w-[70%] p-6 flex flex-col">
            {!imagesLoaded ? (
              /* Placeholder Content */
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center max-w-md">
                  <ImageIcon className="w-16 h-16 mx-auto text-gray-300 mb-6" />
                  <h2 className="text-2xl font-semibold text-gray-600 mb-3">
                    Load and Process Images First
                  </h2>
                  <p className="text-gray-500 leading-relaxed">
                    Select image files from the file browser on the left, then
                    click "Process Images" to analyze them with the SigLIP
                    model. Once processed, you'll be able to search through your
                    images using natural language.
                  </p>
                </div>
              </div>
            ) : (
              /* Processed Content */
              <>
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-800 mb-2">
                    Dhani will not look at your photos 👀
                  </h2>
                </div>

                {/* Search Section */}
                {true && (
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
                )}

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
                              src={image.url}
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
                      <p className="text-gray-400">
                        Try a different search term
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
