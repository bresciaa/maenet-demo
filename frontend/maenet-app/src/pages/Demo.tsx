import React, { useState } from "react";
import "../App.css";
import "./Demo.css";
import UploadComponent from "../components/UploadComponent";
import Overlay from "../components/Overlay";
import JSZip from "jszip";

interface OverlayProps {
  inputImage: string;
  maskImage: string;
  onClose: () => void;
}

interface PredictionResponse {
  file_name: string;
  input_image: string;
  mask_image: string;
}

enum UploadStatus {
  Idle,
  Uploading,
  Success,
  Failure,
}

function Demo() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>(
    UploadStatus.Idle,
  );
  const [showOverlay, setShowOverlay] = useState<boolean>(false);
  
  //
  // Download images
  //
  const handleDownload = async () => {
    

    if (!prediction) return;

    try {
      const zip = new JSZip();

      const inputBlob = await base64ToBlob(prediction.input_image, "image/png");
      const maskBlob = await base64ToBlob(prediction.mask_image, "image/png");


      if (!inputBlob || !maskBlob) {
        throw new Error("Failed to convert base64 to blob.");
      }

      zip.file("input_image.png", inputBlob);
      zip.file("mask_image.png", maskBlob);

      const content = await zip.generateAsync({ type: "blob" });

      const url = URL.createObjectURL(content);
      const a = document.createElement("a");
      a.href = url;
      a.download = `predicted_${prediction.file_name}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading images:", error);
      // Handle the error (e.g., show an error message to the user)
    }
  };

  //
  // Base64 to Blob
  //
  const base64ToBlob = async (base64: string, type: string) => {
    try {
      const response = await fetch(`data:${type};base64,${base64}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch blob: ${response.statusText}`);
      }
      const blob = await response.blob();
      return blob;
    } catch (error) {
      console.error("Error converting base64 to blob:", error);
      return null; // Or handle the error in another way
    }
  };

  //
  // Handle overlay
  //
  const handleOverlay = () => {
    setShowOverlay(true);
  };

  const handleCloseOverlay = () => {
    setShowOverlay(false);
  };

  return (
    <section id="demo">
      <UploadComponent
        onPrediction={setPrediction}
        onUploadStatusChange={setUploadStatus}
      />
      {uploadStatus === UploadStatus.Success && prediction && (
        <div className="prediction-results">
          <div className="prediction-images">
            <div className="image-container">
              <h4>Input Image</h4>
              <img
                src={`data:image/png;base64,${prediction.input_image}`}
                alt="Input"
              />
            </div>
            <div className="image-container">
              <h4>Output Image</h4>
              <img
                src={`data:image/png;base64,${prediction.mask_image}`}
                alt="Output"
              />
            </div>
          </div>
          <div className="prediction-buttons">
            <button onClick={handleOverlay} disabled={!prediction}>Overlay</button>
            <button onClick={handleDownload} disabled={!prediction}>Download</button>
          </div>
        </div>
      )}
      {uploadStatus === UploadStatus.Failure && (
        <div className="prediction-results">
          <p>File upload failed.</p>
        </div>
      )}

      {showOverlay && prediction && (
        <Overlay
          inputImage={prediction.input_image}
          maskImage={prediction.mask_image}
          onClose={handleCloseOverlay}
        />
      )}
    </section>
  );
}

export default Demo;