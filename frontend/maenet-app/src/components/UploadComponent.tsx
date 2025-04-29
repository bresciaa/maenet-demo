import React, { useState, useRef, useCallback } from "react"
import "../App.css";
import "./UploadComponent.css";
import searchIcon from "../assets/magnifyingglass.png"

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

interface UploadComponentProps {
  onPrediction: (prediction: PredictionResponse) => void;
  onUploadStatusChange: (status: UploadStatus) => void;
}

const UploadComponent: React.FC<UploadComponentProps> = ({
  onPrediction,
  onUploadStatusChange,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);

  //
  // Handle file upload
  //
  const handleFiles = useCallback(async (fileToUpload: File) => {
    try {
      const formData = new FormData();
      formData.append("image", fileToUpload);

      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data: PredictionResponse = await response.json();
        onPrediction(data);
        onUploadStatusChange(UploadStatus.Success);
      } else {
        console.error("Error: ", response.statusText);
        onUploadStatusChange(UploadStatus.Failure);
      }
    } catch (error) {
      console.error("Error uploading file: ", error);
      onUploadStatusChange(UploadStatus.Failure);
    }
  }, [onPrediction, onUploadStatusChange]);

  //
  // Handle submit
  //
  const handleSubmit = async () => {
    if (file) {
      onUploadStatusChange(UploadStatus.Uploading);
      await handleFiles(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleClearFile = () => {
    setFile(null);
    onUploadStatusChange(UploadStatus.Idle);
    onPrediction({ file_name: "", input_image: "", mask_image: "" });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <form id="upload-form">
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="upload-box"
      >
        <p>Upload an image here.</p>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={{ display: "none" }}
          accept="image/*"
        />
      </div>

      {file && (
        <div className="selected-file">
          File selected: {file.name}
          <button className="clear-button" type="button" onClick={handleClearFile}>
            Clear
          </button>
        </div>
      )}

      <button className="find-bikelanes-button" type="button" onClick={handleSubmit} disabled={!file}>
        <img src={searchIcon} alt="Search Icon" className="button-icon" />
        Find Bikelanes
      </button>
    </form>
  );
};

export default UploadComponent;
