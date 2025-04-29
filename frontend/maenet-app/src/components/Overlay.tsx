import React, { useEffect, useRef } from "react";
import "./Overlay.css";

interface OverlayProps {
  inputImage: string;
  maskImage: string;
  onClose: () => void;
}

const Overlay: React.FC<OverlayProps> = ({ inputImage, maskImage, onClose }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img1 = new Image();
    img1.src = `data:image/png;base64,${inputImage}`;
    const img2 = new Image();
    img2.src = `data:image/png;base64,${maskImage}`;

    Promise.all([
      new Promise((resolve) => (img1.onload = resolve)),
      new Promise((resolve) => (img2.onload = resolve)),
    ]).then(() => {
      canvas.width = img1.width;
      canvas.height = img1.height;
      ctx.drawImage(img1, 0, 0);
      ctx.globalAlpha = 0.5;
      ctx.drawImage(img2, 0, 0);
      ctx.globalAlpha = 1;
    });
  }, [inputImage, maskImage]);

  return (
    <div className="overlay-popup">
      <div className="overlay-content">
        <canvas ref={canvasRef} />
        <button onClick={onClose} className="close-button">Close</button>
      </div>
    </div>
  );
};

export default Overlay;
