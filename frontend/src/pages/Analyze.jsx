import { useState, useRef } from 'react';
import axios from 'axios';
import './Analyze.css'; // Analyze용 스타일 

function Analyze() {
  const [result, setResult] = useState('');
  const [originalImgUrl, setOriginalImgUrl] = useState('');
  const [gradcamUrl, setGradcamUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef();

  const handleFileChange = async (file) => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      setOriginalImgUrl(URL.createObjectURL(file));
      const response = await axios.post('http://localhost:8000/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data.result);
      setGradcamUrl(`http://localhost:8000/${response.data.gradcam_path}`);
    } catch (error) {
      console.error('업로드 실패', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleZoneClick = () => {
    fileInputRef.current.click();
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0]);
    }
  };

  return (
    <div className="page" onDragEnter={handleDrag}>
      <header className="header">
        Facial Bone X-ray - 분석
      </header>

      <main className="analyze-main">
        <div
          className={`drop-zone ${dragActive ? 'active' : ''}`}
          onClick={handleZoneClick}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
        >
          {loading ? (
            <p>🔎 분석 중입니다...</p>
          ) : (
            <p>파일을 여기에 드래그하거나 클릭하여 업로드하세요</p>
          )}
        </div>

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />

        <div className="analysis-result">
          <div className="image-panel">
            {originalImgUrl ? (
              <>
                <h3>업로드한 X-ray</h3>
                <img src={originalImgUrl} alt="Uploaded X-ray" />
              </>
            ) : (
              <p className="placeholder">여기에 업로드한 X-ray 이미지가 표시됩니다.</p>
            )}
          </div>

          <div className="result-panel">
            {result && (
              <div className={`result-text ${result === "Normal" ? "normal" : "abnormal"}`}>
                분류 결과: {result}
              </div>
            )}
            {gradcamUrl && (
              <>
                <h3>Grad-CAM 결과</h3>
                <img src={gradcamUrl} alt="Grad-CAM Result" />
              </>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        © 2025 Facial Bone X-ray
      </footer>
    </div>
  );
}

export default Analyze;
