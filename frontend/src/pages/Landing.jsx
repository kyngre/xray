import { useNavigate } from 'react-router-dom';
import './Landing.css'; // ✨ 분리된 스타일 가져오기

function Landing() {
  const navigate = useNavigate();

  const goToAnalyze = () => {
    navigate('/analyze');
  };

  return (
    <div className="page">
      <header className="header">
        Facial Bone X-ray
      </header>

      <main className="landing-main">
        {/* Hero Section */}
        <section className="hero">
          <h1>AI 기반 얼굴 뼈 X-ray 자동 분석 솔루션</h1>
          <p className="hero-subtitle">
            합성 데이터를 활용하여 신뢰성 있는 진단 보조 시스템을 제공합니다.
          </p>
          <button className="analyze-btn" onClick={goToAnalyze}>
            지금 분석하러 가기 →
          </button>
        </section>

        {/* About Section */}
        <section className="about">
          <h2>프로젝트 소개</h2>
          <p>
            본 서비스는 AI Hub의 합성 facial bone X-ray 데이터를 기반으로, Normal/Abnormal을 자동 분류하고 
            Grad-CAM을 통해 예측 근거를 시각화합니다. 
            의료진의 진단 정확도를 높이고, 오진을 예방하는 데 도움을 줍니다.
          </p>

          <h2>사용 기술</h2>
          <ul>
            <li>데이터: AI Hub 합성 X-ray 데이터</li>
            <li>모델: EfficientNet-B4, ResNet34</li>
            <li>시각화: Grad-CAM</li>
            <li>환경: Python, PyTorch, FastAPI, React</li>
          </ul>
        </section>
      </main>

      <footer className="footer">
        © 2025 Facial Bone X-ray
      </footer>
    </div>
  );
}

export default Landing;
