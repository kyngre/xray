import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Button,
  Card,
  Container,
  Typography,
  CircularProgress,
  useTheme,
  Stack,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Image as ImageIcon,
  ArrowBack as ArrowBackIcon,
} from '@mui/icons-material';

function Analysis() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const theme = useTheme();

  const onDrop = (acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg'],
    },
    maxFiles: 1,
  });

  const handleAnalysis = async () => {
    if (!file) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('분석 중 오류가 발생했습니다.');
      }

      const result = await response.json();
      navigate('/result', { state: { result } });
    } catch (error) {
      console.error('Error:', error);
      // TODO: 에러 처리
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 64px)',
        background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, ${theme.palette.background.paper} 100%)`,
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 50% 50%, rgba(124, 58, 237, 0.1) 0%, transparent 50%)',
          pointerEvents: 'none',
        },
      }}
    >
      <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ pt: { xs: 6, md: 8 }, pb: { xs: 6, md: 8 } }}>
          <Button
            startIcon={<ArrowBackIcon />}
            onClick={() => navigate('/')}
            sx={{
              mb: 4,
              color: 'text.secondary',
              '&:hover': {
                color: 'primary.main',
                backgroundColor: 'rgba(124, 58, 237, 0.05)',
              },
            }}
          >
            홈으로 돌아가기
          </Button>

          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography
              variant="h2"
              sx={{
                mb: 2,
                background: 'linear-gradient(135deg, #7C3AED 0%, #10B981 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontWeight: 700,
                fontSize: { xs: '2rem', md: '2.5rem' },
              }}
            >
              X-ray 이미지 분석
            </Typography>
            <Typography
              variant="h6"
              color="text.secondary"
              sx={{
                maxWidth: '600px',
                mx: 'auto',
                fontSize: { xs: '1rem', md: '1.1rem' },
                lineHeight: 1.6,
              }}
            >
              안면 골격 X-ray 이미지를 업로드하여 AI 분석을 시작하세요.
            </Typography>
          </Box>

          <Card
            sx={{
              p: 4,
              background: 'linear-gradient(145deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              boxShadow: '0 20px 40px rgba(0, 0, 0, 0.2)',
            }}
          >
            <Box
              {...getRootProps()}
              sx={{
                border: `2px dashed ${
                  isDragActive
                    ? theme.palette.primary.main
                    : 'rgba(255, 255, 255, 0.1)'
                }`,
                borderRadius: 2,
                p: 6,
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease-in-out',
                background: isDragActive
                  ? 'rgba(124, 58, 237, 0.05)'
                  : 'rgba(255, 255, 255, 0.02)',
                '&:hover': {
                  borderColor: theme.palette.primary.main,
                  background: 'rgba(124, 58, 237, 0.05)',
                  transform: 'translateY(-2px)',
                },
              }}
            >
              <input {...getInputProps()} />
              <CloudUploadIcon
                sx={{
                  fontSize: 64,
                  color: isDragActive
                    ? theme.palette.primary.main
                    : 'rgba(255, 255, 255, 0.5)',
                  mb: 2,
                }}
              />
              <Typography
                variant="h6"
                gutterBottom
                sx={{
                  fontWeight: 600,
                  mb: 1,
                }}
              >
                {isDragActive
                  ? '이미지를 여기에 놓으세요'
                  : '이미지를 드래그하거나 클릭하여 업로드하세요'}
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ opacity: 0.7 }}
              >
                PNG, JPG, JPEG 파일만 지원됩니다.
              </Typography>
            </Box>

            {file && (
              <Box
                sx={{
                  mt: 4,
                  p: 3,
                  borderRadius: 2,
                  background: 'rgba(124, 58, 237, 0.05)',
                  border: '1px solid rgba(124, 58, 237, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                }}
              >
                <ImageIcon color="primary" />
                <Typography variant="body1">{file.name}</Typography>
              </Box>
            )}

            <Stack
              direction="row"
              spacing={2}
              justifyContent="center"
              sx={{ mt: 4 }}
            >
              <Button
                variant="contained"
                size="large"
                onClick={handleAnalysis}
                disabled={!file || loading}
                sx={{
                  py: 2,
                  px: 6,
                  fontSize: '1.1rem',
                  background: 'linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
                    transform: 'translateY(-2px)',
                  },
                  '&:disabled': {
                    background: 'rgba(124, 58, 237, 0.3)',
                  },
                  transition: 'all 0.2s ease-in-out',
                }}
              >
                {loading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  '분석 시작하기'
                )}
              </Button>
            </Stack>
          </Card>
        </Box>
      </Container>
    </Box>
  );
}

export default Analysis; 