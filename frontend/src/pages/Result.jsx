import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Container,
  Grid,
  Typography,
  CircularProgress,
  Chip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

function Result() {
  const { id } = useParams();
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const fetchResult = async () => {
      try {
        // TODO: API 엔드포인트로 실제 요청 보내기
        const response = await fetch(`/api/results/${id}`);
        const data = await response.json();
        setResult(data);
      } catch (error) {
        console.error('Error fetching result:', error);
        // TODO: 에러 처리
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [id]);

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (!result) {
    return (
      <Container maxWidth="md">
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Typography variant="h5" color="error">
            결과를 찾을 수 없습니다
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 8 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          분석 결과
        </Typography>

        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  원본 X-ray 이미지
                </Typography>
                <Box sx={{ textAlign: 'center' }}>
                  <img
                    src={result.originalImage}
                    alt="Original X-ray"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '400px',
                      objectFit: 'contain',
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Grad-CAM 시각화
                </Typography>
                <Box sx={{ textAlign: 'center' }}>
                  <img
                    src={result.gradcamImage}
                    alt="Grad-CAM Visualization"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '400px',
                      objectFit: 'contain',
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ mr: 2 }}>
                    분석 결과:
                  </Typography>
                  <Chip
                    icon={result.isNormal ? <CheckCircleIcon /> : <ErrorIcon />}
                    label={result.isNormal ? '정상' : '비정상'}
                    color={result.isNormal ? 'success' : 'error'}
                    sx={{ fontSize: '1rem', py: 2 }}
                  />
                </Box>

                <Typography variant="body1" paragraph>
                  신뢰도: {result.confidence}%
                </Typography>

                {!result.isNormal && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      비정상 영역 분석:
                    </Typography>
                    <Typography variant="body1">
                      {result.abnormalDescription}
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default Result; 