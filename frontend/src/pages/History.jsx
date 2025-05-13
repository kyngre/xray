import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Container,
  Grid,
  Typography,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

function History() {
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        // TODO: API 엔드포인트로 실제 요청 보내기
        const response = await fetch('/api/history');
        const data = await response.json();
        setHistory(data);
      } catch (error) {
        console.error('Error fetching history:', error);
        // TODO: 에러 처리
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

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

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 8 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          분석 기록
        </Typography>

        {history.length === 0 ? (
          <Card sx={{ mt: 4 }}>
            <CardContent>
              <Typography variant="h6" align="center" color="text.secondary">
                아직 분석 기록이 없습니다
              </Typography>
            </CardContent>
          </Card>
        ) : (
          <TableContainer component={Paper} sx={{ mt: 4 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>분석 일시</TableCell>
                  <TableCell>결과</TableCell>
                  <TableCell>신뢰도</TableCell>
                  <TableCell>상세 보기</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {history.map((item) => (
                  <TableRow key={item.id}>
                    <TableCell>
                      {new Date(item.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={item.isNormal ? <CheckCircleIcon /> : <ErrorIcon />}
                        label={item.isNormal ? '정상' : '비정상'}
                        color={item.isNormal ? 'success' : 'error'}
                      />
                    </TableCell>
                    <TableCell>{item.confidence}%</TableCell>
                    <TableCell>
                      <Typography
                        variant="body2"
                        color="primary"
                        sx={{ cursor: 'pointer' }}
                        onClick={() => navigate(`/result/${item.id}`)}
                      >
                        자세히 보기
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Box>
    </Container>
  );
}

export default History; 