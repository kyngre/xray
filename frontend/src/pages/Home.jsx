import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Grid,
  Typography,
  useTheme,
  Stack,
} from '@mui/material';
import {
  Add as AddIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon,
  Science as ScienceIcon,
  ArrowForward as ArrowForwardIcon,
} from '@mui/icons-material';

function Home() {
  const navigate = useNavigate();
  const theme = useTheme();

  const features = [
    {
      icon: <ScienceIcon sx={{ fontSize: 40 }} />,
      title: "AI 기술",
      description: "최신 AI 기술을 활용한 정확한 분석 결과를 제공합니다.",
      gradient: "linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%)",
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40 }} />,
      title: "빠른 분석",
      description: "수 초 내에 X-ray 이미지 분석을 완료합니다.",
      gradient: "linear-gradient(135deg, #10B981 0%, #059669 100%)",
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 40 }} />,
      title: "안전한 보관",
      description: "모든 분석 결과는 안전하게 보관되며 언제든지 확인할 수 있습니다.",
      gradient: "linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)",
    },
  ];

  return (
    <Box
      sx={{
        minHeight: "calc(100vh - 64px)",
        background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, ${theme.palette.background.paper} 100%)`,
        position: "relative",
        overflow: "hidden",
        "&::before": {
          content: '""',
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "radial-gradient(circle at 50% 50%, rgba(124, 58, 237, 0.1) 0%, transparent 50%)",
          pointerEvents: "none",
        },
      }}
    >
      <Container maxWidth="lg" sx={{ position: "relative", zIndex: 1 }}>
        <Box
          sx={{
            textAlign: "center",
            pt: { xs: 8, md: 12 },
            pb: { xs: 8, md: 12 },
          }}
        >
          <Typography
            variant="h1"
            sx={{
              mb: 3,
              background: "linear-gradient(135deg, #7C3AED 0%, #10B981 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              fontWeight: 800,
              fontSize: { xs: "2.5rem", md: "3.5rem" },
              lineHeight: 1.2,
            }}
          >
            안면 골격 X-ray 분석
          </Typography>
          <Typography
            variant="h5"
            color="text.secondary"
            sx={{
              mb: 6,
              maxWidth: "800px",
              mx: "auto",
              fontSize: { xs: "1.1rem", md: "1.25rem" },
              lineHeight: 1.6,
            }}
          >
            AI 기반 안면 골격 X-ray 분석 시스템으로 정확하고 빠른 진단을 받아보세요.
          </Typography>
          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={2}
            justifyContent="center"
            alignItems="center"
          >
            <Button
              variant="contained"
              size="large"
              startIcon={<AddIcon />}
              onClick={() => navigate("/analysis")}
              sx={{
                py: 2,
                px: 4,
                fontSize: "1.1rem",
                background: "linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%)",
                "&:hover": {
                  background: "linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)",
                  transform: "translateY(-2px)",
                },
                transition: "all 0.2s ease-in-out",
              }}
            >
              X-ray 분석 시작하기
            </Button>
            <Button
              variant="outlined"
              size="large"
              endIcon={<ArrowForwardIcon />}
              onClick={() => navigate("/history")}
              sx={{
                py: 2,
                px: 4,
                fontSize: "1.1rem",
                borderColor: "rgba(255, 255, 255, 0.2)",
                color: "text.primary",
                "&:hover": {
                  borderColor: "primary.main",
                  backgroundColor: "rgba(124, 58, 237, 0.05)",
                  transform: "translateY(-2px)",
                },
                transition: "all 0.2s ease-in-out",
              }}
            >
              분석 기록 보기
            </Button>
          </Stack>
        </Box>

        <Grid container spacing={4} sx={{ pb: { xs: 8, md: 12 } }}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card
                sx={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  transition: "all 0.3s ease-in-out",
                  background: "linear-gradient(145deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%)",
                  backdropFilter: "blur(10px)",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  "&:hover": {
                    transform: "translateY(-8px)",
                    boxShadow: "0 20px 40px rgba(0, 0, 0, 0.2)",
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1, p: 4 }}>
                  <Box
                    sx={{
                      display: "inline-flex",
                      p: 2,
                      borderRadius: 2,
                      mb: 3,
                      background: feature.gradient,
                      color: "white",
                      boxShadow: "0 8px 16px rgba(0, 0, 0, 0.1)",
                    }}
                  >
                    {feature.icon}
                  </Box>
                  <Typography
                    variant="h5"
                    component="h2"
                    gutterBottom
                    sx={{
                      fontWeight: 600,
                      mb: 2,
                      background: feature.gradient,
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                    }}
                  >
                    {feature.title}
                  </Typography>
                  <Typography
                    color="text.secondary"
                    sx={{
                      fontSize: "1rem",
                      lineHeight: 1.6,
                    }}
                  >
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}

export default Home; 