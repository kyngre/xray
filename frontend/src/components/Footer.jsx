import { Box, Container, Typography, Link, useTheme } from '@mui/material';

function Footer() {
  const theme = useTheme();
  const currentYear = new Date().getFullYear();

  return (
    <Box
      component="footer"
      sx={{
        py: 2,
        px: 2,
        mt: 'auto',
        backgroundColor: 'rgba(15, 23, 42, 0.8)',
        backdropFilter: 'blur(10px)',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
            © {currentYear} 안면 골격 X-ray 분석 시스템
          </Typography>
          <Box
            sx={{
              display: 'flex',
              gap: 3,
            }}
          >
            <Link
              href="#"
              color="text.secondary"
              underline="hover"
              sx={{ 
                fontSize: '0.8rem',
                opacity: 0.7,
                '&:hover': {
                  opacity: 1,
                },
              }}
            >
              이용약관
            </Link>
            <Link
              href="#"
              color="text.secondary"
              underline="hover"
              sx={{ 
                fontSize: '0.8rem',
                opacity: 0.7,
                '&:hover': {
                  opacity: 1,
                },
              }}
            >
              개인정보처리방침
            </Link>
            <Link
              href="#"
              color="text.secondary"
              underline="hover"
              sx={{ 
                fontSize: '0.8rem',
                opacity: 0.7,
                '&:hover': {
                  opacity: 1,
                },
              }}
            >
              문의하기
            </Link>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}

export default Footer; 