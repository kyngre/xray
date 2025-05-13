import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useTheme,
  Divider,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home as HomeIcon,
  Add as AddIcon,
  History as HistoryIcon,
  Science as ScienceIcon,
} from '@mui/icons-material';
import Footer from './Footer';

const drawerWidth = 240;

function Layout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: '홈', icon: <HomeIcon />, path: '/' },
    { text: 'X-ray 분석', icon: <AddIcon />, path: '/analysis' },
    { text: '분석 기록', icon: <HistoryIcon />, path: '/history' },
  ];

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            color: theme.palette.primary.main,
            fontSize: '1.1rem',
          }}
        >
          안면 골격 X-ray 분석
        </Typography>
      </Box>
      <Divider />
      <List sx={{ px: 2, py: 1 }}>
        {menuItems.map((item) => (
          <ListItem
            button
            key={item.text}
            onClick={() => {
              navigate(item.path);
              setMobileOpen(false);
            }}
            sx={{
              borderRadius: 1,
              mb: 0.5,
              bgcolor: location.pathname === item.path ? `${theme.palette.primary.main}15` : 'transparent',
              color: location.pathname === item.path ? theme.palette.primary.main : theme.palette.text.primary,
              '&:hover': {
                bgcolor: `${theme.palette.primary.main}10`,
              },
            }}
          >
            <ListItemIcon sx={{ 
              color: location.pathname === item.path ? theme.palette.primary.main : 'inherit',
              minWidth: 36,
            }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText 
              primary={item.text} 
              primaryTypographyProps={{
                fontSize: '0.9rem',
                fontWeight: location.pathname === item.path ? 600 : 400,
              }}
            />
          </ListItem>
        ))}
      </List>
      <Box sx={{ mt: 'auto', p: 2 }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center',
          p: 1.5,
          borderRadius: 1,
          bgcolor: `${theme.palette.primary.main}10`,
        }}>
          <ScienceIcon sx={{ 
            color: theme.palette.primary.main,
            mr: 1.5,
            fontSize: '1.2rem',
          }} />
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
            AI 기반 분석 시스템
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          bgcolor: 'rgba(15, 23, 42, 0.8)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Toolbar sx={{ minHeight: '64px !important' }}>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ 
              mr: 2, 
              display: { sm: 'none' },
              color: theme.palette.text.primary,
            }}
          >
            <MenuIcon />
          </IconButton>
          <Typography 
            variant="h6" 
            noWrap 
            component="div"
            sx={{ 
              color: theme.palette.text.primary,
              fontWeight: 600,
              fontSize: '1rem',
            }}
          >
            {location.pathname === '/' ? '안면 골격 X-ray 분석' : 
             location.pathname === '/analysis' ? 'X-ray 이미지 분석' :
             location.pathname === '/history' ? '분석 기록' : '안면 골격 X-ray 분석'}
          </Typography>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderRight: '1px solid rgba(255, 255, 255, 0.1)',
              bgcolor: theme.palette.background.paper,
            },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderRight: '1px solid rgba(255, 255, 255, 0.1)',
              bgcolor: theme.palette.background.paper,
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          bgcolor: theme.palette.background.default,
        }}
      >
        <Toolbar sx={{ minHeight: '64px !important' }} />
        <Box sx={{ flexGrow: 1 }}>
          {children}
        </Box>
        <Footer />
      </Box>
    </Box>
  );
}

export default Layout; 