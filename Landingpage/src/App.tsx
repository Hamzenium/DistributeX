import React, { useEffect, useState } from 'react';
import { LandingPage } from './Components/Landingpage';
import { AuthPage } from './Components/auth';
import { Dashboard } from './Components/Dashboard';
import './App.css';


const getToken = () => localStorage.getItem('token');

const isAuthenticated = () => {
  const token = getToken();
  if (!token) return false;

  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.exp * 1000 > Date.now();
  } catch {
    return false;
  }
};

function App() {
  const [view, setView] = useState<'landing' | 'auth' | 'dashboard'>('landing');
  const [user, setUser] = useState<{ email: string; username: string } | null>(null);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');

    if (savedToken && savedUser) {
      if (isAuthenticated()) {
        console.log('Valid token found, auto-logging in');
        setUser(JSON.parse(savedUser));
        setView('dashboard');
      } else {
        console.log('Token expired, clearing storage');
        localStorage.removeItem('token');
        localStorage.removeItem('user');
      }
    }
  }, []);

  const handleAuth = (authData: any) => {
    console.log('Authentication successful:', authData);
    setUser({ email: authData.email, username: authData.username });
    setView('dashboard');
  };

  const handleLogout = () => {
    console.log('Logging out, clearing token');
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setView('landing');
  };

  return (
    <>
      {view === 'landing' && <LandingPage onNavigate={setView} />}
      {view === 'auth' && <AuthPage onAuth={handleAuth} onNavigate={setView} />}
      {view === 'dashboard' && <Dashboard user={user} onLogout={handleLogout} />}
    </>
  );
}

export default App;