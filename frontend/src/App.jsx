import React, { useState, useEffect, useRef } from 'react';
import './App.css'
import LandingPage from './Components/LandingPage.jsx';
import AuthPage from './Components/AuthPage.jsx';
import Dashboard from './Components/Dashboard.jsx';
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";

import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(useGSAP, ScrollTrigger);

function App() {
  const [view, setView] = useState('landing');
  const [user, setUser] = useState(null);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    if (savedToken && savedUser) {
      setUser(JSON.parse(savedUser));
      setView('dashboard');
    }
  }, []);

  const handleAuth = (authData) => {
    setUser({ email: authData.email, username: authData.username });
    setView('dashboard');
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setView('landing');
  };

  return (
    <>
      {view === 'landing' && <LandingPage onNavigate={setView} />}
      {view === 'auth' && <AuthPage onAuth={handleAuth} />}
      {view === 'dashboard' && <Dashboard user={user} onLogout={handleLogout} />}
    </>
  )
}

export default App
