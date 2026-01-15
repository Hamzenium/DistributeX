import React, { useEffect, useState, useRef } from 'react';
import { gsap } from 'gsap';

const API_BASE = 'http://localhost:8000';

interface AuthPageProps {
    onAuth: (authData: any) => void;
    onNavigate: (view: 'landing' | 'auth' | 'dashboard') => void;
}

export const AuthPage: React.FC<AuthPageProps> = ({ onAuth, onNavigate }) => {
    const [authTab, setAuthTab] = useState('signin');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [formData, setFormData] = useState({ username: '', email: '', password: '' });
    const cardRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (cardRef.current) {
            gsap.from(cardRef.current, {
                opacity: 0,
                scale: 0.95,
                y: 30,
                duration: 0.6,
                ease: 'power3.out'
            });
        }
    }, []);

    const apiCall = async (endpoint: string, method = 'GET', body: any = null) => {
        const headers: any = {
            'Content-Type': 'application/json'
        };

        const options: RequestInit = { method, headers };
        if (body) {
            options.body = JSON.stringify(body);
        }

        try {
            const response = await fetch(`${API_BASE}${endpoint}`, options);

            if (response.status === 401) {
                throw new Error('Session expired. Please login again.');
            }

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Request failed');
            }

            return response.json();
        } catch (error: any) {
            console.error('API Error:', error);
            throw error;
        }
    };

    const handleSubmit = async () => {
        setLoading(true);
        setError(null);
        setSuccess(null);

        const data: any = {
            email: formData.email,
            password: formData.password
        };

        if (authTab === 'signup') {
            data.username = formData.username;
        }

        try {
            const endpoint = `/auth/${authTab}`;
            const result = await apiCall(endpoint, 'POST', data);

            if (authTab === 'signup') {
                setSuccess('Account created! Please sign in.');
                setAuthTab('signin');
                setFormData({ username: '', email: '', password: '' });
            } else {
                localStorage.setItem('token', result.access_token);
                localStorage.setItem('user', JSON.stringify({
                    email: data.email,
                    username: result.username
                }));

                onAuth(result);
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1410 50%, #2a1c10 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }}>
            {/* Animated background orbs */}
            <div style={{
                position: 'absolute',
                top: '-20%',
                right: '-10%',
                width: '600px',
                height: '600px',
                background: 'radial-gradient(circle, rgba(255,140,0,0.2) 0%, transparent 70%)',
                borderRadius: '50%',
                filter: 'blur(80px)',
                animation: 'float 15s ease-in-out infinite'
            }} />

            {/* Back Button */}
            <button
                onClick={() => onNavigate('landing')}
                style={{
                    position: 'absolute',
                    top: '2rem',
                    left: '2rem',
                    padding: '0.75rem 1.5rem',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,140,0,0.2)',
                    borderRadius: '8px',
                    color: '#fff',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    transition: 'all 0.3s',
                    zIndex: 10
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                    e.currentTarget.style.transform = 'translateX(-5px)';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                    e.currentTarget.style.transform = 'translateX(0)';
                }}
            >
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <path d="M12.5 15L7.5 10L12.5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Back
            </button>

            <div ref={cardRef} style={{
                width: '100%',
                maxWidth: '450px',
                margin: '0 2rem',
                zIndex: 2
            }}>
                {/* Header */}
                <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                    <div style={{
                        width: '60px',
                        height: '60px',
                        background: 'linear-gradient(135deg, #ff8c00, #ff4500)',
                        borderRadius: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '32px',
                        margin: '0 auto 1rem'
                    }}>⚡</div>
                    <h1 style={{
                        fontSize: '2rem',
                        fontWeight: '700',
                        color: '#fff',
                        marginBottom: '0.5rem',
                        letterSpacing: '1px'
                    }}>HYPERTUNE</h1>
                    <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.95rem' }}>
                        Distributed ML Training
                    </p>
                </div>

                {/* Auth Card */}
                <div style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,140,0,0.2)',
                    borderRadius: '16px',
                    padding: '2rem',
                    backdropFilter: 'blur(20px)'
                }}>
                    {/* Tabs */}
                    <div style={{
                        display: 'flex',
                        gap: '1rem',
                        marginBottom: '2rem',
                        borderBottom: '1px solid rgba(255,140,0,0.1)'
                    }}>
                        <button
                            onClick={() => {
                                setAuthTab('signin');
                                setError(null);
                                setSuccess(null);
                            }}
                            style={{
                                flex: 1,
                                padding: '0.75rem',
                                background: 'transparent',
                                border: 'none',
                                borderBottom: authTab === 'signin' ? '2px solid #ff8c00' : '2px solid transparent',
                                color: authTab === 'signin' ? '#ff8c00' : 'rgba(255,255,255,0.5)',
                                cursor: 'pointer',
                                fontWeight: '600',
                                fontSize: '1rem',
                                transition: 'all 0.3s'
                            }}
                        >
                            Sign In
                        </button>
                        <button
                            onClick={() => {
                                setAuthTab('signup');
                                setError(null);
                                setSuccess(null);
                            }}
                            style={{
                                flex: 1,
                                padding: '0.75rem',
                                background: 'transparent',
                                border: 'none',
                                borderBottom: authTab === 'signup' ? '2px solid #ff8c00' : '2px solid transparent',
                                color: authTab === 'signup' ? '#ff8c00' : 'rgba(255,255,255,0.5)',
                                cursor: 'pointer',
                                fontWeight: '600',
                                fontSize: '1rem',
                                transition: 'all 0.3s'
                            }}
                        >
                            Sign Up
                        </button>
                    </div>

                    {/* Messages */}
                    {error && (
                        <div style={{
                            padding: '1rem',
                            background: 'rgba(255,69,0,0.2)',
                            border: '1px solid rgba(255,69,0,0.3)',
                            borderRadius: '8px',
                            color: '#ff6b6b',
                            marginBottom: '1.5rem',
                            fontSize: '0.9rem'
                        }}>
                            {error}
                        </div>
                    )}
                    {success && (
                        <div style={{
                            padding: '1rem',
                            background: 'rgba(0,255,0,0.1)',
                            border: '1px solid rgba(0,255,0,0.3)',
                            borderRadius: '8px',
                            color: '#4ade80',
                            marginBottom: '1.5rem',
                            fontSize: '0.9rem'
                        }}>
                            {success}
                        </div>
                    )}

                    {/* Form Fields */}
                    <div>
                        {authTab === 'signup' && (
                            <div style={{ marginBottom: '1.5rem' }}>
                                <label style={{
                                    display: 'block',
                                    marginBottom: '0.5rem',
                                    color: 'rgba(255,255,255,0.8)',
                                    fontSize: '0.9rem',
                                    fontWeight: '500'
                                }}>
                                    Username
                                </label>
                                <input
                                    type="text"
                                    value={formData.username}
                                    onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                                    style={{
                                        width: '100%',
                                        padding: '0.875rem',
                                        background: 'rgba(255,255,255,0.05)',
                                        border: '1px solid rgba(255,140,0,0.2)',
                                        borderRadius: '8px',
                                        color: '#fff',
                                        fontSize: '1rem',
                                        outline: 'none',
                                        transition: 'all 0.3s',
                                        boxSizing: 'border-box'
                                    }}
                                    onFocus={(e) => {
                                        e.currentTarget.style.borderColor = 'rgba(255,140,0,0.5)';
                                        e.currentTarget.style.background = 'rgba(255,255,255,0.08)';
                                    }}
                                    onBlur={(e) => {
                                        e.currentTarget.style.borderColor = 'rgba(255,140,0,0.2)';
                                        e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                                    }}
                                />
                            </div>
                        )}

                        <div style={{ marginBottom: '1.5rem' }}>
                            <label style={{
                                display: 'block',
                                marginBottom: '0.5rem',
                                color: 'rgba(255,255,255,0.8)',
                                fontSize: '0.9rem',
                                fontWeight: '500'
                            }}>
                                Email
                            </label>
                            <input
                                type="email"
                                value={formData.email}
                                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                style={{
                                    width: '100%',
                                    padding: '0.875rem',
                                    background: 'rgba(255,255,255,0.05)',
                                    border: '1px solid rgba(255,140,0,0.2)',
                                    borderRadius: '8px',
                                    color: '#fff',
                                    fontSize: '1rem',
                                    outline: 'none',
                                    transition: 'all 0.3s',
                                    boxSizing: 'border-box'
                                }}
                                onFocus={(e) => {
                                    e.currentTarget.style.borderColor = 'rgba(255,140,0,0.5)';
                                    e.currentTarget.style.background = 'rgba(255,255,255,0.08)';
                                }}
                                onBlur={(e) => {
                                    e.currentTarget.style.borderColor = 'rgba(255,140,0,0.2)';
                                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                                }}
                            />
                        </div>

                        <div style={{ marginBottom: '2rem' }}>
                            <label style={{
                                display: 'block',
                                marginBottom: '0.5rem',
                                color: 'rgba(255,255,255,0.8)',
                                fontSize: '0.9rem',
                                fontWeight: '500'
                            }}>
                                Password
                            </label>
                            <input
                                type="password"
                                value={formData.password}
                                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                                onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
                                style={{
                                    width: '100%',
                                    padding: '0.875rem',
                                    background: 'rgba(255,255,255,0.05)',
                                    border: '1px solid rgba(255,140,0,0.2)',
                                    borderRadius: '8px',
                                    color: '#fff',
                                    fontSize: '1rem',
                                    outline: 'none',
                                    transition: 'all 0.3s',
                                    boxSizing: 'border-box'
                                }}
                                onFocus={(e) => {
                                    e.currentTarget.style.borderColor = 'rgba(255,140,0,0.5)';
                                    e.currentTarget.style.background = 'rgba(255,255,255,0.08)';
                                }}
                                onBlur={(e) => {
                                    e.currentTarget.style.borderColor = 'rgba(255,140,0,0.2)';
                                    e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                                }}
                            />
                        </div>

                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            style={{
                                width: '100%',
                                padding: '1rem',
                                background: loading ? 'rgba(100,100,100,0.3)' : 'linear-gradient(135deg, #ff8c00, #ff4500)',
                                border: 'none',
                                borderRadius: '8px',
                                color: '#fff',
                                fontSize: '1rem',
                                fontWeight: '600',
                                cursor: loading ? 'not-allowed' : 'pointer',
                                transition: 'all 0.3s',
                                boxShadow: loading ? 'none' : '0 10px 30px rgba(255,69,0,0.3)'
                            }}
                            onMouseEnter={(e) => {
                                if (!loading) {
                                    e.currentTarget.style.transform = 'translateY(-2px)';
                                    e.currentTarget.style.boxShadow = '0 15px 40px rgba(255,69,0,0.4)';
                                }
                            }}
                            onMouseLeave={(e) => {
                                if (!loading) {
                                    e.currentTarget.style.transform = 'translateY(0)';
                                    e.currentTarget.style.boxShadow = '0 10px 30px rgba(255,69,0,0.3)';
                                }
                            }}
                        >
                            {loading ? 'Processing...' : (authTab === 'signin' ? 'Sign In →' : 'Create Account →')}
                        </button>
                    </div>

                    <div style={{
                        marginTop: '1.5rem',
                        textAlign: 'center',
                        color: 'rgba(255,255,255,0.6)',
                        fontSize: '0.9rem'
                    }}>
                        {authTab === 'signin' ? "Don't have an account?" : "Already have an account?"}
                        <button
                            onClick={() => {
                                setAuthTab(authTab === 'signin' ? 'signup' : 'signin');
                                setError(null);
                                setSuccess(null);
                            }}
                            style={{
                                background: 'none',
                                border: 'none',
                                color: '#ff8c00',
                                cursor: 'pointer',
                                marginLeft: '0.5rem',
                                fontWeight: '600',
                                fontSize: '0.9rem'
                            }}
                        >
                            {authTab === 'signin' ? 'Sign Up' : 'Sign In'}
                        </button>
                    </div>
                </div>
            </div>

            <style>{`
                @keyframes float {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-30px); }
                }
            `}</style>
        </div>
    );
};