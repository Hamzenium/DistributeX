import React, { useEffect, useState, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const API_BASE = 'http://localhost:8000';


// ============================================================================
// AUTH PAGE COMPONENT
// ============================================================================
const AuthPage = ({ onAuth }) => {
    const [authTab, setAuthTab] = useState('signin');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const cardRef = useRef(null);

    useEffect(() => {
        gsap.from(cardRef.current, {
            opacity: 0,
            scale: 0.9,
            y: 30,
            duration: 0.6,
            ease: 'power3.out'
        });
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setSuccess(null);

        const formData = new FormData(e.target);
        const data = {
            email: formData.get('email'),
            password: formData.get('password')
        };

        if (authTab === 'signup') {
            data.username = formData.get('username');
        }

        try {
            const response = await fetch(`${API_BASE}/auth/${authTab}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Request failed');
            }

            if (authTab === 'signup') {
                setSuccess('Account created! Please sign in.');
                setAuthTab('signin');
            } else {
                localStorage.setItem('token', result.access_token);
                localStorage.setItem('user', JSON.stringify({ email: data.email, username: result.username }));
                onAuth(result);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page-root">
            <div className="fixed-card" ref={cardRef}>
                <div className="header">
                    <div className="logo">
                        <div className="logo-inner">⚡</div>
                    </div>
                    <h1 className="brand">HYPERTUNE</h1>
                    <p className="subtitle">Distributed ML Training</p>
                </div>

                <div className="card">
                    <div className="tabs">
                        <button
                            className={`tab ${authTab === 'signin' ? 'active' : ''}`}
                            onClick={() => setAuthTab('signin')}
                        >
                            Sign In
                            {authTab === 'signin' && <div className="tab-underline" />}
                        </button>
                        <button
                            className={`tab ${authTab === 'signup' ? 'active' : ''}`}
                            onClick={() => setAuthTab('signup')}
                        >
                            Sign Up
                            {authTab === 'signup' && <div className="tab-underline" />}
                        </button>
                    </div>

                    <div className="card-body">
                        {error && <div className="error-msg">{error}</div>}
                        {success && <div className="success-msg">{success}</div>}

                        <form className="form" onSubmit={handleSubmit}>
                            {authTab === 'signup' && (
                                <div className="field">
                                    <label className="label">Username</label>
                                    <input type="text" name="username" className="input" required />
                                </div>
                            )}

                            <div className="field">
                                <label className="label">Email</label>
                                <input type="email" name="email" className="input" required />
                            </div>

                            <div className="field">
                                <label className="label">Password</label>
                                <div className="password-row">
                                    <input type="password" name="password" className="input" required />
                                </div>
                            </div>

                            <div className="actions">
                                <button type="submit" className="primary-btn" disabled={loading}>
                                    {loading ? (
                                        <div className="spinner" />
                                    ) : (
                                        authTab === 'signin' ? 'Sign In →' : 'Create Account →'
                                    )}
                                    <div className="btn-overlay" />
                                </button>
                            </div>
                        </form>
                    </div>

                    <div className="card-footer">
                        <div className="meta">
                            <span>{authTab === 'signin' ? "Don't have an account?" : "Already have an account?"}</span>
                            <button
                                onClick={() => setAuthTab(authTab === 'signin' ? 'signup' : 'signin')}
                                style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', marginLeft: '8px' }}
                            >
                                {authTab === 'signin' ? 'Sign Up' : 'Sign In'}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
export default AuthPage;