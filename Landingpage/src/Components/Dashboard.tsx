import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(useGSAP, ScrollTrigger);


const API_BASE = 'http://localhost:8000';

interface DashboardProps {
    user: { email: string; username: string } | null;
    onLogout: () => void;
}

interface Session {
    _id: string;
    owner_user_id: string;
    num_peers: number;
    status: string;
    created_at: string;
    peers: Array<{
        uid: string;
        results: any;
    }>;
}

export const Dashboard: React.FC<DashboardProps> = ({ user, onLogout }) => {
    const [view, setView] = useState<'create' | 'sessions'>('sessions');
    const [sessions, setSessions] = useState<Session[]>([]);
    const [isOnline, setIsOnline] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Form state
    const [numPeers, setNumPeers] = useState(2);
    const [file, setFile] = useState<File | null>(null);
    const [hyperparameters, setHyperparameters] = useState<any[]>([
        { learning_rate: 0.001, batch_size: 32, epochs: 10 },
        { learning_rate: 0.01, batch_size: 64, epochs: 10 }
    ]);

    const dashRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (dashRef.current) {
            gsap.from(dashRef.current, {
                opacity: 0,
                y: 20,
                duration: 0.6,
                ease: 'power3.out'
            });
        }
    }, []);

    const getToken = () => localStorage.getItem('token');

    const apiCall = async (endpoint: string, method = 'GET', body: any = null) => {
        const token = getToken();
        const headers: any = {
            'Authorization': `Bearer ${token}`
        };

        const options: RequestInit = { method, headers };

        if (body && !(body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(body);
        } else if (body instanceof FormData) {
            options.body = body;
        }

        try {
            const response = await fetch(`${API_BASE}${endpoint}`, options);

            if (response.status === 401) {
                onLogout();
                throw new Error('Session expired');
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

    const handleJoinNetwork = async () => {
        setLoading(true);
        setError(null);
        try {
            await apiCall('/sessions/join', 'POST');
            setIsOnline(true);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleLeaveNetwork = async () => {
        setLoading(true);
        try {
            await apiCall('/sessions/leave', 'POST');
            setIsOnline(false);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleStartSession = async () => {
        if (!file) {
            setError('Please upload a CSV file');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('num_peers', numPeers.toString());
            formData.append('hyperparameters', JSON.stringify(hyperparameters));

            const result = await apiCall('/sessions/start', 'POST', formData);
            setView('sessions');
            fetchSessions();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const fetchSessions = async () => {
        try {
            const data = await apiCall('/sessions', 'GET');
            setSessions(data.sessions || []);
        } catch (err: any) {
            console.error('Failed to fetch sessions:', err);
        }
    };

    useEffect(() => {
        fetchSessions();
    }, []);

    const updateHyperparam = (index: number, key: string, value: any) => {
        const newParams = [...hyperparameters];
        newParams[index] = { ...newParams[index], [key]: value };
        setHyperparameters(newParams);
    };

    return (
        <div ref={dashRef} style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1410 50%, #2a1c10 100%)',
            color: '#fff',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }}>
            {/* Navigation Bar */}
            <nav style={{
                padding: '1.5rem 3rem',
                borderBottom: '1px solid rgba(255,140,0,0.1)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backdropFilter: 'blur(10px)',
                background: 'rgba(0,0,0,0.3)'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{
                        width: '40px',
                        height: '40px',
                        background: 'linear-gradient(135deg, #ff8c00, #ff4500)',
                        borderRadius: '8px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '24px'
                    }}>⚡</div>
                    <span style={{ fontSize: '1.5rem', fontWeight: '700', letterSpacing: '1px' }}>
                        HYPERTUNE
                    </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.5rem 1rem',
                        background: isOnline ? 'rgba(0,255,0,0.1)' : 'rgba(255,255,255,0.05)',
                        borderRadius: '20px',
                        border: `1px solid ${isOnline ? 'rgba(0,255,0,0.3)' : 'rgba(255,255,255,0.1)'}`
                    }}>
                        <div style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            background: isOnline ? '#0f0' : '#666'
                        }} />
                        <span style={{ fontSize: '0.9rem' }}>{isOnline ? 'ONLINE' : 'OFFLINE'}</span>
                    </div>

                    <span style={{ color: 'rgba(255,255,255,0.7)' }}>{user?.username || user?.email}</span>

                    <button
                        onClick={onLogout}
                        style={{
                            padding: '0.5rem 1.5rem',
                            background: 'rgba(255,69,0,0.2)',
                            border: '1px solid rgba(255,69,0,0.3)',
                            borderRadius: '8px',
                            color: '#ff4500',
                            cursor: 'pointer',
                            fontWeight: '500',
                            transition: 'all 0.3s'
                        }}
                    >
                        Logout
                    </button>
                </div>
            </nav>

            {/* Main Content */}
            <div style={{ padding: '3rem' }}>
                {/* Status Card */}
                <div style={{
                    background: 'rgba(255,140,0,0.05)',
                    border: '1px solid rgba(255,140,0,0.2)',
                    borderRadius: '16px',
                    padding: '2rem',
                    marginBottom: '2rem'
                }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Network Status</h2>
                    <p style={{ color: 'rgba(255,255,255,0.7)', marginBottom: '1.5rem' }}>
                        {isOnline
                            ? 'You are connected to the distributed training network'
                            : 'Join the network to contribute compute power and earn rewards'}
                    </p>

                    <button
                        onClick={isOnline ? handleLeaveNetwork : handleJoinNetwork}
                        disabled={loading}
                        style={{
                            padding: '1rem 2rem',
                            background: isOnline ? 'rgba(255,69,0,0.2)' : 'linear-gradient(135deg, #ff8c00, #ff4500)',
                            border: isOnline ? '1px solid rgba(255,69,0,0.3)' : 'none',
                            borderRadius: '8px',
                            color: '#fff',
                            cursor: loading ? 'not-allowed' : 'pointer',
                            fontWeight: '600',
                            transition: 'all 0.3s'
                        }}
                    >
                        {loading ? 'Processing...' : (isOnline ? 'Leave Network' : 'Join Network')}
                    </button>
                </div>

                {error && (
                    <div style={{
                        padding: '1rem',
                        background: 'rgba(255,69,0,0.2)',
                        border: '1px solid rgba(255,69,0,0.3)',
                        borderRadius: '8px',
                        marginBottom: '2rem',
                        color: '#ff6b6b'
                    }}>
                        {error}
                    </div>
                )}

                {/* Tabs */}
                <div style={{
                    display: 'flex',
                    gap: '1rem',
                    marginBottom: '2rem',
                    borderBottom: '1px solid rgba(255,140,0,0.1)'
                }}>
                    <button
                        onClick={() => setView('sessions')}
                        style={{
                            padding: '1rem 2rem',
                            background: view === 'sessions' ? 'rgba(255,140,0,0.1)' : 'transparent',
                            border: 'none',
                            borderBottom: view === 'sessions' ? '2px solid #ff8c00' : '2px solid transparent',
                            color: view === 'sessions' ? '#ff8c00' : 'rgba(255,255,255,0.6)',
                            cursor: 'pointer',
                            fontWeight: '600',
                            transition: 'all 0.3s'
                        }}
                    >
                        My Sessions
                    </button>
                    <button
                        onClick={() => setView('create')}
                        style={{
                            padding: '1rem 2rem',
                            background: view === 'create' ? 'rgba(255,140,0,0.1)' : 'transparent',
                            border: 'none',
                            borderBottom: view === 'create' ? '2px solid #ff8c00' : '2px solid transparent',
                            color: view === 'create' ? '#ff8c00' : 'rgba(255,255,255,0.6)',
                            cursor: 'pointer',
                            fontWeight: '600',
                            transition: 'all 0.3s'
                        }}
                    >
                        Create Session
                    </button>
                </div>

                {/* Content */}
                {view === 'create' ? (
                    <div style={{
                        background: 'rgba(255,255,255,0.02)',
                        border: '1px solid rgba(255,140,0,0.1)',
                        borderRadius: '16px',
                        padding: '2rem',
                        maxWidth: '800px'
                    }}>
                        <h2 style={{ fontSize: '2rem', marginBottom: '2rem' }}>Start Training Session</h2>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'rgba(255,255,255,0.8)' }}>
                                Number of Peers
                            </label>
                            <input
                                type="number"
                                value={numPeers}
                                onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    setNumPeers(val);
                                    setHyperparameters(Array(val).fill(0).map(() => ({
                                        learning_rate: 0.001,
                                        batch_size: 32,
                                        epochs: 10
                                    })));
                                }}
                                min="1"
                                max="10"
                                style={{
                                    width: '100%',
                                    padding: '0.75rem',
                                    background: 'rgba(255,255,255,0.05)',
                                    border: '1px solid rgba(255,140,0,0.2)',
                                    borderRadius: '8px',
                                    color: '#fff',
                                    fontSize: '1rem'
                                }}
                            />
                        </div>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'rgba(255,255,255,0.8)' }}>
                                Dataset (CSV)
                            </label>
                            <input
                                type="file"
                                accept=".csv"
                                onChange={(e) => setFile(e.target.files?.[0] || null)}
                                style={{
                                    width: '100%',
                                    padding: '0.75rem',
                                    background: 'rgba(255,255,255,0.05)',
                                    border: '1px solid rgba(255,140,0,0.2)',
                                    borderRadius: '8px',
                                    color: '#fff',
                                    fontSize: '1rem'
                                }}
                            />
                        </div>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <h3 style={{ marginBottom: '1rem', color: 'rgba(255,255,255,0.8)' }}>
                                Hyperparameters per Peer
                            </h3>
                            {hyperparameters.map((params, idx) => (
                                <div key={idx} style={{
                                    background: 'rgba(255,140,0,0.05)',
                                    border: '1px solid rgba(255,140,0,0.1)',
                                    borderRadius: '8px',
                                    padding: '1rem',
                                    marginBottom: '1rem'
                                }}>
                                    <h4 style={{ marginBottom: '0.75rem' }}>Peer {idx + 1}</h4>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                                        <div>
                                            <label style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.6)' }}>Learning Rate</label>
                                            <input
                                                type="number"
                                                value={params.learning_rate}
                                                onChange={(e) => updateHyperparam(idx, 'learning_rate', parseFloat(e.target.value))}
                                                step="0.001"
                                                style={{
                                                    width: '100%',
                                                    padding: '0.5rem',
                                                    background: 'rgba(0,0,0,0.3)',
                                                    border: '1px solid rgba(255,140,0,0.2)',
                                                    borderRadius: '4px',
                                                    color: '#fff',
                                                    marginTop: '0.25rem'
                                                }}
                                            />
                                        </div>
                                        <div>
                                            <label style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.6)' }}>Batch Size</label>
                                            <input
                                                type="number"
                                                value={params.batch_size}
                                                onChange={(e) => updateHyperparam(idx, 'batch_size', parseInt(e.target.value))}
                                                style={{
                                                    width: '100%',
                                                    padding: '0.5rem',
                                                    background: 'rgba(0,0,0,0.3)',
                                                    border: '1px solid rgba(255,140,0,0.2)',
                                                    borderRadius: '4px',
                                                    color: '#fff',
                                                    marginTop: '0.25rem'
                                                }}
                                            />
                                        </div>
                                        <div>
                                            <label style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.6)' }}>Epochs</label>
                                            <input
                                                type="number"
                                                value={params.epochs}
                                                onChange={(e) => updateHyperparam(idx, 'epochs', parseInt(e.target.value))}
                                                style={{
                                                    width: '100%',
                                                    padding: '0.5rem',
                                                    background: 'rgba(0,0,0,0.3)',
                                                    border: '1px solid rgba(255,140,0,0.2)',
                                                    borderRadius: '4px',
                                                    color: '#fff',
                                                    marginTop: '0.25rem'
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <button
                            onClick={handleStartSession}
                            disabled={loading || !file}
                            style={{
                                width: '100%',
                                padding: '1rem',
                                background: loading || !file ? 'rgba(100,100,100,0.3)' : 'linear-gradient(135deg, #ff8c00, #ff4500)',
                                border: 'none',
                                borderRadius: '8px',
                                color: '#fff',
                                fontSize: '1.1rem',
                                fontWeight: '600',
                                cursor: loading || !file ? 'not-allowed' : 'pointer',
                                transition: 'all 0.3s'
                            }}
                        >
                            {loading ? 'Starting...' : 'Start Training'}
                        </button>
                    </div>
                ) : (
                    <div>
                        <h2 style={{ fontSize: '2rem', marginBottom: '1.5rem' }}>Training Sessions</h2>
                        {sessions.length === 0 ? (
                            <div style={{
                                padding: '3rem',
                                textAlign: 'center',
                                background: 'rgba(255,255,255,0.02)',
                                border: '1px solid rgba(255,140,0,0.1)',
                                borderRadius: '16px',
                                color: 'rgba(255,255,255,0.5)'
                            }}>
                                No sessions yet. Create your first training session!
                            </div>
                        ) : (
                            <div style={{ display: 'grid', gap: '1rem' }}>
                                {sessions.map((session) => (
                                    <div key={session._id} style={{
                                        background: 'rgba(255,255,255,0.02)',
                                        border: '1px solid rgba(255,140,0,0.1)',
                                        borderRadius: '12px',
                                        padding: '1.5rem',
                                        transition: 'all 0.3s'
                                    }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <div>
                                                <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>
                                                    Session {session._id.slice(-6)}
                                                </h3>
                                                <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.9rem' }}>
                                                    {session.num_peers} peers • Created {new Date(session.created_at).toLocaleDateString()}
                                                </p>
                                            </div>
                                            <div style={{
                                                padding: '0.5rem 1rem',
                                                background: session.status === 'RUNNING'
                                                    ? 'rgba(0,255,0,0.1)'
                                                    : 'rgba(255,255,255,0.1)',
                                                border: `1px solid ${session.status === 'RUNNING'
                                                    ? 'rgba(0,255,0,0.3)'
                                                    : 'rgba(255,255,255,0.2)'}`,
                                                borderRadius: '20px',
                                                fontSize: '0.85rem',
                                                color: session.status === 'RUNNING' ? '#0f0' : 'rgba(255,255,255,0.7)'
                                            }}>
                                                {session.status}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};