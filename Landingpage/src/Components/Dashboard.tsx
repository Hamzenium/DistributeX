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
    const [ownedSessions, setOwnedSessions] = useState<Session[]>([]);
    const [joinedSessions, setJoinedSessions] = useState<Session[]>([]);
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
            setOwnedSessions(data.owned_sessions || []);
            setJoinedSessions(data.joined_sessions || []);
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
        <div className="page-root dashboard" ref={dashRef}>
            {/* Header */}
            <div className="dash-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div className="logo">
                        <div className="logo-inner">⚡</div>
                    </div>
                    <h1 className="dash-title">HYPERTUNE</h1>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                    <span className={`status-badge ${isOnline ? 'status-online' : 'status-offline'}`}>
                        {isOnline ? 'ONLINE' : 'OFFLINE'}
                    </span>
                    <span style={{ color: 'var(--muted)' }}>{user?.username || user?.email}</span>
                    <button onClick={onLogout} className="logout-btn">
                        Logout
                    </button>
                </div>
            </div>

            {/* Network Status Card */}
            <div className="card" style={{ marginBottom: '2rem', padding: '2rem' }}>
                <h2 className="section-title">Network Status</h2>
                <p style={{ color: 'var(--muted)', marginBottom: '1.5rem' }}>
                    {isOnline
                        ? 'You are connected to the distributed training network'
                        : 'Join the network to contribute compute power'}
                </p>
                <button
                    onClick={isOnline ? handleLeaveNetwork : handleJoinNetwork}
                    disabled={loading}
                    className="primary-btn"
                    style={{ width: 'auto' }}
                >
                    {loading ? <div className="spinner" /> : (isOnline ? 'Leave Network' : 'Join Network')}
                </button>
            </div>

            {error && <div className="error-msg">{error}</div>}

            {/* Tabs */}
            <div className="tabs" style={{ marginBottom: '2rem' }}>
                <button
                    className={`tab ${view === 'sessions' ? 'active' : ''}`}
                    onClick={() => setView('sessions')}
                >
                    My Sessions
                    {view === 'sessions' && <div className="tab-underline" />}
                </button>
                <button
                    className={`tab ${view === 'create' ? 'active' : ''}`}
                    onClick={() => setView('create')}
                >
                    Create Session
                    {view === 'create' && <div className="tab-underline" />}
                </button>
            </div>

            {/* Content */}
            {view === 'create' ? (
                <div className="new-session-form">
                    <h2 className="section-title">Start Training Session</h2>

                    <div className="field">
                        <label className="label">Number of Peers</label>
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
                            className="input"
                        />
                    </div>

                    <div className="field">
                        <label className="label">Dataset (CSV)</label>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={(e) => setFile(e.target.files?.[0] || null)}
                            className="input file-input"
                        />
                    </div>

                    <div className="field">
                        <label className="label">Hyperparameters per Peer</label>
                        {hyperparameters.map((params, idx) => (
                            <div key={idx} className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                                <h4 style={{ marginBottom: '0.75rem', color: 'var(--accent)' }}>Peer {idx + 1}</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                                    <div className="field">
                                        <label className="label">Learning Rate</label>
                                        <input
                                            type="number"
                                            value={params.learning_rate}
                                            onChange={(e) => updateHyperparam(idx, 'learning_rate', parseFloat(e.target.value))}
                                            step="0.001"
                                            className="input"
                                        />
                                    </div>
                                    <div className="field">
                                        <label className="label">Batch Size</label>
                                        <input
                                            type="number"
                                            value={params.batch_size}
                                            onChange={(e) => updateHyperparam(idx, 'batch_size', parseInt(e.target.value))}
                                            className="input"
                                        />
                                    </div>
                                    <div className="field">
                                        <label className="label">Epochs</label>
                                        <input
                                            type="number"
                                            value={params.epochs}
                                            onChange={(e) => updateHyperparam(idx, 'epochs', parseInt(e.target.value))}
                                            className="input"
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <button
                        onClick={handleStartSession}
                        disabled={loading || !file}
                        className="primary-btn"
                    >
                        {loading ? <div className="spinner" /> : 'Start Training'}
                        <div className="btn-overlay" />
                    </button>
                </div>
            ) : (
                <div className="section">
                    <h2 className="section-title">Your Training Sessions</h2>

                    {ownedSessions.length === 0 && joinedSessions.length === 0 ? (
                        <div className="card" style={{ padding: '3rem', textAlign: 'center' }}>
                            <p style={{ color: 'var(--muted)' }}>
                                No sessions yet. Create your first training session!
                            </p>
                        </div>
                    ) : (
                        <>
                            {ownedSessions.length > 0 && (
                                <>
                                    <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', color: 'var(--accent)' }}>
                                        Owned Sessions
                                    </h3>
                                    <div className="session-grid" style={{ marginBottom: '2rem' }}>
                                        {ownedSessions.map((session) => (
                                            <div key={session._id} className="session-card">
                                                <div className="session-id">ID: {session._id.slice(-8)}</div>
                                                <div className="session-info">
                                                    <strong>Peers:</strong> {session.num_peers}
                                                </div>
                                                <div className="session-info">
                                                    <strong>Created:</strong> {new Date(session.created_at).toLocaleTimeString()}
                                                    <strong style={{ marginLeft: '8px' }}>Date:</strong> {new Date(session.created_at).toLocaleDateString()}
                                                </div>
                                                <span className={`status-badge ${session.status === 'RUNNING' ? 'status-training' :
                                                    session.status === 'COMPLETED' ? 'status-online' : 'status-offline'
                                                    }`}>
                                                    {session.status}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            )}

                            {joinedSessions.length > 0 && (
                                <>
                                    <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', color: 'var(--accent)' }}>
                                        Joined Sessions
                                    </h3>
                                    <div className="session-grid">
                                        {joinedSessions.map((session) => (
                                            <div key={session._id} className="session-card">
                                                <div className="session-id">ID: {session._id.slice(-8)}</div>
                                                <div className="session-info">
                                                    <strong>Peers:</strong> {session.num_peers}
                                                </div>
                                                <div className="session-info">
                                                    <strong>Created:</strong> {new Date(session.created_at).toLocaleDateString()}
                                                </div>
                                                <span className={`status-badge ${session.status === 'RUNNING' ? 'status-training' :
                                                    session.status === 'COMPLETED' ? 'status-online' : 'status-offline'
                                                    }`}>
                                                    {session.status}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            )}
                        </>
                    )}
                </div>
            )}
        </div>
    );
};