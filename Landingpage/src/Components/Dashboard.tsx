import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

gsap.registerPlugin(useGSAP, ScrollTrigger);

const API_BASE = 'http://localhost:8000';

interface DashboardProps {
    user: { email: string; username: string } | null;
    onLogout: () => void;
}

interface TrainingResult {
    epoch: number;
    loss: number;
    accuracy: number;
}

interface Peer {
    uid: string;
    status: string;
    results: TrainingResult[];
    hyperparameters?: any;
}

interface Session {
    _id: string;
    owner_user_id: string;
    num_peers: number;
    status: string;
    created_at: string;
    peers: Peer[];
}

export const Dashboard: React.FC<DashboardProps> = ({ user, onLogout }) => {
    const [view, setView] = useState<'create' | 'sessions'>('sessions');
    const [ownedSessions, setOwnedSessions] = useState<Session[]>([]);
    const [joinedSessions, setJoinedSessions] = useState<Session[]>([]);
    const [selectedSession, setSelectedSession] = useState<Session | null>(null);
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
    const pollingInterval = useRef<number | null>(null);

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

    // Poll for updates when viewing a RUNNING session
    useEffect(() => {
        if (selectedSession && selectedSession.status === 'RUNNING') {
            pollingInterval.current = window.setInterval(() => {
                fetchSessionDetails(selectedSession._id);
            }, 2000);

            return () => {
                if (pollingInterval.current) {
                    clearInterval(pollingInterval.current);
                }
            };
        }
    }, [selectedSession?._id, selectedSession?.status]);

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

            // Don't fetch sessions, directly view the new session
            if (result.session_uid) {
                // Wait 1 second for coordinator to start
                setTimeout(async () => {
                    await fetchSessionDetails(result.session_uid);
                }, 1000);
            }
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

    const fetchSessionDetails = async (sessionId: string) => {
        try {
            const session = await apiCall(`/sessions/${sessionId}`, 'GET');
            setSelectedSession(session);
        } catch (err: any) {
            console.error('Failed to fetch session details:', err);
        }
    };

    useEffect(() => {
        if (view === 'sessions' && !selectedSession) {
            fetchSessions();
        }
    }, [view]);

    const handleNumPeersChange = (newNum: number) => {
        if (newNum < 1 || newNum > 10 || !Number.isInteger(newNum)) {
            return;
        }

        setNumPeers(newNum);

        const newHyperparams = [];
        for (let i = 0; i < newNum; i++) {
            newHyperparams.push({
                learning_rate: 0.001,
                batch_size: 32,
                epochs: 10
            });
        }
        setHyperparameters(newHyperparams);
    };

    const updateHyperparam = (index: number, key: string, value: any) => {
        const newParams = [...hyperparameters];
        newParams[index] = { ...newParams[index], [key]: value };
        setHyperparameters(newParams);
    };

    const viewSessionDetails = (session: Session) => {
        setSelectedSession(session);
    };

    const backToSessions = () => {
        setSelectedSession(null);
        fetchSessions();
    };

    const renderTrainingGraphs = () => {
        if (!selectedSession) return null;

        const colors = ['#f97316', '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899'];
        const hasResults = selectedSession.peers.some(p => p.results && p.results.length > 0);

        return (
            <div style={{ marginTop: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h2 className="section-title">Training Progress</h2>
                    <button
                        onClick={backToSessions}
                        style={{
                            padding: '0.5rem 1rem',
                            background: 'rgba(249,115,22,0.1)',
                            border: '1px solid rgba(249,115,22,0.3)',
                            borderRadius: '8px',
                            color: '#f97316',
                            cursor: 'pointer',
                            fontWeight: '500'
                        }}
                    >
                        ← Back to Sessions
                    </button>
                </div>

                {/* Session Info Card */}
                <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <div className="session-id">Session ID: {selectedSession._id}</div>
                            <div className="session-info">
                                <strong>Peers:</strong> {selectedSession.num_peers}
                            </div>
                            <div className="session-info">
                                <strong>Created:</strong> {new Date(selectedSession.created_at).toLocaleString()}
                            </div>
                        </div>
                        <span className={`status-badge ${selectedSession.status === 'RUNNING' ? 'status-training' :
                            selectedSession.status === 'COMPLETED' ? 'status-online' : 'status-offline'
                            }`}>
                            {selectedSession.status}
                        </span>
                    </div>
                </div>

                {!hasResults && selectedSession.status === 'RUNNING' && (
                    <div className="card" style={{ padding: '2rem', textAlign: 'center', marginBottom: '1.5rem' }}>
                        <div className="spinner" style={{ margin: '0 auto 1rem' }} />
                        <p style={{ color: 'var(--muted)' }}>
                            Training in progress... Waiting for epoch data from peers...
                        </p>
                    </div>
                )}

                {hasResults && (
                    <>
                        {/* Loss Chart */}
                        <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                            <h3 style={{ marginBottom: '1rem', color: '#fff' }}>Loss over Epochs</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                    <XAxis
                                        dataKey="epoch"
                                        type="number"
                                        domain={[1, 'dataMax']}
                                        stroke="#9aa0a6"
                                        label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#9aa0a6' }}
                                    />
                                    <YAxis
                                        stroke="#9aa0a6"
                                        label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#9aa0a6' }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#1a1a1a',
                                            border: '1px solid rgba(249,115,22,0.2)',
                                            borderRadius: '8px',
                                            color: '#fff'
                                        }}
                                    />
                                    <Legend />
                                    {selectedSession.peers.map((peer, idx) => (
                                        peer.results && peer.results.length > 0 && (
                                            <Line
                                                key={peer.uid}
                                                data={peer.results}
                                                type="monotone"
                                                dataKey="loss"
                                                stroke={colors[idx % colors.length]}
                                                name={`Peer ${idx + 1}`}
                                                strokeWidth={2}
                                                dot={{ r: 3 }}
                                                activeDot={{ r: 5 }}
                                            />
                                        )
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Accuracy Chart */}
                        <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                            <h3 style={{ marginBottom: '1rem', color: '#fff' }}>Accuracy over Epochs</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                    <XAxis
                                        dataKey="epoch"
                                        type="number"
                                        domain={[1, 'dataMax']}
                                        stroke="#9aa0a6"
                                        label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#9aa0a6' }}
                                    />
                                    <YAxis
                                        stroke="#9aa0a6"
                                        domain={[0, 1]}
                                        label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', fill: '#9aa0a6' }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            background: '#1a1a1a',
                                            border: '1px solid rgba(249,115,22,0.2)',
                                            borderRadius: '8px',
                                            color: '#fff'
                                        }}
                                    />
                                    <Legend />
                                    {selectedSession.peers.map((peer, idx) => (
                                        peer.results && peer.results.length > 0 && (
                                            <Line
                                                key={peer.uid}
                                                data={peer.results}
                                                type="monotone"
                                                dataKey="accuracy"
                                                stroke={colors[idx % colors.length]}
                                                name={`Peer ${idx + 1}`}
                                                strokeWidth={2}
                                                dot={{ r: 3 }}
                                                activeDot={{ r: 5 }}
                                            />
                                        )
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </>
                )}

                {/* Peer Details */}
                <div className="section-title" style={{ marginBottom: '1rem' }}>Peer Details</div>
                <div style={{ display: 'grid', gap: '1rem' }}>
                    {selectedSession.peers.map((peer, idx) => (
                        <div key={peer.uid} className="card" style={{ padding: '1.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <h4 style={{ color: colors[idx % colors.length], margin: 0 }}>
                                    Peer {idx + 1}
                                </h4>
                                <span className={`status-badge ${peer.status === 'TRAINING' ? 'status-training' :
                                    peer.status === 'COMPLETED' ? 'status-online' : 'status-offline'
                                    }`}>
                                    {peer.status}
                                </span>
                            </div>
                            <div className="session-id" style={{ marginBottom: '0.5rem' }}>
                                UID: {peer.uid}
                            </div>
                            {peer.hyperparameters && (
                                <div style={{
                                    background: 'rgba(249,115,22,0.05)',
                                    padding: '0.75rem',
                                    borderRadius: '8px',
                                    marginBottom: '1rem'
                                }}>
                                    <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                                        LR: {peer.hyperparameters.learning_rate} •
                                        Batch: {peer.hyperparameters.batch_size} •
                                        Epochs: {peer.hyperparameters.epochs}
                                    </div>
                                </div>
                            )}
                            {peer.results && peer.results.length > 0 ? (
                                <div>
                                    <div style={{ fontSize: '0.9rem', color: 'var(--muted)', marginBottom: '0.5rem' }}>
                                        Latest Results (Epoch {peer.results[peer.results.length - 1]?.epoch}):
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                        <div>
                                            <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>Loss</div>
                                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#fff' }}>
                                                {peer.results[peer.results.length - 1]?.loss.toFixed(4)}
                                            </div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>Accuracy</div>
                                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#fff' }}>
                                                {(peer.results[peer.results.length - 1]?.accuracy * 100).toFixed(2)}%
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>
                                    {selectedSession.status === 'RUNNING' ? 'Waiting for training data...' : 'No training data'}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        );
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

            {selectedSession ? (
                renderTrainingGraphs()
            ) : (
                <>
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
                                        if (!isNaN(val)) {
                                            handleNumPeersChange(val);
                                        }
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
                                                    <div
                                                        key={session._id}
                                                        className="session-card"
                                                        style={{ cursor: 'pointer' }}
                                                        onClick={() => viewSessionDetails(session)}
                                                    >
                                                        <div className="session-id">ID: {session._id.slice(-8)}</div>
                                                        <div className="session-info">
                                                            <strong>Peers:</strong> {session.num_peers}
                                                        </div>
                                                        <div className="session-info">
                                                            <strong>Created:</strong> {new Date(session.created_at).toLocaleString()}
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
                                                    <div
                                                        key={session._id}
                                                        className="session-card"
                                                        style={{ cursor: 'pointer' }}
                                                        onClick={() => viewSessionDetails(session)}
                                                    >
                                                        <div className="session-id">ID: {session._id.slice(-8)}</div>
                                                        <div className="session-info">
                                                            <strong>Peers:</strong> {session.num_peers}
                                                        </div>
                                                        <div className="session-info">
                                                            <strong>Created:</strong> {new Date(session.created_at).toLocaleString()}
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
                </>
            )
            }
        </div >
    );
};
