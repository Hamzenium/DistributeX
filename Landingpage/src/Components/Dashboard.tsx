import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { useGSAP } from "@gsap/react";
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';

gsap.registerPlugin(useGSAP, ScrollTrigger);

const API_BASE = 'http://localhost:8000';

// ============================================
// Neural Network Animation Component
// ============================================
interface NeuralNetworkProps {
    isTraining: boolean;
    height?: number;
    numPeers?: number;      // Number of peers determines number of layers
    epochs?: number;        // Epochs shown in the labels
}

const NeuralNetwork: React.FC<NeuralNetworkProps> = ({
    isTraining,
    height = 400,
    numPeers = 4,
    epochs = 10
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Dynamic network configuration based on numPeers
        // Creates layers: Input -> Hidden layers (based on peers) -> Output
        const generateLayers = () => {
            const layers = [];
            const numLayers = Math.max(2, Math.min(numPeers + 2, 8)); // Min 2, Max 8 layers

            // Calculate neuron counts (decreasing from input to output)
            const maxNeurons = 8;
            const minNeurons = 2;

            for (let i = 0; i < numLayers; i++) {
                const progress = i / (numLayers - 1);
                const neurons = Math.round(maxNeurons - (maxNeurons - minNeurons) * progress);

                // Size represents the actual layer size in the network
                let size;
                if (i === 0) size = 784;  // Input layer (e.g., MNIST)
                else if (i === numLayers - 1) size = 10;  // Output layer
                else size = Math.round(128 / (i * 0.5 + 1));  // Hidden layers

                layers.push({
                    neurons,
                    label: `P${i + 1}`,  // P1, P2, P3... for Peer layers
                    size,
                    epoch: Math.round((epochs / numLayers) * (i + 1))  // Distribute epochs across layers
                });
            }
            return layers;
        };

        const layers = generateLayers();

        const padding = 60;
        const layerSpacing = (canvas.width - padding * 2) / (layers.length - 1);

        // Animation state
        let pulsePhase = 0;
        let connectionPulses: Array<{
            from: number;
            to: number;
            fromNeuron: number;
            toNeuron: number;
            progress: number;
            speed: number;
        }> = [];

        // Calculate neuron positions
        const neuronPositions = layers.map((layer, layerIdx) => {
            const x = padding + layerIdx * layerSpacing;
            const neuronSpacing = (canvas.height - padding * 2) / (layer.neurons - 1);

            return Array.from({ length: layer.neurons }, (_, neuronIdx) => ({
                x,
                y: padding + neuronIdx * neuronSpacing,
                active: false,
                activation: 0
            }));
        });

        // Create initial connection pulses
        const createPulse = () => {
            if (!isTraining) return;

            const fromLayer = Math.floor(Math.random() * (layers.length - 1));
            const toLayer = fromLayer + 1;
            const fromNeuron = Math.floor(Math.random() * layers[fromLayer].neurons);
            const toNeuron = Math.floor(Math.random() * layers[toLayer].neurons);

            connectionPulses.push({
                from: fromLayer,
                to: toLayer,
                fromNeuron,
                toNeuron,
                progress: 0,
                speed: 0.02 + Math.random() * 0.03
            });
        };

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            pulsePhase += 0.02;

            // Update and filter pulses
            connectionPulses = connectionPulses.filter(pulse => {
                pulse.progress += pulse.speed;
                return pulse.progress <= 1;
            });

            // Create new pulses randomly
            if (isTraining && Math.random() < 0.3) {
                createPulse();
            }

            // Draw connections
            layers.forEach((layer, layerIdx) => {
                if (layerIdx < layers.length - 1) {
                    neuronPositions[layerIdx].forEach((fromPos, fromIdx) => {
                        neuronPositions[layerIdx + 1].forEach((toPos, toIdx) => {
                            // Base connection
                            ctx.strokeStyle = 'rgba(249, 115, 22, 0.1)';
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(fromPos.x, fromPos.y);
                            ctx.lineTo(toPos.x, toPos.y);
                            ctx.stroke();

                            // Draw active pulses
                            const activePulses = connectionPulses.filter(
                                p => p.from === layerIdx && p.to === layerIdx + 1 &&
                                    p.fromNeuron === fromIdx && p.toNeuron === toIdx
                            );

                            activePulses.forEach(pulse => {
                                const x = fromPos.x + (toPos.x - fromPos.x) * pulse.progress;
                                const y = fromPos.y + (toPos.y - fromPos.y) * pulse.progress;

                                // Glowing pulse
                                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
                                gradient.addColorStop(0, 'rgba(249, 115, 22, 0.8)');
                                gradient.addColorStop(0.5, 'rgba(249, 115, 22, 0.4)');
                                gradient.addColorStop(1, 'rgba(249, 115, 22, 0)');

                                ctx.fillStyle = gradient;
                                ctx.beginPath();
                                ctx.arc(x, y, 15, 0, Math.PI * 2);
                                ctx.fill();
                            });
                        });
                    });
                }
            });

            // Draw neurons
            neuronPositions.forEach((layerNeurons, layerIdx) => {
                layerNeurons.forEach((pos, neuronIdx) => {
                    const isActive = connectionPulses.some(
                        p => (p.from === layerIdx && p.fromNeuron === neuronIdx) ||
                            (p.to === layerIdx && p.toNeuron === neuronIdx)
                    );

                    // Neuron glow when training
                    if (isTraining) {
                        const glowIntensity = isActive ? 0.6 : 0.2 + Math.sin(pulsePhase + neuronIdx) * 0.1;
                        const glowRadius = isActive ? 12 : 8;

                        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);
                        gradient.addColorStop(0, `rgba(249, 115, 22, ${glowIntensity})`);
                        gradient.addColorStop(1, 'rgba(249, 115, 22, 0)');

                        ctx.fillStyle = gradient;
                        ctx.beginPath();
                        ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
                        ctx.fill();
                    }

                    // Neuron circle
                    ctx.fillStyle = isActive && isTraining ? '#f97316' : 'rgba(249, 115, 22, 0.3)';
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
                    ctx.fill();

                    if (isActive && isTraining) {
                        ctx.strokeStyle = 'rgba(249, 115, 22, 0.8)';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    }
                });

                // Layer labels
                const layer = layers[layerIdx];
                const firstNeuron = layerNeurons[0];

                ctx.fillStyle = '#9aa0a6';
                ctx.font = '14px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(layer.label, firstNeuron.x, canvas.height - 25);
                ctx.font = '11px monospace';
                ctx.fillStyle = '#6b7280';
                ctx.fillText(`E${layer.epoch}`, firstNeuron.x, canvas.height - 10);
            });

            animationRef.current = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isTraining, numPeers, epochs]);

    return (
        <div style={{
            position: 'relative',
            width: '100%',
            height: `${height}px`,
            background: isTraining
                ? 'radial-gradient(ellipse at center, rgba(249, 115, 22, 0.05) 0%, transparent 70%)'
                : 'transparent',
            borderRadius: '12px',
            overflow: 'hidden',
            transition: 'background 0.5s ease'
        }}>
            <canvas
                ref={canvasRef}
                width={1200}
                height={height}
                style={{
                    width: '100%',
                    height: '100%',
                    display: 'block'
                }}
            />
            <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                fontSize: height > 250 ? '48px' : '24px',
                fontWeight: '700',
                color: isTraining ? '#f97316' : 'rgba(249, 115, 22, 0.3)',
                textShadow: isTraining ? '0 0 20px rgba(249, 115, 22, 0.5)' : 'none',
                letterSpacing: '0.1em',
                fontFamily: 'monospace',
                transition: 'all 0.5s ease',
                pointerEvents: 'none',
                animation: isTraining ? 'pulse 2s ease-in-out infinite' : 'none'
            }}>
                {isTraining ? '[TRAINING]' : '[READY]'}
            </div>
            <style>{`
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.7; }
                }
            `}</style>
        </div>
    );
};

// ============================================
// Dashboard Interfaces
// ============================================
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

interface EpochResult {
    epoch: number;
    loss: number;
    accuracy: number;
    timestamp: string;
}

interface PeerData {
    hyperparameters: {
        lr?: number;
        learning_rate?: number;
        batch_size: number;
        epochs: number;
    };
    epochs: EpochResult[];
}

interface FullResultsResponse {
    session_id: string;
    status: string;
    peers: Record<string, PeerData>;
}

// ============================================
// Enhanced Peer Card Component with Charts
// ============================================
interface PeerCardProps {
    peerId: string;
    peer: PeerData;
    index: number;
    color: string;
    status: string;
}

const PeerCard: React.FC<PeerCardProps> = ({ peerId, peer, index, color, status }) => {
    const latestEpoch = peer.epochs[peer.epochs.length - 1];
    const lr = peer.hyperparameters.lr || peer.hyperparameters.learning_rate || 0;

    return (
        <div className="card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h4 style={{ color: color, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        backgroundColor: color,
                        boxShadow: `0 0 10px ${color}40`
                    }} />
                    Peer {index + 1}
                </h4>
                <span className={`status-badge ${peer.epochs.length > 0 ? 'status-training' : 'status-offline'}`}>
                    {peer.epochs.length > 0 ? 'TRAINING' : 'WAITING'}
                </span>
            </div>

            <div className="session-id" style={{ marginBottom: '0.5rem', fontSize: '0.75rem' }}>
                UID: {peerId}
            </div>

            <div style={{
                background: 'rgba(249,115,22,0.05)',
                padding: '0.75rem',
                borderRadius: '8px',
                marginBottom: '1rem',
                border: '1px solid rgba(249,115,22,0.1)'
            }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--muted)', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    <span>LR: <strong style={{ color: '#fff' }}>{lr}</strong></span>
                    <span>Batch: <strong style={{ color: '#fff' }}>{peer.hyperparameters.batch_size}</strong></span>
                    <span>Epochs: <strong style={{ color: '#fff' }}>{peer.hyperparameters.epochs}</strong></span>
                </div>
            </div>

            {peer.epochs.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                    {/* Mini Loss Chart */}
                    <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '8px', padding: '0.75rem' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '0.5rem' }}>Loss</div>
                        <ResponsiveContainer width="100%" height={80}>
                            <AreaChart data={peer.epochs}>
                                <defs>
                                    <linearGradient id={`lossGrad-${index}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={color} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <Area
                                    type="monotone"
                                    dataKey="loss"
                                    stroke={color}
                                    fill={`url(#lossGrad-${index})`}
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                        <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#fff' }}>
                            {latestEpoch?.loss.toFixed(4)}
                        </div>
                    </div>

                    {/* Mini Accuracy Chart */}
                    <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '8px', padding: '0.75rem' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginBottom: '0.5rem' }}>Accuracy</div>
                        <ResponsiveContainer width="100%" height={80}>
                            <AreaChart data={peer.epochs}>
                                <defs>
                                    <linearGradient id={`accGrad-${index}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <Area
                                    type="monotone"
                                    dataKey="accuracy"
                                    stroke="#10b981"
                                    fill={`url(#accGrad-${index})`}
                                    strokeWidth={2}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                        <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#10b981' }}>
                            {latestEpoch ? `${(latestEpoch.accuracy * 100).toFixed(1)}%` : '--'}
                        </div>
                    </div>
                </div>
            )}

            {!latestEpoch && (
                <div style={{ color: 'var(--muted)', fontSize: '0.9rem', textAlign: 'center', padding: '1rem' }}>
                    {status === 'RUNNING' ? (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                            <div className="spinner" style={{ width: '16px', height: '16px' }} />
                            Waiting for training data...
                        </div>
                    ) : 'No training data'}
                </div>
            )}
        </div>
    );
};

// ============================================
// Main Dashboard Component
// ============================================
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

    // Real-time monitoring state
    const [resultsData, setResultsData] = useState<FullResultsResponse | null>(null);
    const [monitoringSessionId, setMonitoringSessionId] = useState<string | null>(null);

    const dashRef = useRef<HTMLDivElement>(null);
    const pollingInterval = useRef<number | null>(null);

    useEffect(() => {
        if (dashRef.current) {
            gsap.from(dashRef.current, {
                opacity: 10,
                y: 20,
                duration: 0.6,
                ease: 'power3.out'
            });
        }
    }, []);

    // Poll for real-time updates when monitoring a session
    useEffect(() => {
        if (monitoringSessionId && resultsData?.status === 'RUNNING') {
            pollingInterval.current = window.setInterval(() => {
                fetchFullResults(monitoringSessionId);
            }, 1000);

            return () => {
                if (pollingInterval.current) {
                    clearInterval(pollingInterval.current);
                }
            };
        }
    }, [monitoringSessionId, resultsData?.status]);

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

    const fetchFullResults = async (sessionId: string) => {
        try {
            const data = await apiCall(`/sessions/${sessionId}/full-results`, 'GET');
            setResultsData(data);
            setError(null);
        } catch (err: any) {
            console.error('Failed to fetch full results:', err);
            setError(err.message);
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

            // Create proper hyperparameters array with lr key (not learning_rate)
            const hyperparamsForBackend = hyperparameters.map(param => ({
                lr: param.learning_rate,
                batch_size: param.batch_size,
                epochs: param.epochs
            }));

            formData.append('hyperparameters', JSON.stringify(hyperparamsForBackend));

            console.log('Sending hyperparameters:', hyperparamsForBackend);

            const result = await apiCall('/sessions/start', 'POST', formData);

            if (result.session_uid) {
                // Start monitoring the new session
                setMonitoringSessionId(result.session_uid);

                // Wait 1 second then fetch initial data
                setTimeout(async () => {
                    await fetchFullResults(result.session_uid);
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

    const viewSessionDetails = (session: Session) => {
        setMonitoringSessionId(session._id);
        fetchFullResults(session._id);
    };

    const backToSessions = () => {
        setMonitoringSessionId(null);
        setResultsData(null);
        if (pollingInterval.current) {
            clearInterval(pollingInterval.current);
        }
        fetchSessions();
    };

    useEffect(() => {
        if (view === 'sessions' && !monitoringSessionId) {
            fetchSessions();
        }
    }, [view]);

    useEffect(() => {
        // Cleanup on unmount
        return () => {
            if (pollingInterval.current) {
                clearInterval(pollingInterval.current);
            }
        };
    }, []);

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

    const colors = ['#f97316', '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899'];

    const renderTrainingGraphs = () => {
        if (!resultsData) return null;

        const peerEntries = Object.entries(resultsData.peers);
        const hasResults = peerEntries.some(([_, peer]) => peer.epochs.length > 0);
        const isTraining = resultsData.status === 'RUNNING';

        return (
            <div style={{ marginTop: '2rem' }}>
                {/* Header with back button */}
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
                            fontWeight: '500',
                            transition: 'all 0.2s ease'
                        }}
                        onMouseOver={(e) => {
                            e.currentTarget.style.background = 'rgba(249,115,22,0.2)';
                        }}
                        onMouseOut={(e) => {
                            e.currentTarget.style.background = 'rgba(249,115,22,0.1)';
                        }}
                    >
                        ← Back to Sessions
                    </button>
                </div>

                {/* Neural Network Animation - Dynamic based on session data */}
                <div className="card" style={{ marginBottom: '2rem', padding: '1rem', overflow: 'hidden' }}>
                    <NeuralNetwork
                        isTraining={isTraining}
                        height={300}
                        numPeers={peerEntries.length}
                        epochs={peerEntries[0]?.[1]?.hyperparameters?.epochs || 10}
                    />
                </div>

                {/* Session Info Card */}
                <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                            <div className="session-id">Session ID: {resultsData.session_id}</div>
                            <div className="session-info">
                                <strong>Peers:</strong> {peerEntries.length}
                            </div>
                        </div>
                        <span className={`status-badge ${isTraining ? 'status-training' :
                            resultsData.status === 'COMPLETED' ? 'status-online' : 'status-offline'
                            }`}>
                            {resultsData.status}
                        </span>
                    </div>
                </div>

                {!hasResults && isTraining && (
                    <div className="card" style={{ padding: '2rem', textAlign: 'center', marginBottom: '1.5rem' }}>
                        <div className="spinner" style={{ margin: '0 auto 1rem' }} />
                        <p style={{ color: 'var(--muted)' }}>
                            Training in progress... Waiting for epoch data from peers...
                        </p>
                    </div>
                )}

                {hasResults && (
                    <>
                        {/* Loss Chart with Gradient Fill */}
                        <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                            <h3 style={{ marginBottom: '1rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <span style={{ color: '#f97316' }}>📉</span> Loss over Epochs
                            </h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart>
                                    <defs>
                                        {peerEntries.map(([peerId], idx) => (
                                            <linearGradient key={`lossGradient-${idx}`} id={`lossGradient-${idx}`} x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={colors[idx % colors.length]} stopOpacity={0.3} />
                                                <stop offset="95%" stopColor={colors[idx % colors.length]} stopOpacity={0} />
                                            </linearGradient>
                                        ))}
                                    </defs>
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
                                    {peerEntries.map(([peerId, peer], idx) => (
                                        peer.epochs.length > 0 && (
                                            <Area
                                                key={peerId}
                                                data={peer.epochs}
                                                type="monotone"
                                                dataKey="loss"
                                                stroke={colors[idx % colors.length]}
                                                fill={`url(#lossGradient-${idx})`}
                                                name={`Peer ${idx + 1}`}
                                                strokeWidth={2}
                                            />
                                        )
                                    ))}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Accuracy Chart with Gradient Fill */}
                        <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                            <h3 style={{ marginBottom: '1rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <span style={{ color: '#10b981' }}>📈</span> Accuracy over Epochs
                            </h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart>
                                    <defs>
                                        {peerEntries.map(([peerId], idx) => (
                                            <linearGradient key={`accGradient-${idx}`} id={`accGradient-${idx}`} x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={colors[idx % colors.length]} stopOpacity={0.3} />
                                                <stop offset="95%" stopColor={colors[idx % colors.length]} stopOpacity={0} />
                                            </linearGradient>
                                        ))}
                                    </defs>
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
                                        formatter={(value: number | undefined) =>
                                            typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : 'Accuracy'
                                        }
                                    />
                                    <Legend />
                                    {peerEntries.map(([peerId, peer], idx) => (
                                        peer.epochs.length > 0 && (
                                            <Area
                                                key={peerId}
                                                data={peer.epochs}
                                                type="monotone"
                                                dataKey="accuracy"
                                                stroke={colors[idx % colors.length]}
                                                fill={`url(#accGradient-${idx})`}
                                                name={`Peer ${idx + 1}`}
                                                strokeWidth={2}
                                            />
                                        )
                                    ))}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </>
                )}

                {/* Peer Details with Enhanced Cards */}
                <div className="section-title" style={{ marginBottom: '1rem' }}>Peer Details</div>
                <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))' }}>
                    {peerEntries.map(([peerId, peer], idx) => (
                        <PeerCard
                            key={peerId}
                            peerId={peerId}
                            peer={peer}
                            index={idx}
                            color={colors[idx % colors.length]}
                            status={resultsData.status}
                        />
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

            {monitoringSessionId && resultsData ? (
                renderTrainingGraphs()
            ) : (
                <>
                    {/* Network Status Card with Neural Network Animation */}
                    <div className="card" style={{ marginBottom: '2rem', padding: '2rem' }}>
                        <h2 className="section-title">Network Status</h2>

                        {/* Mini Neural Network Animation - Uses form state for preview */}
                        <div style={{ marginBottom: '1.5rem' }}>
                            <NeuralNetwork
                                isTraining={isOnline}
                                height={200}
                                numPeers={numPeers}
                                epochs={hyperparameters[0]?.epochs || 10}
                            />
                        </div>

                        <p style={{ color: 'var(--muted)', marginBottom: '1.5rem', textAlign: 'center' }}>
                            {isOnline
                                ? 'You are connected to the distributed training network'
                                : 'Join the network to contribute compute power'}
                        </p>
                        <div style={{ display: 'flex', justifyContent: 'center' }}>
                            <button
                                onClick={isOnline ? handleLeaveNetwork : handleJoinNetwork}
                                disabled={loading}
                                className="primary-btn"
                                style={{ width: 'auto' }}
                            >
                                {loading ? <div className="spinner" /> : (isOnline ? 'Leave Network' : 'Join Network')}
                            </button>
                        </div>
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
                                        <h4 style={{ marginBottom: '0.75rem', color: colors[idx % colors.length], display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                            <span style={{
                                                width: '10px',
                                                height: '10px',
                                                borderRadius: '50%',
                                                backgroundColor: colors[idx % colors.length]
                                            }} />
                                            Peer {idx + 1}
                                        </h4>
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
            )}
        </div>
    );
};