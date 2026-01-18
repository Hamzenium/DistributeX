// PeerNodeDashboard.tsx - Updated with conditional timer and real backend logs
import React, { useState, useEffect, useRef } from 'react';
import { SystemLog } from '../Peer/Systemlogs';
import type { FullResultsResponse } from '../types';

interface PeerNodeDashboardProps {
    resultsData: FullResultsResponse | null;
    onLogout: () => void;
    onChangeRole?: () => void;
    user: { email: string; username: string } | null;
}

export const PeerNodeDashboard: React.FC<PeerNodeDashboardProps> = ({
    resultsData,
    onLogout,
    onChangeRole,
    user
}) => {
    const [uptime, setUptime] = useState(0);
    const [isPaused, setIsPaused] = useState(false);
    const uptimeInterval = useRef<number | null>(null);

    // Timer ONLY starts when resultsData exists AND status is RUNNING
    useEffect(() => {
        if (resultsData?.status === 'RUNNING' && !isPaused) {
            uptimeInterval.current = window.setInterval(() => {
                setUptime(prev => prev + 1);
            }, 1000);
        } else {
            // Clear interval if not training
            if (uptimeInterval.current) {
                clearInterval(uptimeInterval.current);
            }
        }

        return () => {
            if (uptimeInterval.current) {
                clearInterval(uptimeInterval.current);
            }
        };
    }, [resultsData?.status, isPaused]);

    const formatUptime = (seconds: number) => {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    };

    const isTraining = resultsData?.status === 'RUNNING';
    const sessionId = resultsData?.session_id || 'N/A';
    const peerEntries = resultsData?.peers ? Object.entries(resultsData.peers) : [];
    const currentPeerData = peerEntries[0]?.[1];
    const latestEpoch = currentPeerData?.epochs[currentPeerData.epochs.length - 1];
    const batchNum = latestEpoch ? latestEpoch.epoch * (currentPeerData.hyperparameters?.batch_size || 32) : 0;
    const totalEpochs = currentPeerData?.hyperparameters?.epochs || 50;

    // Neural network nodes (simple static visualization)
    const renderNeuralNetwork = () => (
        <div style={{
            position: 'relative',
            width: '100%',
            height: '320px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 3rem'
        }}>
            {/* Input layer */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {[0, 1, 2, 3, 4].map(i => (
                    <div key={`input-${i}`} style={{
                        width: i === 1 || i === 3 ? '12px' : '8px',
                        height: i === 1 || i === 3 ? '12px' : '8px',
                        borderRadius: '50%',
                        background: i === 1 || i === 3 ? '#fff' : '#6b7280',
                        boxShadow: i === 1 || i === 3 ? '0 0 10px #fff' : 'none'
                    }} />
                ))}
            </div>

            {/* Connections layer 1 */}
            <div style={{
                flex: 1,
                height: '128px',
                position: 'relative',
                margin: '0 1rem'
            }}>
                <div style={{
                    position: 'absolute',
                    top: '20%',
                    left: 0,
                    right: 0,
                    height: '1px',
                    background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,85,0,0.2) 50%, rgba(255,255,255,0.05) 100%)',
                    backgroundSize: '200% 100%',
                    animation: isTraining ? 'dataFlow 3s linear infinite' : 'none',
                    opacity: 0.4,
                    transform: 'rotate(5deg)'
                }} />
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: 0,
                    right: 0,
                    height: '1px',
                    background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,85,0,0.2) 50%, rgba(255,255,255,0.05) 100%)',
                    backgroundSize: '200% 100%',
                    animation: isTraining ? 'dataFlow 3s linear infinite' : 'none',
                    opacity: 0.8,
                    transform: 'rotate(-2deg)'
                }} />
                <div style={{
                    position: 'absolute',
                    top: '80%',
                    left: 0,
                    right: 0,
                    height: '1px',
                    background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,85,0,0.2) 50%, rgba(255,255,255,0.05) 100%)',
                    backgroundSize: '200% 100%',
                    animation: isTraining ? 'dataFlow 3s linear infinite' : 'none',
                    opacity: 0.3,
                    transform: 'rotate(-8deg)'
                }} />
            </div>

            {/* Hidden layer 1 */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', transform: 'translateY(1rem)' }}>
                {[0, 1, 2].map(i => (
                    <div key={`hidden1-${i}`} style={{
                        width: i === 1 ? '16px' : '12px',
                        height: i === 1 ? '16px' : '12px',
                        borderRadius: '50%',
                        background: i === 1 ? '#FF5500' : i === 0 ? 'rgba(255, 85, 0, 0.8)' : 'rgba(255, 85, 0, 0.6)',
                        boxShadow: i === 1 ? '0 0 15px #FF5500' : 'none',
                        animation: i === 1 && isTraining ? 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite' : i === 0 && isTraining ? 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
                    }} />
                ))}
            </div>

            {/* Connections layer 2 */}
            <div style={{
                flex: 1,
                height: '128px',
                position: 'relative',
                margin: '0 1rem'
            }}>
                <div style={{
                    position: 'absolute',
                    top: '30%',
                    left: 0,
                    right: 0,
                    height: '1px',
                    background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,85,0,0.2) 50%, rgba(255,255,255,0.05) 100%)',
                    backgroundSize: '200% 100%',
                    animation: isTraining ? 'dataFlow 3s linear infinite' : 'none',
                    opacity: 0.6,
                    transform: 'rotate(-3deg)'
                }} />
                <div style={{
                    position: 'absolute',
                    top: '60%',
                    left: 0,
                    right: 0,
                    height: '1px',
                    background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,85,0,0.2) 50%, rgba(255,255,255,0.05) 100%)',
                    backgroundSize: '200% 100%',
                    animation: isTraining ? 'dataFlow 3s linear infinite' : 'none',
                    opacity: 0.8,
                    transform: 'rotate(4deg)'
                }} />
            </div>

            {/* Hidden layer 2 */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', transform: 'translateY(-0.5rem)' }}>
                {[0, 1, 2].map(i => (
                    <div key={`hidden2-${i}`} style={{
                        width: i === 1 ? '20px' : '16px',
                        height: i === 1 ? '20px' : '16px',
                        borderRadius: '50%',
                        background: i === 1 ? '#FF5500' : '#fff',
                        boxShadow: i === 1 ? '0 0 20px rgba(255, 85, 0, 0.5)' : '0 0 15px #fff',
                        animation: i === 1 && isTraining ? 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
                    }} />
                ))}
            </div>

            {/* Connections layer 3 */}
            <div style={{
                flex: 1,
                height: '128px',
                position: 'relative',
                margin: '0 1rem'
            }}>
                <div style={{
                    position: 'absolute',
                    top: '40%',
                    left: 0,
                    right: 0,
                    height: '1px',
                    background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,85,0,0.2) 50%, rgba(255,255,255,0.05) 100%)',
                    backgroundSize: '200% 100%',
                    animation: isTraining ? 'dataFlow 3s linear infinite' : 'none',
                    opacity: 0.9
                }} />
            </div>

            {/* Output layer */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
                {[0, 1, 2].map(i => (
                    <div key={`output-${i}`} style={{
                        width: i === 1 ? '24px' : '8px',
                        height: i === 1 ? '24px' : '8px',
                        borderRadius: '50%',
                        background: i === 1 ? 'transparent' : '#6b7280',
                        border: i === 1 ? '2px solid #FF5500' : 'none',
                        boxShadow: i === 1 ? '0 0 20px rgba(255, 85, 0, 0.5)' : 'none',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        {i === 1 && (
                            <div style={{
                                width: '8px',
                                height: '8px',
                                borderRadius: '50%',
                                background: '#FF5500',
                                animation: isTraining ? 'ping 1s cubic-bezier(0, 0, 0.2, 1) infinite' : 'none'
                            }} />
                        )}
                    </div>
                ))}
            </div>

            {/* Status indicators */}
            <div style={{
                position: 'absolute',
                bottom: '1rem',
                left: '1.5rem',
                display: 'flex',
                gap: '1.5rem'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: '#10b981',
                        animation: isTraining ? 'pulse 2s ease-in-out infinite' : 'none'
                    }} />
                    <span style={{
                        fontSize: '10px',
                        fontFamily: 'monospace',
                        textTransform: 'uppercase',
                        color: '#9ca3af'
                    }}>
                        Gradient Flow: Optimal
                    </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: '#FF5500',
                        animation: isTraining ? 'pulse 2s ease-in-out infinite' : 'none'
                    }} />
                    <span style={{
                        fontSize: '10px',
                        fontFamily: 'monospace',
                        textTransform: 'uppercase',
                        color: '#9ca3af'
                    }}>
                        Backprop Active
                    </span>
                </div>
            </div>
        </div>
    );

    return (
        <div style={{
            minHeight: '100vh',
            background: '#020202',
            color: '#fff',
            position: 'relative'
        }}>
            {/* Background effects */}
            <div style={{
                position: 'fixed',
                inset: 0,
                backgroundImage: 'url(\'data:image/svg+xml,%3Csvg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"%3E%3Cpath d="M0 0h40v40H0V0zm1 1h38v38H1V1z" fill="%23FFFFFF" fill-opacity="0.03" fill-rule="evenodd"/%3E%3C/svg%3E\')',
                opacity: 0.2,
                pointerEvents: 'none',
                zIndex: 0
            }} />
            <div style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 0,
                overflow: 'hidden'
            }}>
                <div style={{
                    position: 'absolute',
                    top: '-20%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '800px',
                    height: '800px',
                    background: '#FF5500',
                    borderRadius: '50%',
                    mixBlendMode: 'screen',
                    filter: 'blur(180px)',
                    opacity: 0.05,
                    animation: 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite'
                }} />
            </div>

            {/* Header */}
            <nav style={{
                position: 'fixed',
                top: 0,
                width: '100%',
                zIndex: 50,
                background: 'rgba(15, 15, 15, 0.4)',
                backdropFilter: 'blur(12px)',
                borderBottom: '1px solid rgba(255, 255, 255, 0.08)'
            }}>
                <div style={{
                    maxWidth: '1280px',
                    margin: '0 auto',
                    padding: '0 1.5rem'
                }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        height: '80px'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ fontSize: '1.5rem', color: '#FF5500' }}>⚡</span>
                            <span style={{
                                fontWeight: '700',
                                fontSize: '1.25rem',
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em'
                            }}>
                                HYPERTUNE
                            </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                            {isTraining && (
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.5rem',
                                    padding: '0.5rem 0.75rem',
                                    borderRadius: '6px',
                                    border: '1px solid rgba(255, 85, 0, 0.3)',
                                    background: 'rgba(255, 85, 0, 0.1)'
                                }}>
                                    <div style={{
                                        width: '8px',
                                        height: '8px',
                                        borderRadius: '50%',
                                        background: '#FF5500',
                                        animation: 'pulse 2s ease-in-out infinite'
                                    }} />
                                    <span style={{
                                        fontSize: '0.75rem',
                                        fontFamily: 'monospace',
                                        textTransform: 'uppercase',
                                        color: '#FF5500',
                                        fontWeight: '700'
                                    }}>
                                        Training Active
                                    </span>
                                </div>
                            )}
                            <button style={{
                                fontSize: '0.75rem',
                                fontFamily: 'monospace',
                                textTransform: 'uppercase',
                                color: '#9ca3af',
                                background: 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                padding: '0.5rem'
                            }}>
                                Node Settings
                            </button>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main style={{
                position: 'relative',
                paddingTop: '6rem',
                paddingBottom: '3rem',
                paddingLeft: '1.5rem',
                paddingRight: '1.5rem',
                zIndex: 10
            }}>
                <div style={{ maxWidth: '1536px', margin: '0 auto' }}>
                    {/* Title Section */}
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-end',
                        marginBottom: '2rem',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                        paddingBottom: '1.5rem'
                    }}>
                        <div>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.75rem',
                                marginBottom: '0.5rem'
                            }}>
                                <span style={{
                                    fontSize: '0.75rem',
                                    fontFamily: 'monospace',
                                    color: '#FF5500',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.1em',
                                    border: '1px solid rgba(255, 85, 0, 0.3)',
                                    padding: '0.25rem 0.5rem',
                                    borderRadius: '4px',
                                    background: 'rgba(255, 85, 0, 0.05)'
                                }}>
                                    Session ID: {sessionId.substring(0, 8)}
                                </span>
                                <span style={{
                                    fontSize: '0.75rem',
                                    fontFamily: 'monospace',
                                    color: '#6b7280',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.1em'
                                }}>
                                    v2.4.1
                                </span>
                            </div>
                            <h1 style={{
                                fontSize: '2.25rem',
                                fontWeight: '700',
                                marginBottom: '0.25rem'
                            }}>
                                Active Training <span style={{
                                    background: 'linear-gradient(135deg, #FF9500 0%, #FF5500 100%)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                    textShadow: '0 0 30px rgba(255, 85, 0, 0.6)'
                                }}>Node</span>
                            </h1>
                            <p style={{
                                color: '#9ca3af',
                                fontFamily: 'monospace',
                                fontSize: '0.875rem'
                            }}>
                                {latestEpoch
                                    ? `Processing shard batch #${batchNum} • Epoch ${latestEpoch.epoch}/${totalEpochs}`
                                    : 'Waiting for training to start...'}
                            </p>
                        </div>
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            <button style={{
                                padding: '0.5rem 1.5rem',
                                borderRadius: '6px',
                                background: 'rgba(239, 68, 68, 0.1)',
                                border: '1px solid rgba(239, 68, 68, 0.3)',
                                color: '#ef4444',
                                fontSize: '0.75rem',
                                fontFamily: 'monospace',
                                textTransform: 'uppercase',
                                fontWeight: '700',
                                letterSpacing: '0.1em',
                                cursor: 'pointer',
                                transition: 'all 0.2s'
                            }}
                                onMouseOver={(e) => {
                                    e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)';
                                }}
                                onMouseOut={(e) => {
                                    e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)';
                                }}
                            >
                                Stop Node
                            </button>
                            <button
                                onClick={() => setIsPaused(!isPaused)}
                                style={{
                                    padding: '0.5rem 1.5rem',
                                    borderRadius: '6px',
                                    background: 'rgba(255, 255, 255, 0.05)',
                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                    color: '#fff',
                                    fontSize: '0.75rem',
                                    fontFamily: 'monospace',
                                    textTransform: 'uppercase',
                                    fontWeight: '700',
                                    letterSpacing: '0.1em',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s'
                                }}
                                onMouseOver={(e) => {
                                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                                }}
                                onMouseOut={(e) => {
                                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                                }}
                            >
                                {isPaused ? 'Resume' : 'Pause'}
                            </button>
                        </div>
                    </div>

                    {/* Neural Network Visualization */}
                    <div style={{
                        position: 'relative',
                        width: '100%',
                        background: 'rgba(15, 15, 15, 0.4)',
                        backdropFilter: 'blur(12px)',
                        border: '1px solid rgba(255, 85, 0, 0.2)',
                        borderRadius: '16px',
                        marginBottom: '2rem',
                        overflow: 'hidden'
                    }}>
                        <div style={{
                            position: 'absolute',
                            inset: 0,
                            background: 'linear-gradient(to bottom, transparent, rgba(255, 85, 0, 0.05), transparent)',
                            opacity: 0.3,
                            pointerEvents: 'none'
                        }} />
                        {renderNeuralNetwork()}
                    </div>

                    {/* System Log with Uptime */}
                    <SystemLog
                        resultsData={resultsData}
                        uptime={uptime}
                        sessionId={resultsData?.session_id || null}
                    />
                </div>
            </main>

            <style>{`
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                @keyframes ping {
                    75%, 100% {
                        transform: scale(2);
                        opacity: 0;
                    }
                }
                @keyframes dataFlow {
                    0% { background-position: 0% 50%; }
                    100% { background-position: 100% 50%; }
                }
            `}</style>
        </div>
    );
};