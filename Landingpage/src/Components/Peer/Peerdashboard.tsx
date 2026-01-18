// PeerDashboard.tsx - Directly loads active node dashboard without ready screen
import React, { useState, useEffect, useRef } from 'react';
import { PeerNodeDashboard } from './PeerNodeDashboard';
import type { FullResultsResponse, DashboardProps } from '../types';
import { sessionAPI } from '../apiService';

export const PeerDashboard: React.FC<DashboardProps> = ({
    user,
    onLogout,
    userRole,
    onChangeRole
}) => {
    const [resultsData, setResultsData] = useState<FullResultsResponse | null>(null);
    const [monitoringSessionId, setMonitoringSessionId] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [joinedSessions, setJoinedSessions] = useState<any[]>([]);

    const pollingInterval = useRef<number | null>(null);

    // Fetch joined sessions on mount
    useEffect(() => {
        fetchJoinedSessions();
    }, []);

    // Polling for results
    useEffect(() => {
        if (monitoringSessionId) {
            // Immediate fetch
            fetchFullResults(monitoringSessionId);

            // Then poll every second
            pollingInterval.current = window.setInterval(() => {
                fetchFullResults(monitoringSessionId);
            }, 1000);

            return () => {
                if (pollingInterval.current) {
                    clearInterval(pollingInterval.current);
                }
            };
        }
    }, [monitoringSessionId]);

    const fetchJoinedSessions = async () => {
        try {
            const data = await sessionAPI.fetchSessions();
            const joined = data.joined_sessions || [];
            setJoinedSessions(joined);

            // If there's an active session, monitor it
            const activeSession = joined.find((s: any) => s.status === 'RUNNING');
            if (activeSession) {
                setMonitoringSessionId(activeSession._id);
            }

            setLoading(false);
        } catch (err: any) {
            if (err.message === 'SESSION_EXPIRED') {
                onLogout();
            } else {
                console.error('Failed to fetch sessions:', err);
                setError(err.message);
                setLoading(false);
            }
        }
    };

    const fetchFullResults = async (sessionId: string) => {
        try {
            const data = await sessionAPI.fetchFullResults(sessionId);
            setResultsData(data);
            setError(null);
        } catch (err: any) {
            if (err.message === 'SESSION_EXPIRED') {
                onLogout();
            } else {
                console.error('Failed to fetch results:', err);
                // Don't set error for 400s during polling, just log it
                if (!err.message.includes('400')) {
                    setError(err.message);
                }
            }
        }
    };

    // Show loading state
    if (loading) {
        return (
            <div style={{
                minHeight: '100vh',
                background: '#020202',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
            }}>
                <div style={{
                    textAlign: 'center'
                }}>
                    <div style={{
                        width: '48px',
                        height: '48px',
                        border: '3px solid rgba(255, 85, 0, 0.2)',
                        borderTop: '3px solid #FF5500',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite',
                        margin: '0 auto 1rem'
                    }} />
                    <p style={{ color: '#9ca3af' }}>Loading peer dashboard...</p>
                </div>
                <style>{`
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                `}</style>
            </div>
        );
    }

    // Render the active node dashboard (timer only starts when resultsData exists)
    return (
        <PeerNodeDashboard
            resultsData={resultsData}
            onLogout={onLogout}
            onChangeRole={onChangeRole}
            user={user}
        />
    );
};