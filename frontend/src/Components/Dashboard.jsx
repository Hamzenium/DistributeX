import React, { useEffect, useState, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const API_BASE = 'http://localhost:8000'; // added

// ============================================================================
// DASHBOARD COMPONENT
// ============================================================================
const Dashboard = ({ user, onLogout }) => {
    const [userStatus, setUserStatus] = useState('OFFLINE');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [sessions, setSessions] = useState([]);
    const dashRef = useRef(null);

    useEffect(() => {
        if (dashRef.current) {
            gsap.from(dashRef.current.children, {
                opacity: 0,
                y: 30,
                stagger: 0.1,
                duration: 0.8,
                ease: 'power3.out'
            });
        }
    }, []);

    const apiCall = async (endpoint, method = 'GET', body = null, isFormData = false) => {
        const headers = { 'Authorization': `Bearer ${localStorage.getItem('token')}` };

        if (!isFormData && body) {
            headers['Content-Type'] = 'application/json';
        }

        const options = { method, headers };
        if (body) {
            options.body = isFormData ? body : JSON.stringify(body);
        }

        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    };

    const handleJoinSession = async () => {
        try {
            await apiCall('/sessions/join', 'POST');
            setUserStatus('ONLINE');
            setSuccess('You are now online and ready to receive tasks!');
        } catch (err) {
            setError(err.message);
        }
    };

    const handleLeaveSession = async () => {
        try {
            await apiCall('/sessions/leave', 'POST');
            setUserStatus('OFFLINE');
            setSuccess('You are now offline.');
        } catch (err) {
            setError(err.message);
        }
    };

    const handleSessionSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const formData = new FormData(e.target);
            const data = await apiCall('/sessions/start', 'POST', formData, true);
            setSuccess(`Session ${data.session_uid} started successfully!`);
            e.target.reset();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page-root">
            <div className="dashboard" ref={dashRef}>
                <div className="dash-header">
                    <div>
                        <h1 className="dash-title">Welcome, {user?.username || 'User'}</h1>
                        <span className={`status-badge status-${userStatus.toLowerCase()}`}>
                            {userStatus}
                        </span>
                    </div>
                    <button className="logout-btn" onClick={onLogout}>Logout</button>
                </div>

                {error && <div className="error-msg">{error}</div>}
                {success && <div className="success-msg">{success}</div>}

                <div className="section">
                    <h2 className="section-title">Worker Status</h2>
                    <div className="card" style={{ padding: '20px', maxWidth: '400px' }}>
                        {userStatus === 'OFFLINE' ? (
                            <>
                                <p style={{ margin: '0 0 16px', color: 'var(--muted)' }}>
                                    Start accepting training tasks from other users
                                </p>
                                <button className="primary-btn" onClick={handleJoinSession}>
                                    Go Online
                                    <div className="btn-overlay" />
                                </button>
                            </>
                        ) : (
                            <>
                                <p style={{ margin: '0 0 16px', color: '#86efac' }}>
                                    ✓ You're online and ready to receive tasks
                                </p>
                                <button
                                    className="primary-btn"
                                    style={{ background: '#ef4444' }}
                                    onClick={handleLeaveSession}
                                >
                                    Go Offline
                                </button>
                            </>
                        )}
                    </div>
                </div>

                <div className="section">
                    <h2 className="section-title">Start New Training Session</h2>
                    <form className="new-session-form" onSubmit={handleSessionSubmit}>
                        <div className="field">
                            <label className="label">Dataset (CSV)</label>
                            <input type="file" name="file" className="input file-input" accept=".csv" required />
                        </div>

                        <div className="field">
                            <label className="label">Number of Peers</label>
                            <input type="number" name="num_peers" className="input" min="1" max="10" defaultValue="2" required />
                        </div>

                        <div className="field">
                            <label className="label">Hyperparameters (JSON Array)</label>
                            <textarea
                                name="hyperparameters"
                                className="input textarea"
                                defaultValue={`[\n  {"learning_rate": 0.01, "batch_size": 32},\n  {"learning_rate": 0.001, "batch_size": 64}\n]`}
                                required
                            />
                        </div>

                        <button type="submit" className="primary-btn" disabled={loading}>
                            {loading ? <div className="spinner" /> : 'Start Training Session →'}
                            <div className="btn-overlay" />
                        </button>
                    </form>
                </div>

                <div className="section">
                    <h2 className="section-title">Your Sessions</h2>
                    <div className="session-grid">
                        {sessions.length === 0 ? (
                            <p style={{ color: 'var(--muted)' }}>No sessions yet. Start a new training session above.</p>
                        ) : (
                            sessions.map((s) => (
                                <div key={s.id} className="session-card">
                                    <div className="session-id">ID: {s.id}</div>
                                    <div className="session-info"><strong>Status:</strong> {s.status}</div>
                                    <div className="session-info"><strong>Peers:</strong> {s.num_peers}</div>
                                    <div className="session-info"><strong>Created:</strong> {new Date(s.created_at).toLocaleString()}</div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
export default Dashboard;