import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

interface LandingPageProps {
    onNavigate: (view: 'landing' | 'auth' | 'dashboard') => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigate }) => {
    const heroRef = useRef<HTMLDivElement>(null);
    const titleRef = useRef<HTMLHeadingElement>(null);

    useEffect(() => {
        if (heroRef.current && titleRef.current) {
            gsap.from(heroRef.current, {
                opacity: 0,
                duration: 1.2,
                ease: 'power3.out'
            });

            gsap.from(titleRef.current, {
                opacity: 0,
                y: 50,
                duration: 1,
                delay: 0.3,
                ease: 'power3.out'
            });
        }
    }, []);

    return (
        <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #1a1a1a 0%, #2d1810 50%, #4a2c1a 100%)',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }}>
            {/* Animated gradient orbs */}
            <div style={{
                position: 'absolute',
                top: '-20%',
                right: '-10%',
                width: '800px',
                height: '800px',
                background: 'radial-gradient(circle, rgba(255,140,0,0.3) 0%, transparent 70%)',
                borderRadius: '50%',
                filter: 'blur(100px)',
                animation: 'float 20s ease-in-out infinite'
            }} />

            <div style={{
                position: 'absolute',
                bottom: '-20%',
                left: '-10%',
                width: '600px',
                height: '600px',
                background: 'radial-gradient(circle, rgba(255,69,0,0.2) 0%, transparent 70%)',
                borderRadius: '50%',
                filter: 'blur(100px)',
                animation: 'float 15s ease-in-out infinite reverse'
            }} />

            {/* Navigation */}
            <nav style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                padding: '2rem 4rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                zIndex: 10
            }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem'
                }}>
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
                    <span style={{
                        fontSize: '1.5rem',
                        fontWeight: '700',
                        color: '#fff',
                        letterSpacing: '1px'
                    }}>HYPERTUNE</span>
                </div>

                <div style={{ display: 'flex', gap: '1rem' }}>
                    <button
                        onClick={() => onNavigate('auth')}
                        style={{
                            padding: '0.75rem 2rem',
                            background: 'rgba(255,255,255,0.1)',
                            border: '1px solid rgba(255,255,255,0.2)',
                            borderRadius: '8px',
                            color: '#fff',
                            cursor: 'pointer',
                            fontWeight: '500',
                            backdropFilter: 'blur(10px)',
                            transition: 'all 0.3s'
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'rgba(255,255,255,0.15)';
                            e.currentTarget.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'rgba(255,255,255,0.1)';
                            e.currentTarget.style.transform = 'translateY(0)';
                        }}
                    >
                        Sign In
                    </button>
                </div>
            </nav>

            {/* Hero Content */}
            <div ref={heroRef} style={{
                maxWidth: '1200px',
                padding: '0 2rem',
                zIndex: 2
            }}>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '4rem',
                    alignItems: 'center'
                }}>
                    {/* Left Content */}
                    <div>
                        <h1 ref={titleRef} style={{
                            fontSize: '4rem',
                            fontWeight: '700',
                            color: '#fff',
                            marginBottom: '1.5rem',
                            lineHeight: '1.1',
                            letterSpacing: '-2px'
                        }}>
                            A new era of<br />
                            <span style={{
                                background: 'linear-gradient(135deg, #ff8c00, #ff4500)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                backgroundClip: 'text'
                            }}>
                                distributed ML
                            </span>
                        </h1>

                        <p style={{
                            fontSize: '1.25rem',
                            color: 'rgba(255,255,255,0.7)',
                            marginBottom: '2rem',
                            lineHeight: '1.6'
                        }}>
                            The world's most advanced platform to help you train models faster,
                            leverage distributed compute, and achieve breakthrough performance.
                        </p>

                        <button
                            onClick={() => onNavigate('auth')}
                            style={{
                                padding: '1rem 3rem',
                                background: 'linear-gradient(135deg, #ff8c00, #ff4500)',
                                border: 'none',
                                borderRadius: '12px',
                                color: '#fff',
                                fontSize: '1.1rem',
                                fontWeight: '600',
                                cursor: 'pointer',
                                boxShadow: '0 10px 40px rgba(255,69,0,0.3)',
                                transition: 'all 0.3s',
                                position: 'relative',
                                overflow: 'hidden'
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.transform = 'translateY(-2px)';
                                e.currentTarget.style.boxShadow = '0 15px 50px rgba(255,69,0,0.4)';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = '0 10px 40px rgba(255,69,0,0.3)';
                            }}
                        >
                            GET STARTED
                        </button>
                    </div>

                    {/* Right Visual */}
                    <div style={{
                        position: 'relative',
                        height: '500px'
                    }}>
                        <div style={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '400px',
                            height: '400px',
                            background: 'linear-gradient(135deg, rgba(255,140,0,0.2), rgba(255,69,0,0.2))',
                            borderRadius: '50%',
                            filter: 'blur(60px)',
                            animation: 'pulse 4s ease-in-out infinite'
                        }} />

                        {/* Silhouette placeholder */}
                        <div style={{
                            position: 'relative',
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <div style={{
                                width: '300px',
                                height: '300px',
                                background: 'rgba(0,0,0,0.3)',
                                borderRadius: '50%',
                                border: '2px solid rgba(255,140,0,0.3)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '6rem',
                                animation: 'rotate 20s linear infinite'
                            }}>
                                ⚡
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) scale(1); }
          50% { transform: translateY(-50px) scale(1.05); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
          50% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.1); }
        }
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
        </div>
    );
};