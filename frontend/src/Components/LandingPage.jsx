import React, { useEffect, useState, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const API_BASE = 'http://localhost:8000';

// ============================================================================
// LANDING PAGE COMPONENT
// ============================================================================
const LandingPage = ({ onNavigate }) => {
    const heroRef = useRef(null);
    const windmillRef = useRef(null);
    const featuresRef = useRef(null);

    useEffect(() => {
        const ctx = gsap.context(() => {
            // Hero title animation
            gsap.from('.hero-title', {
                opacity: 0,
                y: 50,
                duration: 1,
                ease: 'power3.out'
            });

            gsap.from('.hero-subtitle', {
                opacity: 0,
                y: 30,
                duration: 1,
                delay: 0.3,
                ease: 'power3.out'
            });

            gsap.from('.cta-group', {
                opacity: 0,
                y: 30,
                duration: 1,
                delay: 0.6,
                ease: 'power3.out'
            });

            // Windmill rotation on scroll
            gsap.to(windmillRef.current, {
                rotation: 360,
                scrollTrigger: {
                    trigger: heroRef.current,
                    start: 'top top',
                    end: 'bottom top',
                    scrub: 1,
                }
            });

            // Features stagger animation
            gsap.from('.feature-card', {
                opacity: 0,
                y: 50,
                stagger: 0.2,
                scrollTrigger: {
                    trigger: featuresRef.current,
                    start: 'top 80%',
                    end: 'bottom 60%',
                    toggleActions: 'play none none reverse'
                }
            });

            // Parallax background grid
            gsap.to('.page-root', {
                backgroundPosition: '0px 200px',
                ease: 'none',
                scrollTrigger: {
                    trigger: heroRef.current,
                    start: 'top top',
                    end: 'bottom top',
                    scrub: true
                }
            });
        });

        return () => ctx.revert();
    }, []);

    return (
        <div className="page-root">
            <div className="landing" ref={heroRef}>
                {/* Windmill Animation */}
                <div className="windmill-container">
                    <svg
                        ref={windmillRef}
                        width="120"
                        height="120"
                        viewBox="0 0 200 200"
                        style={{ transformOrigin: '50% 50%' }}
                    >
                        <defs>
                            <linearGradient id="g" x1="0" x2="1">
                                <stop offset="0" stopColor="#fff" stopOpacity="0.9" />
                                <stop offset="1" stopColor="#f97316" stopOpacity="0.9" />
                            </linearGradient>
                        </defs>
                        <circle cx="100" cy="100" r="44" fill="url(#g)" opacity="0.08" />
                        <g transform="translate(100 100)" fill="#f97316" opacity="0.9">
                            <path d="M0-6 L70-24 L56 8 Z" />
                            <path d="M0-6 L24 70 L-8 56 Z" transform="rotate(90)" />
                            <path d="M0-6 L-70 24 L-56 -8 Z" transform="rotate(180)" />
                            <path d="M0-6 L-24 -70 L8 -56 Z" transform="rotate(270)" />
                        </g>
                        <circle cx="100" cy="100" r="10" fill="#f97316" />
                    </svg>
                </div>

                <h1 className="hero-title">HypertuneAI</h1>
                <p className="hero-subtitle">
                    Distributed machine learning training platform. Harness the power of collaborative hyperparameter optimization.
                </p>

                <div className="cta-group">
                    <button className="cta-btn cta-primary" onClick={() => onNavigate('auth')}>
                        Get Started
                    </button>
                    <button className="cta-btn cta-secondary" onClick={() => onNavigate('auth')}>
                        Sign In
                    </button>
                </div>

                <div className="features" ref={featuresRef}>
                    <div className="feature-card">
                        <div className="feature-icon">⚡</div>
                        <h3 className="feature-title">Lightning Fast</h3>
                        <p className="feature-desc">Distribute training across multiple peers for faster convergence</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">🔒</div>
                        <h3 className="feature-title">Secure & Private</h3>
                        <p className="feature-desc">Your data stays encrypted with enterprise-grade security</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">📊</div>
                        <h3 className="feature-title">Real-time Analytics</h3>
                        <p className="feature-desc">Monitor training progress and results in real-time</p>
                    </div>
                </div>
            </div>
        </div>
    );
};
export default LandingPage;