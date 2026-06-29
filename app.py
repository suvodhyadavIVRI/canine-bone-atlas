import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from collections import defaultdict, Counter
import tensorflow as tf
import keras

# Set page configuration
st.set_page_config(
    page_title="Canine Osteology AI System",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# STYLING (Premium Cyber-Academic Dashboard CSS v2.0)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* ===== CSS Custom Properties ===== */
    :root {
        --primary-gradient: linear-gradient(135deg, #ff79c6 0%, #bd93f9 100%);
        --accent-cyan: #8be9fd;
        --accent-green: #50fa7b;
        --accent-orange: #ffb86c;
        --accent-pink: #ff79c6;
        --accent-purple: #bd93f9;
        --accent-red: #ff5555;
        --bg-deep: #030712;
        --bg-surface: rgba(17, 24, 39, 0.6);
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
    }

    /* ===== Keyframe Animations ===== */
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    @keyframes pulse-glow {
        0% { box-shadow: 0 4px 15px rgba(189, 147, 249, 0.3); }
        100% { box-shadow: 0 8px 30px rgba(255, 121, 198, 0.6); }
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    @keyframes borderGlow {
        0%, 100% { border-color: rgba(189, 147, 249, 0.25); }
        50% { border-color: rgba(255, 121, 198, 0.45); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes scanLine {
        0% { top: -5%; }
        100% { top: 105%; }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* ===== Layout and Spacing ===== */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    [data-testid="stHeader"] {
        display: none !important;
    }

    /* ===== Custom Scrollbars ===== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.15); border-radius: 10px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-pink), var(--accent-purple));
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-purple), var(--accent-cyan));
    }

    /* ===== Global Base ===== */
    .stApp {
        background: radial-gradient(ellipse at 20% 0%, #1a1035 0%, #111827 40%, #030712 100%) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ===== Typography ===== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }

    /* ===== Premium Header Banner ===== */
    .app-header {
        position: relative;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(3, 7, 18, 0.85) 100%);
        backdrop-filter: blur(24px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 32px 36px 28px;
        margin-bottom: 28px;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }

    .app-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: linear-gradient(to bottom, var(--accent-pink), var(--accent-purple), var(--accent-cyan));
        animation: gradientShift 3s ease infinite;
        background-size: 100% 200%;
    }

    .app-header::after {
        content: '';
        position: absolute;
        top: -50%; right: -30%; width: 80%; height: 200%;
        background: radial-gradient(circle, rgba(189, 147, 249, 0.04) 0%, transparent 70%);
        pointer-events: none;
    }

    .app-title {
        background: linear-gradient(135deg, #ffb86c 0%, #ff79c6 30%, #bd93f9 60%, #8be9fd 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        margin: 0; padding: 0;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.15;
        animation: shimmer 4s linear infinite;
        font-family: 'Outfit', sans-serif;
    }

    .app-subtitle {
        color: var(--text-secondary);
        font-size: 0.88rem;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'Outfit', sans-serif;
    }

    .app-subtitle::after {
        content: '';
        flex-grow: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.15), transparent);
        margin-left: 10px;
    }

    .header-meta {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(80, 250, 123, 0.1);
        border: 1px solid rgba(80, 250, 123, 0.2);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 600;
        color: var(--accent-green);
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }

    .status-dot {
        width: 7px; height: 7px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse-dot 1.5s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(80, 250, 123, 0.5);
        display: inline-block;
    }

    .version-badge {
        background: rgba(189, 147, 249, 0.1);
        border: 1px solid rgba(189, 147, 249, 0.2);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--accent-purple);
        font-family: 'Fira Code', monospace;
        letter-spacing: 0.5px;
    }

    .thesis-badge {
        background: rgba(255, 184, 108, 0.1);
        border: 1px solid rgba(255, 184, 108, 0.2);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--accent-orange);
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #030712 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* ===== Tab Bar ===== */
    [data-baseweb="tab-list"] {
        background: var(--glass-bg) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        border: 1px solid var(--glass-border) !important;
        gap: 5px !important;
    }

    [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        background: transparent !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #ffffff !important;
    }

    [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(189, 147, 249, 0.3) !important;
        border-bottom: none !important;
    }

    /* ===== Buttons ===== */
    /* Target only our application buttons, avoiding pollution of Streamlit's internal buttons */
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button, div.stLinkButton a {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: var(--text-primary) !important;
        border-radius: 30px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 0.85rem !important;
        font-family: 'Outfit', sans-serif !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-decoration: none !important;
    }

    div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover, div.stLinkButton a:hover {
        background: var(--primary-gradient) !important;
        border-color: transparent !important;
        box-shadow: 0 0 25px rgba(189, 147, 249, 0.4) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }

    /* ===== File Uploader ===== */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(189, 147, 249, 0.3) !important;
        background: rgba(17, 24, 39, 0.4) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: all 0.4s ease !important;
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.2) !important;
        animation: borderGlow 3s ease-in-out infinite;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-purple) !important;
        background: rgba(189, 147, 249, 0.06) !important;
        box-shadow: inset 0 0 30px rgba(189, 147, 249, 0.08), 0 0 20px rgba(189, 147, 249, 0.15) !important;
    }

    /* ===== Landing Page Hero ===== */
    .landing-hero {
        text-align: center;
        padding: 50px 40px 40px;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.5) 0%, rgba(3, 7, 18, 0.5) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out;
    }

    .landing-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(189, 147, 249, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }

    .landing-hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f3f4f6 0%, #d1d5db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }

    .landing-hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.92rem;
        max-width: 520px;
        margin: 0 auto 28px;
        line-height: 1.7;
        font-family: 'Inter', sans-serif;
    }

    .landing-upload-cta {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: var(--primary-gradient);
        padding: 14px 32px;
        border-radius: 40px;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: white;
        box-shadow: 0 8px 30px rgba(189, 147, 249, 0.35);
        animation: float 3s ease-in-out infinite;
        letter-spacing: 0.5px;
    }

    /* ===== Feature Cards Grid ===== */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        margin-top: 32px;
    }

    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }

    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.01) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 26px 18px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out both;
    }

    .feature-card:nth-child(2) { animation-delay: 0.15s; }
    .feature-card:nth-child(3) { animation-delay: 0.3s; }

    .feature-card:hover {
        transform: translateY(-6px);
        border-color: rgba(189, 147, 249, 0.3);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
        display: block;
    }

    .feature-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: var(--text-primary);
        margin-bottom: 8px;
    }

    .feature-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    /* ===== Bone Type Badges Showcase ===== */
    .bone-showcase {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-top: 28px;
        animation: fadeInUp 0.8s ease-out 0.35s both;
    }

    .bone-type-badge {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 7px 18px;
        border-radius: 24px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-secondary);
        transition: all 0.3s ease;
        cursor: default;
    }

    .bone-type-badge:hover {
        background: var(--primary-gradient);
        color: white;
        border-color: transparent;
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(189, 147, 249, 0.3);
    }

    /* ===== Clean Image Frame ===== */
    .image-frame {
        border: 1px solid var(--glass-border);
        background: rgba(3, 7, 18, 0.5);
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }

    /* Hide Streamlit's fullscreen button on images inside the frame */
    .image-frame button[title="View fullscreen"] {
        top: 8px !important;
        right: 8px !important;
        background: rgba(0,0,0,0.5) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        padding: 4px 6px !important;
    }

    /* ===== Console / Coordinate Box ===== */
    .sys-coord-box {
        margin-top: 12px;
        font-family: 'Fira Code', monospace;
        font-size: 0.76rem;
        color: var(--accent-cyan);
        background: rgba(3, 7, 18, 0.6) !important;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid rgba(139, 233, 253, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }

    /* ===== Landmark Cards ===== */
    .landmark-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(3, 7, 18, 0.75) 100%) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-left-width: 6px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .landmark-card:hover {
        transform: translateY(-3px) !important;
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.35) !important;
    }

    /* ===== Bone Classification Badge ===== */
    .bone-badge-container {
        display: flex;
        align-items: center;
        margin-bottom: 24px;
        gap: 15px;
    }

    .bone-badge {
        background: var(--primary-gradient);
        padding: 10px 24px;
        border-radius: 40px;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 25px rgba(189, 147, 249, 0.3);
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        animation: pulse-glow 2.5s infinite alternate;
    }

    .classifier-confidence-badge {
        background: rgba(255, 184, 108, 0.15) !important;
        color: var(--accent-orange) !important;
        border: 1px solid rgba(255, 184, 108, 0.3) !important;
        padding: 8px 20px;
        border-radius: 40px;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
    }

    /* ===== Confidence Badges ===== */
    .conf-badge {
        background: rgba(80, 250, 123, 0.1) !important;
        color: var(--accent-green) !important;
        border: 1px solid rgba(80, 250, 123, 0.2) !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        float: right;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }

    .conf-badge-medium {
        background: rgba(255, 184, 108, 0.1) !important;
        color: var(--accent-orange) !important;
        border: 1px solid rgba(255, 184, 108, 0.2) !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        float: right;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }

    .conf-badge-low {
        background: rgba(255, 85, 85, 0.1) !important;
        color: var(--accent-red) !important;
        border: 1px solid rgba(255, 85, 85, 0.2) !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        float: right;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }

    /* ===== Metric Boxes ===== */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .metric-box {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px !important;
        padding: 16px 12px !important;
        text-align: center;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease;
    }

    .metric-box:hover {
        border-color: rgba(255, 255, 255, 0.15) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        transform: translateY(-2px);
    }

    .metric-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--accent-cyan);
        line-height: 1;
    }

    .metric-lbl {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-top: 6px;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* ===== Custom Slider ===== */
    .stSlider {
        margin-bottom: 20px !important;
        padding-bottom: 10px !important;
    }
    .stSlider [data-baseweb="slider"] {
        height: 6px !important;
        padding-left: 0px !important;
        padding-right: 0px !important;
    }
    /* Fix slider value overflow in sidebar */
    [data-testid="stSidebar"] .stSlider > div {
        overflow: visible !important;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"],
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    [data-testid="stSidebar"] .stSlider {
        padding-left: 4px !important;
        padding-right: 4px !important;
    }

    /* ===== Hide Streamlit deploy button ===== */
    [data-testid="stStatusWidget"],
    .stDeployButton,
    [data-testid="stToolbar"] button[kind="header"] {
        display: none !important;
    }

    /* ===== Sidebar Model Status Cards ===== */
    .model-status-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: all 0.3s ease;
    }

    .model-status-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
    }

    .model-dot-active {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        box-shadow: 0 0 8px rgba(80, 250, 123, 0.5);
        animation: pulse-dot 2s ease-in-out infinite;
        display: inline-block;
        flex-shrink: 0;
    }

    .model-dot-inactive {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent-orange);
        box-shadow: 0 0 8px rgba(255, 184, 108, 0.3);
        display: inline-block;
        flex-shrink: 0;
    }

    .model-status-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* ===== Sidebar Section Dividers ===== */
    .sidebar-section-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--text-secondary);
        margin: 20px 0 10px;
        padding-top: 16px;
        border-top: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ===== Sidebar Footer ===== */
    .sidebar-footer {
        text-align: center;
        padding: 16px 0 8px;
        border-top: 1px solid var(--border-subtle);
        margin-top: 24px;
    }

    .sidebar-footer-text {
        font-family: 'Outfit', sans-serif;
        font-size: 0.68rem;
        color: var(--text-secondary);
        letter-spacing: 0.5px;
        line-height: 1.6;
    }

    /* ===== Tabs custom overrides ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid var(--border-subtle) !important;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border: none !important;
        padding: 10px 16px !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-purple) !important;
        border-bottom: 2px solid var(--accent-purple) !important;
    }

    /* ===== Upload Zone Placeholder ===== */
    .upload-zone-placeholder {
        border: 2px dashed rgba(189, 147, 249, 0.2);
        border-radius: 20px;
        padding: 50px 30px;
        text-align: center;
        background: radial-gradient(ellipse at 50% 50%, rgba(189, 147, 249, 0.03), transparent 70%);
        animation: borderGlow 3s ease-in-out infinite;
    }

    .upload-zone-icon {
        font-size: 2.8rem;
        display: block;
        margin-bottom: 14px;
        animation: float 3s ease-in-out infinite;
        opacity: 0.5;
    }

    .upload-zone-text {
        color: var(--text-secondary);
        font-family: 'Outfit', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* ===== Diagnostic Confidence Gauge ===== */
    .diag-gauge {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 20px;
        animation: fadeInUp 0.5s ease-out;
    }

    .diag-gauge-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .diag-gauge-label {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: var(--accent-pink);
        font-size: 1.15rem;
    }

    .diag-gauge-value {
        font-family: 'Fira Code', monospace;
        font-weight: 600;
        color: var(--accent-purple);
        font-size: 1rem;
    }

    .diag-gauge-track {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        height: 10px;
        width: 100%;
        overflow: hidden;
    }

    .diag-gauge-fill {
        background: var(--primary-gradient);
        height: 100%;
        border-radius: 10px;
        box-shadow: 0 0 12px rgba(189, 147, 249, 0.6);
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ===== System Diagnostics Console ===== */
    .diagnostics-console {
        font-family: 'Fira Code', monospace;
        font-size: 0.78rem;
        background: linear-gradient(135deg, #030712 0%, #0a0f1a 100%);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 18px;
        color: var(--accent-cyan);
        line-height: 1.6;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
    }

    .diagnostics-console .log-line {
        margin-bottom: 2px;
    }

    /* ===== Rejection Panels ===== */
    .rejection-panel {
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 24px;
        animation: fadeInUp 0.5s ease-out;
    }

    .rejection-panel-error {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .rejection-panel-warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# IMPORT VISUALIZATION LIBRARIES
# ---------------------------------------------------------
try:
    from anatomy_visualizer import (
        make_publication_card,
        REGION_COLORS,
        REGION_DESC,
        color_for
    )
except ImportError:
    st.error("Failed to import anatomy_visualizer.py. Make sure it is in the same directory.")

# ---------------------------------------------------------
# LEGACY KERAS COMPATIBILITY REGISTER
# ---------------------------------------------------------
@keras.saving.register_keras_serializable(package="Custom")
class TrueDivide(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, *args, **kwargs):
        clean_args = [args[0]]
        return super().__call__(*clean_args, **kwargs)
    def call(self, x):
        return x / 127.5

@keras.saving.register_keras_serializable(package="Custom")
class CustomSubtract(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, *args, **kwargs):
        clean_args = [args[0]]
        return super().__call__(*clean_args, **kwargs)
    def call(self, x):
        return x - 1.0

# ---------------------------------------------------------
# LOAD CACHED MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_cnn_classifier():
    try:
        model_path = "bone_mobilenetv2_model.h5"
        if os.path.exists(model_path):
            model = keras.models.load_model(
                model_path,
                custom_objects={
                    "TrueDivide": TrueDivide,
                    "Subtract": CustomSubtract
                }
            )
            return model
    except Exception as e:
        st.error(f"Error loading Keras CNN model: {e}")
    return None

@st.cache_resource
def load_yolo_model(weights_path):
    try:
        from ultralytics import YOLO
        if os.path.exists(weights_path):
            return YOLO(weights_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
    return None

# ---------------------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------------------
def classify_bone_image(model, image_pil):
    if model is None:
        return "Unknown", 0.0
    
    # Preprocess for MobileNetV2 input size (224x224x3)
    img_resized = image_pil.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    confidence = float(preds[0][pred_idx])
    
    # Map from class_names.json indices to standard labels
    class_map = {
        0: "Femur",
        1: "Fibula",
        2: "Humerus",
        3: "Radius-Ulna",
        4: "Scapula",
        5: "Tibia",
        6: "Os-Coxae" # Map "os coxae ima" to "Os-Coxae"
    }
    
    return class_map.get(pred_idx, "Unknown"), confidence

def detect_all_landmarks(yolo_model, image_pil, conf_threshold):
    if yolo_model is None:
        return []
        
    results = yolo_model(image_pil, conf=conf_threshold, verbose=False)
    if not results or len(results) == 0:
        return []
        
    res = results[0]
    names = res.names
    
    annotations = []
    
    if res.masks is not None:
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confidences = res.boxes.conf.cpu().numpy()
        polygons = res.masks.xy
        
        for i, (box, cls_id, conf, poly) in enumerate(zip(boxes, classes, confidences, polygons)):
            if len(poly) < 3:
                continue
                
            full_class_name = names.get(cls_id, f"Unknown(id={cls_id})")
            
            # Extract bone prefix (e.g. 'Scapula' from 'Scapula: Acromion Process')
            if ": " in full_class_name:
                bone_prefix, landmark_name = full_class_name.split(": ", 1)
            else:
                bone_prefix, landmark_name = "Unknown", full_class_name
                
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            poly_flat = poly.flatten().tolist()
            
            annotations.append({
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "segmentation": [poly_flat],
                "label": landmark_name,
                "confidence": float(conf),
                "bone_prefix": bone_prefix
            })
            
    return annotations

def predict_bone_landmarks(yolo_model, image_pil, target_bone, conf_threshold):
    if yolo_model is None:
        return []
        
    results = yolo_model(image_pil, conf=conf_threshold, verbose=False)
    if not results or len(results) == 0:
        return []
        
    res = results[0]
    names = res.names
    
    annotations = []
    
    if res.masks is not None:
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confidences = res.boxes.conf.cpu().numpy()
        polygons = res.masks.xy
        
        for i, (box, cls_id, conf, poly) in enumerate(zip(boxes, classes, confidences, polygons)):
            if len(poly) < 3:
                continue
                
            full_class_name = names.get(cls_id, f"Unknown(id={cls_id})")
            
            # Extract bone prefix (e.g. 'Scapula' from 'Scapula: Acromion Process')
            if ": " in full_class_name:
                bone_prefix, landmark_name = full_class_name.split(": ", 1)
            else:
                bone_prefix, landmark_name = "Unknown", full_class_name
                
            # Filter segmentations to only include the target bone's landmarks
            # Compare normalization of names (e.g. 'Os-Coxae' vs 'Os-Coxae' or 'os coxae ima')
            norm_prefix = bone_prefix.lower().replace("-", "").replace(" ", "")
            norm_target = target_bone.lower().replace("-", "").replace(" ", "")
            
            if norm_prefix != norm_target:
                continue
                
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            poly_flat = poly.flatten().tolist()
            
            annotations.append({
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "segmentation": [poly_flat],
                "label": landmark_name,
                "confidence": float(conf)
            })
            
    return annotations

# ---------------------------------------------------------
# TEXTBOOK BONE DESCRIPTIONS
# ---------------------------------------------------------
BONE_TEXTBOOK_DESCRIPTIONS = {
    "Femur": {
        "description": "The femur is the heaviest, strongest, and longest long bone of the canine hindlimb. It forms the skeleton of the thigh, extending from the hip joint to the stifle (knee) joint. It articulates proximally with the acetabulum of the pelvis and distally with the tibia and patella.",
        "features": [
            "**Femoral Head**: The smooth, hemispherical projection articulating with the pelvic acetabulum to form the hip joint.",
            "**Greater Trochanter**: A massive lateral process serving as the insertion point for the middle and deep gluteal muscles.",
            "**Lesser & Third Trochanters**: Bony landmarks on the shaft serving as insertions for the iliopsoas and superficial gluteal muscles.",
            "**Trochlea of Femur**: A smooth, deep cranial groove where the patella glides during flexion and extension.",
            "**Medial & Lateral Condyles**: Articular surfaces on the distal extremity articulating with the tibia and menisci."
        ],
        "clinical": "Medial patellar luxation (MPL) is a common hereditary orthopedic condition in small-breed dogs, often caused by a shallow femoral trochlea. Femoral neck fractures or head fractures (e.g. aseptic necrosis of the femoral head) are treated surgically, sometimes via Femoral Head Osteotomy (FHO). Stifle stability relies heavily on the cruciate ligaments connecting the femur and tibia.",
        "species": "In dogs, the third trochanter is present on the lateral aspect of the shaft, which is absent in cats. The femoral shaft is straight in cats, but curved cranially in dogs."
    },
    "Fibula": {
        "description": "The fibula is a long, slender, reduced bone running parallel and lateral to the tibia in the canine leg. It acts primarily as an anchor for muscles and lateral collateral ligaments of the stifle and hock, bearing negligible weight.",
        "features": [
            "**Head of Fibula**: The proximal expansion articulating with the lateral condyle of the tibia.",
            "**Neck of Fibula**: The constricted portion just below the head.",
            "**Lateral Malleolus**: The distal extremity forming the outer side of the hock joint, articulating with the tibial cochlea and the talus."
        ],
        "clinical": "Fibular fractures rarely occur in isolation and usually happen alongside tibial fractures. Because the tibia is the primary weight-bearing bone, isolated fibular fractures can often be treated conservatively, but instability of the lateral malleolus disrupts hock joint function.",
        "species": "In dogs, the tibia and fibula are separate, articulating at proximal and distal tibiofibular joints. In horses, the fibula is vestigial and fuses halfway down the tibia. In cattle, the head fuses with the tibia, and only the distal end remains as a separate malleolar bone."
    },
    "Humerus": {
        "description": "The humerus is the long bone of the brachium (upper arm) in dogs. It articulates proximally with the scapula to form the shoulder joint and distally with the radius and ulna to form the elbow joint. It has an S-shape and features the supratrochlear foramen.",
        "features": [
            "**Humeral Head**: The smooth, rounded articular surface on the proximal end that articulates with the scapula's glenoid cavity.",
            "**Greater & Lesser Tubercles**: Bony projections on the proximal end for muscle attachments.",
            "**Deltoid Ridge**: A prominent ridge on the craniolateral aspect of the humerus where the deltoid muscle inserts.",
            "**Supratrochlear Foramen**: An opening in the bony septum between the radial and olecranon fossae. Unique to dogs; no major vessels pass through it.",
            "**Olecranon Fossa**: A deep cavity on the caudal surface of the distal extremity which receives the anconeal process of the ulna."
        ],
        "clinical": "Intercondylar fractures (Y- or T-fractures) are common in spaniel breeds due to incomplete ossification of the humeral condyle (IOHC). The supratrochlear foramen is a site of structural weakness, making distal fractures common during traumatic falls.",
        "species": "The supratrochlear foramen is present in dogs but absent in cats. In cats, a supracondylar foramen is present on the medial side of the distal shaft, through which the median nerve and brachial artery pass (a critical landmark to avoid during surgery)."
    },
    "Radius-Ulna": {
        "description": "The radius and ulna are the paired bones of the canine forelimb (forearm). The radius is the cranial, larger, and major weight-bearing bone. The ulna is longer, situated caudolaterally, and provides the olecranon process (elbow point) which serves as a lever arm for elbow extension.",
        "features": [
            "**Olecranon Process**: The large projection at the proximal end of the ulna, forming the point of the elbow (attachment for triceps).",
            "**Anconeal Process**: A sharp, beak-like projection on the proximal end of the ulna that fits into the olecranon fossa of the humerus.",
            "**Trochlear Notch**: The semi-circular groove on the ulna that receives the trochlea of the humerus.",
            "**Body of Radius**: The shaft of the radius, curved cranially to support forelimb weight.",
            "**Styloid Process of Radius & Ulna**: Pointed projections at the distal medial end of the radius and distal lateral end of the ulna."
        ],
        "clinical": "Elbow dysplasia is a major orthopedic issue in large-breed dogs, often involving: Ununited Anconeal Process (UAP), Fragmented Medial Coronoid Process (FMCP), or osteochondrosis dissecans (OCD). Radius-ulna fractures are common in toy breeds, where poor vascularization of the distal radius can lead to non-union.",
        "species": "In dogs, the radius and ulna are fused by interosseous ligaments, allowing minimal rotation (pronation/supination). In cats, the bones remain distinct and permit extensive rotation of the paw."
    },
    "Scapula": {
        "description": "The scapula (shoulder blade) is a flat, triangular bone of the canine shoulder girdle. Uniquely in dogs, it has no bony connection to the thorax (known as synsarcosis) and is held entirely by muscles, absorbing shock during movement.",
        "features": [
            "**Scapular Spine**: A prominent shelf-like plate of bone dividing the lateral surface of the scapula.",
            "**Acromion Process**: The lateral extension of the scapular spine. In dogs, it is well-developed and serves as the origin for the acromiodeltoideus muscle.",
            "**Glenoid Cavity**: A shallow articular depression that articulates with the head of the humerus to form the shoulder joint.",
            "**Supraspinous Fossa**: The area cranial to the scapular spine, occupied by the supraspinatus muscle.",
            "**Infraspinous Fossa**: The large area caudal to the scapular spine, occupied by the infraspinatus muscle.",
            "**Subscapular Fossa**: The large, shallow depression on the medial surface occupied by the subscapularis muscle."
        ],
        "clinical": "Fractures of the scapular body or spine are rare due to the protective muscle cover, but fractures of the glenoid neck or cavity are clinically significant as they involve the joint and lead to osteoarthritic changes. Congenital shoulder luxation can also occur in toy breeds.",
        "species": "In dogs, the spine divides the lateral surface into two nearly equal fossae (ratio 1:1), and terminates in a prominent acromion process. Cats have a metacromion process projecting caudally from the acromion, which dogs lack."
    },
    "Tibia": {
        "description": "The tibia (shin bone) is the large weight-bearing bone of the leg. It is accompanied laterally by the thin fibula. Proximally it articulates with the femur and patella, and distally with the talus.",
        "features": [
            "**Tibial Tuberosity**: A large cranial projection at the proximal end of the tibia for attachment of the patellar ligament.",
            "**Tibial Crest**: A prominent, palpable cranial border of the tibial shaft extending distally from the tuberosity.",
            "**Medial & Lateral Condyles**: The articular surfaces on the proximal tibia that articulate with the femur and meniscus.",
            "**Medial Malleolus**: The prominent bony projection on the medial side of the distal extremity of the tibia.",
            "**Tibial Cochlea**: The distal articular surface of the tibia articulating with the talus of the ankle."
        ],
        "clinical": "Tibial Plateau Leveling Osteotomy (TPLO) is a very common surgical procedure in veterinary medicine to treat cranial cruciate ligament (CCL) rupture by altering the angle of the tibial plateau. Avulsion fracture of the tibial tuberosity is common in growing large breed dogs.",
        "species": "The tibia and fibula are separate in dogs, articulating at the proximal and distal tibiofibular joints. In sheep and goats, the distal fibula is a separate malleolar bone."
    },
    "Os-Coxae": {
        "description": "The os coxae (hip bone or pelvis) is formed by the fusion of three bones: the ilium, ischium, and pubis. These meet at the acetabulum, which forms the hip joint socket with the femur. The pelvis supports the abdominal organs and transmits hindlimb forces.",
        "features": [
            "**Cotyloid Cavity (Acetabulum)**: The deep cup-shaped socket that articulates with the head of the femur.",
            "**Ilium**: The largest and most cranial of the pelvic bones, forming the wing of the ilium.",
            "**Obturator Foramen**: The large opening on the pelvic floor bounded by the pubis and ischium.",
            "**Pelvic Symphysis**: The median joint uniting the two halves of the pelvis.",
            "**Ischiatic Tuberosity**: The thick caudal angle of the ischium, forming the pin bone."
        ],
        "clinical": "Hip dysplasia is a common hereditary disease in large dogs, characterized by joint laxity and subluxation. Triple Pelvic Osteotomy (TPO) is performed in young dogs to improve femoral head coverage. Pelvic fractures often result from vehicular trauma and require surgical stabilization if they collapse the pelvic canal.",
        "species": "The dog pelvic symphysis ossifies completely with age, whereas it remains cartilaginous longer in smaller species. The wing of the ilium is oriented vertically in dogs, but more horizontally in cattle."
    },
    "Skull": {
        "description": "The skull of the dog is a complex skeletal structure enclosing the brain and supporting the facial features. It is divided into the cranium (protecting the brain) and the face (supporting the nasal cavity and mouth).",
        "features": [
            "**Zygomatic Arch**: The bony arch forming the lateral boundary of the orbit, serving as the origin for the masseter muscle.",
            "**Occipital Condyle**: The articular prominences at the base of the skull that articulate with the first cervical vertebra (atlas).",
            "**External Sagittal Crest**: A median ridge on the dorsal aspect of the skull, prominent in dolichocephalic breeds.",
            "**Nasal Bone**: The paired bones forming the dorsal roof of the nasal cavity.",
            "**Mandible**: The lower jaw bone, forming the temporomandibular joint (TMJ)."
        ],
        "clinical": "Mandibular fractures are common in small breed dogs due to periodontal disease weakening the bone. Craniomandibular osteopathy (CMO) is a non-neoplastic disease of young dogs (especially Westies) causing bilateral bony proliferation of the mandible and tympanic bullae.",
        "species": "Dog skulls vary dramatically in shape by breed: dolichocephalic (long, e.g. Greyhound), mesocephalic (medium, e.g. Beagle), and brachycephalic (short/flat, e.g. Pug). Cats have much more uniform mesocephalic-like skulls with larger orbital cavities."
    }
}

# ---------------------------------------------------------
# STREAMLIT INTERFACE MAIN
# ---------------------------------------------------------
def main():
    # ─── Premium Header Banner ───
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🐕 Canine Osteology AI System</h1>
        <div class="app-subtitle">Section of Veterinary Anatomy, IVRI Bareilly</div>
        <div class="header-meta">
            <span class="status-badge"><span class="status-dot"></span> System Online</span>
            <span class="version-badge">v2.0 — AI Engine</span>
            <span class="thesis-badge">MVSc Thesis Project</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load classification and segmentation models
    _base = os.path.dirname(os.path.abspath(__file__))
    selected_weights_path = os.path.join(_base, "best.pt")
    
    cnn_model = load_cnn_classifier()
    yolo_model = load_yolo_model(selected_weights_path)
    
    # ─── Sidebar: IVRI Branding ───
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/2/23/Indian_Veterinary_Research_Institute_logo.png", width=100)
    
    # ─── Sidebar: Model Status Section ───
    st.sidebar.markdown('<div class="sidebar-section-label">🤖 AI Model Status</div>', unsafe_allow_html=True)
    
    if cnn_model:
        st.sidebar.markdown("""
        <div class="model-status-card">
            <span class="model-dot-active"></span>
            <span class="model-status-text">CNN Classifier — Loaded</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="model-status-card">
            <span class="model-dot-inactive"></span>
            <span class="model-status-text">CNN Classifier — Missing</span>
        </div>
        """, unsafe_allow_html=True)
        
    if yolo_model:
        st.sidebar.markdown("""
        <div class="model-status-card">
            <span class="model-dot-active"></span>
            <span class="model-status-text">YOLOv11 Segmenter — Loaded</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="model-status-card">
            <span class="model-dot-inactive"></span>
            <span class="model-status-text">YOLOv11 Segmenter — Missing</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ─── Sidebar: Appearance Section ───
    st.sidebar.markdown('<div class="sidebar-section-label">🎨 Appearance</div>', unsafe_allow_html=True)
    
    # Theme configuration
    theme_choice = st.sidebar.selectbox("Select Theme Style:", [
        "Midnight Obsidian (Dark)",
        "Clinical Light Mode",
        "Cyberpunk Amber (Tactical)",
        "Emerald Laboratory (Medical)"
    ], index=1)
    
    # ─── Sidebar: Deployment Section ───
    st.sidebar.markdown('<div class="sidebar-section-label">🚀 Deployment</div>', unsafe_allow_html=True)
    st.sidebar.link_button("🌐 Deploy to Streamlit Cloud", "https://share.streamlit.io/deploy")
    st.sidebar.link_button("🤗 Deploy to Hugging Face Spaces", "https://huggingface.co/new-space")
    
    # Inject Conditional CSS overrides based on selected theme
    if theme_choice == "Clinical Light Mode":
        st.markdown("""
        <style>
            .stApp {
                background: radial-gradient(circle at 50% 0%, #f1f3f5 0%, #e9ecef 100%) !important;
                color: #212529 !important;
            }
            h1, h2, h3, h4, h5, h6, p, li, span, strong, div,
            [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
            .stMarkdown, .stMarkdown p {
                color: #212529 !important;
            }
            .app-header {
                background: linear-gradient(135deg, rgba(222, 226, 230, 0.9) 0%, rgba(233, 236, 239, 0.9) 100%) !important;
                border: 1px solid rgba(0, 0, 0, 0.08) !important;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06) !important;
            }
            .app-header::before {
                background: linear-gradient(to bottom, #1d3557, #457b9d, #1d3557) !important;
            }
            .app-title {
                background: linear-gradient(135deg, #1d3557 0%, #457b9d 50%, #1d3557 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }
            .app-subtitle, .app-subtitle::after { color: #495057 !important; }
            .status-badge { background: rgba(40, 167, 69, 0.1) !important; color: #28a745 !important; border-color: rgba(40, 167, 69, 0.2) !important; }
            .status-dot { background: #28a745 !important; box-shadow: 0 0 8px rgba(40, 167, 69, 0.5) !important; }
            .version-badge { background: rgba(29, 53, 87, 0.1) !important; color: #1d3557 !important; border-color: rgba(29, 53, 87, 0.2) !important; }
            .thesis-badge { background: rgba(69, 123, 157, 0.1) !important; color: #457b9d !important; border-color: rgba(69, 123, 157, 0.2) !important; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%) !important;
                border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
            }
            [data-testid="stSidebar"] * { color: #212529 !important; }
            .sidebar-section-label { color: #495057 !important; border-top-color: rgba(0,0,0,0.08) !important; }
            .model-status-card { background: rgba(0, 0, 0, 0.03) !important; border-color: rgba(0,0,0,0.06) !important; }
            .model-status-text { color: #212529 !important; }
            .landmark-card {
                background: rgba(255, 255, 255, 0.9) !important;
                border: 1px solid rgba(0, 0, 0, 0.06) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
            }
            .landmark-card:hover {
                background: #ffffff !important;
                border-color: rgba(0, 0, 0, 0.12) !important;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
            }
            .bone-badge {
                background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%) !important;
                color: #ffffff !important;
                box-shadow: 0 4px 15px rgba(29, 53, 87, 0.2) !important;
            }
            .classifier-confidence-badge {
                background: rgba(29, 53, 87, 0.1) !important;
                color: #1d3557 !important;
                border-color: rgba(29, 53, 87, 0.3) !important;
            }
            .metric-box {
                background: rgba(255, 255, 255, 0.7) !important;
                border: 1px solid rgba(0, 0, 0, 0.06) !important;
            }
            .metric-box:hover { background: rgba(255, 255, 255, 0.9) !important; }
            .metric-val { color: #1d3557 !important; }
            .metric-lbl { color: #495057 !important; }
            .diag-gauge { background: rgba(0, 0, 0, 0.03) !important; border-color: rgba(0,0,0,0.06) !important; }
            .diag-gauge-label { color: #1d3557 !important; }
            .diag-gauge-value { color: #457b9d !important; }
            .diag-gauge-track { background: rgba(0,0,0,0.06) !important; }
            .diag-gauge-fill { background: linear-gradient(90deg, #1d3557 0%, #457b9d 100%) !important; box-shadow: 0 0 8px rgba(29, 53, 87, 0.3) !important; }
            .landing-hero { background: rgba(255, 255, 255, 0.6) !important; border-color: rgba(0,0,0,0.06) !important; }
            .landing-hero::before { background: radial-gradient(ellipse at 50% 0%, rgba(29, 53, 87, 0.05) 0%, transparent 70%) !important; }
            .landing-hero-title { background: linear-gradient(135deg, #212529 0%, #495057 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
            .landing-hero-subtitle { color: #495057 !important; }
            .landing-upload-cta { background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%) !important; box-shadow: 0 8px 25px rgba(29, 53, 87, 0.2) !important; }
            .feature-card { background: rgba(255, 255, 255, 0.5) !important; border-color: rgba(0,0,0,0.06) !important; }
            .feature-card:hover { background: rgba(255, 255, 255, 0.8) !important; border-color: rgba(29, 53, 87, 0.2) !important; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08) !important; }
            .feature-title { color: #212529 !important; }
            .feature-desc { color: #495057 !important; }
            .bone-type-badge { background: rgba(0, 0, 0, 0.04) !important; color: #495057 !important; border-color: rgba(0,0,0,0.08) !important; }
            .bone-type-badge:hover { background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%) !important; color: white !important; border-color: transparent !important; }
            .image-frame { background: rgba(255, 255, 255, 0.5) !important; border-color: rgba(0,0,0,0.06) !important; box-shadow: 0 8px 25px rgba(0,0,0,0.06) !important; }
            .image-frame::before { border-color: #457b9d !important; }
            .image-frame::after { border-color: #457b9d !important; }
            .upload-zone-placeholder { border-color: rgba(29, 53, 87, 0.2) !important; background: radial-gradient(ellipse at 50% 50%, rgba(29, 53, 87, 0.03), transparent 70%) !important; }
            .upload-zone-text { color: #495057 !important; }
            .diagnostics-console { background: linear-gradient(135deg, #212529 0%, #343a40 100%) !important; border-color: rgba(0,0,0,0.1) !important; }
            [data-baseweb="tab-list"] { background: rgba(0, 0, 0, 0.03) !important; border: 1px solid rgba(0, 0, 0, 0.05) !important; }
            [data-baseweb="tab"] { color: #495057 !important; }
            [data-baseweb="tab"]:hover { background: rgba(0, 0, 0, 0.04) !important; color: #212529 !important; }
            [aria-selected="true"] {
                background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%) !important;
                color: #ffffff !important;
                box-shadow: 0 4px 12px rgba(29, 53, 87, 0.2) !important;
            }
            .sidebar-footer { border-top-color: rgba(0,0,0,0.08) !important; }
            .sidebar-footer-text { color: #495057 !important; }
            .rejection-panel-error { background: rgba(239, 68, 68, 0.06) !important; border-color: rgba(239, 68, 68, 0.15) !important; }
            .rejection-panel-warning { background: rgba(245, 158, 11, 0.06) !important; border-color: rgba(245, 158, 11, 0.15) !important; }
        </style>
        """, unsafe_allow_html=True)
    elif theme_choice == "Cyberpunk Amber (Tactical)":
        st.markdown("""
        <style>
            .stApp {
                background: radial-gradient(ellipse at 20% 0%, #1c1508 0%, #0d0a03 40%, #080602 100%) !important;
                color: #ffeedd !important;
            }
            h1, h2, h3, h4, h5, h6 { color: #ffb86c !important; }
            .app-header {
                background: linear-gradient(135deg, rgba(30, 20, 10, 0.5) 0%, rgba(8, 6, 2, 0.5) 100%) !important;
                border: 1px solid rgba(255, 184, 108, 0.15) !important;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
            }
            .app-header::before { background: linear-gradient(to bottom, #ffb86c, #ff5555, #ffb86c) !important; }
            .app-title {
                background: linear-gradient(135deg, #ffb86c 0%, #ff5555 50%, #ffb86c 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }
            .app-subtitle { color: #ffb86c !important; }
            .status-badge { background: rgba(255, 184, 108, 0.1) !important; color: #ffb86c !important; border-color: rgba(255, 184, 108, 0.2) !important; }
            .status-dot { background: #ffb86c !important; box-shadow: 0 0 8px rgba(255, 184, 108, 0.5) !important; }
            .version-badge { background: rgba(255, 85, 85, 0.1) !important; color: #ff5555 !important; border-color: rgba(255, 85, 85, 0.2) !important; }
            .thesis-badge { background: rgba(255, 184, 108, 0.1) !important; color: #ffb86c !important; border-color: rgba(255, 184, 108, 0.2) !important; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0d0a05 0%, #080602 100%) !important;
                border-right: 1px solid rgba(255, 184, 108, 0.15) !important;
            }
            .sidebar-section-label { color: #ffb86c !important; border-top-color: rgba(255, 184, 108, 0.1) !important; }
            .landmark-card {
                background: rgba(255, 184, 108, 0.03) !important;
                border: 1px solid rgba(255, 184, 108, 0.1) !important;
            }
            .landmark-card:hover {
                background: rgba(255, 184, 108, 0.06) !important;
                border-color: rgba(255, 184, 108, 0.25) !important;
            }
            .bone-badge {
                background: linear-gradient(135deg, #ffb86c 0%, #ff5555 100%) !important;
                color: #ffffff !important;
                box-shadow: 0 4px 15px rgba(255, 184, 108, 0.25) !important;
            }
            .classifier-confidence-badge { background: rgba(255, 184, 108, 0.1) !important; color: #ffb86c !important; border-color: rgba(255, 184, 108, 0.3) !important; }
            .metric-box { background: rgba(255, 184, 108, 0.02) !important; border: 1px solid rgba(255, 184, 108, 0.08) !important; }
            .metric-val { color: #ffb86c !important; }
            .metric-lbl { color: #ffeedd !important; }
            .diag-gauge { background: rgba(255, 184, 108, 0.03) !important; border-color: rgba(255, 184, 108, 0.1) !important; }
            .diag-gauge-label { color: #ffb86c !important; }
            .diag-gauge-value { color: #ff5555 !important; }
            .diag-gauge-fill { background: linear-gradient(90deg, #ffb86c 0%, #ff5555 100%) !important; box-shadow: 0 0 12px rgba(255, 184, 108, 0.5) !important; }
            .landing-hero { background: rgba(255, 184, 108, 0.03) !important; border-color: rgba(255, 184, 108, 0.1) !important; }
            .landing-upload-cta { background: linear-gradient(135deg, #ffb86c 0%, #ff5555 100%) !important; }
            .feature-card { background: rgba(255, 184, 108, 0.02) !important; border-color: rgba(255, 184, 108, 0.08) !important; }
            .feature-card:hover { border-color: rgba(255, 184, 108, 0.25) !important; }
            .bone-type-badge { border-color: rgba(255, 184, 108, 0.12) !important; color: #ffeedd !important; }
            .bone-type-badge:hover { background: linear-gradient(135deg, #ffb86c 0%, #ff5555 100%) !important; }
            .image-frame { border-color: rgba(255, 184, 108, 0.1) !important; }
            .image-frame::before, .image-frame::after { border-color: #ffb86c !important; }
            [data-baseweb="tab-list"] { background: rgba(255, 184, 108, 0.02) !important; border: 1px solid rgba(255, 184, 108, 0.08) !important; }
            [data-baseweb="tab"] { color: #ffeedd !important; }
            [data-baseweb="tab"]:hover { background: rgba(255, 184, 108, 0.04) !important; color: #ffb86c !important; }
            [aria-selected="true"] {
                background: linear-gradient(135deg, #ffb86c 0%, #ff5555 100%) !important;
                color: #ffffff !important;
                box-shadow: 0 4px 12px rgba(255, 184, 108, 0.25) !important;
            }
        </style>
        """, unsafe_allow_html=True)
    elif theme_choice == "Emerald Laboratory (Medical)":
        st.markdown("""
        <style>
            .stApp {
                background: radial-gradient(ellipse at 20% 0%, #0d1e16 0%, #071210 40%, #040b08 100%) !important;
                color: #e2f9ea !important;
            }
            h1, h2, h3, h4, h5, h6 { color: #50fa7b !important; }
            .app-header {
                background: linear-gradient(135deg, rgba(16, 40, 30, 0.5) 0%, rgba(4, 11, 8, 0.5) 100%) !important;
                border: 1px solid rgba(80, 255, 130, 0.15) !important;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
            }
            .app-header::before { background: linear-gradient(to bottom, #50fa7b, #8be9fd, #50fa7b) !important; }
            .app-title {
                background: linear-gradient(135deg, #50fa7b 0%, #8be9fd 50%, #50fa7b 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }
            .app-subtitle { color: #50fa7b !important; }
            .status-badge { background: rgba(80, 250, 123, 0.1) !important; color: #50fa7b !important; border-color: rgba(80, 250, 123, 0.2) !important; }
            .status-dot { background: #50fa7b !important; }
            .version-badge { background: rgba(139, 233, 253, 0.1) !important; color: #8be9fd !important; border-color: rgba(139, 233, 253, 0.2) !important; }
            .thesis-badge { background: rgba(80, 250, 123, 0.1) !important; color: #50fa7b !important; border-color: rgba(80, 250, 123, 0.2) !important; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #060e0a 0%, #040b08 100%) !important;
                border-right: 1px solid rgba(80, 255, 130, 0.15) !important;
            }
            .sidebar-section-label { color: #50fa7b !important; border-top-color: rgba(80, 250, 123, 0.1) !important; }
            .landmark-card {
                background: rgba(80, 255, 130, 0.03) !important;
                border: 1px solid rgba(80, 255, 130, 0.1) !important;
            }
            .landmark-card:hover {
                background: rgba(80, 255, 130, 0.06) !important;
                border-color: rgba(80, 255, 130, 0.25) !important;
            }
            .bone-badge {
                background: linear-gradient(135deg, #50fa7b 0%, #8be9fd 100%) !important;
                color: #080b10 !important;
                box-shadow: 0 4px 15px rgba(80, 255, 130, 0.25) !important;
            }
            .classifier-confidence-badge { background: rgba(80, 255, 130, 0.1) !important; color: #50fa7b !important; border-color: rgba(80, 255, 130, 0.3) !important; }
            .metric-box { background: rgba(80, 255, 130, 0.02) !important; border: 1px solid rgba(80, 255, 130, 0.08) !important; }
            .metric-val { color: #50fa7b !important; }
            .metric-lbl { color: #e2f9ea !important; }
            .diag-gauge { background: rgba(80, 250, 123, 0.03) !important; border-color: rgba(80, 250, 123, 0.1) !important; }
            .diag-gauge-label { color: #50fa7b !important; }
            .diag-gauge-value { color: #8be9fd !important; }
            .diag-gauge-fill { background: linear-gradient(90deg, #50fa7b 0%, #8be9fd 100%) !important; box-shadow: 0 0 12px rgba(80, 250, 123, 0.5) !important; }
            .landing-hero { background: rgba(80, 250, 123, 0.03) !important; border-color: rgba(80, 250, 123, 0.1) !important; }
            .landing-upload-cta { background: linear-gradient(135deg, #50fa7b 0%, #8be9fd 100%) !important; color: #080b10 !important; }
            .feature-card { background: rgba(80, 250, 123, 0.02) !important; border-color: rgba(80, 250, 123, 0.08) !important; }
            .feature-card:hover { border-color: rgba(80, 250, 123, 0.25) !important; }
            .bone-type-badge { border-color: rgba(80, 250, 123, 0.12) !important; color: #e2f9ea !important; }
            .bone-type-badge:hover { background: linear-gradient(135deg, #50fa7b 0%, #8be9fd 100%) !important; color: #080b10 !important; }
            .image-frame { border-color: rgba(80, 250, 123, 0.1) !important; }
            .image-frame::before, .image-frame::after { border-color: #50fa7b !important; }
            [data-baseweb="tab-list"] { background: rgba(80, 255, 130, 0.02) !important; border: 1px solid rgba(80, 255, 130, 0.08) !important; }
            [data-baseweb="tab"] { color: #e2f9ea !important; }
            [data-baseweb="tab"]:hover { background: rgba(80, 255, 130, 0.04) !important; color: #50fa7b !important; }
            [aria-selected="true"] {
                background: linear-gradient(135deg, #50fa7b 0%, #8be9fd 100%) !important;
                color: #080b10 !important;
                box-shadow: 0 4px 12px rgba(80, 250, 123, 0.25) !important;
            }
        </style>
        """, unsafe_allow_html=True)

    # ─── Sidebar: Visual Effects Section ───
    st.sidebar.markdown('<div class="sidebar-section-label">🎬 Visual Effects</div>', unsafe_allow_html=True)
    enable_bg_video = st.sidebar.toggle("Enable Animated BG", value=False)
    video_bg_url = st.sidebar.text_input("Background Video URL:", "https://assets.mixkit.co/videos/preview/mixkit-futuristic-digital-background-loop-42207-large.mp4")
    
    if enable_bg_video:
        st.markdown(f"""
        <style>
            .stApp {{
                background: transparent !important;
            }}
            #bg-video-container {{
                position: fixed;
                top: 0; left: 0; width: 100%; height: 100%;
                z-index: -1;
                overflow: hidden;
                pointer-events: none;
            }}
            #bg-video {{
                width: 100%; height: 100%;
                object-fit: cover;
                opacity: 0.10;
                filter: grayscale(80%) contrast(110%);
            }}
        </style>
        <div id="bg-video-container">
            <video autoplay loop muted playsinline id="bg-video">
                <source src="{video_bg_url}" type="video/mp4">
            </video>
        </div>
        """, unsafe_allow_html=True)
    
    # ─── Sidebar: AI Settings Section ───
    st.sidebar.markdown('<div class="sidebar-section-label">⚙️ AI Settings</div>', unsafe_allow_html=True)
    
    conf_threshold = st.sidebar.slider("Landmark Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    
    # Bone Selection Mode (Auto CNN or Manual Override)
    bone_selection_mode = st.sidebar.selectbox("Bone Selection Mode:", [
        "Auto-detect (CNN Model)",
        "Override: Femur",
        "Override: Fibula",
        "Override: Humerus",
        "Override: Radius-Ulna",
        "Override: Scapula",
        "Override: Tibia",
        "Override: Os-Coxae",
        "Override: Skull"
    ], index=0)
    
    # ─── Sidebar: Image Input Section ───
    st.sidebar.markdown('<div class="sidebar-section-label">📷 Image Input</div>', unsafe_allow_html=True)
    
    input_method = st.sidebar.radio("Select Image Source:", ["Upload Your Own Image", "Sample Images (Testing)"])
    
    # ─── Sidebar: Model Metrics Section ───
    st.sidebar.markdown('<div class="sidebar-section-label">📊 Model Metrics</div>', unsafe_allow_html=True)
    
    with st.sidebar.expander("📊 MobileNetV2 CNN Classifier"):
        st.markdown("""
        * **Framework:** Keras 3 / TensorFlow 2.20
        * **Input Resolution:** `224 × 224 × 3`
        * **Accuracy (Validation):** **`94.2%`**
        * **Loss:** `0.18`
        * **Epochs Trained:** `60`
        """)
        
    with st.sidebar.expander("📊 YOLOv11 Segmenter (Landmarks)"):
        st.markdown("""
        * **Precision (mAP@50):** **`70.7%`**
        * **Landmark Coverage:** 91 classes
        * **Active Filter:** Prevents cross-bone landmark leakage using CNN classification outputs.
        """)

    # ─── Sidebar: Footer ───
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-text">
            🐕 Canine Osteology AI System<br>
            Dept. of Veterinary Anatomy, IVRI Bareilly<br>
            © 2026 — MVSc Thesis Project
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Setup Image paths
    selected_img_path = None
    uploaded_file = None
    
    if input_method == "Sample Images (Testing)":
        # Find test sample images from folder splits
        _base_samples = os.path.dirname(os.path.abspath(__file__))
        sample_dirs = {
            "Sample Scapula": os.path.join(_base_samples, "sample_images", "scapula"),
            "Sample Humerus": os.path.join(_base_samples, "sample_images", "humerus"),
            "Sample Os Coxae": os.path.join(_base_samples, "sample_images", "os_coxae"),
            "Sample Radius-Ulna": os.path.join(_base_samples, "sample_images", "radius_ulna")
        }
        
        sample_images = {}
        for name, folder in sample_dirs.items():
            if os.path.exists(folder):
                jpgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))]
                if jpgs:
                    sample_images[name] = os.path.join(folder, jpgs[0])
                    
        if not sample_images:
            st.sidebar.info("No test samples found on disk. Please upload your own image.")
        else:
            chosen_sample = st.sidebar.selectbox("Choose a test bone:", list(sample_images.keys()))
            selected_img_path = sample_images[chosen_sample]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a Canine Bone Image (.jpg, .png, .webp)", type=["jpg", "jpeg", "png", "webp"], key="sidebar_uploader")

    # Load Image first to decide layout
    img = None
    img_name = ""
    
    # Check if either the sidebar uploader or main uploader has a file
    actual_upload = uploaded_file
    if actual_upload is None:
        actual_upload = st.session_state.get("main_uploader")
        
    if actual_upload is not None:
        img = Image.open(actual_upload).convert("RGB")
        img_name = actual_upload.name
    elif selected_img_path and os.path.exists(selected_img_path):
        img = Image.open(selected_img_path).convert("RGB")
        img_name = os.path.basename(selected_img_path)
    
    if img is None:
        # ─── Full-Width Premium Landing State ───
        # Hero top section
        st.markdown('<div class="landing-hero">', unsafe_allow_html=True)
        st.markdown('<span class="upload-zone-icon">🦴</span>', unsafe_allow_html=True)
        st.markdown('<div class="landing-hero-title">Upload a Bone to Begin Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="landing-hero-subtitle">Select a sample image or upload your own canine bone photograph to activate AI-powered anatomical identification.</div>', unsafe_allow_html=True)
        
        # Real upload button that functions immediately
        st.file_uploader("UPLOAD an image", type=["jpg", "jpeg", "png", "webp"], key="main_uploader")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature cards — each in its own st.markdown to avoid Streamlit truncation
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown('<div class="feature-card"><span class="feature-icon">🧠</span><div class="feature-title">Bone Identification</div><div class="feature-desc">CNN-powered classification of 7 canine bone types with 94.2% accuracy</div></div>', unsafe_allow_html=True)
        with fc2:
            st.markdown('<div class="feature-card"><span class="feature-icon">📍</span><div class="feature-title">Landmark Detection</div><div class="feature-desc">YOLOv11 segmentation of 91 anatomical landmark classes</div></div>', unsafe_allow_html=True)
        with fc3:
            st.markdown('<div class="feature-card"><span class="feature-icon">📖</span><div class="feature-title">Study Notes</div><div class="feature-desc">Complete textbook descriptions, clinical relevance & species comparisons</div></div>', unsafe_allow_html=True)
        
        # Bone showcase badges
        st.markdown('<div class="bone-showcase"><span class="bone-type-badge">🦴 Femur</span><span class="bone-type-badge">🦴 Tibia</span><span class="bone-type-badge">🦴 Humerus</span><span class="bone-type-badge">🦴 Scapula</span><span class="bone-type-badge">🦴 Radius-Ulna</span><span class="bone-type-badge">🦴 Os-Coxae</span><span class="bone-type-badge">🦴 Fibula</span></div>', unsafe_allow_html=True)
        
        # How it works section
        st.markdown("---")
        hw1, hw2, hw3 = st.columns(3)
        with hw1:
            st.markdown('<div style="text-align:center; padding:20px 10px;"><div style="font-size:1.8rem; margin-bottom:8px;">1️⃣</div><div style="font-family:Outfit,sans-serif; font-weight:700; font-size:0.95rem; margin-bottom:6px;">Upload Photo</div><div style="font-size:0.82rem; color:#6b7280;">Take or select a clear photograph of a canine bone specimen</div></div>', unsafe_allow_html=True)
        with hw2:
            st.markdown('<div style="text-align:center; padding:20px 10px;"><div style="font-size:1.8rem; margin-bottom:8px;">2️⃣</div><div style="font-family:Outfit,sans-serif; font-weight:700; font-size:0.95rem; margin-bottom:6px;">AI Analysis</div><div style="font-size:0.82rem; color:#6b7280;">CNN identifies the bone type, YOLO detects anatomical landmarks</div></div>', unsafe_allow_html=True)
        with hw3:
            st.markdown('<div style="text-align:center; padding:20px 10px;"><div style="font-size:1.8rem; margin-bottom:8px;">3️⃣</div><div style="font-family:Outfit,sans-serif; font-weight:700; font-size:0.95rem; margin-bottom:6px;">Study & Learn</div><div style="font-size:0.82rem; color:#6b7280;">View annotated diagrams, textbook descriptions, and clinical notes</div></div>', unsafe_allow_html=True)
    else:
        # 3. Layout: Left for Image, Right for dynamic tabs (only when image is loaded)
        col_img, col_tabs = st.columns([7, 5])
        
        with col_img:
            st.write("### Bone Landmark Annotation Canvas")
            # 4. Perform CNN Bone Classification & YOLO Detection (Hybrid Self-Correction)
            predicted_bone = "Unknown"
            classifier_confidence = 0.0
            
            # Run YOLO Landmark detector for ALL landmarks first
            all_anns = []
            if yolo_model:
                with st.spinner("Analyzing anatomical features via YOLO segmenter..."):
                    all_anns = detect_all_landmarks(
                        yolo_model=yolo_model,
                        image_pil=img,
                        conf_threshold=0.15 # lower threshold for validation/rejection
                    )
            
            # Group YOLO detections by bone prefix to find dominant prefix
            from collections import Counter
            prefix_counts = Counter()
            prefix_conf_sum = {}
            for ann in all_anns:
                prefix = ann["bone_prefix"]
                prefix_counts[prefix] += 1
                prefix_conf_sum[prefix] = prefix_conf_sum.get(prefix, 0.0) + ann["confidence"]
                
            yolo_predicted_bone = None
            if prefix_counts:
                yolo_predicted_bone = max(prefix_conf_sum, key=prefix_conf_sum.get)
            
            # Determine CNN Bone prediction
            cnn_predicted_bone = "Unknown"
            if bone_selection_mode.startswith("Override: "):
                predicted_bone = bone_selection_mode.replace("Override: ", "").strip()
                classifier_confidence = 1.0
                st.info(f"🔧 Bone type manually set to: **{predicted_bone}** (CNN bypassed)")
            else:
                # Run Keras CNN model prediction
                if cnn_model:
                    with st.spinner("Analyzing image shape via CNN classifier..."):
                        cnn_predicted_bone, classifier_confidence = classify_bone_image(cnn_model, img)
                        predicted_bone = cnn_predicted_bone
                else:
                    # Fallback to name hint or default Scapula if classifier model not found
                    st.warning("⚠️ CNN Classifier not loaded. Defaulting detection to Scapula.")
                    cnn_predicted_bone = "Scapula"
                    predicted_bone = "Scapula"
                    classifier_confidence = 0.0
                
                # Hybrid Self-Correction: if YOLO found landmarks for a different bone with high confidence, trust YOLO!
                if yolo_predicted_bone and yolo_predicted_bone != predicted_bone:
                    norm_yolo = yolo_predicted_bone.lower().replace("-", "").replace(" ", "")
                    norm_cnn = predicted_bone.lower().replace("-", "").replace(" ", "")
                    if norm_yolo != norm_cnn:
                        yolo_conf_avg = prefix_conf_sum[yolo_predicted_bone] / prefix_counts[yolo_predicted_bone]
                        if yolo_conf_avg > 0.25:
                            st.info(f"🔄 **AI Self-Correction:** CNN classified as *{predicted_bone}*, but YOLO validated *{yolo_predicted_bone}* landmarks. Overriding prediction.")
                            predicted_bone = yolo_predicted_bone

            # Filter target annotations for the final predicted bone type
            target_anns = []
            norm_target = predicted_bone.lower().replace("-", "").replace(" ", "")
            for ann in all_anns:
                norm_prefix = ann["bone_prefix"].lower().replace("-", "").replace(" ", "")
                if norm_prefix == norm_target and ann["confidence"] >= conf_threshold:
                    target_anns.append(ann)
            
            # 5. Apply Rejection Heuristics (Bone vs. Non-Bone & Untrained Bone warnings)
            is_non_bone = len(all_anns) == 0
            is_untrained_or_mismatch = (len(target_anns) == 0 and len(all_anns) > 0)
            
            # Diagnostic Metrics calculations
            avg_conf = np.mean([ann["confidence"] for ann in target_anns]) if target_anns else 0.0
            detected_labels = list(set([ann["label"] for ann in target_anns]))
            
            if is_non_bone:
                # Case A: Rejected as Non-Bone
                st.markdown(f"""
                <div class="rejection-panel rejection-panel-error">
                    <h3 style="color: #ef4444; margin-top: 0; font-family: 'Outfit', sans-serif; font-weight: 700;">❌ Non-Bone or Unrecognized Object Detected</h3>
                    <p style="color: #fca5a5; font-size: 0.92rem; margin-bottom: 0; line-height: 1.6;">
                        The uploaded image does not contain recognizable canine skeletal structures. 
                        Please ensure you upload a clear photo of one of the 7 canine bones (Scapula, Humerus, Pelvis, Femur, Tibia, etc.).
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='image-frame' style='border-color: rgba(239, 68, 68, 0.3);'>", unsafe_allow_html=True)
                st.image(img, caption="Original Photo (Unrecognized Structure)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                with col_tabs:
                    st.markdown("### 📖 Anatomical Diagnostics Panel")
                    st.info("💡 Awaiting valid canine bone image input to display diagnostics.")
                    
            elif is_untrained_or_mismatch:
                # Case B: Rejected as Untrained Bone (e.g. Ribs) or Mismatch
                st.markdown(f"""
                <div class="rejection-panel rejection-panel-warning">
                    <h3 style="color: #f59e0b; margin-top: 0; font-family: 'Outfit', sans-serif; font-weight: 700;">⚠️ Unrecognized Skeletal Element</h3>
                    <p style="color: #fde047; font-size: 0.92rem; margin-bottom: 0; line-height: 1.6;">
                        The structure detected does not match the anatomical features of a canine <strong>{predicted_bone}</strong>. 
                        This may be an untrained bone (such as a rib or vertebra) or a non-skeletal element.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='image-frame' style='border-color: rgba(245, 158, 11, 0.3);'>", unsafe_allow_html=True)
                st.image(img, caption="Original Photo (Unrecognized Skeletal Element)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                with col_tabs:
                    st.markdown("### 📖 Anatomical Diagnostics Panel")
                    st.warning(f"Classification returned '{predicted_bone}', but no matching anatomical landmarks were validated. Please verify the bone type.")
                    
            else:
                # Case C: Valid Bone Detections
                # Render publication-quality overlay image card
                card_img = make_publication_card(
                    image_rgb=np.array(img),
                    annotations=target_anns,
                    file_name=img_name,
                    img_w=img.width,
                    img_h=img.height,
                    card_width=1100,
                    top_height=740,
                    desc_height=340,
                    isolate=False,
                    bone_type=predicted_bone
                )
                
                st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
                st.image(card_img, caption="AI Annotated Anatomical Map", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Dynamic Download button
                import io
                buf = io.BytesIO()
                Image.fromarray(card_img).save(buf, format="JPEG", quality=95)
                st.download_button(
                    "⬇️ Download Publication-Quality Diagram",
                    data=buf.getvalue(),
                    file_name=f"{predicted_bone}_annotated_card.jpg",
                    mime="image/jpeg"
                )
                
                # 6. Tab layout in Right Column
                with col_tabs:
                    st.markdown("### 📖 Anatomical Diagnostics Panel")
                    
                    # Modern Confidence Gauge — explicitly labeled as CNN Classifier confidence
                    st.markdown(f"""
                    <div class="diag-gauge">
                        <div class="diag-gauge-header">
                            <span class="diag-gauge-label">🐕 Canine {predicted_bone}</span>
                            <span class="diag-gauge-value">CNN: {classifier_confidence*100:.1f}%</span>
                        </div>
                        <div class="diag-gauge-track">
                            <div class="diag-gauge-fill" style="width: {classifier_confidence*100}%;"></div>
                        </div>
                        <div style="font-size:0.72rem; color:#6b7280; margin-top:6px; font-family:'Fira Code',monospace;">Bone Classification Confidence (MobileNetV2 CNN)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    tab_labels, tab_desc = st.tabs(["📍 Labeled Bone Parts", "📖 Bone Description"])
                    
                    # TAB 1: LABELED BONE PARTS LISTING
                    with tab_labels:
                        st.write("#### 📍 Detected Landmarks & Coordinates")
                        
                        st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-box'>
                                <div class='metric-val'>{len(target_anns)}</div>
                                <div class='metric-lbl'>Total Marks</div>
                            </div>
                            <div class='metric-box'>
                                <div class='metric-val'>{avg_conf*100:.0f}%</div>
                                <div class='metric-lbl'>YOLO Avg Conf</div>
                            </div>
                            <div class='metric-box'>
                                <div class='metric-val'>{len(detected_labels)}</div>
                                <div class='metric-lbl'>Unique Regions</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Find the highest confidence instance for each label
                        best_conf = {}
                        for ann in target_anns:
                            lbl = ann["label"]
                            conf = ann["confidence"]
                            if lbl not in best_conf or conf > best_conf[lbl]:
                                best_conf[lbl] = conf
                        
                        # Fixed height scrollable wrapper with modern padding
                        st.markdown("<div style='max-height: 520px; overflow-y: auto; padding-right: 8px;'>", unsafe_allow_html=True)
                        
                        # List each landmark with details
                        for label in sorted(detected_labels):
                            desc = REGION_DESC.get(label, "Anatomical description not available for this landmark.")
                            conf = best_conf.get(label, 1.0)
                            conf_pct = f"{conf*100:.0f}%"
                            
                            # Assign confidence classes
                            if conf >= 0.70:
                                badge_html = f"<span class='conf-badge'>High Conf: {conf_pct}</span>"
                            elif conf >= 0.40:
                                badge_html = f"<span class='conf-badge-medium'>Med Conf: {conf_pct}</span>"
                            else:
                                badge_html = f"<span class='conf-badge-low'>Low Conf: {conf_pct}</span>"
                            
                            c_rgb = color_for(label)
                            c_hex = "#%02x%02x%02x" % c_rgb
                            
                            # Extract spatial bounding box coordinates
                            ann_item = next((a for a in target_anns if a["label"] == label), None)
                            coord_badge = ""
                            if ann_item and "bbox" in ann_item:
                                x, y, w, h = ann_item["bbox"]
                                coord_badge = f"<span style='background: rgba(139, 233, 253, 0.08); border: 1px solid rgba(139, 233, 253, 0.15); padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-family: \"Fira Code\", monospace; color: #8be9fd;'>X:{x:.0f} Y:{y:.0f} W:{w:.0f} H:{h:.0f}</span>"
                            
                            st.markdown(f"""
                            <div class='landmark-card' style='border-left-color: {c_hex}; padding: 12px 16px !important; margin-bottom: 10px !important;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; gap: 10px;'>
                                    <strong style='color:{c_hex}; font-size:1rem;'>📍 {label}</strong>
                                    {badge_html}
                                </div>
                                <div style='color:#9ca3af; font-size:0.86rem; margin-top: 6px; line-height: 1.4;'>{desc}</div>
                                {f"<div style='margin-top: 8px;'>{coord_badge}</div>" if coord_badge else ""}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Close scrollable wrapper
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # TAB 2: TEXTBOOK BONE DESCRIPTION
                    with tab_desc:
                        desc_obj = BONE_TEXTBOOK_DESCRIPTIONS.get(predicted_bone)
                        
                        if not desc_obj:
                            st.info("No textbook description is available for this bone class yet.")
                        else:
                            st.markdown(f"### 📖 Textbook Profile: Canine {predicted_bone}")
                            st.write(desc_obj["description"])
                            
                            st.markdown("#### 📍 Key Anatomical Landmarks")
                            for f in desc_obj["features"]:
                                st.markdown(f"- {f}")
                                
                            st.markdown("#### 🩺 Clinical & Surgical Relevance")
                            st.write(desc_obj["clinical"])
                            
                            st.markdown("#### 🐕 Species-Specific Notes (Dog vs. Cat)")
                            st.write(desc_obj["species"])

            # 7. System Diagnostics Console Panel
            st.write("---")
            st.markdown("### 💻 System Diagnostics Log")
            st.markdown(f"""
            <div class="diagnostics-console">
                <div class="log-line"><span style="color: #50fa7b;">[SYS_INFO]</span> Initializing canine osteology diagnostics session...</div>
                <div class="log-line"><span style="color: #50fa7b;">[SYS_INFO]</span> Active models: MobileNetV2 (Classifier), YOLOv11 (Segmenter)</div>
                <div class="log-line"><span style="color: #ff79c6;">[ST_PARAMS]</span> Confidence Threshold: {conf_threshold} | Override Mode: {bone_selection_mode}</div>
                <div class="log-line"><span style="color: #ffb86c;">[INPUT_FILE]</span> Loaded image: {img_name} ({img.width}x{img.height} px)</div>
                <div class="log-line"><span style="color: #bd93f9;">[CNN_PRED]</span> Classification output: {predicted_bone} (confidence: {classifier_confidence:.4f})</div>
                <div class="log-line"><span style="color: #f1fa8c;">[YOLO_VERIFY]</span> Total detected landmarks: {len(all_anns)} | Matching target landmarks: {len(target_anns)}</div>
                {"<div class='log-line'><span style='color: #ff5555;'>[ALERT]</span> Bone vs. Non-Bone verification failed: Rejected as Non-Bone.</div>" if is_non_bone else ""}
                {"<div class='log-line'><span style='color: #ffb86c;'>[WARNING]</span> Skeletal mismatch detected: Rejected as Unrecognized/Untrained skeletal element.</div>" if is_untrained_or_mismatch else ""}
            </div>
            """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
