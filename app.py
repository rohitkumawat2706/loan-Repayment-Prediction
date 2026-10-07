import streamlit as st
import os
from src.api import ensure_api_server_running

# Configure Streamlit page layout
st.set_page_config(
    page_title="FinSight AI - Underwriting Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Start API server in background thread if not already running
api_port = ensure_api_server_running()

# Hide default Streamlit elements and overlay custom SPA full-screen
st.markdown("""
<style>
/* Hide default Streamlit elements */
header[data-testid="stHeader"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    display: none !important;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
}
div[data-testid="stVerticalBlock"] > div:has(iframe) {
    padding: 0 !important;
    margin: 0 !important;
}
iframe {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    z-index: 99999;
}
</style>
""", unsafe_allow_html=True)

# Load SPA HTML template and render
html_path = os.path.join(os.path.dirname(__file__), 'src', 'frontend', 'index.html')
try:
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Replace {{API_PORT}} placeholder with the dynamic active port
    templated_html = html_content.replace('{{API_PORT}}', str(api_port))
    
    # Render full-screen iframe component
    st.components.v1.html(templated_html, height=1200, scrolling=True)
except Exception as e:
    st.error(f"Error loading dashboard frontend assets: {str(e)}")