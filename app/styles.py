# app/styles.py

CUSTOM_CSS = """
<style>
/* Import Handcrafted & Organic Fonts */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=Quicksand:wght@400;600&display=swap');

/* Base App Styling - Warm Parchment & Organic Texture */
.stApp {
    background-color: #f4ebd9;
    background-image: radial-gradient(#e8dac1 1px, transparent 1px);
    background-size: 20px 20px;
    font-family: 'Quicksand', sans-serif;
    color: #3e362e !important;
}

/* FORCE ALL text to be dark brown so it's readable */
.stApp p, .stApp span, .stApp label, .stApp div {
    color: #3e362e;
}

/* Headers - Handcrafted Serif */
h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #2b2520 !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}

/* 3D Neumorphic Card Style */
.3d-card {
    background: #f4ebd9;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 12px 12px 24px #d1c8b7, 
               -12px -12px 24px #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.4);
    margin-bottom: 20px;
    transition: transform 0.2s ease;
}

.3d-card:hover {
    transform: translateY(-5px);
}

/* --- SIDEBAR FIXES --- */
[data-testid="stSidebar"] {
    background-color: #ebdccb;
    box-shadow: inset -5px 0px 15px rgba(0,0,0,0.05);
}
[data-testid="stSidebar"] * {
    color: #3e362e !important; /* Force dark text in sidebar */
}

/* --- DROPDOWN (SELECTBOX) FIXES --- */
div[data-baseweb="select"] > div {
    background-color: #f4ebd9 !important;
    color: #3e362e !important;
    border: 2px solid #8b5a2b !important;
    border-radius: 8px !important;
}
/* Dropdown menu list */
div[data-baseweb="popover"] ul {
    background-color: #f4ebd9 !important;
    color: #3e362e !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: #ebdccb !important;
}

/* --- FILE UPLOADER FIXES --- */
[data-testid="stFileUploader"] {
    background: #f4ebd9;
    border-radius: 15px;
    padding: 20px;
    box-shadow: inset 5px 5px 10px #d1c8b7, inset -5px -5px 10px #ffffff;
    border: none;
}
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    border: 2px dashed #8b5a2b !important;
}
/* Make the uploaded file chip (the black box) brown */
[data-testid="stFileUploaderFile"] {
    background-color: #8b5a2b !important;
    border-radius: 8px;
}
[data-testid="stFileUploaderFile"] span {
    color: #ffffff !important; /* White text inside the brown chip */
}

/* --- PROGRESS BAR FIXES --- */
.stProgress > div > div > div > div {
    background-color: #8b5a2b !important;
}
.stProgress p {
    color: #3e362e !important; /* Dark text for percentage */
    font-weight: 600 !important;
}

/* Hide standard Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom 3D Result Box */
.result-box {
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
    box-shadow: 8px 8px 16px #d1c8b7, -8px -8px 16px #ffffff;
}

.result-real {
    background-color: #e3f0e3;
    color: #2e5c2e !important;
    border: 2px solid #4b8b4b;
}

.result-fake {
    background-color: #f0e3e3;
    color: #5c2e2e !important;
    border: 2px solid #8b4b4b;
}
/* Audio Player Override */
audio {
    width: 100%;
    border-radius: 10px;
    box-shadow: 5px 5px 10px #d1c8b7, -5px -5px 10px #ffffff;
}
</style>
"""