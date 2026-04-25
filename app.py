import os, re, io, time
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["MPLBACKEND"] = "Agg"

import streamlit as st
from PIL import Image
import numpy as np
import requests
from ultralytics import YOLO

# ── MODEL ────────────────────────────────────────────────────────
MODEL_PATH    = "best_model.pt"
GDRIVE_FILE_ID = "1FYO7H9UnLDuw5FwAqVpLSvEnPC1dTmod"

def download_model(fid, dest):
    s   = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={fid}"
    r   = s.get(url, stream=True)
    tok = next((v for k,v in r.cookies.items() if "warning" in k), None)
    if tok: r = s.get(url+f"&confirm={tok}", stream=True)
    if "text/html" in r.headers.get("Content-Type",""):
        m = re.search(r'confirm=([0-9A-Za-z_-]+)', r.text)
        if m: r = s.get(url+f"&confirm={m.group(1)}", stream=True)
    with open(dest,"wb") as f:
        for chunk in r.iter_content(65536):
            if chunk: f.write(chunk)

if not (os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000):
    st.set_page_config(page_title="WasteLens", layout="centered")
    with st.spinner("Setting up model..."):
        download_model(GDRIVE_FILE_ID, MODEL_PATH)
    st.rerun()

st.set_page_config(page_title="WasteLens", layout="wide", initial_sidebar_state="collapsed")

# ── CONSTANTS ────────────────────────────────────────────────────
REC_KW   = {"can","glass","paper","plastic","cardboard","bottle","metal"}
NON_REC  = {"foodwaste","food","organic","food_waste"}
TIPS = {
    "cans":           "Rinse and crush before placing in the recycling bin.",
    "glass":          "Remove lids. Sort by color if your facility requires it.",
    "paperwaste":     "Keep dry. Remove any staples or plastic coating.",
    "plasticbottles": "Empty and rinse. Check the resin code on the base.",
    "foodwaste":      "Compostable bag recommended. Use the organic bin.",
}
COLORS = {
    "plasticbottles":"#60a5fa",
    "glass":"#a78bfa",
    "paperwaste":"#fbbf24",
    "cans":"#34d399",
    "foodwaste":"#f87171",
}

def class_maps(mdl):
    rec, names = set(), {}
    for i, n in mdl.names.items():
        lo = n.lower()
        names[lo] = lo.replace("_"," ").replace("waste","").strip().title()
        if any(k in lo for k in REC_KW) and lo not in NON_REC:
            rec.add(lo)
    return rec, names

@st.cache_resource
def load_model(): return YOLO(MODEL_PATH)
model = load_model()

if "history" not in st.session_state: st.session_state.history = []
if "result"  not in st.session_state: st.session_state.result  = None
if "img_path" not in st.session_state: st.session_state.img_path = None

# ── STYLES ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #030303;
    color: #d4d4d4;
}
#MainMenu, footer, header, [data-testid="stSidebar"] { display: none !important; visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stImage"] img { border-radius: 12px; }

/* Upload */
[data-testid="stFileUploadDropzone"] {
    background: #0a0a0a !important;
    border: 1px dashed #1e1e1e !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] span { color: #1f1f1f !important; font-size:0.8rem !important; }
[data-testid="stFileUploadDropzone"] svg  { stroke: #1a1a1a !important; }
[data-testid="stFileUploadDropzone"]:hover { border-color: #22c55e !important; }

/* Analyse button */
.stButton > button {
    background: #22c55e !important; color: #000 !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.8rem !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; height: 3rem !important;
    width: 100% !important; box-shadow: 0 0 20px rgba(34,197,94,.25) !important;
    transition: all .15s !important;
}
.stButton > button:hover { background: #16a34a !important; box-shadow: 0 0 30px rgba(34,197,94,.4) !important; }
.stButton > button:disabled { background: #111 !important; color: #222 !important; box-shadow: none !important; }

/* Download */
[data-testid="stDownloadButton"] > button {
    background: transparent !important; color: #22c55e !important;
    border: 1px solid #14532d !important; border-radius: 10px !important;
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing:.06em !important; text-transform: uppercase !important;
    height: 2.6rem !important; width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover { background: #071a0d !important; }

/* Radio */
[data-testid="stRadio"] > label { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────
st.markdown("""
<div style="background:#080808;border-bottom:1px solid #111;padding:1.2rem 3rem;
    display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:baseline;gap:.05em;">
        <span style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;
            color:#fff;letter-spacing:-.04em;">Waste</span>
        <span style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;
            color:#22c55e;letter-spacing:-.04em;">Lens</span>
        <span style="font-size:.5rem;font-weight:700;letter-spacing:.18em;
            text-transform:uppercase;color:#22c55e;background:#071a0d;
            border:1px solid #14532d;padding:3px 9px;border-radius:100px;
            margin-left:.7rem;position:relative;top:-.25rem;">AI</span>
    </div>
    <span style="font-size:.68rem;color:#1a1a1a;letter-spacing:.12em;text-transform:uppercase;">
        Detect · Classify · Reduce Impact
    </span>
</div>
""", unsafe_allow_html=True)

# ── BODY ─────────────────────────────────────────────────────────
st.markdown('<div style="padding:2.5rem 3rem;">', unsafe_allow_html=True)

L, R = st.columns([1, 1.4], gap="large")

# LEFT
with L:
    st.markdown('<p style="font-size:.6rem;font-weight:600;letter-spacing:.22em;text-transform:uppercase;color:#1f1f1f;margin-bottom:.8rem;">Input</p>', unsafe_allow_html=True)

    mode = st.radio("m", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")

    # Tab pills
    st.markdown(f"""
    <div style="display:flex;gap:.4rem;margin-bottom:1rem;">
        {"".join(f'''<div style="flex:1;text-align:center;padding:.55rem;border-radius:9px;
            font-size:.76rem;font-weight:600;
            background:{"#071a0d" if m==mode else "#0c0c0c"};
            color:{"#22c55e" if m==mode else "#1a1a1a"};
            border:1px solid {"#14532d" if m==mode else "#111"};">{m}</div>'''
        for m in ["Upload","Camera"])}
    </div>
    """, unsafe_allow_html=True)

    src = None
    if mode == "Upload":
        up = st.file_uploader("f", type=["jpg","jpeg","png","webp","bmp"], label_visibility="collapsed")
        if up:
            p = f"/tmp/wl_{up.name}"
            with open(p,"wb") as f: f.write(up.getvalue())
            src = Image.open(p).convert("RGB")
            st.session_state.img_path = p
    else:
        cam = st.camera_input("c", label_visibility="collapsed")
        if cam:
            p = "/tmp/wl_cam.jpg"
            with open(p,"wb") as f: f.write(cam.getvalue())
            src = Image.open(p).convert("RGB")
            st.session_state.img_path = p

    if src:
        st.image(src, use_column_width=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    go = st.button("Analyze Image", disabled=not src, use_container_width=True, type="primary")

# RIGHT
with R:
    st.markdown('<p style="font-size:.6rem;font-weight:600;letter-spacing:.22em;text-transform:uppercase;color:#1f1f1f;margin-bottom:.8rem;">Live Analysis</p>', unsafe_allow_html=True)

    if not src:
        st.markdown("""
        <div style="background:#080808;border:1px solid #111;border-radius:16px;
            min-height:400px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;text-align:center;padding:3rem;">
            <div style="font-size:2.5rem;color:#111;margin-bottom:1.2rem;">⬡</div>
            <div style="font-size:.62rem;font-weight:700;letter-spacing:.22em;
                text-transform:uppercase;color:#111;margin-bottom:.5rem;">Awaiting scan</div>
            <div style="font-size:.78rem;color:#0f0f0f;line-height:1.7;max-width:220px;">
                Upload or capture an image, then press Analyze
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif go:
        with st.spinner("Scanning..."):
            t0 = time.time()
            inp = st.session_state.img_path or np.array(src)
            res = model.predict(inp, conf=0.25, iou=0.45, imgsz=640, verbose=False)
            elapsed = time.time() - t0

        boxes = res[0].boxes
        names = res[0].names
        n     = len(boxes)
        dets  = [{"cls": names[int(b.cls[0])].lower(), "conf": float(b.conf[0])} for b in boxes]

        REC, FNAME = class_maps(model)
        rec    = [d for d in dets if d["cls"] in REC]
        nonrec = [d for d in dets if d["cls"] not in REC]
        verdict = ("none" if n==0 else "recyclable" if rec and not nonrec
                   else "non-recyclable" if nonrec and not rec else "mixed")

        avg_c  = int(sum(d["conf"] for d in dets)/n*100) if n else 0
        score  = int((len(rec)/n*.6 + avg_c/100*.4)*100) if n else 0

        ann = Image.fromarray(res[0].plot())
        buf = io.BytesIO(); ann.save(buf, "PNG")

        st.session_state.result = {
            "ann": ann, "buf": buf.getvalue(), "dets": dets,
            "verdict": verdict, "elapsed": elapsed,
            "n": n, "rec": len(rec), "nonrec": len(nonrec),
            "avg_c": avg_c, "score": score,
        }
        st.session_state.history.append({"n": n, "verdict": verdict})

    r = st.session_state.result
    if r and src:
        REC, FNAME = class_maps(model)

        # Annotated image
        st.image(r["ann"], use_column_width=True)

        # Score color
        sc = "#22c55e" if r["score"]>=60 else "#f59e0b" if r["score"]>=30 else "#ef4444"

        # 4 metric cards
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin:.9rem 0 1.2rem;">
            {''.join(f"""<div style="background:#080808;border:1px solid #111;border-radius:12px;
                padding:.9rem .6rem;text-align:center;">
                <div style="font-size:.48rem;color:#141414;text-transform:uppercase;
                    letter-spacing:.18em;font-weight:700;margin-bottom:4px;">{lbl}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;
                    color:{col};line-height:1;">{val}</div>
                <div style="font-size:.55rem;color:#111;margin-top:3px;">{sub}</div>
            </div>""" for lbl,val,col,sub in [
                ("Score", r["score"], sc, "/100"),
                ("Objects", r["n"], "#fff", "detected"),
                ("Confidence", f'{r["avg_c"]}%', "#22c55e", "average"),
                ("Speed", f'{r["elapsed"]*1000:.0f}ms', "#333", "inference"),
            ])}
        </div>
        """, unsafe_allow_html=True)

        # Verdict banner
        vmap = {
            "recyclable":     ("#071a0d","#14532d","#22c55e","♻","Recyclable","Goes in the recycling bin"),
            "non-recyclable": ("#1a0505","#7f1d1d","#ef4444","✕","Non-Recyclable","Goes in the general waste bin"),
            "mixed":          ("#1a1200","#92400e","#f59e0b","⚠","Mixed Waste","Separate before disposal"),
            "none":           ("#0a0a0a","#111","#252525","?","Nothing Detected","Try a clearer image"),
        }
        vbg,vbd,vc,vi,vt,vs = vmap.get(r["verdict"], vmap["none"])
        st.markdown(f"""
        <div style="background:{vbg};border:1px solid {vbd};border-radius:12px;
            padding:1rem 1.1rem;display:flex;align-items:center;gap:.8rem;margin-bottom:1rem;">
            <div style="width:38px;height:38px;border-radius:10px;border:1px solid {vbd};
                display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;color:{vc};flex-shrink:0;">{vi}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-weight:800;
                    font-size:.95rem;color:{vc};">{vt}</div>
                <div style="font-size:.68rem;color:{vbd};margin-top:2px;">{vs}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-class breakdown
        if r["dets"]:
            seen, counts = set(), {}
            for d in r["dets"]: counts[d["cls"]] = counts.get(d["cls"],0)+1

            for cls, cnt in counts.items():
                if cls in seen: continue
                seen.add(cls)
                col   = COLORS.get(cls,"#555")
                lbl   = FNAME.get(cls, cls.title())
                conf  = int(sum(d["conf"] for d in r["dets"] if d["cls"]==cls)/cnt*100)
                is_r  = cls in REC
                cc,cb,cbg = ("#22c55e","#14532d","#071a0d") if is_r else ("#ef4444","#7f1d1d","#1a0505")
                tip   = TIPS.get(cls, "Seal and place in general waste.")

                st.markdown(f"""
                <div style="background:#080808;border:1px solid #111;border-radius:11px;
                    padding:.8rem .95rem;margin-bottom:.5rem;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                        <div style="display:flex;align-items:center;gap:.5rem;">
                            <div style="width:8px;height:8px;border-radius:50%;
                                background:{col};box-shadow:0 0 6px {col}70;"></div>
                            <span style="font-size:.82rem;font-weight:600;color:#e5e5e5;">{lbl}</span>
                            <span style="font-size:.52rem;font-weight:700;padding:2px 7px;
                                border-radius:100px;background:{cbg};color:{cc};
                                border:1px solid {cb};">{"Recyclable" if is_r else "Non-recyclable"}</span>
                        </div>
                        <span style="font-size:.82rem;font-weight:700;color:{col};">{conf}%</span>
                    </div>
                    <div style="background:#0f0f0f;border-radius:3px;height:3px;margin-bottom:7px;">
                        <div style="height:3px;border-radius:3px;background:{col};width:{conf}%;"></div>
                    </div>
                    <div style="font-size:.68rem;color:#1a1a1a;line-height:1.6;">{tip}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:.8rem;'>", unsafe_allow_html=True)
        st.download_button("Download Report", data=r["buf"],
            file_name="wastelens.png", mime="image/png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
