import re
import streamlit as st
import os
import json
import uuid
import random
import pandas as pd
import sqlite3
from datetime import datetime

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# --------------------------------------------------
# Load Reports from CSV
# --------------------------------------------------
# Adjust this path if your CSV is located elsewhere
REPORT_CSV = "CTPE_Radiology_all_rewrite.csv"
if os.path.exists(REPORT_CSV):
    df_reports = pd.read_csv(REPORT_CSV)
    report_dict = {
        str(row["PAT_ENC_CSN_ID"]): {
            "gt": row["ground_truth_report"],
            "gen": row["generated_report"]
        }
        for _, row in df_reports.iterrows()
    }
else:
    report_dict = {}  # fallback to file‐based loading if empty

# --------------------------------------------------
# 0. Authentication Setup (must be first)
# --------------------------------------------------
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

if "credentials" not in config or "usernames" not in config["credentials"]:
    st.error("❌ Your config.yaml must include a 'credentials → usernames' section.")
    st.stop()

config["credentials"] = Hasher.hash_passwords(config["credentials"])
authenticator = stauth.Authenticate(
    credentials        = config["credentials"],
    cookie_name        = config["cookie"]["name"],
    key                = config["cookie"]["key"],
    cookie_expiry_days = config["cookie"]["expiry_days"],
    preauthorized      = config.get("preauthorized", [])
)
authenticator.login(location="sidebar", key="login")

name                  = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username              = st.session_state.get("username")

if not authentication_status:
    if authentication_status is False:
        st.error("❌ Username/password is incorrect")
    else:
        st.warning("⚠️ Please enter your username and password")
    st.stop()

# --------------------------------------------------
# 0. Database Setup for Queryable Logs
# --------------------------------------------------
DB_DIR  = "logs"
DB_PATH = os.path.join(DB_DIR, "logs.db")

def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS progress_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      category TEXT,
      progress_json TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS annotations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      case_id TEXT,
      annotations_json TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# --------------------------------------------------
# Helper: Prevent Duplicate SQLite Inserts
# --------------------------------------------------
def should_log(session_id: str, category: str, new_progress: dict) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT progress_json FROM progress_logs "
        "WHERE session_id=? AND category=? "
        "ORDER BY timestamp DESC LIMIT 1",
        (session_id, category)
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return True
    last = json.loads(row[0])
    if "last_case" in new_progress:
        return last.get("last_case") != new_progress.get("last_case")
    if category == "ai_edit" and "case_id" in new_progress:
        return last.get("case_id") != new_progress.get("case_id")
    return True

# --------------------------------------------------
# 1. Generate & Store Unique Session ID
# --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --------------------------------------------------
# 2. Sidebar: Display Session ID
# --------------------------------------------------
st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id}`")

# --------------------------------------------------
# 3. Utility: Save Progress per Category & Session
# --------------------------------------------------
def save_progress(category: str, progress: dict):
    sid = st.session_state.session_id
    if not should_log(sid, category, progress):
        return
    os.makedirs(DB_DIR, exist_ok=True)
    # JSON
    jpath = os.path.join(DB_DIR, f"{category}_{sid}_progress.json")
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = data if isinstance(data, list) else [data]
    else:
        data = []
    data.append(progress)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    # CSV
    cpath = os.path.join(DB_DIR, f"{category}_{sid}_progress.csv")
    df = pd.DataFrame([progress])
    if os.path.exists(cpath):
        df.to_csv(cpath, index=False, mode="a", header=False)
    else:
        df.to_csv(cpath, index=False)
    # SQLite
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO progress_logs(session_id, category, progress_json) VALUES (?, ?, ?)",
        (sid, category, json.dumps(progress))
    )
    conn.commit()
    conn.close()

# --------------------------------------------------
# 4. Utility: Save Annotations per Case
# --------------------------------------------------
def save_annotations(case_id: str, annotations: list):
    os.makedirs("evaluations", exist_ok=True)
    path = os.path.join("evaluations", f"{case_id}_annotations.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = data if isinstance(data, list) else [data]
    else:
        data = []
    data.extend(annotations)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO annotations(case_id, annotations_json) VALUES (?, ?)",
        (case_id, json.dumps(annotations))
    )
    conn.commit()
    conn.close()

# --------------------------------------------------
# 5. Initialize per-workflow Session State
# --------------------------------------------------
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# Turing Test
init_state("last_case_turing",    0)
init_state("current_slice_turing",0)
init_state("assignments_turing",  {})
init_state("initial_eval_turing", None)
init_state("final_eval_turing",   None)
init_state("viewed_images_turing",False)

# Standard Evaluation
init_state("last_case_standard",    0)
init_state("current_slice_standard",0)
init_state("assignments_standard",  {})
init_state("corrections_standard",  [])

# AI Edit
init_state("last_case_ai",    0)
init_state("current_slice_ai",0)
init_state("corrections_ai",  [])
init_state("assembled_ai",    "")

# --------------------------------------------------
# 6. Routing Setup
# --------------------------------------------------
params = st.experimental_get_query_params()
if "page" in params:
    st.session_state.page = params["page"][0]
elif "page" not in st.session_state:
    st.session_state.page = "index"

# ── ADAPT THIS: point to your parent folder containing 123,456,789 ──
BASE_IMAGE_DIR = "cases"

cases = sorted([
    d for d in os.listdir(BASE_IMAGE_DIR)
    if os.path.isdir(os.path.join(BASE_IMAGE_DIR, d))
])
total_cases = len(cases)

# --------------------------------------------------
# 7. Helpers for Text & Carousel
# --------------------------------------------------
def load_text(path):
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

def display_carousel(category, case_id):
    key = f"current_slice_{category}"
    bone_folder = os.path.join(BASE_IMAGE_DIR, case_id, "bone_tissue_window")
    lung_folder = os.path.join(BASE_IMAGE_DIR, case_id, "lung_tissue_window")
    soft_folder = os.path.join(BASE_IMAGE_DIR, case_id, "soft_tissue_window")

    bone_imgs = sorted([os.path.join(bone_folder, f)
                        for f in os.listdir(bone_folder)
                        if f.lower().endswith((".png",".jpg",".jpeg"))]) if os.path.exists(bone_folder) else []
    lung_imgs = sorted([os.path.join(lung_folder, f)
                        for f in os.listdir(lung_folder)
                        if f.lower().endswith((".png",".jpg",".jpeg"))]) if os.path.exists(lung_folder) else []
    soft_imgs = sorted([os.path.join(soft_folder, f)
                        for f in os.listdir(soft_folder)
                        if f.lower().endswith((".png",".jpg",".jpeg"))]) if os.path.exists(soft_folder) else []

    max_slices = max(len(bone_imgs), len(lung_imgs), len(soft_imgs), 1)
    idx = st.session_state.get(key, 0)
    idx = max(0, min(idx, max_slices - 1))
    st.session_state[key] = idx

    c_prev, c1, c2, c3, c_next = st.columns([1,3,3,3,1])
    with c_prev:
        if st.button("⟨ Prev", key=f"prev_{category}_{case_id}") and idx>0:
            st.session_state[key] = idx - 1
            st.rerun()
    with c1:
        if bone_imgs:
            st.image(bone_imgs[idx], caption="Bone", use_column_width=True)
        else:
            st.info("No bone images.")
    with c2:
        if lung_imgs:
            st.image(lung_imgs[idx], caption="Lung", use_column_width=True)
        else:
            st.info("No lung images.")
    with c3:
        if soft_imgs:
            st.image(soft_imgs[idx], caption="Soft Tissue", use_column_width=True)
        else:
            st.info("No soft‐tissue images.")
    with c_next:
        if st.button("Next ⟩", key=f"next_{category}_{case_id}") and idx<max_slices-1:
            st.session_state[key] = idx + 1
            st.rerun()

# --------------------------------------------------
# 8. Pages
# --------------------------------------------------
def index():
    st.title("Survey App")
    if total_cases == 0:
        st.error("No cases found.")
        return
    st.markdown("### Your Progress")
    st.markdown(f"- **Turing Test**: Case {st.session_state.last_case_turing+1}/{total_cases}")
    st.markdown(f"- **Standard Eval**: Case {st.session_state.last_case_standard+1}/{total_cases}")
    st.markdown(f"- **AI Edit**: Case {st.session_state.last_case_ai+1}/{total_cases}")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    if c1.button("Turing Test"):
        st.experimental_set_query_params(page="turing_test"); st.session_state.page="turing_test"; st.rerun()
    if c2.button("Standard Eval"):
        st.experimental_set_query_params(page="standard_eval"); st.session_state.page="standard_eval"; st.rerun()
    if c3.button("AI Edit"):
        st.experimental_set_query_params(page="ai_edit"); st.session_state.page="ai_edit"; st.rerun()
    if c4.button("View All Results"):
        st.experimental_set_query_params(page="view_results"); st.session_state.page="view_results"; st.rerun()

def turing_test():
    # initialize index
    idx = st.session_state.get("last_case_turing", 0)

    # all done?
    if idx >= total_cases:
        st.success("Turing Test complete!")
        if st.button("Home"):
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()
        return

    # ── INITIALIZE RANDOM A/B ASSIGNMENTS ──
    if "assignments_turing" not in st.session_state:
        # one random flip per case
        flags = [random.choice([True, False]) for _ in cases]
        st.session_state.assignments_turing = dict(zip(cases, flags))

    assigns = st.session_state.assignments_turing
    case = cases[idx]

    # header & back button
    st.header(f"Turing Test: {case} ({idx+1}/{total_cases})")
    if st.button("Save & Back"):
        st.session_state.page = "index"
        st.experimental_set_query_params(page="index")
        st.rerun()

    # ── LOAD REPORT TEXT ──
    reports = report_dict.get(case, {})
    gt_report  = reports.get("gt",  load_text(os.path.join("images", case, "text.txt")))
    gen_report = reports.get("gen", load_text(os.path.join("images", case, "pred.txt")))

    # ── ASSIGN A vs B ──
    if assigns[case]:
        A, B = gt_report, gen_report
    else:
        A, B = gen_report, gt_report

    st.subheader("Report A")
    st.text_area("A", A, height=200, key=f"A_t_{case}")
    st.subheader("Report B")
    st.text_area("B", B, height=200, key=f"B_t_{case}")

    # ── INITIAL EVAL ──
    if st.session_state.get("initial_eval_turing") is None:
        choice = st.radio(
            "Which is ground truth?",
            ["A", "B", "Not sure"],
            key=f"ch_t_{case}",
            index=2
        )
        if st.button("Submit Initial Evaluation"):
            st.session_state.initial_eval_turing = choice
            st.session_state.viewed_images_turing = True
            st.success("Recorded initial eval.")
            st.rerun()

    # ── SHOW IMAGES & FINAL EVAL ──
    if st.session_state.get("viewed_images_turing", False):
        st.markdown("#### Images")
        display_carousel("turing", case)

        st.markdown(f"**Initial Eval:** {st.session_state.initial_eval_turing}")
        up = st.radio("Keep or Update?", ["Keep", "Update"], key=f"up_t_{case}")
        final = st.session_state.initial_eval_turing
        if up == "Update":
            final = st.radio(
                "New choice:",
                ["A", "B", "Not sure"],
                key=f"new_t_{case}",
                index=2
            )
        st.session_state.final_eval_turing = final

        if st.button("Finalize & Next"):
            prog = {
                "case_id":       case,
                "last_case":     idx,
                "assignments":   st.session_state.assignments_turing,
                "initial_eval":  st.session_state.initial_eval_turing,
                "final_eval":    st.session_state.final_eval_turing,
                "viewed_images": st.session_state.viewed_images_turing
            }
            save_progress("turing_test", prog)

            # advance
            st.session_state.last_case_turing    = idx + 1
            st.session_state.initial_eval_turing  = None
            st.session_state.final_eval_turing    = None
            st.session_state.viewed_images_turing = False
            st.rerun()

def evaluate_case():
    idx = st.session_state.last_case_standard
    if idx >= total_cases:
        st.success("Standard Eval complete!")
        if st.button("Home"):
            st.session_state.page="index"; st.experimental_set_query_params(page="index"); st.rerun()
        return
    case = cases[idx]
    st.header(f"Standard Eval: {case} ({idx+1}/{total_cases})")

    if st.button("Save & Back"):
        st.session_state.page="index"; st.experimental_set_query_params(page="index"); st.rerun()

    # ── MODIFIED: load from CSV, fallback to files ──
    reports = report_dict.get(case, {})
    gt_report  = reports.get("gt",  load_text(os.path.join(BASE_IMAGE_DIR, case, "text.txt")))
    gen_report = reports.get("gen", load_text(os.path.join(BASE_IMAGE_DIR, case, "pred.txt")))

    assigns = st.session_state.assignments_standard
    if case not in assigns:
        assigns[case] = random.choice([True, False])
        st.session_state.assignments_standard = assigns
    if assigns[case]:
        A, B = gen_report, gt_report
    else:
        A, B = gt_report, gen_report

    st.subheader("Report A")
    st.text_area("A", A, height=150, key=f"A_s_{case}")
    st.subheader("Report B")
    st.text_area("B", B, height=150, key=f"B_s_{case}")

    st.markdown("#### Images")
    display_carousel("standard", case)

    organ  = st.selectbox("Organ", [""] + ["LIVER","PANCREAS","KIDNEY","OTHER"], key=f"org_s_{case}")
    reason = st.text_input("Reason", key=f"rsn_s_{case}")
    details= st.text_area("Details", key=f"dtl_s_{case}")
    if st.button("Add Correction") and organ:
        st.session_state.corrections_standard.append({
            "case_id": case, "organ": organ, "reason": reason, "details": details
        })
        st.success("Added correction")
        st.rerun()

    cors = [c for c in st.session_state.corrections_standard if c["case_id"] == case]
    if cors:
        st.table(pd.DataFrame(cors).drop(columns=["case_id"]))

    choice = st.radio("Best report?", ["A","B","Corrected","Equal"], key=f"ch_s_{case}")
    if st.button("Submit & Next"):
        if cors:
            save_annotations(case, cors)
        prog = {
            "case_id": case,
            "last_case": idx,
            "assignments": st.session_state.assignments_standard,
            "corrections": st.session_state.corrections_standard
        }
        save_progress("standard_evaluation", prog)
        st.session_state.corrections_standard = [
            c for c in st.session_state.corrections_standard if c["case_id"] != case
        ]
        st.session_state.last_case_standard += 1
        st.session_state.current_slice_standard = 0
        st.rerun()

def ai_edit():
    idx = st.session_state.last_case_ai
    if idx >= total_cases:
        st.success("AI Edit complete!")
        if st.button("Home"):
            st.session_state.page="index"; st.experimental_set_query_params(page="index"); st.rerun()
        return
    case = cases[idx]
    st.header(f"AI Edit: {case} ({idx+1}/{total_cases})")

    if st.button("Save & Back"):
        prog = {
            "case_id": case,
            "mode": st.session_state.get("last_mode_ai", "Free"),
            "assembled": st.session_state.assembled_ai,
            "corrections": st.session_state.corrections_ai
        }
        save_progress("ai_edit", prog)
        st.session_state.page="index"; st.experimental_set_query_params(page="index"); st.rerun()

    orig = report_dict.get(case, {}).get("gen",
           load_text(os.path.join(BASE_IMAGE_DIR, case, "pred.txt")))
    st.subheader("Original AI Report")
    st.text_area("orig", orig, height=150, disabled=True)
    st.markdown("#### Images")
    display_carousel("ai", case)

    mode = st.radio("Mode", ["Free","Organ"], key=f"md_ai_{case}")
    st.session_state["last_mode_ai"] = mode

    if mode == "Free":
        text = st.session_state.assembled_ai or orig
        new  = st.text_area("Edit", text, height=200, key=f"free_ai_{case}")
        st.session_state.assembled_ai = new
    else:
        organ  = st.selectbox("Organ", [""]+["LIVER","PANCREAS","KIDNEY","OTHER"], key=f"org_ai_{case}")
        reason = st.text_input("Reason", key=f"rsn_ai_{case}")
        details= st.text_area("Details", key=f"dtl_ai_{case}")
        if st.button("Add Corr AI") and organ:
            st.session_state.corrections_ai.append({
                "case_id": case, "organ": organ, "reason": reason, "details": details
            })
            st.success("Added")
            st.rerun()

        cors = [c for c in st.session_state.corrections_ai if c["case_id"] == case]
        if cors:
            st.table(pd.DataFrame(cors).drop(columns=["case_id"]))
            if st.button("Assemble"):
                txt = "\n".join(f"- {c['organ']}: {c['reason']} — {c['details']}" for c in cors)
                st.session_state.assembled_ai = txt
                st.success("Assembled")
                st.rerun()

    if st.button("Submit & Next"):
        prog = {
            "case_id": case,
            "mode": mode,
            "assembled": st.session_state.assembled_ai,
            "corrections": st.session_state.corrections_ai
        }
        save_progress("ai_edit", prog)
        st.session_state.corrections_ai = [c for c in st.session_state.corrections_ai if c["case_id"] != case]
        st.session_state.assembled_ai = ""
        st.session_state.last_case_ai += 1
        st.session_state.current_slice_ai = 0
        st.rerun()

def view_all_results():
    st.title("All Saved Results")
    if st.button("Home"):
        st.session_state.page="index"; st.experimental_set_query_params(page="index"); st.rerun()

    conn = get_db_connection()

    # Sessions
    df_sessions = pd.read_sql_query(
        "SELECT DISTINCT session_id FROM progress_logs ORDER BY session_id", conn
    )
    st.subheader("All Sessions with Saved Progress")
    for sid in df_sessions["session_id"]:
        st.write(f"- {sid}")

    # Turing & Standard
    for cat,label in [
        ("turing_test","Turing Test Logs"),
        ("standard_evaluation","Standard Eval Logs")
    ]:
        st.subheader(label)
        df = pd.read_sql_query(
            "SELECT session_id, progress_json, timestamp FROM progress_logs WHERE category=? ORDER BY timestamp",
            conn, params=(cat,)
        )
        if not df.empty:
            df_expanded = pd.concat([
                df.drop(columns=["progress_json"]),
                df["progress_json"].apply(json.loads).apply(pd.Series)
            ], axis=1)
            for col in df_expanded.columns:
                if df_expanded[col].apply(lambda x: isinstance(x, (dict,list))).any():
                    df_expanded[col] = df_expanded[col].apply(json.dumps)
            if "last_case" in df_expanded.columns:
                df_expanded["Case"] = df_expanded["last_case"] + 1
                df_expanded = df_expanded.drop(columns=["last_case"])
                cols = ["Case"] + [c for c in df_expanded.columns if c!="Case"]
                st.dataframe(df_expanded[cols])
            else:
                st.dataframe(df_expanded)
        else:
            st.write("— no entries —")

    # AI Report Edit Logs
    st.subheader("AI Report Edit Logs")
    df_ai = pd.read_sql_query(
        "SELECT session_id, progress_json, timestamp FROM progress_logs WHERE category='ai_edit' ORDER BY timestamp", conn
    )
    if not df_ai.empty:
        df_ai_expanded = pd.concat([
            df_ai.drop(columns=["progress_json"]),
            df_ai["progress_json"].apply(json.loads).apply(pd.Series)
        ], axis=1)
        for col in df_ai_expanded.columns:
            if df_ai_expanded[col].apply(lambda x: isinstance(x, (dict,list))).any():
                df_ai_expanded[col] = df_ai_expanded[col].apply(json.dumps)
        st.dataframe(df_ai_expanded)
    else:
        st.write("— no AI edit logs found —")

    conn.close()

# --------------------------------------------------
# 9. Main Router
# --------------------------------------------------
page = st.session_state.page
if page == "turing_test":
    turing_test()
elif page == "standard_eval":
    evaluate_case()
elif page == "ai_edit":
    ai_edit()
elif page == "view_results":
    view_all_results()
else:
    index()
