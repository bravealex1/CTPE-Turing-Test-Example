import re
import streamlit as st
import os
import json
import uuid
import random
import pandas as pd
import psycopg2
from datetime import datetime

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# --------------------------------------------------
# Load Reports from CSV (Normal & Abnormal)
# --------------------------------------------------
# Paths to the CSVs
NORMAL_CSV = r"update_normal_top_15.csv"
ABNORMAL_CSV = r"update_abnormal_top_15.csv"

# Read both CSVs and combine
if os.path.exists(NORMAL_CSV) and os.path.exists(ABNORMAL_CSV):
    df_normal = pd.read_csv(NORMAL_CSV)
    df_abnormal = pd.read_csv(ABNORMAL_CSV)
    df_reports = pd.concat([df_normal, df_abnormal], ignore_index=True)
    # Build a dictionary keyed by case ID
    report_dict = {
        str(row["id"]): {
            "gt": row["gt"],
            "gen": row["parsed_output"]
        }
        for _, row in df_reports.iterrows()
    }
else:
    report_dict = {}  # fallback if CSVs are missing

# --------------------------------------------------
# 0. Authentication Setup (must be first) - SIMPLIFIED
# --------------------------------------------------
# Create default config if file doesn't exist
if not os.path.exists("config.yaml"):
    # Create 10 test users with hashed passwords
    passwords = Hasher(['pass1', 'pass2', 'pass3', 'pass4', 'pass5',
                        'pass6', 'pass7', 'pass8', 'pass9', 'pass10']).generate()
    
    config = {
        'credentials': {
            'usernames': {
                f'tester{i+1}': {
                    'email': f'tester{i+1}@example.com',
                    'name': f'Test User {i+1}',
                    'password': pwd
                } for i, pwd in enumerate(passwords)
            }
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'your_cookie_key',
            'name': 'auth_cookie'
        },
        'preauthorized': []
    }
    
    with open("config.yaml", "w") as file:
        yaml.dump(config, file)

# Load the YAML config
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Set up Streamlit-Authenticator
authenticator = stauth.Authenticate(
    credentials        = config['credentials'],
    cookie_name        = config['cookie']['name'],
    key                = config['cookie']['key'],
    cookie_expiry_days = config['cookie']['expiry_days'],
    preauthorized      = config.get('preauthorized', [])
)

# Render login widget in the sidebar
authenticator.login(location="sidebar", key="login")

name                  = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username              = st.session_state.get("username")

# ADDED: Logout button if logged in
if authentication_status:
    authenticator.logout('Logout', 'sidebar', key='logout_button')

if not authentication_status:
    if authentication_status is False:
        st.error("❌ Username/password is incorrect")
    else:
        st.warning("⚠️ Please enter your username and password")
    st.stop()

# --------------------------------------------------
# 0. Database Setup for Queryable Logs
# --------------------------------------------------
def get_db_connection():
    # Use Streamlit secrets to get database credentials
    conn = psycopg2.connect(**st.secrets["postgres"])
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # 1) Create progress_logs if it doesn't exist (PostgreSQL syntax)
    c.execute('''
    CREATE TABLE IF NOT EXISTS progress_logs (
      id SERIAL PRIMARY KEY,
      session_id TEXT,
      username TEXT,
      category TEXT,
      progress_json TEXT,
      timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 2) Create annotations table (PostgreSQL syntax)
    c.execute('''
    CREATE TABLE IF NOT EXISTS annotations (
      id SERIAL PRIMARY KEY,
      case_id TEXT,
      annotations_json TEXT,
      timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3) Create user_progress table (PostgreSQL syntax)
    c.execute('''
    CREATE TABLE IF NOT EXISTS user_progress (
      username TEXT PRIMARY KEY,
      last_case_turing INTEGER DEFAULT 0,
      last_case_standard INTEGER DEFAULT 0,
      last_case_ai INTEGER DEFAULT 0
    )
    ''')

    conn.commit()
    conn.close()

# Initialize the database
init_db()

# --------------------------------------------------
# Helper: Prevent Duplicate Inserts (PostgreSQL version)
# --------------------------------------------------
def should_log(session_id: str, category: str, new_progress: dict) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT progress_json FROM progress_logs "
        "WHERE session_id=%s AND category=%s "
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
# 2. Sidebar: Display Session ID & Username
# --------------------------------------------------
st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.markdown(f"**User:** {name}")

# --------------------------------------------------
# 3. Utility: Save Progress per Category & Session
# --------------------------------------------------
def save_progress(category: str, progress: dict):
    sid = st.session_state.session_id
    if not should_log(sid, category, progress):
        return

    # Save to Database (PostgreSQL)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO progress_logs(session_id, username, category, progress_json) VALUES (%s, %s, %s, %s)",
        (sid, username, category, json.dumps(progress))
    )
    conn.commit()
    conn.close()

# --------------------------------------------------
# 4. Utility: Save Annotations per Case
# --------------------------------------------------
def save_annotations(case_id: str, annotations: list):
    # Save to Database (PostgreSQL)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO annotations(case_id, annotations_json) VALUES (%s, %s)",
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

# Load user progress from database on first login
def load_user_progress():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT last_case_turing, last_case_standard, last_case_ai "
        "FROM user_progress WHERE username = %s",
        (username,)
    )
    row = c.fetchone()
    conn.close()
    if row:
        st.session_state.last_case_turing  = row[0]
        st.session_state.last_case_standard = row[1]
        st.session_state.last_case_ai       = row[2]
    else:
        # Initialize to 0 if no record exists
        st.session_state.last_case_turing  = 0
        st.session_state.last_case_standard = 0
        st.session_state.last_case_ai       = 0

# Save user progress to database whenever updated
def save_user_progress():
    conn = get_db_connection()
    c = conn.cursor()
    # Use "ON CONFLICT" for PostgreSQL to handle updates
    c.execute(
        """
        INSERT INTO user_progress (username, last_case_turing, last_case_standard, last_case_ai)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (username) DO UPDATE SET
            last_case_turing = EXCLUDED.last_case_turing,
            last_case_standard = EXCLUDED.last_case_standard,
            last_case_ai = EXCLUDED.last_case_ai
        """,
        (username,
         st.session_state.last_case_turing,
         st.session_state.last_case_standard,
         st.session_state.last_case_ai)
    )
    conn.commit()
    conn.close()

# Load user progress exactly once after login
if 'progress_loaded' not in st.session_state:
    load_user_progress()
    st.session_state.progress_loaded = True

# Other session state initializations
init_state("current_slice_turing", 0)
init_state("assignments_turing",  {})
init_state("initial_eval_turing", None)
init_state("final_eval_turing",   None)
init_state("viewed_images_turing", False)

init_state("current_slice_standard", 0)
init_state("assignments_standard",  {})
init_state("corrections_standard",  [])

init_state("current_slice_ai", 0)
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

# ── ADAPTED: point to the folders containing normal & abnormal cases ──
NORMAL_IMAGE_DIR   = r"sampled_normal"
ABNORMAL_IMAGE_DIR = r"sampled_abnormal"

# Ensure those directories exist
if not os.path.isdir(NORMAL_IMAGE_DIR) or not os.path.isdir(ABNORMAL_IMAGE_DIR):
    st.error(f"Image directories not found: {NORMAL_IMAGE_DIR} / {ABNORMAL_IMAGE_DIR}")
    st.stop()

# Get list of case IDs from both directories
cases_normal   = sorted([d for d in os.listdir(NORMAL_IMAGE_DIR)   if os.path.isdir(os.path.join(NORMAL_IMAGE_DIR, d))])
cases_abnormal = sorted([d for d in os.listdir(ABNORMAL_IMAGE_DIR) if os.path.isdir(os.path.join(ABNORMAL_IMAGE_DIR, d))])
cases = sorted(cases_normal + cases_abnormal)
total_cases = len(cases)

# --------------------------------------------------
# 7. Helpers for Text & Carousel (adjusted folder names)
# --------------------------------------------------
def load_text(path):
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

def display_carousel(category, case_id):
    """
    Display a slider-based carousel showing only "lung" and "soft tissue" images.
    """
    key = f"current_slice_{category}_{case_id}"
    # Determine which base folder this case lives in
    if case_id in cases_normal:
        base_dir = NORMAL_IMAGE_DIR
    else:
        base_dir = ABNORMAL_IMAGE_DIR

    # Define lung & soft tissue folders
    lung_folder = os.path.join(base_dir, case_id, "lung")
    soft_folder = os.path.join(base_dir, case_id, "soft")

    lung_imgs = sorted([
        os.path.join(lung_folder, f)
        for f in os.listdir(lung_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]) if os.path.exists(lung_folder) else []

    soft_imgs = sorted([
        os.path.join(soft_folder, f)
        for f in os.listdir(soft_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]) if os.path.exists(soft_folder) else []

    max_slices = max(len(lung_imgs), len(soft_imgs), 1)
    # Use a slider to pick the slice index; starts at 0
    idx = st.slider(
        "Slice index",
        min_value=0,
        max_value=max_slices - 1,
        value=st.session_state.get(key, 0),
        key=key
    )

    # Display side-by-side columns for lung & soft tissue
    c1, c2 = st.columns(2)
    with c1:
        if lung_imgs:
            # If idx >= len(lung_imgs), show the last available image
            i = min(idx, len(lung_imgs) - 1)
            st.image(lung_imgs[i], caption="Lung", use_column_width=True)
        else:
            st.info("No lung images available.")

    with c2:
        if soft_imgs:
            # If idx >= len(soft_imgs), show the last available image
            i = min(idx, len(soft_imgs) - 1)
            st.image(soft_imgs[i], caption="Soft Tissue", use_column_width=True)
        else:
            st.info("No soft-tissue images available.")

# --------------------------------------------------
# 8. Pages
# --------------------------------------------------
def index():
    st.title("Survey App")
    if total_cases == 0:
        st.error("No cases found.")
        return
    st.markdown("### Your Progress")
    st.markdown(f"- **Turing Test**: Case {st.session_state.last_case_turing + 1}/{total_cases}")
    st.markdown(f"- **Standard Eval**: Case {st.session_state.last_case_standard + 1}/{total_cases}")
    st.markdown(f"- **AI Edit**: Case {st.session_state.last_case_ai + 1}/{total_cases}")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Turing Test"):
        st.experimental_set_query_params(page="turing_test")
        st.session_state.page = "turing_test"
        st.rerun()
    if c2.button("Standard Eval"):
        st.experimental_set_query_params(page="standard_eval")
        st.session_state.page = "standard_eval"
        st.rerun()
    if c3.button("AI Edit"):
        st.experimental_set_query_params(page="ai_edit")
        st.session_state.page = "ai_edit"
        st.rerun()
    if c4.button("View All Results"):
        st.experimental_set_query_params(page="view_results")
        st.session_state.page = "view_results"
        st.rerun()

def turing_test():
    idx = st.session_state.last_case_turing
    if idx >= total_cases:
        st.success("Turing Test complete!")
        if st.button("Home"):
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()
        return
    case = cases[idx]
    st.header(f"Turing Test: {case} ({idx + 1}/{total_cases})")

    if st.button("Save & Back"):
        st.session_state.page = "index"
        st.experimental_set_query_params(page="index")
        st.rerun()

    # ── Load reports from the combined dictionary ──
    reports = report_dict.get(case, {})
    gt_report  = reports.get("gt",  load_text(os.path.join(NORMAL_IMAGE_DIR, case, "text.txt")))
    gen_report = reports.get("gen", load_text(os.path.join(NORMAL_IMAGE_DIR, case, "pred.txt")))

    assigns = st.session_state.assignments_turing
    if case not in assigns:
        assigns[case] = random.choice([True, False])
        st.session_state.assignments_turing = assigns
    if assigns[case]:
        A, B = gen_report, gt_report
    else:
        A, B = gt_report, gen_report

    st.subheader("Report A")
    st.text_area("A", A, height=200, key=f"A_t_{case}")
    st.subheader("Report B")
    st.text_area("B", B, height=200, key=f"B_t_{case}")

    if st.session_state.initial_eval_turing is None:
        choice = st.radio("Which is ground truth?", ["A","B","Not sure"], key=f"ch_t_{case}", index=2)
        if st.button("Submit initial evaluation"):
            st.session_state.initial_eval_turing = choice
            st.session_state.viewed_images_turing = True
            st.success("Recorded initial eval.")
            st.rerun()

    if st.session_state.viewed_images_turing:
        st.markdown("#### Images")
        display_carousel("turing", case)
        st.markdown(f"**Initial Eval:** {st.session_state.initial_eval_turing}")
        up = st.radio("Keep or Update?", ["Keep","Update"], key=f"up_t_{case}")
        final = st.session_state.initial_eval_turing
        if up == "Update":
            final = st.radio("New choice:", ["A","B","Not sure"], key=f"new_t_{case}", index=2)
        st.session_state.final_eval_turing = final

        if st.button("Finalize & Next"):
            prog = {
                "case_id": case,
                "last_case": idx,
                "assignments": st.session_state.assignments_turing,
                "initial_eval": st.session_state.initial_eval_turing,
                "final_eval": st.session_state.final_eval_turing,
                "viewed_images": st.session_state.viewed_images_turing
            }
            save_progress("turing_test", prog)
            st.session_state.last_case_turing += 1
            save_user_progress()  # ADDED: Save progress
            st.session_state.current_slice_turing = 0
            st.session_state.initial_eval_turing = None
            st.session_state.final_eval_turing   = None
            st.session_state.viewed_images_turing = False
            st.rerun()

def evaluate_case():
    idx = st.session_state.last_case_standard
    if idx >= total_cases:
        st.success("Standard Eval complete!")
        if st.button("Home"):
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()
        return
    case = cases[idx]
    st.header(f"Standard Eval: {case} ({idx + 1}/{total_cases})")

    if st.button("Save & Back"):
        st.session_state.page = "index"
        st.experimental_set_query_params(page="index")
        st.rerun()

    # ── Load reports from the combined dictionary ──
    reports = report_dict.get(case, {})
    gt_report  = reports.get("gt",  load_text(os.path.join(NORMAL_IMAGE_DIR, case, "text.txt")))
    gen_report = reports.get("gen", load_text(os.path.join(NORMAL_IMAGE_DIR, case, "pred.txt")))

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
        save_user_progress()  # ADDED: Save progress
        st.session_state.current_slice_standard = 0
        st.rerun()

def ai_edit():
    idx = st.session_state.last_case_ai
    if idx >= total_cases:
        st.success("AI Edit complete!")
        if st.button("Home"):
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()
        return
    case = cases[idx]
    st.header(f"AI Edit: {case} ({idx + 1}/{total_cases})")

    if st.button("Save & Back"):
        prog = {
            "case_id": case,
            "mode": st.session_state.get("last_mode_ai", "Free"),
            "assembled": st.session_state.assembled_ai,
            "corrections": st.session_state.corrections_ai
        }
        save_progress("ai_edit", prog)
        st.session_state.page = "index"
        st.experimental_set_query_params(page="index")
        st.rerun()

    orig = report_dict.get(case, {}).get("gen",
           load_text(os.path.join(NORMAL_IMAGE_DIR, case, "pred.txt")))
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
        save_user_progress()  # ADDED: Save progress
        st.session_state.current_slice_ai = 0
        st.rerun()

def view_all_results():
    st.title("All Saved Results")
    if st.button("Home"):
        st.session_state.page = "index"
        st.experimental_set_query_params(page="index")
        st.rerun()

    conn = get_db_connection()

    # Sessions
    df_sessions = pd.read_sql_query(
        "SELECT DISTINCT session_id, username FROM progress_logs ORDER BY session_id", conn
    )
    st.subheader("All Sessions with Saved Progress")
    st.dataframe(df_sessions)

    # Turing & Standard
    for cat, label in [
        ("turing_test", "Turing Test Logs"),
        ("standard_evaluation", "Standard Eval Logs")
    ]:
        st.subheader(label)
        df = pd.read_sql_query(
            "SELECT session_id, username, progress_json, timestamp FROM progress_logs WHERE category=%s ORDER BY timestamp",
            conn, params=(cat,)
        )
        if not df.empty:
            # Expand the JSON data
            expanded_data = []
            for _, row in df.iterrows():
                progress = json.loads(row['progress_json'])
                progress['session_id'] = row['session_id']
                progress['username'] = row['username']
                progress['timestamp'] = row['timestamp']
                expanded_data.append(progress)
            
            st.dataframe(pd.DataFrame(expanded_data))
        else:
            st.write("— no entries —")

    # AI Report Edit Logs
    st.subheader("AI Report Edit Logs")
    df_ai = pd.read_sql_query(
        "SELECT session_id, username, progress_json, timestamp FROM progress_logs WHERE category='ai_edit' ORDER BY timestamp", conn
    )
    if not df_ai.empty:
        # Expand the JSON data
        expanded_data = []
        for _, row in df_ai.iterrows():
            progress = json.loads(row['progress_json'])
            progress['session_id'] = row['session_id']
            progress['username'] = row['username']
            progress['timestamp'] = row['timestamp']
            expanded_data.append(progress)
        
        st.dataframe(pd.DataFrame(expanded_data))
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
