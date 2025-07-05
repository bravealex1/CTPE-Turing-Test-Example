import re
import streamlit as st
import os
import json
import uuid
import random
import pandas as pd
import sqlite3  # Changed from psycopg2 to sqlite3
from datetime import datetime

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# --------------------------------------------------
# Load Reports from CSV (Combined)
# --------------------------------------------------
# Path to the combined CSV
CTPA_CSV = "CTPA_list_30_remove15_23.csv" # Corrected filename from user

# Read the CSV
if os.path.exists(CTPA_CSV):
    df_reports_csv = pd.read_csv(CTPA_CSV)
    # Build a dictionary keyed by case ID
    report_dict = {
        str(row["id"]): { # Ensure ID is string for consistency
            "gt": row["gt"],
            "gen": row["parsed_output"] # Assuming 'parsed_output' is the generated report
        }
        for _, row in df_reports_csv.iterrows()
    }
    cases = sorted([str(case_id) for case_id in df_reports_csv["id"].unique()])
    total_cases = len(cases)
else:
    st.error(f"Error: The file {CTPA_CSV} was not found. Please ensure it's uploaded.")
    report_dict = {}
    cases = []
    total_cases = 0

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
    credentials          = config['credentials'],
    cookie_name          = config['cookie']['name'],
    key                  = config['cookie']['key'],
    cookie_expiry_days   = config['cookie']['expiry_days'],
    preauthorized        = config.get('preauthorized', [])
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
# 0. Database Setup for Queryable Logs - SQLITE VERSION
# --------------------------------------------------
DB_DIR  = "logs"
DB_PATH = os.path.join(DB_DIR, "logs.db")

def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # 1) Create progress_logs if it doesn't exist (with username column included)
    c.execute('''
    CREATE TABLE IF NOT EXISTS progress_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      username TEXT,
      category TEXT,
      progress_json TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 2) Create annotations table
    c.execute('''
    CREATE TABLE IF NOT EXISTS annotations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      case_id TEXT,
      annotations_json TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3) Create user_progress table
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

    # Save to SQLite Database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO progress_logs(session_id, username, category, progress_json) VALUES (?, ?, ?, ?)",
        (sid, username, category, json.dumps(progress))
    conn.commit()
    conn.close()

# --------------------------------------------------
# 4. Utility: Save Annotations per Case
# --------------------------------------------------
def save_annotations(case_id: str, annotations: list):
    # Save to SQLite Database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO annotations(case_id, annotations_json) VALUES (?, ?)",
        (case_id, json.dumps(annotations))
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
        "FROM user_progress WHERE username = ?",
        (username,)
    )
    row = c.fetchone()
    conn.close()
    if row:
        st.session_state.last_case_turing  = row[0]
        st.session_state.last_case_standard = row[1]
        st.session_state.last_case_ai        = row[2]
    else:
        # Initialize to 0 if no record exists
        st.session_state.last_case_turing  = 0
        st.session_state.last_case_standard = 0
        st.session_state.last_case_ai        = 0

# Save user progress to database whenever updated
def save_user_progress():
    conn = get_db_connection()
    c = conn.cursor()
    # Use "INSERT OR REPLACE" for SQLite to handle updates
    c.execute(
        """
        INSERT OR REPLACE INTO user_progress (username, last_case_turing, last_case_standard, last_case_ai)
        VALUES (?, ?, ?, ?)
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
init_state("final_eval_turing",    None)
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

# ── ADAPTED: Dummy folders for image display - actual images will not be loaded from here ──
# The `display_carousel` function will look for images within these paths
# but the case IDs are now derived from the CSV, not the file system.
# You will need to ensure these image paths are correctly populated by your deployment
# or replace this logic with actual image serving if they are stored remotely.
IMAGE_BASE_DIR = r"images_for_ctpa" # A single base directory for all CTPA images

# Ensure the dummy directory exists for the pathing logic to function without error,
# though actual image presence depends on deployment.
if not os.path.isdir(IMAGE_BASE_DIR):
    # This is a placeholder for local testing if you don't have the images
    # In a real deployment, these directories should exist and contain the images.
    try:
        os.makedirs(IMAGE_BASE_DIR, exist_ok=True)
        # You might also want to create subdirectories like 'case_id/lung' and 'case_id/soft'
        # for a few example cases to test display_carousel.
    except Exception as e:
        st.warning(f"Could not create dummy image base directory: {e}. Image display might fail if images are not present.")


# --------------------------------------------------
# 7. Helpers for Text & Carousel (adjusted to single image base dir)
# --------------------------------------------------
# Removed load_text as reports come from CSV
def display_carousel(category, case_id):
    """
    Display a slider-based carousel showing only "lung" and "soft tissue" images.
    Assumes images are structured as IMAGE_BASE_DIR/case_id/lung/*.png and IMAGE_BASE_DIR/case_id/soft/*.png
    """
    key = f"current_slice_{category}_{case_id}"
    
    # Define lung & soft tissue folders
    lung_folder = os.path.join(IMAGE_BASE_DIR, case_id, "lung")
    soft_folder = os.path.join(IMAGE_BASE_DIR, case_id, "soft")

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
    
    if max_slices == 0 and not (lung_imgs or soft_imgs):
        st.info("No images available for this case in the specified directories.")
        return

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
        st.error("No cases found in the provided CSV file.")
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

    st.info("""
    In this Turing Test, you will evaluate two radiology reports, Report A and Report B.
    One is a 'Ground-truth' report, which is a clinician's interpretation of the images.
    The other is an 'AI-generated' report, produced by a large language model.
    Your task is to determine which report is the original 'Ground-truth' report.
    """)

    if st.button("Save & Back"):
        st.session_state.page = "index"
        st.experimental_set_query_params(page="index")
        st.rerun()

    # ── Load reports from the combined dictionary ──
    reports = report_dict.get(case, {})
    gt_report  = reports.get("gt", "Ground-truth report not found in CSV.")
    gen_report = reports.get("gen", "Generated report not found in CSV.")

    assigns = st.session_state.assignments_turing
    if case not in assigns:
        assigns[case] = random.choice([True, False]) # True: A=gen, B=gt; False: A=gt, B=gen
        st.session_state.assignments_turing = assigns
    
    if assigns[case]:
        # If assigns[case] is True, Report A is generated, Report B is ground truth
        report_label_A = "AI-Generated Report"
        report_label_B = "Ground-truth Report"
        A, B = gen_report, gt_report
    else:
        # If assigns[case] is False, Report A is ground truth, Report B is generated
        report_label_A = "Ground-truth Report"
        report_label_B = "AI-Generated Report"
        A, B = gt_report, gen_report

    st.subheader(f"Report A ({report_label_A})")
    st.text_area("A", A, height=400, key=f"A_t_{case}", label_visibility="collapsed") # Increased height
    st.subheader(f"Report B ({report_label_B})")
    st.text_area("B", B, height=400, key=f"B_t_{case}", label_visibility="collapsed") # Increased height

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
        st.markdown(f"**Initial Eval (before images):** {st.session_state.initial_eval_turing}")
        
        # Simplified "Keep or Update" options
        final_choice_options = ["Keep current choice", "Change to A", "Change to B", "Change to Not sure"]
        final_choice_selected = st.radio("Review your choice after viewing images:", final_choice_options, key=f"final_ch_t_{case}")

        final = st.session_state.initial_eval_turing # Default to initial choice
        if final_choice_selected == "Change to A":
            final = "A"
        elif final_choice_selected == "Change to B":
            final = "B"
        elif final_choice_selected == "Change to Not sure":
            final = "Not sure"
        
        st.session_state.final_eval_turing = final

        st.markdown(f"**Final Choice:** {final}") # Display current final choice

        if st.button("Finalize & Next"):
            prog = {
                "case_id": case,
                "last_case_idx": idx, # Renamed to avoid confusion with last_case in db
                "assignments": st.session_state.assignments_turing[case], # Only log assignment for current case
                "initial_eval": st.session_state.initial_eval_turing,
                "final_eval": st.session_state.final_eval_turing,
                "viewed_images": st.session_state.viewed_images_turing
            }
            save_progress("turing_test", prog)
            st.session_state.last_case_turing += 1
            save_user_progress()  # ADDED: Save progress
            st.session_state.current_slice_turing = 0
            st.session_state.initial_eval_turing = None
            st.session_state.final_eval_turing    = None
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
    gt_report  = reports.get("gt", "Ground-truth report not found in CSV.")
    gen_report = reports.get("gen", "Generated report not found in CSV.")

    assigns = st.session_state.assignments_standard
    if case not in assigns:
        assigns[case] = random.choice([True, False]) # True: A=gen, B=gt; False: A=gt, B=gen
        st.session_state.assignments_standard = assigns
    
    if assigns[case]:
        # If assigns[case] is True, Report A is generated, Report B is ground truth
        report_label_A = "AI-Generated Report"
        report_label_B = "Ground-truth Report"
        A, B = gen_report, gt_report
    else:
        # If assigns[case] is False, Report A is ground truth, Report B is generated
        report_label_A = "Ground-truth Report"
        report_label_B = "AI-Generated Report"
        A, B = gt_report, gen_report

    st.subheader(f"Report A ({report_label_A})")
    st.text_area("A", A, height=200, key=f"A_s_{case}", label_visibility="collapsed")
    st.subheader(f"Report B ({report_label_B})")
    st.text_area("B", B, height=200, key=f"B_s_{case}", label_visibility="collapsed")

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
            save_annotations(case, cors) # Save specific corrections to annotations table
        prog = {
            "case_id": case,
            "last_case_idx": idx, # Renamed
            "assignments": st.session_state.assignments_standard[case], # Log current case assignment
            "evaluation_choice": choice, # Log the choice
            "corrections_made": cors # Log the corrections that were added for this case
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
            "last_case_idx": idx, # Renamed
            "mode": st.session_state.get("last_mode_ai", "Free"),
            "assembled_report": st.session_state.assembled_ai, # Renamed
            "corrections_applied": st.session_state.corrections_ai # Renamed
        }
        save_progress("ai_edit", prog)
        st.session_state.page = "index"
        st.experimental_set_query_params(page="index")
        st.rerun()

    orig = report_dict.get(case, {}).get("gen", "Original AI report not found in CSV.")
    st.subheader("Original AI Report")
    st.text_area("orig", orig, height=200, disabled=True, label_visibility="collapsed")
    st.markdown("#### Images")
    display_carousel("ai", case)

    mode = st.radio("Mode", ["Free","Organ"], key=f"md_ai_{case}")
    st.session_state["last_mode_ai"] = mode

    if mode == "Free":
        text = st.session_state.assembled_ai or orig
        new  = st.text_area("Edit", text, height=300, key=f"free_ai_{case}", label_visibility="collapsed") # Increased height
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
            if st.button("Assemble into Report"): # Changed button text
                txt = "\n".join(f"- {c['organ']}: {c['reason']} — {c['details']}" for c in cors)
                st.session_state.assembled_ai = txt
                st.success("Assembled")
                st.rerun()

    if st.button("Submit & Next"):
        prog = {
            "case_id": case,
            "last_case_idx": idx, # Renamed
            "mode": mode,
            "final_report_content": st.session_state.assembled_ai, # Renamed
            "corrections_applied": st.session_state.corrections_ai
        }
        save_progress("ai_edit", prog)
        st.session_state.corrections_ai = [c for c in st.session_state.corrections_ai if c["case_id"] != case]
        st.session_state.assembled_ai = "" # Clear for next case
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
            "SELECT session_id, username, progress_json, timestamp FROM progress_logs WHERE category=? ORDER BY timestamp",
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
    
    # Annotations Table
    st.subheader("Annotations Table")
    df_annotations = pd.read_sql_query(
        "SELECT case_id, annotations_json, timestamp FROM annotations ORDER BY timestamp", conn
    )
    if not df_annotations.empty:
        # Expand JSON
        expanded_annotations_data = []
        for _, row in df_annotations.iterrows():
            annotations = json.loads(row['annotations_json'])
            for ann in annotations:
                ann['case_id'] = row['case_id']
                ann['timestamp'] = row['timestamp']
                expanded_annotations_data.append(ann)
        st.dataframe(pd.DataFrame(expanded_annotations_data))
    else:
        st.write("— no annotations found —")

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
