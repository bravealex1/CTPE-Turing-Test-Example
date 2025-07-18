import streamlit as st
import os
import json
import uuid
import random
import pandas as pd
import sqlite3
import logging
from datetime import datetime
import time
import re  # For extracting folder names

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# --------------------------------------------------
# Configure logging
# --------------------------------------------------
logging.basicConfig(
    filename='app_errors.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------------------------------------
# Load Reports from CSV (Corrected)
# --------------------------------------------------
CTPA_CSV = "CTPA_list_30_remove15_23.csv"

def load_reports():
    """
    Loads reports from the CSV, using the 'id' column as the folder name,
    and builds the report dictionary and case list.
    """
    try:
        if not os.path.exists(CTPA_CSV):
            st.error(f"Error: The file {CTPA_CSV} was not found. Please ensure it's in the correct directory.")
            return {}, [], 0

        df_reports_csv = pd.read_csv(CTPA_CSV)
        report_dict = {}

        # Process the DataFrame
        for _, row in df_reports_csv.iterrows():
            # The 'id' column directly corresponds to the folder name, as per the requirement.
            case_id = str(row["id"])
            folder_name = case_id
            
            report_dict[case_id] = {
                "gt": row["gt"],
                "gen": row["parsed_output"],
                "folder": folder_name  # Use the 'id' as the folder name
            }

        cases = sorted(report_dict.keys())
        total_cases = len(cases)
        
        return report_dict, cases, total_cases

    except Exception as e:
        st.error(f"Failed to load or process the CSV report file: {str(e)}")
        logging.error(f"Failed to load reports: {str(e)}")
        return {}, [], 0

# Load the reports using the corrected function
report_dict, cases, total_cases = load_reports()

# --------------------------------------------------
# 0. Authentication Setup
# --------------------------------------------------
def setup_authentication():
    try:
        if not os.path.exists("config.yaml"):
            # Create 20 test users with hashed passwords
            plain_passwords = [f'pass{i+1}' for i in range(20)]
            hashed_passwords = Hasher(plain_passwords).generate()
            
            credentials_dict = {}
            for i in range(20):
                username = f'tester{i+1}'
                credentials_dict[username] = {
                    'email': f'{username}@example.com',
                    'name': f'Test User {i+1}',
                    'password': hashed_passwords[i]
                }
            
            config = {
                'credentials': {
                    'usernames': credentials_dict
                },
                'cookie': {
                    'expiry_days': 180,
                    'key': 'your_cookie_key',
                    'name': 'auth_cookie'
                },
                'preauthorized': []
            }
            
            with open("config.yaml", "w") as file:
                yaml.dump(config, file)

        with open("config.yaml", "r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=SafeLoader)

        return stauth.Authenticate(
            credentials=config['credentials'],
            cookie_name=config['cookie']['name'],
            key=config['cookie']['key'],
            cookie_expiry_days=config['cookie']['expiry_days'],
            preauthorized=config.get('preauthorized', [])
        )
    except Exception as e:
        st.error(f"Authentication setup failed: {str(e)}")
        logging.error(f"Authentication setup failed: {str(e)}")
        st.stop()

authenticator = setup_authentication()
authenticator.login(location="sidebar", key="login")

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if authentication_status:
    authenticator.logout('Logout', 'sidebar', key='logout_button')

if not authentication_status:
    if authentication_status is False:
        st.error("❌ Username/password is incorrect")
    else:
        st.warning("⚠️ Please enter your username and password")
    st.stop()

# --------------------------------------------------
# Database Setup
# --------------------------------------------------
DB_DIR = "logs"
DB_PATH = os.path.join(DB_DIR, "logs.db")

def get_db_connection():
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        return sqlite3.connect(DB_PATH, check_same_thread=False)
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        logging.error(f"Database connection failed: {str(e)}")
        st.stop()

def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS progress_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            username TEXT,
            category TEXT,
            progress_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        c.execute("PRAGMA table_info(progress_logs)")
        columns = [col[1] for col in c.fetchall()]
        if 'username' not in columns:
            c.execute("ALTER TABLE progress_logs ADD COLUMN username TEXT")
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT,
            annotations_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS user_progress (
            username TEXT PRIMARY KEY,
            last_case_turing INTEGER DEFAULT 0,
            last_case_standard INTEGER DEFAULT 0,
            last_case_ai INTEGER DEFAULT 0
        )''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")
        logging.error(f"Database initialization failed: {str(e)}")
        st.stop()

init_db()

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def should_log(session_id: str, category: str, new_progress: dict) -> bool:
    try:
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
    except Exception as e:
        logging.error(f"should_log failed: {str(e)}")
        return True

def save_progress(category: str, progress: dict):
    try:
        sid = st.session_state.session_id
        if not should_log(sid, category, progress):
            return

        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO progress_logs(session_id, username, category, progress_json) VALUES (?, ?, ?, ?)",
            (sid, username, category, json.dumps(progress))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"save_progress failed: {str(e)}")

def save_annotations(case_id: str, annotations: list):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO annotations(case_id, annotations_json) VALUES (?, ?)",
            (case_id, json.dumps(annotations))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"save_annotations failed: {str(e)}")

def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def load_user_progress():
    try:
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
            st.session_state.last_case_turing  = row[0] if row[0] is not None else 0
            st.session_state.last_case_standard = row[1] if row[1] is not None else 0
            st.session_state.last_case_ai       = row[2] if row[2] is not None else 0
        else:
            st.session_state.last_case_turing  = 0
            st.session_state.last_case_standard = 0
            st.session_state.last_case_ai       = 0
    except Exception as e:
        logging.error(f"load_user_progress failed: {str(e)}")
        st.session_state.last_case_turing  = 0
        st.session_state.last_case_standard = 0
        st.session_state.last_case_ai       = 0

def save_user_progress():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO user_progress (username, last_case_turing, last_case_standard, last_case_ai) "
            "VALUES (?, ?, ?, ?)",
            (username,
             st.session_state.last_case_turing,
             st.session_state.last_case_standard,
             st.session_state.last_case_ai)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"save_user_progress failed: {str(e)}")

# --------------------------------------------------
# Session Initialization
# --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.markdown(f"**User:** {name}")

if 'progress_loaded' not in st.session_state:
    load_user_progress()
    st.session_state.progress_loaded = True

# Initialize all session state variables
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

init_state("image_cache", {})

# --------------------------------------------------
# Routing
# --------------------------------------------------
params = st.experimental_get_query_params()
st.session_state.page = params.get("page", ["index"])[0]

# --------------------------------------------------
# Image Handling
# --------------------------------------------------
def find_case_folder(folder_name):
    """Find the folder containing images for a given folder name"""
    try:
        search_dirs = ["sampled_normal", "sampled_abnormal"]
        for base_dir in search_dirs:
            # Check if base directory exists
            if not os.path.isdir(base_dir):
                logging.warning(f"Search directory not found: {os.path.abspath(base_dir)}")
                continue

            # Construct path to the specific case folder
            case_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(case_path):
                logging.info(f"Found matching folder: {case_path}")
                return case_path
                
        logging.warning(f"No folder found for: {folder_name} in directories {search_dirs}")
        return None
    except Exception as e:
        logging.error(f"find_case_folder failed: {str(e)}")
        return None

def get_images_from_folder(folder_path):
    """Get sorted image paths from a folder with detailed logging"""
    if not os.path.exists(folder_path):
        logging.warning(f"Image folder not found: {folder_path}")
        return []
    
    images = []
    try:
        for f in sorted(os.listdir(folder_path)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder_path, f)
                images.append(img_path)
        
        logging.info(f"Found {len(images)} images in {folder_path}")
        return images
    except Exception as e:
        logging.error(f"Error loading images from {folder_path}: {str(e)}")
        return []

def display_carousel(category, case_id):
    """
    Displays a side-by-side image carousel for lung and soft tissue scans
    with a slider and navigation buttons.
    """
    # Key for storing the current slice index for a given category (turing, standard, ai)
    slice_state_key = f"current_slice_{category}"

    try:
        folder_name = report_dict.get(case_id, {}).get("folder", "")
        if not folder_name:
            st.info("No folder mapping for this case.")
            logging.warning(f"No folder mapping for case: {case_id}")
            return
            
        cache_key = f"{case_id}_{category}"
        if cache_key in st.session_state.image_cache:
            lung_imgs, soft_imgs = st.session_state.image_cache[cache_key]
        else:
            case_folder = find_case_folder(folder_name)
            if not case_folder:
                st.warning(f"Image folder '{folder_name}' not found. Check that the 'sampled_abnormal' and 'sampled_normal' directories are present.")
                logging.warning(f"Case folder not found for: {folder_name}")
                return
            
            lung_folder = os.path.join(case_folder, "lung")
            soft_folder = os.path.join(case_folder, "soft")
            lung_imgs = get_images_from_folder(lung_folder)
            soft_imgs = get_images_from_folder(soft_folder)
            st.session_state.image_cache[cache_key] = (lung_imgs, soft_imgs)
        
        num_slices = max(len(lung_imgs), len(soft_imgs))
        if num_slices == 0:
            st.info("No images are available for this case.")
            return

        idx = st.session_state.get(slice_state_key, 0)
        if idx >= num_slices: # Reset if index is out of bounds for the new case
            idx = 0
            st.session_state[slice_state_key] = 0

        # --- UI for Image Navigation ---
        new_idx = st.slider(
            f"Slice ({idx + 1}/{num_slices})",
            min_value=0,
            max_value=num_slices - 1,
            value=idx,
            key=f"slider_{category}_{case_id}"
        )
        if new_idx != idx:
            st.session_state[slice_state_key] = new_idx
            st.rerun()

        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("⬅️ Left", key=f"left_{category}_{case_id}", use_container_width=True):
                if st.session_state[slice_state_key] > 0:
                    st.session_state[slice_state_key] -= 1
                    st.rerun()
        with col2:
            if st.button("Right ➡️", key=f"right_{category}_{case_id}", use_container_width=True):
                if st.session_state[slice_state_key] < num_slices - 1:
                    st.session_state[slice_state_key] += 1
                    st.rerun()
        
        st.markdown("---")
        
        # --- Display Images ---
        current_index = st.session_state[slice_state_key]
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Lung")
            if lung_imgs:
                img_index = min(current_index, len(lung_imgs) - 1)
                st.image(lung_imgs[img_index], use_container_width=True)
            else:
                st.info("No lung images available.")

        with c2:
            st.caption("Soft Tissue")
            if soft_imgs:
                img_index = min(current_index, len(soft_imgs) - 1)
                st.image(soft_imgs[img_index], use_container_width=True)
            else:
                st.info("No soft-tissue images available.")

    except Exception as e:
        st.error(f"Could not display images: {e}")
        logging.error(f"display_carousel failed for case {case_id}: {e}", exc_info=True)

# --------------------------------------------------
# Page Definitions
# --------------------------------------------------
def index():
    st.title("Survey App")
    if total_cases == 0:
        return
    st.markdown("### Your Progress")
    st.markdown(f"- **Turing Test**: Case {st.session_state.last_case_turing + 1}/{total_cases}")
    st.markdown(f"- **Standard Eval**: Case {st.session_state.last_case_standard + 1}/{total_cases}")
    st.markdown(f"- **AI Edit**: Case {st.session_state.last_case_ai + 1}/{total_cases}")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Turing Test"):
        st.session_state.page = "turing_test"
        st.experimental_set_query_params(page="turing_test")
        st.rerun()
    if col2.button("Standard Eval"):
        st.session_state.page = "standard_eval"
        st.experimental_set_query_params(page="standard_eval")
        st.rerun()
    if col3.button("AI Edit"):
        st.session_state.page = "ai_edit"
        st.experimental_set_query_params(page="ai_edit")
        st.rerun()
    if col4.button("View Results"):
        st.session_state.page = "view_results"
        st.experimental_set_query_params(page="view_results")
        st.rerun()

def turing_test():
    try:
        idx = st.session_state.last_case_turing
        if idx >= total_cases:
            st.success("Turing Test complete!")
            if st.button("Home"):
                st.session_state.page = "index"
                st.experimental_set_query_params(page="index")
                st.rerun()
            return
        
        case = cases[idx]
        st.header(f"Turing Test: Case {idx + 1}/{total_cases}")
        st.info("""
        You will evaluate two radiology reports, Report A and Report B. 
        One is a clinician's interpretation (ground truth), the other is AI-generated.
        Your task is to determine which report is the human-generated one.
        """)

        if st.button("Save & Back"):
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()

        reports = report_dict.get(case, {})
        gt_report = reports.get("gt", "Ground-truth report not found.")
        gen_report = reports.get("gen", "Generated report not found.")

        assigns = st.session_state.assignments_turing
        if case not in assigns:
            assigns[case] = random.choice([True, False])
            st.session_state.assignments_turing = assigns
        
        if assigns[case]:
            A, B = gen_report, gt_report
        else:
            A, B = gt_report, gen_report

        st.subheader("Report A")
        st.text_area("A", A, height=300, key=f"A_t_{case}", label_visibility="collapsed")
        st.subheader("Report B")
        st.text_area("B", B, height=300, key=f"B_t_{case}", label_visibility="collapsed")

        if st.session_state.initial_eval_turing is None:
            choice = st.radio("Which report is human-generated?", ["A", "B", "Not sure"], index=2, key=f"ch_t_{case}")
            if st.button("Submit initial evaluation"):
                st.session_state.initial_eval_turing = choice
                st.session_state.viewed_images_turing = True
                st.rerun()

        if st.session_state.viewed_images_turing:
            st.markdown("#### Images")
            display_carousel("turing", case)
            st.markdown(f"**Initial Choice:** {st.session_state.initial_eval_turing}")
            
            final_choice = st.radio(
                "Review after viewing images:",
                ["Keep choice", "Change to A", "Change to B", "Change to Not sure"],
                key=f"final_ch_t_{case}"
            )
            
            if final_choice == "Change to A":
                final = "A"
            elif final_choice == "Change to B":
                final = "B"
            elif final_choice == "Change to Not sure":
                final = "Not sure"
            else:
                final = st.session_state.initial_eval_turing
                
            st.session_state.final_eval_turing = final

            if st.button("Finalize & Next"):
                save_progress("turing_test", {
                    "case_id": case,
                    "assignments": assigns[case],
                    "initial_eval": st.session_state.initial_eval_turing,
                    "final_eval": final
                })
                st.session_state.last_case_turing += 1
                save_user_progress()
                # Reset for next case
                st.session_state.initial_eval_turing = None
                st.session_state.final_eval_turing = None
                st.session_state.viewed_images_turing = False
                st.session_state.current_slice_turing = 0
                st.rerun()
    except Exception as e:
        st.error(f"Error in Turing Test: {str(e)}")
        logging.error(f"Turing Test error: {str(e)}")

def evaluate_case():
    try:
        idx = st.session_state.last_case_standard
        if idx >= total_cases:
            st.success("Standard Eval complete!")
            if st.button("Home"):
                st.session_state.page = "index"
                st.experimental_set_query_params(page="index")
                st.rerun()
            return
        
        case = cases[idx]
        st.header(f"Standard Eval: Case {idx + 1}/{total_cases}")

        if st.button("Save & Back"):
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()

        reports = report_dict.get(case, {})
        gt_report = reports.get("gt", "Ground-truth report not found.")
        gen_report = reports.get("gen", "Generated report not found.")

        assigns = st.session_state.assignments_standard
        if case not in assigns:
            assigns[case] = random.choice([True, False])
            st.session_state.assignments_standard = assigns
        
        if assigns[case]:
            report_label_A, report_label_B = "AI-Generated", "Ground-truth"
            A, B = gen_report, gt_report
        else:
            report_label_A, report_label_B = "Ground-truth", "AI-Generated"
            A, B = gt_report, gen_report

        st.subheader(f"Report A ({report_label_A})")
        st.text_area("A", A, height=200, key=f"A_s_{case}", label_visibility="collapsed")
        st.subheader(f"Report B ({report_label_B})")
        st.text_area("B", B, height=200, key=f"B_s_{case}", label_visibility="collapsed")

        st.markdown("#### Images")
        display_carousel("standard", case)

        st.markdown("#### Corrections")
        organ = st.selectbox("Organ", ["", "LIVER", "PANCREAS", "KIDNEY", "OTHER"], key=f"org_s_{case}")
        reason = st.text_input("Reason", key=f"rsn_s_{case}")
        details = st.text_area("Details", key=f"dtl_s_{case}")
        
        if st.button("Add Correction") and organ:
            st.session_state.corrections_standard.append({
                "case_id": case, "organ": organ, "reason": reason, "details": details
            })
            st.rerun()

        cors = [c for c in st.session_state.corrections_standard if c["case_id"] == case]
        if cors:
            st.table(pd.DataFrame(cors).drop(columns=["case_id"]))

        choice = st.radio("Best report?", ["A", "B", "Corrected", "Equal"], key=f"ch_s_{case}")
        
        if st.button("Submit & Next"):
            if cors:
                save_annotations(case, cors)
            save_progress("standard_evaluation", {
                "case_id": case,
                "assignments": assigns[case],
                "evaluation_choice": choice,
                "corrections_made": cors
            })
            st.session_state.corrections_standard = [
                c for c in st.session_state.corrections_standard if c["case_id"] != case
            ]
            st.session_state.last_case_standard += 1
            save_user_progress()
            st.session_state.current_slice_standard = 0
            st.rerun()
    except Exception as e:
        st.error(f"Error in Standard Eval: {str(e)}")
        logging.error(f"Standard Eval error: {str(e)}")

def ai_edit():
    try:
        idx = st.session_state.last_case_ai
        if idx >= total_cases:
            st.success("AI Edit complete!")
            if st.button("Home"):
                st.session_state.page = "index"
                st.experimental_set_query_params(page="index")
                st.rerun()
            return
        
        case = cases[idx]
        st.header(f"AI Edit: Case {idx + 1}/{total_cases}")

        if st.button("Save & Back"):
            save_progress("ai_edit", {
                "case_id": case,
                "mode": st.session_state.get("last_mode_ai", "Free"),
                "assembled_report": st.session_state.assembled_ai,
                "corrections_applied": st.session_state.corrections_ai
            })
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()

        orig = report_dict.get(case, {}).get("gen", "Original AI report not found.")
        st.subheader("Original AI Report")
        st.text_area("Original", orig, height=200, disabled=True, label_visibility="collapsed")
        st.markdown("#### Images")
        display_carousel("ai", case)

        mode = st.radio("Mode", ["Free", "Organ"], key=f"md_ai_{case}")
        st.session_state["last_mode_ai"] = mode

        if mode == "Free":
            text = st.session_state.assembled_ai or orig
            new = st.text_area("Edit", text, height=300, key=f"free_ai_{case}", label_visibility="collapsed")
            st.session_state.assembled_ai = new
        else:
            organ = st.selectbox("Organ", ["", "LIVER", "PANCREAS", "KIDNEY", "OTHER"], key=f"org_ai_{case}")
            reason = st.text_input("Reason", key=f"rsn_ai_{case}")
            details = st.text_area("Details", key=f"dtl_ai_{case}")
            
            if st.button("Add Correction") and organ:
                st.session_state.corrections_ai.append({
                    "case_id": case, "organ": organ, "reason": reason, "details": details
                })
                st.rerun()

            cors = [c for c in st.session_state.corrections_ai if c["case_id"] == case]
            if cors:
                st.table(pd.DataFrame(cors).drop(columns=["case_id"]))
                if st.button("Assemble Report"):
                    txt = "\n".join(f"- {c['organ']}: {c['reason']} — {c['details']}" for c in cors)
                    st.session_state.assembled_ai = txt
                    st.rerun()

        if st.button("Submit & Next"):
            save_progress("ai_edit", {
                "case_id": case,
                "mode": mode,
                "final_report_content": st.session_state.assembled_ai,
                "corrections_applied": st.session_state.corrections_ai
            })
            st.session_state.corrections_ai = [c for c in st.session_state.corrections_ai if c["case_id"] != case]
            st.session_state.assembled_ai = ""
            st.session_state.last_case_ai += 1
            save_user_progress()
            st.session_state.current_slice_ai = 0
            st.rerun()
    except Exception as e:
        st.error(f"Error in AI Edit: {str(e)}")
        logging.error(f"AI Edit error: {str(e)}")

def view_all_results():
    try:
        st.title("All Saved Results")
        if st.button("Home"):
            st.session_state.page = "index"
            st.experimental_set_query_params(page="index")
            st.rerun()

        conn = get_db_connection()
        
        st.subheader("Sessions")
        sessions_df = pd.read_sql("SELECT DISTINCT session_id, username FROM progress_logs", conn)
        st.dataframe(sessions_df)
        
        st.subheader("Progress Logs")
        progress_df = pd.read_sql("SELECT * FROM progress_logs", conn)
        st.dataframe(progress_df)
        
        st.subheader("Annotations")
        annotations_df = pd.read_sql("SELECT * FROM annotations", conn)
        st.dataframe(annotations_df)
        
        st.subheader("User Progress")
        user_progress_df = pd.read_sql("SELECT * FROM user_progress", conn)
        st.dataframe(user_progress_df)
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        logging.error(f"View Results error: {str(e)}")

# --------------------------------------------------
# Main App Router
# --------------------------------------------------
try:
    if st.session_state.page == "turing_test":
        turing_test()
    elif st.session_state.page == "standard_eval":
        evaluate_case()
    elif st.session_state.page == "ai_edit":
        ai_edit()
    elif st.session_state.page == "view_results":
        view_all_results()
    else:
        index()
except Exception as e:
    st.error(f"A critical error occurred: {str(e)}")
    logging.error(f"Critical error in router: {str(e)}")
