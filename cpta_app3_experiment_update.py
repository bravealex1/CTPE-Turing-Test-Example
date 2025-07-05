import streamlit as st
import os
import json
import uuid
import random
import pandas as pd
import sqlite3
import logging
from datetime import datetime

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

# --------------------------------------------------
# Configure logging
# --------------------------------------------------
logging.basicConfig(filename='app_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------
# Load Reports from CSV (Combined)
# --------------------------------------------------
CTPA_CSV = "CTPA_list_30_remove15_23.csv"

def load_reports():
    try:
        if os.path.exists(CTPA_CSV):
            df_reports_csv = pd.read_csv(CTPA_CSV)
            report_dict = {
                str(row["id"]): {
                    "gt": row["gt"],
                    "gen": row["parsed_output"]
                }
                for _, row in df_reports_csv.iterrows()
            }
            cases = sorted([str(case_id) for case_id in df_reports_csv["id"].unique()])
            total_cases = len(cases)
            return report_dict, cases, total_cases
        else:
            st.error(f"Error: The file {CTPA_CSV} was not found. Please ensure it's uploaded.")
            return {}, [], 0
    except Exception as e:
        st.error(f"Failed to load reports: {str(e)}")
        logging.error(f"Failed to load reports: {str(e)}")
        return {}, [], 0

report_dict, cases, total_cases = load_reports()

# --------------------------------------------------
# 0. Authentication Setup
# --------------------------------------------------
def setup_authentication():
    try:
        if not os.path.exists("config.yaml"):
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

        with open("config.yaml", "r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=SafeLoader)

        return stauth.Authenticate(
            credentials          = config['credentials'],
            cookie_name          = config['cookie']['name'],
            key                  = config['cookie']['key'],
            cookie_expiry_days   = config['cookie']['expiry_days'],
            preauthorized        = config.get('preauthorized', [])
        )
    except Exception as e:
        st.error(f"Authentication setup failed: {str(e)}")
        logging.error(f"Authentication setup failed: {str(e)}")
        st.stop()

authenticator = setup_authentication()
authenticator.login(location="sidebar", key="login")

name                  = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username              = st.session_state.get("username")

if authentication_status:
    authenticator.logout('Logout', 'sidebar', key='logout_button')

if not authentication_status:
    if authentication_status is False:
        st.error("❌ Username/password is incorrect")
    else:
        st.warning("⚠️ Please enter your username and password")
    st.stop()

# --------------------------------------------------
# Database Setup - SQLite ONLY
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
        
        # Create tables if they don't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS progress_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            username TEXT,
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
init_state("final_eval_turing",    None)
init_state("viewed_images_turing", False)

init_state("current_slice_standard", 0)
init_state("assignments_standard",  {})
init_state("corrections_standard",  [])

init_state("current_slice_ai", 0)
init_state("corrections_ai",  [])
init_state("assembled_ai",    "")

# --------------------------------------------------
# Routing
# --------------------------------------------------
params = st.experimental_get_query_params()
st.session_state.page = params.get("page", ["index"])[0]

# --------------------------------------------------
# Image Handling
# --------------------------------------------------
IMAGE_BASE_DIR = "images_for_ctpa"

def display_carousel(category, case_id):
    key = f"current_slice_{category}_{case_id}"
    
    try:
        lung_folder = os.path.join(IMAGE_BASE_DIR, case_id, "lung")
        soft_folder = os.path.join(IMAGE_BASE_DIR, case_id, "soft")

        lung_imgs = []
        soft_imgs = []
        
        if os.path.exists(lung_folder):
            lung_imgs = sorted([
                os.path.join(lung_folder, f)
                for f in os.listdir(lung_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
        
        if os.path.exists(soft_folder):
            soft_imgs = sorted([
                os.path.join(soft_folder, f)
                for f in os.listdir(soft_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

        # Calculate max slices between both image sets
        max_slices = max(len(lung_imgs), len(soft_imgs))
        
        if max_slices == 0:
            st.info("No images available for this case.")
            return
            
        # Only show slider if there are multiple slices
        if max_slices > 1:
            idx = st.slider(
                "Slice index",
                0,
                max_slices - 1,
                st.session_state.get(key, 0),
                key=key
            )
        else:
            idx = 0

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
    except Exception as e:
        st.warning(f"Could not load images: {str(e)}")
        logging.error(f"display_carousel failed: {str(e)}")

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
            # Report A is generated, Report B is ground truth
            A, B = gen_report, gt_report
        else:
            # Report A is ground truth, Report B is generated
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
        st.header(f"Standard Eval: {case} ({idx + 1}/{total_cases})")

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
        st.header(f"AI Edit: {case} ({idx + 1}/{total_cases})")

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
