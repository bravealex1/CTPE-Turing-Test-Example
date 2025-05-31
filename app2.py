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
# Load Reports from CSV (Normal & Abnormal)
# --------------------------------------------------
# Paths to the CSVs
NORMAL_CSV = r"normal_top_15.csv"
ABNORMAL_CSV = r"abnormal_top_15.csv"

# Read both CSVs and combine
if os.path.exists(NORMAL_CSV) and os.path.exists(ABNORMAL_CSV):
    df_normal = pd.read_csv(NORMAL_CSV)
    df_abnormal = pd.read_csv(ABNORMAL_CSV)
    df_reports = pd.concat([df_normal, df_abnormal], ignore_index=True)
    # Build a dictionary keyed by case ID
    report_dict = {
        str(row["id"]): {
            "gt": row["gt"],
            "gen": row["generated_output"]
        }
        for _, row in df_reports.iterrows()
    }
else:
    st.warning("One or both report CSV files (normal_top_15.csv, abnormal_top_15.csv) not found. Report loading will be affected.")
    report_dict = {}  # fallback if CSVs are missing

# --------------------------------------------------
# 0. Authentication Setup (must be first)
# --------------------------------------------------
try:
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("❌ Config.yaml not found. Please create it with user credentials.")
    st.stop()


if "credentials" not in config or "usernames" not in config["credentials"]:
    st.error("❌ Your config.yaml must include a 'credentials → usernames' section.")
    st.stop()

# Hash passwords in the config dictionary if they are not already hashed
# This is a one-time operation if you store plain passwords in YAML and let the script hash them.
# For production, you might pre-hash them.
config["credentials"]["usernames"] = Hasher.hash_passwords(config["credentials"]["usernames"]) # Hasher now returns a tuple (bool, dict)
if hashed_creds[0]: # Check if hashing was successful / necessary
    config["credentials"] = hashed_creds[1]
else: # If passwords were already hashed, Hasher.hash_passwords returns (False, original_credentials)
    config["credentials"] = hashed_creds[1]


authenticator = stauth.Authenticate(
    credentials        = config["credentials"],
    cookie_name        = config["cookie"]["name"],
    key                = config["cookie"]["key"],
    cookie_expiry_days = config["cookie"]["expiry_days"],
    preauthorized      = config.get("preauthorized", [])
)

# Render login on the main page or sidebar
# authenticator.login() # Renders on main page
# OR for sidebar:
name, authentication_status, username = authenticator.login(location="main")


if authentication_status is False:
    st.error("❌ Username/password is incorrect")
    st.stop()
elif authentication_status is None:
    st.warning("⚠️ Please enter your username and password")
    st.stop()

# If authenticated, show welcome message and logout button in sidebar
st.sidebar.title(f"Welcome {name}")
st.sidebar.markdown(f"**Username:** `{username}`")
authenticator.logout("Logout", "sidebar", key="logout_button")


# --------------------------------------------------
# 0. Database Setup for Queryable Logs
# --------------------------------------------------
DB_DIR  = "user_data" # Changed from "logs" to "user_data" to be more descriptive
DB_PATH = os.path.join(DB_DIR, "progress_data.db") # Changed db name

def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Added username column
    c.execute('''
    CREATE TABLE IF NOT EXISTS progress_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT NOT NULL,
      session_id TEXT,
      category TEXT,
      progress_json TEXT,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    # Added username column
    c.execute('''
    CREATE TABLE IF NOT EXISTS annotations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT NOT NULL,
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
def should_log(username_to_check: str, category: str, new_progress: dict) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT progress_json FROM progress_logs "
        "WHERE username=? AND category=? " # Use username
        "ORDER BY timestamp DESC LIMIT 1",
        (username_to_check, category)
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return True
    last = json.loads(row[0])
    # Check if the content of the progress is meaningfully different
    # This is a simple check; more sophisticated diffing might be needed for complex states
    if "last_case" in new_progress and "last_case" in last:
         if last.get("last_case") != new_progress.get("last_case"):
             return True
         # If last_case is the same, check other relevant fields for changes
         if category == "turing_test":
             return (last.get("initial_eval") != new_progress.get("initial_eval") or
                     last.get("final_eval") != new_progress.get("final_eval"))
         if category == "standard_evaluation": # Check based on actual saved keys
             return last.get("corrections") != new_progress.get("corrections") # Assuming corrections is a list
         if category == "ai_edit":
             return (last.get("assembled") != new_progress.get("assembled") or
                     last.get("corrections") != new_progress.get("corrections")) # Assuming corrections is a list
         return False # If last_case is same and no other relevant diff, don't log

    if category == "ai_edit" and "case_id" in new_progress and "case_id" in last: # For AI Edit save & back
        if last.get("case_id") != new_progress.get("case_id"):
            return True
        return (last.get("assembled") != new_progress.get("assembled") or
                last.get("corrections") != new_progress.get("corrections"))

    return True # Default to log if conditions above aren't met

# --------------------------------------------------
# 1. Generate & Store Unique Session ID (still useful for session-level tracking if needed)
# --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.sidebar.markdown(f"**Session ID (for debugging):** `{st.session_state.session_id}`")

# --------------------------------------------------
# Utility: Save Progress per Category & User
# --------------------------------------------------
def save_progress(category: str, progress: dict):
    current_username = st.session_state.username
    sid = st.session_state.session_id

    if not should_log(current_username, category, progress):
        # st.sidebar.warning(f"No changes detected in {category} for case {progress.get('case_id', 'N/A')}. Not saving duplicate.")
        return

    user_log_dir = os.path.join(DB_DIR, current_username, "logs")
    os.makedirs(user_log_dir, exist_ok=True)

    # JSON (optional, as DB is primary)
    jpath = os.path.join(user_log_dir, f"{category}_progress.json")
    # For JSON, you might want to store a list of all progress steps or overwrite with the latest
    # Current implementation appends, which might grow large.
    # Consider if this detailed file logging is necessary alongside the database.
    # For simplicity, let's save the current progress state, overwriting previous file for this user/category.
    # Or, append with timestamps if history is needed in JSON.
    # For now, let's keep appending as per original logic, but user-specific.
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                data = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append({"timestamp": datetime.now().isoformat(), **progress})
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # CSV (optional) - Similar consideration for appending vs. overwriting
    cpath = os.path.join(user_log_dir, f"{category}_progress.csv")
    df_new_row = pd.DataFrame([{"timestamp": datetime.now().isoformat(), **progress}])
    if os.path.exists(cpath):
        df_new_row.to_csv(cpath, index=False, mode="a", header=False)
    else:
        df_new_row.to_csv(cpath, index=False)

    # SQLite (Primary persistent storage)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO progress_logs(username, session_id, category, progress_json) VALUES (?, ?, ?, ?)",
        (current_username, sid, category, json.dumps(progress))
    )
    conn.commit()
    conn.close()
    st.sidebar.success(f"Progress for {category} saved!")


# --------------------------------------------------
# Utility: Save Annotations per Case & User
# --------------------------------------------------
def save_annotations(case_id: str, annotations: list):
    current_username = st.session_state.username
    user_eval_dir = os.path.join(DB_DIR, current_username, "evaluations")
    os.makedirs(user_eval_dir, exist_ok=True)

    # JSON
    path = os.path.join(user_eval_dir, f"{case_id}_annotations.json")
    # This will overwrite previous annotations for THIS case by THIS user if file exists.
    # If you need to append, adjust logic similar to save_progress.
    # For this function, saving the current set of annotations for the case seems appropriate.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    # SQLite
    conn = get_db_connection()
    c = conn.cursor()
    # Delete old annotations for this user and case_id to avoid duplicates if re-submitting
    c.execute(
        "DELETE FROM annotations WHERE username=? AND case_id=?",
        (current_username, case_id)
    )
    # Insert new ones
    c.execute(
        "INSERT INTO annotations(username, case_id, annotations_json) VALUES (?, ?, ?)",
        (current_username, case_id, json.dumps(annotations))
    )
    conn.commit()
    conn.close()
    st.sidebar.success(f"Annotations for case {case_id} saved!")

# --------------------------------------------------
# Initialize or Load User-Specific Session State
# --------------------------------------------------
def init_or_load_user_state(current_username: str):
    # Define default states
    default_states = {
        "last_case_turing": 0, "current_slice_turing": 0, "assignments_turing": {},
        "initial_eval_turing": None, "final_eval_turing": None, "viewed_images_turing": False,
        "last_case_standard": 0, "current_slice_standard": 0, "assignments_standard": {},
        "corrections_standard": [],
        "last_case_ai": 0, "current_slice_ai": 0, "corrections_ai": [], "assembled_ai": "",
        "last_mode_ai": "Free" # Default mode for AI Edit
    }

    # Initialize with defaults
    for key, default_val in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

    # Attempt to load saved progress from DB
    conn = get_db_connection()
    c = conn.cursor()

    categories_config = {
        "turing_test": ["last_case_turing", "assignments_turing", "initial_eval_turing", "final_eval_turing", "viewed_images_turing"],
        "standard_evaluation": ["last_case_standard", "assignments_standard", "corrections_standard"],
        "ai_edit": ["last_case_ai", "corrections_ai", "assembled_ai", "last_mode_ai"]
    }

    loaded_something = False
    for category, state_keys_to_load in categories_config.items():
        c.execute(
            "SELECT progress_json FROM progress_logs "
            "WHERE username=? AND category=? "
            "ORDER BY timestamp DESC LIMIT 1",
            (current_username, category)
        )
        row = c.fetchone()
        if row:
            try:
                progress_data = json.loads(row[0])
                for key in state_keys_to_load:
                    if key in progress_data:
                        st.session_state[key] = progress_data[key]
                        loaded_something = True
                # Ensure current_slice is reset for the loaded case
                if f"current_slice_{category.split('_')[0]}" in st.session_state: # e.g. current_slice_turing
                     st.session_state[f"current_slice_{category.split('_')[0]}"] = 0

            except json.JSONDecodeError:
                st.sidebar.warning(f"Could not decode progress for {category}. Using defaults.")

    conn.close()
    if loaded_something:
        st.sidebar.success(f"Resumed progress for user {current_username}.")
    else:
        st.sidebar.info(f"No prior saved progress found for {current_username}. Starting fresh.")


# Call after successful authentication
init_or_load_user_state(username)


# --------------------------------------------------
# Routing Setup
# --------------------------------------------------
# st.experimental_get_query_params is deprecated. Use st.query_params
params = st.query_params
if "page" in params:
    st.session_state.page = params["page"][0]
elif "page" not in st.session_state:
    st.session_state.page = "index"

# ── ADAPTED: point to the folders containing normal & abnormal cases ──
NORMAL_IMAGE_DIR   = r"sampled_normal"
ABNORMAL_IMAGE_DIR = r"sampled_abnormal"

# Get list of case IDs from both directories
cases_normal   = []
cases_abnormal = []
if os.path.exists(NORMAL_IMAGE_DIR):
    cases_normal   = sorted([d for d in os.listdir(NORMAL_IMAGE_DIR)   if os.path.isdir(os.path.join(NORMAL_IMAGE_DIR, d))])
else:
    st.warning(f"Directory not found: {NORMAL_IMAGE_DIR}")

if os.path.exists(ABNORMAL_IMAGE_DIR):
    cases_abnormal = sorted([d for d in os.listdir(ABNORMAL_IMAGE_DIR) if os.path.isdir(os.path.join(ABNORMAL_IMAGE_DIR, d))])
else:
    st.warning(f"Directory not found: {ABNORMAL_IMAGE_DIR}")

cases = sorted(list(set(cases_normal + cases_abnormal))) # Use set to avoid duplicates if case IDs overlap by mistake
total_cases = len(cases)
if total_cases == 0:
    st.error("No cases found in specified image directories. Please check paths and content.")
    # st.stop() # Uncomment if you want the app to halt if no cases

# --------------------------------------------------
# Helpers for Text & Carousel (adjusted folder names)
# --------------------------------------------------
def load_text(path):
    try:
        return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""
    except Exception as e:
        st.warning(f"Could not load text from {path}: {e}")
        return ""

def display_carousel(category_key_suffix, case_id): # Changed 'category' to 'category_key_suffix' to avoid conflict
    key = f"current_slice_{category_key_suffix}" # e.g. current_slice_turing

    # Determine which base folder this case lives in
    base_dir = None
    potential_normal_path = os.path.join(NORMAL_IMAGE_DIR, case_id)
    potential_abnormal_path = os.path.join(ABNORMAL_IMAGE_DIR, case_id)

    if os.path.exists(potential_normal_path):
        base_dir = NORMAL_IMAGE_DIR
    elif os.path.exists(potential_abnormal_path):
        base_dir = ABNORMAL_IMAGE_DIR
    else:
        st.error(f"Case ID {case_id} not found in normal or abnormal image directories.")
        return

    bone_folder = os.path.join(base_dir, case_id, "bone")
    lung_folder = os.path.join(base_dir, case_id, "lung")
    soft_folder = os.path.join(base_dir, case_id, "soft")

    bone_imgs = sorted([
        os.path.join(bone_folder, f)
        for f in os.listdir(bone_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]) if os.path.exists(bone_folder) else []
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

    if not any([bone_imgs, lung_imgs, soft_imgs]):
        st.info(f"No images found for case {case_id} in any window (bone, lung, soft).")
        # return # Optionally return if no images at all

    max_slices = max(len(bone_imgs), len(lung_imgs), len(soft_imgs), 1) # Ensure max_slices is at least 1
    idx = st.session_state.get(key, 0) # Get current slice index for this category
    idx = max(0, min(idx, max_slices - 1)) # Clamp index to valid range
    st.session_state[key] = idx

    c_prev, c1, c2, c3, c_next = st.columns([1, 3, 3, 3, 1])
    with c_prev:
        if st.button("⟨ Prev", key=f"prev_{category_key_suffix}_{case_id}"):
            if idx > 0:
                st.session_state[key] = idx - 1
                st.rerun()
    with c1:
        st.subheader("Bone")
        if bone_imgs and idx < len(bone_imgs):
            st.image(bone_imgs[idx], caption=f"Bone Slice {idx+1}/{len(bone_imgs)}", use_column_width=True)
        else:
            st.info("No bone images for this slice or case.")
    with c2:
        st.subheader("Lung")
        if lung_imgs and idx < len(lung_imgs):
            st.image(lung_imgs[idx], caption=f"Lung Slice {idx+1}/{len(lung_imgs)}", use_column_width=True)
        else:
            st.info("No lung images for this slice or case.")
    with c3:
        st.subheader("Soft Tissue")
        if soft_imgs and idx < len(soft_imgs):
            st.image(soft_imgs[idx], caption=f"Soft Tissue Slice {idx+1}/{len(soft_imgs)}", use_column_width=True)
        else:
            st.info("No soft‐tissue images for this slice or case.")
    with c_next:
        if st.button("Next ⟩", key=f"next_{category_key_suffix}_{case_id}"):
            if idx < max_slices - 1:
                st.session_state[key] = idx + 1
                st.rerun()
    st.caption(f"Showing slice {st.session_state[key] + 1} of {max_slices}")


# --------------------------------------------------
# 8. Pages
# --------------------------------------------------
def set_page(page_name):
    st.query_params["page"] = page_name # st.experimental_set_query_params is deprecated
    st.session_state.page = page_name
    # st.rerun() # Usually rerun is called after this by button logic

def index():
    st.title("Survey App")
    if total_cases == 0:
        st.error("No cases found. Please check application setup.")
        return

    # Display progress relative to total_cases, ensuring it doesn't exceed total_cases
    progress_turing = min(st.session_state.last_case_turing, total_cases)
    progress_standard = min(st.session_state.last_case_standard, total_cases)
    progress_ai = min(st.session_state.last_case_ai, total_cases)

    st.markdown("### Your Progress")
    st.markdown(f"- **Turing Test**: Case {progress_turing}/{total_cases} completed.")
    st.markdown(f"- **Standard Eval**: Case {progress_standard}/{total_cases} completed.")
    st.markdown(f"- **AI Edit**: Case {progress_ai}/{total_cases} completed.")
    st.markdown("---")

    cols = st.columns(4)
    if cols[0].button("Turing Test", key="btn_turing"):
        set_page("turing_test")
        st.rerun()
    if cols[1].button("Standard Eval", key="btn_standard"):
        set_page("standard_eval")
        st.rerun()
    if cols[2].button("AI Edit", key="btn_ai_edit"):
        set_page("ai_edit")
        st.rerun()
    if cols[3].button("View My Results", key="btn_my_results"): # Changed from "View All Results"
        set_page("view_my_results")
        st.rerun()

def turing_test():
    idx = st.session_state.last_case_turing
    if total_cases == 0: # Handle case where no images are loaded
        st.warning("No cases available for Turing Test.")
        if st.button("Home", key="home_turing_no_cases"):
            set_page("index")
            st.rerun()
        return

    if idx >= total_cases:
        st.success("Turing Test complete! Thank you.")
        if st.button("Home", key="home_turing_complete"):
            set_page("index")
            st.rerun()
        return

    case = cases[idx]
    st.header(f"Turing Test: Case {case} ({idx + 1}/{total_cases})")

    if st.button("Save & Back to Home", key="back_turing"):
        # Save current state before going back
        prog = {
            "last_case_turing": idx, # current case index being viewed, not yet completed
            "case_id": case,
            "assignments_turing": st.session_state.assignments_turing,
            "initial_eval_turing": st.session_state.initial_eval_turing,
            "final_eval_turing": st.session_state.final_eval_turing,
            "viewed_images_turing": st.session_state.viewed_images_turing
        }
        save_progress("turing_test", prog)
        set_page("index")
        st.rerun()

    # Load reports
    case_reports = report_dict.get(str(case), {})
    gt_report  = case_reports.get("gt", f"Ground truth report for case {case} not found.")
    gen_report = case_reports.get("gen", f"Generated report for case {case} not found.")


    # Ensure assignment is made for the current case
    if case not in st.session_state.assignments_turing:
        st.session_state.assignments_turing[case] = random.choice([True, False])

    is_gen_A = st.session_state.assignments_turing[case]
    report_A_text = gen_report if is_gen_A else gt_report
    report_B_text = gt_report if is_gen_A else gen_report

    st.subheader("Report A")
    st.text_area("Report A Text", report_A_text, height=200, key=f"A_t_{case}", disabled=True)
    st.subheader("Report B")
    st.text_area("Report B Text", report_B_text, height=200, key=f"B_t_{case}", disabled=True)

    if not st.session_state.viewed_images_turing: # Only show initial eval if images not yet viewed for this case
        initial_choice_options = ["A is Human", "B is Human", "Not Sure"]
        current_initial_eval = st.session_state.initial_eval_turing # This should be specific to the case, managed by load/save
        initial_choice_idx = 0 # Default to "A is Human"
        if current_initial_eval == "A is Human": initial_choice_idx = 0
        elif current_initial_eval == "B is Human": initial_choice_idx = 1
        elif current_initial_eval == "Not Sure": initial_choice_idx = 2


        initial_choice = st.radio(
            "Which report is written by a human radiologist? (Initial Guess)",
            initial_choice_options,
            key=f"initial_choice_t_{case}",
            index=initial_choice_idx # Use previously saved choice if available
        )
        if st.button("Submit Initial Guess & View Images", key=f"submit_initial_t_{case}"):
            st.session_state.initial_eval_turing = initial_choice
            st.session_state.viewed_images_turing = True # Mark images as viewed for this case
            st.success("Initial guess recorded. Now showing images.")
            # Save progress here as an intermediate step
            prog = {
                "last_case_turing": idx, "case_id": case,
                "assignments_turing": st.session_state.assignments_turing,
                "initial_eval_turing": initial_choice,
                "final_eval_turing": st.session_state.final_eval_turing, # Might be None
                "viewed_images_turing": True
            }
            save_progress("turing_test", prog)
            st.rerun()

    if st.session_state.viewed_images_turing:
        st.markdown("---")
        st.markdown("#### Images (Scroll through slices using Next/Prev under images)")
        display_carousel("turing", case)
        st.markdown(f"**Your Initial Guess:** {st.session_state.initial_eval_turing}")

        final_choice_options = ["A is Human", "B is Human", "Not Sure"]
        current_final_eval = st.session_state.final_eval_turing
        final_choice_idx = 0 # Default
        if st.session_state.final_eval_turing == "A is Human": final_choice_idx = 0
        elif st.session_state.final_eval_turing == "B is Human": final_choice_idx = 1
        elif st.session_state.final_eval_turing == "Not Sure": final_choice_idx = 2
        else: # If no final eval yet, default to initial eval
            if st.session_state.initial_eval_turing == "A is Human": final_choice_idx = 0
            elif st.session_state.initial_eval_turing == "B is Human": final_choice_idx = 1
            elif st.session_state.initial_eval_turing == "Not Sure": final_choice_idx = 2


        final_choice = st.radio(
            "After viewing images, which report is written by a human radiologist?",
            final_choice_options,
            key=f"final_choice_t_{case}",
            index=final_choice_idx
        )
        st.session_state.final_eval_turing = final_choice # Update session state continuously

        if st.button("Confirm Final Answer & Next Case", key=f"finalize_t_{case}"):
            prog = {
                "last_case_turing": idx + 1, # Progress to next case index
                "case_id": case, # The case that was just completed
                "assignments_turing": st.session_state.assignments_turing, # Save all assignments
                "initial_eval_turing": st.session_state.initial_eval_turing,
                "final_eval_turing": final_choice,
                "viewed_images_turing": True # This case is done with images
            }
            save_progress("turing_test", prog)

            # Reset for the next case
            st.session_state.last_case_turing += 1
            st.session_state.current_slice_turing = 0
            st.session_state.initial_eval_turing = None # Reset for next case
            st.session_state.final_eval_turing = None   # Reset for next case
            st.session_state.viewed_images_turing = False # Reset for next case
            st.rerun()

def evaluate_case(): # Standard Evaluation
    idx = st.session_state.last_case_standard
    if total_cases == 0:
        st.warning("No cases available for Standard Evaluation.")
        if st.button("Home", key="home_std_no_cases"):
            set_page("index")
            st.rerun()
        return

    if idx >= total_cases:
        st.success("Standard Evaluation complete! Thank you.")
        if st.button("Home", key="home_std_complete"):
            set_page("index")
            st.rerun()
        return

    case = cases[idx]
    st.header(f"Standard Evaluation: Case {case} ({idx + 1}/{total_cases})")

    if st.button("Save & Back to Home", key="back_std"):
        prog = {
            "last_case_standard": idx, "case_id": case,
            "assignments_standard": st.session_state.assignments_standard,
            "corrections_standard": st.session_state.corrections_standard, # Save current list of corrections
        }
        save_progress("standard_evaluation", prog)
        set_page("index")
        st.rerun()

    case_reports = report_dict.get(str(case), {})
    gt_report  = case_reports.get("gt", f"Ground truth report for case {case} not found.")
    gen_report = case_reports.get("gen", f"Generated report for case {case} not found.")

    if case not in st.session_state.assignments_standard:
        st.session_state.assignments_standard[case] = random.choice([True, False])

    is_gen_A = st.session_state.assignments_standard[case]
    report_A_text = gen_report if is_gen_A else gt_report
    report_B_text = gt_report if is_gen_A else gen_report

    st.subheader("Report A")
    st.text_area("Report A Text", report_A_text, height=150, key=f"A_s_{case}", disabled=True)
    st.subheader("Report B")
    st.text_area("Report B Text", report_B_text, height=150, key=f"B_s_{case}", disabled=True)

    st.markdown("---")
    st.markdown("#### Images (Scroll through slices using Next/Prev under images)")
    display_carousel("standard", case)
    st.markdown("---")

    st.subheader("Identify Errors/Corrections (if any)")
    # Ensure corrections_standard for the current case are handled correctly
    # Upon loading, st.session_state.corrections_standard contains all corrections from last save.
    # We need to display/manage only those for the current case.
    current_case_corrections = [c for c in st.session_state.corrections_standard if c.get("case_id") == case]


    with st.form(key=f"correction_form_s_{case}"):
        organ  = st.selectbox("Organ/System with Error", [""] + ["LIVER","PANCREAS","KIDNEY","LUNG","BONE","SOFT TISSUE", "VESSELS", "LYMPH NODES", "OTHER"], key=f"org_s_{case}")
        error_type = st.selectbox("Type of Error", ["", "Missing Finding (False Negative)", "Hallucinated Finding (False Positive)", "Incorrect Description/Severity", "Laterality Error", "Other"], key=f"err_s_{case}")
        description = st.text_area("Describe the error and correction", key=f"desc_s_{case}")
        report_choice = st.radio("Which report has this error (or is worse)?", ["Report A", "Report B", "Both are equally problematic here", "N/A - No error of this type"], key=f"rep_choice_s_{case}")
        submitted_correction = st.form_submit_button("Add Correction")

        if submitted_correction and organ and error_type and description:
            new_correction = {
                "case_id": case, "organ": organ, "error_type": error_type,
                "description": description, "report_with_error": report_choice,
                "correction_id": str(uuid.uuid4()) # Unique ID for each correction
            }
            st.session_state.corrections_standard.append(new_correction)
            st.success("Correction added!")
            # Do not rerun here, let user add more or submit. Form will clear.
            # Instead, just update the display of corrections.
            current_case_corrections.append(new_correction) # Update local list for immediate display


    if current_case_corrections:
        st.subheader("Added Corrections for this Case:")
        # Display with option to remove
        for i, cor in enumerate(reversed(current_case_corrections)): # Show newest first
            col1, col2 = st.columns([4,1])
            with col1:
                st.markdown(f"**Organ:** {cor['organ']} | **Error Type:** {cor['error_type']} | **Report:** {cor['report_with_error']}")
                st.markdown(f"> {cor['description']}")
            with col2:
                if st.button(f"Remove##{cor['correction_id']}", key=f"del_s_{cor['correction_id']}"):
                    st.session_state.corrections_standard = [c for c in st.session_state.corrections_standard if c.get('correction_id') != cor['correction_id']]
                    st.rerun() # Rerun to update the list
            st.markdown("---")


    st.subheader("Overall Assessment")
    overall_preference_options = [
        "Report A is significantly better",
        "Report A is slightly better",
        "Reports are of comparable quality",
        "Report B is slightly better",
        "Report B is significantly better",
        "Neither is acceptable, my corrections are essential"
    ]
    overall_preference = st.selectbox(
        "Which report is clinically preferable overall, or are your corrections essential?",
        overall_preference_options, key=f"overall_s_{case}"
    )

    if st.button("Submit Evaluation & Next Case", key=f"submit_s_{case}"):
        # Save only corrections for the current case as annotations
        final_case_corrections = [c for c in st.session_state.corrections_standard if c.get("case_id") == case]
        if final_case_corrections:
            save_annotations(case, final_case_corrections)

        prog = {
            "last_case_standard": idx + 1, # Progress to next case index
            "case_id": case, # The case that was just completed
            "assignments_standard": st.session_state.assignments_standard, # Save all assignments
            "corrections_standard": st.session_state.corrections_standard, # Save all corrections made so far by user (can be filtered later)
            "overall_preference": overall_preference,
            "is_gen_A": is_gen_A # Store which report was A for this case
        }
        save_progress("standard_evaluation", prog)

        # Clear corrections for the *completed* case from the main list if you want them isolated per save
        # Or keep them all and filter by case_id on load/display. The current load_user_state loads the whole list.
        # For simplicity and to avoid very long lists in session_state if not managed carefully:
        st.session_state.corrections_standard = [c for c in st.session_state.corrections_standard if c.get("case_id") != case]


        st.session_state.last_case_standard += 1
        st.session_state.current_slice_standard = 0
        # st.session_state.corrections_standard = [] # Reset for next case; already filtered above
        st.rerun()

def ai_edit():
    idx = st.session_state.last_case_ai
    if total_cases == 0:
        st.warning("No cases available for AI Edit.")
        if st.button("Home", key="home_ai_no_cases"):
            set_page("index")
            st.rerun()
        return

    if idx >= total_cases:
        st.success("AI Edit task complete! Thank you.")
        if st.button("Home", key="home_ai_complete"):
            set_page("index")
            st.rerun()
        return

    case = cases[idx]
    st.header(f"AI-Assisted Report Editing: Case {case} ({idx + 1}/{total_cases})")

    # Initialize case-specific session state for editing if not present
    if f"ai_edit_text_{case}" not in st.session_state:
        original_ai_report = report_dict.get(str(case), {}).get("gen", f"AI-generated report for case {case} not found.")
        st.session_state[f"ai_edit_text_{case}"] = original_ai_report # Start with original AI report
    if f"ai_edit_corrections_{case}" not in st.session_state:
         st.session_state[f"ai_edit_corrections_{case}"] = [] # Case-specific corrections

    if st.button("Save & Back to Home", key="back_ai"):
        prog = {
            "last_case_ai": idx, "case_id": case,
            "assembled_ai": st.session_state.get(f"ai_edit_text_{case}", ""), # Save current text
            "corrections_ai": st.session_state.get(f"ai_edit_corrections_{case}", []), # Save current structured corrections
            "last_mode_ai": st.session_state.get("last_mode_ai", "Freeform Edit")
        }
        save_progress("ai_edit", prog)
        set_page("index")
        st.rerun()

    original_ai_report = report_dict.get(str(case), {}).get("gen", f"AI-generated report for case {case} not found.")
    st.subheader("Original AI Report (for reference)")
    st.text_area("Original AI-Generated Report", original_ai_report, height=150, key=f"orig_ai_{case}", disabled=True)
    st.markdown("---")
    st.markdown("#### Images (Scroll through slices using Next/Prev under images)")
    display_carousel("ai", case) # Suffix "ai" for carousel state
    st.markdown("---")

    # Use a case-specific key for mode as well if it needs to be remembered per case
    # For now, 'last_mode_ai' is global.
    mode_options = ["Freeform Edit", "Structured Findings Input"]
    mode = st.radio("Editing Mode:", mode_options, key=f"mode_ai_{case}", index=mode_options.index(st.session_state.get("last_mode_ai", "Freeform Edit")))
    st.session_state.last_mode_ai = mode


    if mode == "Freeform Edit":
        st.subheader("Edit the AI Report Directly:")
        edited_text = st.text_area(
            "Your Edited Report:",
            value=st.session_state[f"ai_edit_text_{case}"],
            height=300,
            key=f"free_edit_ai_{case}"
        )
        st.session_state[f"ai_edit_text_{case}"] = edited_text # Continuously update
        # Clear structured corrections if switching to freeform after making some? Or merge them?
        # For now, they are separate.
    else: # Structured Findings Input
        st.subheader("Add or Modify Findings Structurally:")
        with st.form(key=f"structured_form_ai_{case}"):
            finding_type = st.selectbox("Finding/Impression Type", ["", "Impression", "Organ Finding", "Recommendation"], key=f"ftype_ai_{case}")
            finding_detail = st.text_area("Describe the finding or statement:", key=f"fdetail_ai_{case}")
            add_structured = st.form_submit_button("Add Structured Finding")

            if add_structured and finding_type and finding_detail:
                new_finding = {
                    "case_id": case, "type": finding_type, "detail": finding_detail,
                    "finding_id": str(uuid.uuid4())
                }
                st.session_state[f"ai_edit_corrections_{case}"].append(new_finding)
                st.success("Structured finding added!")
                # Form clears, new finding is in state

        current_findings = st.session_state[f"ai_edit_corrections_{case}"]
        if current_findings:
            st.markdown("##### Current Structured Findings for this Case:")
            for i, find in enumerate(reversed(current_findings)):
                fcol1, fcol2 = st.columns([4,1])
                fcol1.markdown(f"**{find['type']}:** {find['detail']}")
                if fcol2.button(f"Del##{find['finding_id']}", key=f"del_ai_{find['finding_id']}"):
                    st.session_state[f"ai_edit_corrections_{case}"] = [f for f in current_findings if f['finding_id'] != find['finding_id']]
                    st.rerun()
                st.markdown("---")

            if st.button("Assemble Structured Findings into Report Text", key=f"assemble_ai_{case}"):
                assembled_text_parts = []
                for find in current_findings:
                    assembled_text_parts.append(f"**{find['type']}:** {find['detail']}")
                st.session_state[f"ai_edit_text_{case}"] = "\n\n".join(assembled_text_parts)
                st.success("Report text updated from structured findings. You can now switch to Freeform Edit to refine.")
                st.rerun()
        st.markdown("##### Current Report Text (auto-updated if assembled):")
        st.text_area("", value=st.session_state[f"ai_edit_text_{case}"], height=200, key=f"display_assembled_ai_{case}", disabled=True)


    if st.button("Submit Edited Report & Next Case", key=f"submit_ai_{case}"):
        final_text_for_case = st.session_state.get(f"ai_edit_text_{case}", original_ai_report)
        final_corrections_for_case = st.session_state.get(f"ai_edit_corrections_{case}", [])

        prog = {
            "last_case_ai": idx + 1, # Progress to next case index
            "case_id": case, # The case that was just completed
            "assembled_ai": final_text_for_case, # The final edited text
            "corrections_ai": final_corrections_for_case, # The structured findings if that mode was used
            "last_mode_ai": mode, # Mode used for this submission
            "original_report_gen": original_ai_report # Keep a copy of original for comparison
        }
        save_progress("ai_edit", prog)

        # Clean up session state for the completed case
        if f"ai_edit_text_{case}" in st.session_state:
            del st.session_state[f"ai_edit_text_{case}"]
        if f"ai_edit_corrections_{case}" in st.session_state:
            del st.session_state[f"ai_edit_corrections_{case}"]

        st.session_state.last_case_ai += 1
        st.session_state.current_slice_ai = 0
        # st.session_state.assembled_ai = "" # Reset for next case (handled by case-specific state)
        # st.session_state.corrections_ai = [] # Reset for next case (handled by case-specific state)
        st.rerun()


def view_my_results(): # Changed from view_all_results
    st.title(f"My Saved Results - {st.session_state.username}")
    if st.button("Home", key="home_results"):
        set_page("index")
        st.rerun()

    conn = get_db_connection()
    current_username = st.session_state.username

    st.subheader(f"Progress for User: {current_username}")

    for cat_key, label in [
        ("turing_test", "Turing Test Submissions"),
        ("standard_evaluation", "Standard Evaluation Submissions"),
        ("ai_edit", "AI Edit Submissions")
    ]:
        st.markdown(f"#### {label}")
        df = pd.read_sql_query(
            "SELECT progress_json, timestamp FROM progress_logs WHERE username=? AND category=? ORDER BY timestamp DESC",
            conn, params=(current_username, cat_key)
        )
        if not df.empty:
            all_data = []
            for _, row in df.iterrows():
                try:
                    data = json.loads(row["progress_json"])
                    data["timestamp"] = row["timestamp"]
                    # Ensure "case_id" is at a consistent place if it exists
                    if "case_id" not in data and "last_case" in data and isinstance(data["last_case"], int) and data["last_case"] > 0 and data["last_case"] <= len(cases):
                        # Try to infer case_id if it's a "next case" save
                        # This logic is tricky because last_case refers to the *next* index to start
                        # The actual completed case_id would be cases[data["last_case"]-1]
                        # This part might need refinement based on exact save logic for "case_id"
                        if data["last_case"] > 0 and (data["last_case"]-1) < len(cases):
                           data["completed_case_id_inferred"] = cases[data["last_case"]-1]
                    all_data.append(data)
                except json.JSONDecodeError:
                    all_data.append({"error": "Could not parse progress_json", "timestamp": row["timestamp"]})

            if all_data:
                df_expanded = pd.DataFrame(all_data)
                # Attempt to bring important columns to front
                cols_to_front = ["timestamp", "case_id", "completed_case_id_inferred"]
                existing_cols_front = [c for c in cols_to_front if c in df_expanded.columns]
                other_cols = [c for c in df_expanded.columns if c not in existing_cols_front]
                df_display = df_expanded[existing_cols_front + other_cols]
                st.dataframe(df_display)
            else:
                st.write("— no entries found —")

        else:
            st.write("— no entries found —")

    st.subheader("My Annotations (from Standard Evaluation)")
    df_annotations = pd.read_sql_query(
        "SELECT case_id, annotations_json, timestamp FROM annotations WHERE username=? ORDER BY timestamp DESC",
        conn, params=(current_username,)
    )
    if not df_annotations.empty:
        anno_data = []
        for _, row in df_annotations.iterrows():
            try:
                annotations = json.loads(row["annotations_json"])
                if isinstance(annotations, list): # Expecting a list of annotation dicts
                    for anno in annotations:
                        anno_data.append({
                            "case_id": row["case_id"],
                            "timestamp": row["timestamp"],
                            **anno # Unpack the annotation dict
                        })
                else: # Single annotation dict (legacy or error)
                     anno_data.append({
                        "case_id": row["case_id"],
                        "timestamp": row["timestamp"],
                        "annotation_data": json.dumps(annotations) # dump as string if not list
                    })
            except json.JSONDecodeError:
                anno_data.append({
                    "case_id": row["case_id"],
                    "timestamp": row["timestamp"],
                    "error": "Could not parse annotations_json"
                })
        if anno_data:
            st.dataframe(pd.DataFrame(anno_data))
        else:
            st.write("— no annotation entries found —")

    else:
        st.write("— no annotation entries found —")

    conn.close()


# --------------------------------------------------
# 9. Main Router
# --------------------------------------------------
if authentication_status: # Only proceed if authenticated
    page_to_display = st.session_state.get("page", "index")
    if page_to_display == "turing_test":
        turing_test()
    elif page_to_display == "standard_eval":
        evaluate_case()
    elif page_to_display == "ai_edit":
        ai_edit()
    elif page_to_display == "view_my_results":
        view_my_results()
    else:
        index()
