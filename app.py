import streamlit as st
import requests
import random
import pytesseract
from PIL import Image
import io
import re
import concurrent.futures
import json
import os
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import speech_recognition as sr
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create LLM Object using the new model
llm = ChatGroq(
    model="mistral-saba-24b", 
    temperature=0
)

# -------------------------
# PyDrive Setup for Google Drive Storage
# -------------------------
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def init_drive():
    """Authenticate and return a GoogleDrive instance using PyDrive."""
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    gauth.SaveCredentialsFile("mycreds.txt")
    return GoogleDrive(gauth)

drive = init_drive()

# Google Drive file ID for users.json (if already uploaded)
# If not present, the code below will create a new file.
USERS_JSON_FILE_ID = st.secrets.get("users_json_file_id", None)
USER_DATA_FILE = "users.json"  # Local fallback

def load_users():
    """Load user data from users.json stored on Google Drive."""
    global USERS_JSON_FILE_ID
    try:
        if USERS_JSON_FILE_ID:
            file = drive.CreateFile({'id': USERS_JSON_FILE_ID})
            file.GetContentFile(USER_DATA_FILE)
        elif os.path.exists(USER_DATA_FILE):
            pass  # Use local file if exists
        else:
            return {}
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading users.json: {e}")
        return {}

def save_users(users):
    """Save user data to users.json on Google Drive by updating an existing file if it exists."""
    global USERS_JSON_FILE_ID
    # Write the updated users data to the local file.
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)
    try:
        # If we already have a known file ID, use it.
        if USERS_JSON_FILE_ID:
            file_drive = drive.CreateFile({'id': USERS_JSON_FILE_ID})
        else:
            # Otherwise, search for an existing file with the same name.
            file_list = drive.ListFile({
                'q': f"title='{USER_DATA_FILE}' and trashed=false"
            }).GetList()
            if file_list:
                file_drive = file_list[0]
                USERS_JSON_FILE_ID = file_drive['id']
            else:
                # If not found, create a new file.
                file_drive = drive.CreateFile({'title': USER_DATA_FILE})
        # Update the file content and upload.
        file_drive.SetContentFile(USER_DATA_FILE)
        file_drive.Upload()
        st.success("Dr R Pathmini MD, CMCH, Coimbatore <nrtyasri@gmail.com>")
    except Exception as e:
        st.error(f"Error saving... : {e}")

# -------------------------
# Helper Validation Functions
# -------------------------
def is_valid_email(email):
    """Validate email using regex."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def is_valid_phone(phone):
    """Validate phone number in E.164 format: must start with '+' followed by 10-15 digits."""
    pattern = r'^\+\d{10,15}$'
    return re.match(pattern, phone)

def is_valid_address(address):
    """
    Validate that the address includes both a city and a country.
    It requires at least one comma and non-empty values on both sides.
    """
    if ',' not in address:
        return False
    parts = address.split(',')
    if len(parts) < 2:
        return False
    city = parts[0].strip()
    country = parts[1].strip()
    return bool(city) and bool(country)

# -------------------------
# User Authentication and Registration
# -------------------------

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.login_attempted = False

# Force download from Google Drive before every credential check
def force_refresh_users_from_drive():
    global USERS_JSON_FILE_ID
    try:
        if USERS_JSON_FILE_ID:
            # Explicitly download the latest version from Google Drive
            file = drive.CreateFile({'id': USERS_JSON_FILE_ID})
            # Remove local file first if it exists
            if os.path.exists(USER_DATA_FILE):
                os.remove(USER_DATA_FILE)
            # Download the latest version
            file.GetContentFile(USER_DATA_FILE)
            
            # Read the freshly downloaded file
            with open(USER_DATA_FILE, "r") as file:
                return json.load(file)
        else:
            # Search for the file on Google Drive
            file_list = drive.ListFile({
                'q': f"title='{USER_DATA_FILE}' and trashed=false"
            }).GetList()
            if file_list:
                file_drive = file_list[0]
                USERS_JSON_FILE_ID = file_drive['id']
                # Remove local file first if it exists
                if os.path.exists(USER_DATA_FILE):
                    os.remove(USER_DATA_FILE)
                file_drive.GetContentFile(USER_DATA_FILE)
                
                with open(USER_DATA_FILE, "r") as file:
                    return json.load(file)
    except Exception as e:
        st.error(f"Error refreshing data from Google Drive: {e}")
    
    # If we couldn't load from Google Drive, try local file as fallback
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r") as file:
                return json.load(file)
    except Exception as e:
        st.error(f"Error loading local user data: {e}")
    
    # Return empty dict if all else fails
    return {}

def attempt_auto_login():
    # Force refresh from Google Drive every time
    current_users = force_refresh_users_from_drive()
    
    default_username = "dcs1"
    default_password = "DCS1"
    
    # Log the current users for debugging
    st.sidebar.write(f"Debug - Available users: {list(current_users.keys())}")
    
    # Explicitly check if the username exists in the current data
    if default_username not in current_users:
        st.sidebar.error("🚫 User 'dcs1' does not exist in Google Drive data. Please register.")
        return False
        
    # Then verify password only if the user exists
    if current_users[default_username]["password"] == default_password:
        st.session_state.authenticated = True
        st.session_state.username = default_username
        return True
    else:
        st.sidebar.error("🚫 Invalid credentials!")
        return False

def register():
    st.sidebar.title("📝 User Registration")
    new_username = st.sidebar.text_input("Username", key="register_username")
    new_password = st.sidebar.text_input("Password", type="password", key="register_password")
    new_occupation = st.sidebar.text_input("Occupation", key="register_occupation")
    new_email = st.sidebar.text_input("Email", key="register_email")
    new_phone = st.sidebar.text_input("Phone Number (+91)", key="register_phone")
    new_address = st.sidebar.text_area("Address (City, Country)", key="register_address")
    register_button = st.sidebar.button("Register")
    
    if register_button:
        # Force refresh from Google Drive every time
        current_users = force_refresh_users_from_drive()
        
        # Check for missing fields
        if not new_username or not new_password or not new_email or not new_phone or not new_address:
            st.sidebar.error("🚨 All fields are required!")
            return
        # Check for duplicate username
        if new_username in current_users:
            st.sidebar.error("🚫 Username already exists! Choose another.")
            return
        # Validate email
        if not is_valid_email(new_email):
            st.sidebar.error("🚫 Please provide a valid email address!")
            return
        # Validate phone number
        if not is_valid_phone(new_phone):
            st.sidebar.error("🚫 Please provide a valid phone number in E.164 format (e.g., +12345678901)!")
            return
        # Validate address (ensure both city and country are present)
        if not is_valid_address(new_address):
            st.sidebar.error("🚫 Address must include a valid city and country (format: City, Country)!")
            return
        
        # If all validations pass, save the new user
        current_users[new_username] = {
            "password": new_password,
            "occupation": new_occupation,
            "email": new_email,
            "phone": new_phone,
            "address": new_address
        }
        save_users(current_users)
        st.sidebar.success("✅ Registration successful! Please log in.")

def login():
    st.sidebar.title("🔐 User Login")
    # Auto-fill with default credentials
    username = st.sidebar.text_input("Username", value="dcs1", key="login_username")
    password = st.sidebar.text_input("Password", value="DCS1", type="password", key="login_password")
    login_button = st.sidebar.button("Login")
    
    # Auto-login attempt when the page loads (only once)
    if not st.session_state.login_attempted:
        st.session_state.login_attempted = True
        
        # Force refresh from Google Drive every time
        current_users = force_refresh_users_from_drive()
        
        # Log the current users for debugging
        st.sidebar.write(f"Debug - Available users: {list(current_users.keys())}")
        
        # Explicit check for user existence and password match
        if username not in current_users:
            st.sidebar.error("🚫 User does not exist in Google Drive data. Please register.")
        elif current_users[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success(f"✅ Logged in as {username}")
        else:
            st.sidebar.error("🚫 Invalid credentials!")
    
    # Handle manual login button click
    elif login_button:
        # Force refresh from Google Drive every time
        current_users = force_refresh_users_from_drive()
        
        # Log the current users for debugging
        st.sidebar.write(f"Debug - Available users: {list(current_users.keys())}")
        
        if username not in current_users:
            st.sidebar.error("🚫 User does not exist in Google Drive data. Please register.")
        elif current_users[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success(f"✅ Logged in as {username}")
        else:
            st.sidebar.error("🚫 Invalid credentials!")

def logout():
    st.session_state.authenticated = False
    st.session_state.login_attempted = False
    st.sidebar.warning("Logged out. Please refresh.")

# First page load - try auto-login
if not st.session_state.authenticated:
    if not st.session_state.login_attempted:
        attempt_auto_login()
    
    # If auto-login failed or already attempted, show the login form
    if not st.session_state.authenticated:
        register()
        login()
        st.stop()

# -------------------------
# Close Sidebar on Successful Login
# -------------------------
if st.session_state.authenticated:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
        """, unsafe_allow_html=True
    )
st.markdown(
                """
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-weight: bold; font-size: 14px;'>nrtyasri@gmail.com</span>
                    <span style='font-weight: bold; font-size: 14px;'>dr.pathmini md coimbatore</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ✅ ECG-style animated title
            
st.markdown(
                """
                <style>
                @keyframes typing {
                  from { width: 0 }
                  to { width: 100% }
                }

                @keyframes blink-caret {
                  0%, 100% { border-color: transparent }
                  50% { border-color: #00FF00; }
                }

                @keyframes flash {
                  0% { opacity: 0.8; text-shadow: 0 0 5px #00FF00, 0 0 10px #00FF00; }
                  50% { opacity: 1; text-shadow: 0 0 20px #00FF00, 0 0 30px #00FF00; }
                  100% { opacity: 0.8; text-shadow: 0 0 5px #00FF00, 0 0 10px #00FF00; }
                }
                .ecg-title h1 {
                  overflow: hidden;
                  white-space: nowrap;
                  border-right: 2px solid #00FF00;
                  animation:
                    typing 4s steps(40, end),
                    blink-caret 0.75s step-end infinite,
                    flash 1s infinite;
                  font-family: "Courier New", Courier, monospace;
                  color: #00FF00;
                  font-size: 44px;
                  margin: 20px auto;
                  width: fit-content;
                  text-align: center;
                }
                </style>

                <div class="ecg-title">
                    <h1>Medical Assistant #VelanAI</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
# -------------------------
# Setup Redis Cache (if available)
# -------------------------
redis_client = None
try:
    redis_config = st.secrets.get("redis", None)
except Exception:
    redis_config = None

if redis_config:
    try:
        import redis
        redis_client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            password=redis_config.get("password", None),
            decode_responses=True
        )
    except Exception as e:
        st.error("Error connecting to Redis: " + str(e))
        redis_client = None

# -------------------------
# Global Requests Session
# -------------------------
session = requests.Session()

# FDA Drug Label Fields to Fetch
FDA_FIELDS = [
    "purpose", "dosage_and_administration","adverse_reactions", "drug_and_or_laboratory_test_interactions", "drug_interactions",
    "ask_doctor", "ask_doctor_or_pharmacist", "do_not_use", "information_for_patients",
    "instructions_for_use", "other_safety_information", "patient_medication_information",
    "spl_medguide", "spl_patient_package_insert", "stop_use", "when_using", "boxed_warning",
    "general_precautions", "precautions", "user_safety_warnings", "warnings", "contraindications",
    "geriatric_use", "labor_and_delivery", "mechanism_of_action", "nursing_mothers", "overdosage",
    "pediatric_use", "pregnancy", "pregnancy_or_breast_feeding", "safe_handling_warning",
    "use_in_specific_populations"
]

# Mapping for RxNav class types
class_type_mapping = {
    "ci_with": "Contraindications",
    "ci_moa": "Contraindications (MoA)",
    "ci_pe": "Contraindications (Effects)",
    "ci_chemclass": "Contraindications (Chem)",
    "has_pe": "Effects",
    "has_moa": "MoA",
    "has_epc": "Drug Class",
    "may_treat": "To Treat"
}

ordered_class_types = [
    "ci_with", "ci_moa", "ci_pe", "ci_chemclass", "has_pe", "has_moa", "has_epc", "may_treat"
]

# List of Jokes
jokes = [
    "Aristotle: To actualize its potential.",
    "Plato: For the greater good.",
    "Socrates: To examine the other side.",
    "Descartes: It had sufficient reason to believe it was dreaming.",
    "Hume: Out of habit.",
    "Kant: Out of a sense of duty.",
    "Nietzsche: Because if you gaze too long across the road, the road gazes also across you.",
    "Hegel: To fulfill the dialectical progression.",
    "Marx: It was a historical inevitability.",
    "Sartre: In order to act in good faith and be true to itself.",
    "Camus: One must imagine Sisyphus happy and the chicken crossing the road.",
    "Wittgenstein: The meaning of 'cross' was in the use, not in the action.",
    "Derrida: The chicken was making a deconstructive statement on the binary opposition of 'this side' and 'that side.'",
    "Heidegger: To authentically dwell in the world.",
    "Foucault: Because of the societal structures and power dynamics at play.",
    "Chomsky: For a syntactic, not pragmatic, purpose.",
    "Buddha: If you meet the chicken on the road, kill it.",
    "Laozi: The chicken follows its path naturally.",
    "Confucius: The chicken crossed the road to reach the state of Ren.",
    "Leibniz: In the best of all possible worlds, the chicken would cross the road."
]

import re
import streamlit.components.v1 as components

# -------------------------
# Helper function to format text for enhanced readability
# -------------------------
def format_text(raw_text):
    """
    Format raw text to improve readability while preserving numbered lists.
    
    Steps:
    1. Strip leading/trailing whitespace.
    2. Insert a newline before any occurrence of a numbered list item (e.g., "1.").
       This uses a negative lookbehind to ensure a newline is added only when not already present.
    3. Replace punctuation (.?!), followed by whitespace, with the punctuation plus a newline.
    4. Collapse multiple newlines into a single newline.
    5. Finally, join each non-empty line with a double newline for clarity.
    """
    # 1. Remove any leading/trailing whitespace.
    raw_text = raw_text.strip()
    
# 3. Replace punctuation (., ?, !) followed by whitespace with punctuation plus a newline.
    raw_text = re.sub(r'([.?!])\s+', r'\1\n', raw_text)

# Code for removing a newline immediately after numbered list items.
    raw_text = re.sub(r'(\d+\.\s*)\n', r'\1', raw_text)

    # 2. Insert newline before numbered list items if not already preceded by a newline.
    #raw_text = re.sub(r'(?<!\n)(\d+\.\s*)', r'\n\1', raw_text)
    
    
    
    # 4. Collapse multiple newlines into a single newline.
    raw_text = re.sub(r'\n+', '\n', raw_text)
    
    # 5. Join lines with a double newline for enhanced readability.
    formatted_text = "\n\n".join(line.strip() for line in raw_text.split("\n") if line.strip())
    
    return formatted_text

# -------------------------
# Fetch RxNav Data
# -------------------------
def fetch_rxnav_data(drug_name):
    """Fetch RxNav drug class information."""
    class_types = {rela: set() for rela in ordered_class_types}
    for rela in ordered_class_types:
        url = f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName.json?drugName={drug_name}&relaSource=ALL&relas={rela}"
        response = session.get(url)
        if response.status_code != 200:
            return {'error': 'Failed to fetch data from RxClass API'}
        data = response.json()
        if 'rxclassDrugInfoList' in data:
            drug_classes = data['rxclassDrugInfoList'].get('rxclassDrugInfo', [])
            for cls in drug_classes:
                class_name = cls['rxclassMinConceptItem']['className']
                class_types[rela].add(class_name)
    mapped_classes = {class_type_mapping[rela]: list(class_types[rela]) for rela in ordered_class_types}
    return {'drug_name': drug_name, 'classes': mapped_classes}

# -------------------------
# Fetch FDA Drug Label Data with Redis Caching
# -------------------------
def fetch_fda_data(drug_name):
    """Fetch FDA drug label information, using Redis for caching."""
    cache_key = f"fda:{drug_name.lower()}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
        except UnicodeError as e:
            st.warning("Redis cache key encoding error: " + str(e))
            cached = None
        if cached:
            return json.loads(cached)
    
    url = f'https://api.fda.gov/drug/label.json?search=openfda.generic_name:"{drug_name}"&limit=1'
    response = session.get(url)
    
    if response.status_code != 200:
        return {'error': 'Failed to fetch data from the FDA API'}

    data = response.json()
    if "results" in data and data["results"]:
        fda_data = data["results"][0]
        
        # Merge "ask_doctor_or_pharmacist" into "ask_doctor"
        if "ask_doctor_or_pharmacist" in fda_data:
            doc_val = fda_data.get("ask_doctor", "")
            if isinstance(doc_val, list):
                doc_val = "\n".join(doc_val)
            pharm_val = fda_data["ask_doctor_or_pharmacist"]
            if isinstance(pharm_val, list):
                pharm_val = "\n".join(pharm_val)
            fda_data["ask_doctor"] = doc_val + "\n" + pharm_val
            del fda_data["ask_doctor_or_pharmacist"]
        
        # Merge "stop_use" into "do_not_use"
        if "stop_use" in fda_data:
            not_use_val = fda_data.get("do_not_use", "")
            if isinstance(not_use_val, list):
                not_use_val = "\n".join(not_use_val)
            stop_val = fda_data["stop_use"]
            if isinstance(stop_val, list):
                stop_val = "\n".join(stop_val)
            fda_data["do_not_use"] = not_use_val + "\n" + stop_val
            del fda_data["stop_use"]

        if redis_client:
            redis_client.setex(cache_key, 3600, json.dumps(fda_data))  # Cache for 1 hour
        return fda_data
    
    return {'error': 'No FDA data available.'}

# -------------------------
# Combined function to fetch both FDA and RxNav data for a drug
# -------------------------
def fetch_drug_data(drug_name):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_rxnav = executor.submit(fetch_rxnav_data, drug_name)
        future_fda = executor.submit(fetch_fda_data, drug_name)
        rxnav_data = future_rxnav.result()
        fda_data = future_fda.result()
    if 'error' in rxnav_data or 'error' in fda_data:
        return {'drug_name': drug_name, 'error': rxnav_data.get('error') or fda_data.get('error')}
    return {
        "drug_name": drug_name,
        "rxnav": rxnav_data,
        "fda": fda_data
    }

# -------------------------
# Extract text from uploaded image using OCR (Tesseract)
# -------------------------
def extract_text_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# -------------------------
# Streamlit UI
# -------------------------
# st.write("# **#VelanAI_khel**")
# st.write("## **Physician Pocket Reference**")
st.markdown(
    """
    <p style='text-align: right; color: white; font-size: 20px;'>
        ....for the just-in-time review of clinical signs
    </p>
    """,
    unsafe_allow_html=True
)

# st.title(" **VelanAI_Khel**")
# st.write("### **Why did the Chicken cross the road?!**")
# st.write(f"**{random.choice(jokes)}**")

# Option to input drug name manually
# drug_name_input = st.text_input("#### **Enter drug name(s) (comma-separated)**")

# Option to upload an image containing drug label information
# uploaded_image = st.file_uploader("#### **Or upload your Prescription**", type=["png", "jpg", "jpeg"])

# Determine input source: text or image (image takes precedence if uploaded)
# if uploaded_image:
    # st.info("Extracting text from the uploaded image...")
    # extracted_text = extract_text_from_image(uploaded_image)
    # st.write("Extracted Text:", extracted_text)
    # Split by comma or newline to handle multi-line OCR output
    # drug_name_input = ",".join([name.strip() for name in re.split(r"[,\n]+", extracted_text) if name.strip()])

# if st.button("Fetch"):
    # if not drug_name_input:
        # st.error("Please provide a drug name or upload an image.")
    # else:
        # Split input into drug names using comma and newline as delimiters
        # drug_names = [name.strip() for name in re.split(r"[,\n]+", drug_name_input) if name.strip()]
        # placeholder = st.empty()  # Placeholder for incremental updates
        # results_markdown = ""
        
        # Use ThreadPoolExecutor to fetch data concurrently for each drug
        # with concurrent.futures.ThreadPoolExecutor(max_workers=len(drug_names)) as executor:
          #  future_to_drug = {executor.submit(fetch_drug_data, drug): drug for drug in drug_names}
           # for future in concurrent.futures.as_completed(future_to_drug):
            #    result = future.result()
             #   if "error" in result:
              #      results_markdown += f"### {result['drug_name']}\n**Error:** {result['error']}\n\n"
               # else:
                #    md_text = f"### {result['drug_name']}\n"
                    # Display RxNav classification data
                 #   for category, items in result["rxnav"]["classes"].items():
                  #      if items:
                   #         md_text += f"- **{category}:** {', '.join(items)}\n"
                    # Display FDA data fields with enhanced text formatting
                    # for field in FDA_FIELDS:
                      #  if field in result["fda"]:
                       #     field_value = result["fda"][field]
                        #    if field_value and field_value != "No data available":
                                # If field_value is a list, join the items first
                         #       if isinstance(field_value, list):
                          #          combined_text = "\n".join(field_value)
                           #     else:
                            #        combined_text = field_value
                                # Format the text for enhanced readability
                             #   formatted_field = format_text(combined_text)
                              #  md_text += f"<details><summary>{field.replace('_', ' ').capitalize()}</summary>\n"
                               # md_text += formatted_field
                                # md_text += "\n</details>\n"
                   # md_text += "\n"
                   # results_markdown += md_text
                # Update UI incrementally as each drug's result is appended
                # placeholder.markdown(results_markdown, unsafe_allow_html=True)
# st.title("Regional GenAI 'Medical Assistant' Chatbot")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "translations" not in st.session_state:
    st.session_state.translations = []

# --- Translation Options ---
translation_language = st.selectbox(
    "Translate responses to:",
    ["None", "Tamil", "Kannada", "Malayalam", "Telugu", "Hindi"],
    key="lang_selector"
)

lang_map = {
    "Tamil": "ta",
    "Malayalam": "ml",
    "Telugu": "te",
    "Hindi": "hi",
    "Kannada": "kn"
}

# Custom CSS for styling
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 5rem !important; /* Add extra padding at bottom */
}
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin: 1.5rem 0;
}
.stChatMessage {
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
}
.stChatMessage [data-testid="StChatMessageContent"] {
    padding: 1rem !important;
}
/* Larger and more readable title */
h1 {
    font-size: 2.5rem !important;
    margin-bottom: 1.5rem !important;
    color: #1E3A8A !important;
    text-align: center;
}
/* More attractive buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}
/* Better selectbox styling */
.stSelectbox {
    margin-bottom: 1rem !important;
}
/* Chat input styling */
[data-testid="stChatInput"] {
    border-radius: 10px !important;
    padding: 0.5rem !important;
    margin-top: 1rem !important;
}
/* Add space at bottom of page */
.download-section {
    margin-bottom: 4rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

def render_chat_message(role, content, index):
    """Render chat message with enhanced formatting and translation"""
    translation = ""
    if translation_language != "None" and role == "Assistant":
        try:
            translation = GoogleTranslator(source='auto', target=lang_map[translation_language]).translate(content)
            # Store translations for download
            if index >= len(st.session_state.translations):
                st.session_state.translations.append({"role": role, "content": translation})
            else:
                st.session_state.translations[index] = {"role": role, "content": translation}
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            translation = "Translation failed"
            if index >= len(st.session_state.translations):
                st.session_state.translations.append({"role": role, "content": translation})
            else:
                st.session_state.translations[index] = {"role": role, "content": translation}
    elif role == "User" and index >= len(st.session_state.translations):
        # Add user messages to translation list too
        st.session_state.translations.append({"role": role, "content": content})
    
    with st.container():
        # Display content without role title
        st.markdown(content)
        if translation:
            st.markdown(f"**{translation_language} Translation:**")
            st.markdown(translation)

def process_user_input(user_input, display_input=True):
    """Process and store user input with error handling, with option to hide the input"""
    if not user_input.strip():
        return

    try:
        # Generate response
        response = llm.invoke([{
            "role": "system",
            "content": "You are a helpful medical assistant. Provide clear, structured responses about medications."
        }, {
            "role": "user",
            "content": user_input
        }]).content

        # Store messages with unique IDs
        if display_input:
            st.session_state.chat_history.extend([
                {"role": "User", "content": user_input},
                {"role": "Assistant", "content": response}
            ])
        else:
            # Only add the assistant's response, not the original prompt
            st.session_state.chat_history.append(
                {"role": "Assistant", "content": response}
            )

    except Exception as e:
        st.error(f"Error processing request: {str(e)}")

# UI Components with improved layout
col1, col2, col3 = st.columns([1, 1, 1])
# with col1:
    # if st.button("🎤 Voice Input", use_container_width=True):
        # try:
            # r = sr.Recognizer()
            # with sr.Microphone() as source:
                # st.info("Listening... Please speak now.")
                # audio = r.listen(source)
                # user_input = r.recognize_google(audio)
                # process_user_input(user_input)
        # except Exception as e:
            # st.error(f"Voice recognition error: {str(e)}")

with col1:
    if st.button("👨‍⚕️ Physician Review", use_container_width=True):
        # Extract drug names from the most recent user input
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            recent_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "User"]
            if recent_messages:
                user_input = recent_messages[-1]["content"]
                # Form the physician prompt
                physician_prompt = f"""Act as a clinical decision support tool. For the drug {user_input}mentioned in the user query:

1. Identify its primary therapeutic use based on FDA labeling and clinical guidelines.
2. Systematically categorize symptoms to monitor regularly into:
   - Medication Ineffectiveness (failure to achieve intended effect)
   - Unintended Side Effects (adverse drug reactions)
   - Disease Progression (worsening underlying condition)
3. Rank symptoms within each category by clinical urgency (Critical/High/Moderate) using these criteria:
   - Likelihood of severe harm if untreated
   - Time sensitivity for intervention
   - Strength of association with the drug/condition
4. For each symptom, provide:
   - Rationale: Pathophysiological basis + evidence from ≥1 peer-reviewed study
   - Monitoring Guidance: Frequency, tools (e.g., lab tests, validated scales), and red flags
5. Format results in a structured table with columns:
	| Rank | Symptom (SNOMED-CT) | Features | Priority | Rationale | Monitoring Guidance |
6. Prioritize symptoms mentioned in drug monographs (e.g., FDA Black Box Warnings)
7. Include patient-specific considerations: Age, comorbidities, concurrent medications.

Example Output Structure for Metformin (Physician Prompt):
1. Primary Therapeutic Use
Metformin is primarily used for the management of type 2 diabetes mellitus. It is indicated to improve glycemic control in adults with type 2 diabetes mellitus. It is also used in combination with diet and exercise to lower blood sugar levels.

2. Categorized Symptoms to Monitor

| Rank | Symptom (SNOMED-CT) | Features | Priority | Rationale | Monitoring Guidance |
|---|---|---|---|---|---|
| 1 | Lactic acidosis (SNOMED-CT: 230141003) | Severe metabolic acidosis, often with renal impairment | Critical | Rare but fatal | SCr/eGFR baseline + q3mo, avoid in renal impairment (SCr > 1.5 mg/dL in males, > 1.4 mg/dL in females) |
| 2 | Persistent GI distress (SNOMED-CT: 22298006) | Nausea and/or diarrhea | High | 25% experience, may lead to non-adherence | Symptom diary, diet modifications, consider dose reduction or discontinuation if severe |
| 3 | Hypoglycemia (SNOMED-CT: 230141003) | Low blood glucose levels | High | Risk of severe hypoglycemia, especially with sulfonylureas or insulin | Blood glucose monitoring, educate on symptoms and treatment |
| 4 | Vitamin B12 deficiency (SNOMED-CT: 230141003) | Low serum vitamin B12 levels, may cause neurological symptoms | Moderate | Long-term use may cause deficiency | Annual CBC, serum vitamin B12 levels, consider supplementation |
| 5 | Weight loss (SNOMED-CT: 230141003) | Unintentional weight loss | Moderate | May indicate poor glycemic control or other underlying issues | Regular weight monitoring, assess dietary intake and adherence |
| 6 | Rash (SNOMED-CT: 230141003) | Skin irritation or redness | Moderate | Allergic reaction, may indicate hypersensitivity | Monitor for skin changes, discontinue if severe rash or itching |
| 7 | Muscle pain (SNOMED-CT: 230141003) | Muscle discomfort or soreness | Moderate | May indicate lactic acidosis or myopathy | Monitor for muscle weakness, discontinue if severe pain or weakness |
| 8 | Fatigue (SNOMED-CT: 230141003) | Persistent tiredness or exhaustion | Moderate | May indicate anemia or other underlying issues | Monitor for persistent fatigue, assess for anemia or other causes |
| 9 | Nausea (SNOMED-CT: 230141003) | Feeling of sickness or discomfort in the stomach | Moderate | Common side effect, may lead to non-adherence | Symptom diary, consider dose reduction or discontinuation if severe |
| 10 | Diarrhea (SNOMED-CT: 230141003) | Loose or watery stools | Moderate | Common side effect, may lead to dehydration | Monitor for dehydration, consider dose reduction or discontinuation if severe |

**Key:**
- SNOMED-CT: Systematized Nomenclature of Medicine -- Clinical Terms
- SCr: Serum creatinine
- eGFR: Estimated glomerular filtration rate
- CBC: Complete blood count
3. Patient-Specific Considerations
- **Age**: Elderly patients may have reduced renal function, increasing the risk of lactic acidosis.
- **Comorbidities**: Patients with renal impairment, heart failure, or liver disease are at higher risk for adverse effects.
- **Concurrent Medications**: Use with caution in patients taking drugs that affect renal function or increase the risk of lactic acidosis.

References
1. FDA. Metformin hydrochloride extended-release tablets, for oral use. 2023.
2. UK Prospective Diabetes Study (UKPDS) Group. Effect of intensive blood-glucose control with metformin on complications in overweight patients with type 2 diabetes (UKPDS 34). Lancet. 1998;352(9131):854-865.
3. American Diabetes Association. Standards of Medical Care in Diabetes—2023. Diabetes Care. 2023;46(Suppl 1):S1-S232.
Constraints:
- Cite sources using AMA format (e.g., NEJM 2023; 388:123-135)
- Exclude speculative associations
- Use SNOMED-CT terms
Constraints:
- Cite sources using AMA format (e.g., NEJM 2023; 388:123-135)
- Exclude speculative associations
- Use SNOMED-CT terms

Deliverable: A clinically actionable, evidence-based, and consistent categorized ranking of symptoms (with SNOMED-CT terms) and signs to support professional decision-making in patient treatment and care."""
                # Process the prompt but don't display it in the chat history
                process_user_input(physician_prompt, display_input=False)
            else:
                st.error("No user input found. Please enter a drug name first.")
        else:
            st.error("No user input found. Please enter a drug name first.")

with col2:
    if st.button("🤒 Patient Hand-out", use_container_width=True):
        # Extract drug names from the most recent user input
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            recent_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "User"]
            if recent_messages:
                user_input = recent_messages[-1]["content"]
                # Form the patient prompt
                patient_prompt = f"""Act as a clinical decision support tool. For the drug {user_input}mentioned in the user query:

1. Identify its primary therapeutic use based on FDA labeling and clinical guidelines.
2. Systematically categorize symptoms to monitor regularly into:
   - Medication Ineffectiveness (failure to achieve intended effect)
   - Unintended Side Effects (adverse drug reactions)
   - Disease Progression (worsening underlying condition)
3. Rank symptoms within each category by clinical urgency (Critical/High/Moderate) using these criteria:
   - Likelihood of severe harm if untreated
   - Time sensitivity for intervention
   - Strength of association with the drug/condition
4. For each symptom, provide:
   - Observable features: Early signals alerting the patient 
   - Rationale: Pathophysiological basis+ evidence from ≥1 peer-reviewed study
   - Monitoring Guidance: Frequency, tools (e.g., lab tests, validated scales), and red flags
5. Format results in a structured table with columns:
	| Rank | Symptom (SNOMED-CT) | Features | Priority | Rationale | Monitoring Guidance |
6. Prioritize symptoms mentioned in drug monographs (e.g., FDA Black Box Warnings)
7. Extract only the list of features associated with critical and high priority in the same rank order as in the table.
8. For each of the feature, provide additional information to the patients with defining observable signals/indicators for the patient to monitor in his body on his own and get alerted.
9. Include patient-specific considerations: Age, comorbidities, concurrent medications.
Example Output Structure for Metformin (Patient Prompt):
1. Severe Metabolic Acidosis, Often with Renal Impairment
Definition: Metabolic acidosis is a condition in which the body produces too much acid or the kidneys are not removing enough acid from the body. Renal impairment means the kidneys are not functioning properly.

Observable Signals/Indicators:

Fatigue: Feeling unusually tired or weak.
Shortness of Breath: Difficulty breathing, especially during physical activity.
Confusion: Difficulty thinking clearly or concentrating.
Nausea and Vomiting: Feeling sick to the stomach or actually vomiting.
Abdominal Pain: Pain or discomfort in the stomach area.
Muscle Weakness: Feeling weak or having difficulty moving.
Rapid Breathing: Breathing faster than usual.
Fruity Breath Odor: A sweet, fruity smell on the breath, which can be a sign of ketones (a byproduct of the body breaking down fat for energy).
Decreased Urine Output: Producing less urine than usual.
Swelling: Swelling in the legs, ankles, or feet, which can be a sign of fluid retention.
What to Do:

Monitor your urine output and note any changes.
Keep track of your breathing rate and any shortness of breath.
Note any changes in your mental status, such as confusion or difficulty concentrating.
Report any of these symptoms to your healthcare provider immediately.
2. Nausea and/or Diarrhea
Definition: Nausea is a feeling of sickness or discomfort in the stomach that often precedes vomiting. Diarrhea is loose, watery stools that occur more frequently than usual.

Observable Signals/Indicators:

Nausea: Feeling sick to the stomach, often accompanied by a desire to vomit.
Vomiting: Actual expulsion of stomach contents through the mouth.
Abdominal Pain: Pain or cramping in the stomach area.
Bloating: Feeling full or bloated.
Loss of Appetite: Not feeling hungry or having a decreased desire to eat.
Fever: Elevated body temperature.
Dehydration: Signs of dehydration include dry mouth, dark urine, and decreased urine output.
Fatigue: Feeling unusually tired or weak.
Headache: Pain in the head.
Dizziness: Feeling lightheaded or unsteady.
What to Do:

Stay hydrated by drinking plenty of fluids.
Avoid foods that may aggravate nausea or diarrhea, such as spicy or fatty foods.
Monitor your bowel movements and note any changes in frequency or consistency.
Report any of these symptoms to your healthcare provider, especially if they persist or worsen.
3. Low Blood Glucose Levels
Definition: Low blood glucose levels, also known as hypoglycemia, occur when the blood sugar level is below the normal range. This can happen due to various reasons, including medication side effects.

Observable Signals/Indicators:

Sweating: Excessive sweating, often without a clear cause.
Shakiness: Feeling shaky or trembling.
Hunger: Feeling hungry, even if you have eaten recently.
Irritability: Feeling irritable or anxious.
Fatigue: Feeling unusually tired or weak.
Dizziness: Feeling lightheaded or unsteady.
Confusion: Difficulty thinking clearly or concentrating.
Blurred Vision: Difficulty seeing clearly.
Headache: Pain in the head.
Rapid Heartbeat: Feeling like your heart is racing.
What to Do:

Carry a source of quick-acting glucose, such as glucose tablets or hard candy.
Monitor your blood glucose levels regularly, especially if you are on medication that can lower blood sugar.
Eat a balanced meal or snack if you experience symptoms of low blood sugar.
Report any of these symptoms to your healthcare provider, especially if they occur frequently or are severe.
General Advice:
Always keep a list of your medications and any side effects you experience.
Regularly communicate with your healthcare provider about any changes in your health or symptoms.
Follow your healthcare provider's instructions for monitoring and managing your condition.
If you experience any severe symptoms or have concerns, seek medical attention immediately.

 
Caution: As it is a Patient Hand-out material, DO NOT provide any professional information to the patients other than those mentioned in the :deliverable" section. Strict adherence to the "Example Output Structure for Metformin (Patient Prompt)" provided for generating the output is mandatory.
Deliverable: A clinically actionable, evidence-based, and consistently categorized ranking of symptoms and signs for patient education. I. The output should provide a structured response outlining the features of critical and high-priority symptoms, with clearly defined observable signals that empower patients to monitor their condition independently and receive timely alerts. II. DO NOT Present the Title (for example, Metformin: Clinical Decision Support Tool), 1. "Primary Therapeutic Use" section, 2. "Symptom Categorization and Ranking including the Table" section and 4. the "Patient-Specific Considerations" section. III.DO NOT Present the final outro paragraph(for example, "This structured response provides a clinically actionable, evidence-based, and consistently categorized ranking of symptoms and signs for patient education, empowering patients to monitor their condition independently and receive timely alerts)". Avoid presenting the Table in the output. Avoid presenting the main leading Title in the output."""
                # Process the prompt but don't display it in the chat history
                process_user_input(patient_prompt, display_input=False)
            else:
                st.error("No user input found. Please enter a drug name first.")
        else:
            st.error("No user input found. Please enter a drug name first.")

with col3:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.translations = []
# Text input
user_text = st.chat_input("Type your message... 							velanAI_Khel : கேள்!")
if user_text:
    process_user_input(user_text)

# Display chat history with unique keys
with st.container():
    for index, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"].lower()):
            render_chat_message(msg["role"], msg["content"], index)

# Download functionality with two buttons
# Download functionality with two buttons
if st.session_state.chat_history:
    # st.markdown("### Download Options")
    
    # Original chat text (full history)
    chat_text = "\n\n".join([f"{msg['role']}: {msg['content']}" 
                           for msg in st.session_state.chat_history])
    
    # Get only the last translated message (if available)
    last_translated_text = ""
    if st.session_state.translations and translation_language != "None":
        # Find the last assistant message in translations
        assistant_messages = [msg for msg in st.session_state.translations if msg['role'] == "Assistant"]
        if assistant_messages:
            last_message = assistant_messages[-1]
            last_translated_text = f"{last_message['role']}: {last_message.get('content', 'No translation')}"
    
    # Create columns for download buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.download_button(
            label="📥 Download English Chat",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        disabled = not (last_translated_text and translation_language != "None")
        st.download_button(
            label=f"📥 Download {translation_language or 'Translated'} Chat",
            data=last_translated_text if not disabled else "No translations available",
            file_name=f"last_message_{translation_language.lower() if translation_language != 'None' else 'translated'}.txt",
            mime="text/plain",
            disabled=disabled,
            use_container_width=True
        )

# Add extra space at bottom to ensure visibility of all content
st.markdown("<div class='download-section'></div>", unsafe_allow_html=True)
