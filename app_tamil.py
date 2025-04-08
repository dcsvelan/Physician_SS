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

from deep_translator import GoogleTranslator  # Added import for translation


from dotenv import load_dotenv
from langchain_groq import ChatGroq
import speech_recognition as sr
import tempfile


# Load environment variables
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create a LLM Object
llm = ChatGroq(
    model="llama3-70b-8192", 
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
def load_user_credentials():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

AUTHORIZED_USERS = load_users()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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
        # Check for missing fields
        if not new_username or not new_password or not new_email or not new_phone or not new_address:
            st.sidebar.error("🚨 All fields are required!")
            return
        # Check for duplicate username
        if new_username in AUTHORIZED_USERS:
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
        AUTHORIZED_USERS[new_username] = {
            "password": new_password,
            "occupation": new_occupation,
            "email": new_email,
            "phone": new_phone,
            "address": new_address
        }
        save_users(AUTHORIZED_USERS)
        st.sidebar.success("✅ Registration successful! Please log in.")

def login():
    st.sidebar.title("🔐 User Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    login_button = st.sidebar.button("Login")
    if login_button:
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success(f"✅ Logged in as {username}")
        else:
            st.sidebar.error("🚫 Invalid credentials!")

def logout():
    st.session_state.authenticated = False
    st.sidebar.warning("Logged out. Please refresh.")

register()

if not st.session_state.authenticated:
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
st.title("# velanAI_khel")
st.title("Physician Pocket Reference")
st.write("### **Why did the Chicken cross the road?!**")
st.write(f"**{random.choice(jokes)}**")

# Option to input drug name manually
drug_name_input = st.text_input("Enter drug name(s) (comma-separated)")

# Option to upload an image containing drug label information
uploaded_image = st.file_uploader("Or upload your Prescription", type=["png", "jpg", "jpeg"])

# Determine input source: text or image (image takes precedence if uploaded)
if uploaded_image:
    st.info("Extracting text from the uploaded image...")
    extracted_text = extract_text_from_image(uploaded_image)
    st.write("Extracted Text:", extracted_text)
    # Split by comma or newline to handle multi-line OCR output
    drug_name_input = ",".join([name.strip() for name in re.split(r"[,\n]+", extracted_text) if name.strip()])

if st.button("Fetch"):
    if not drug_name_input:
        st.error("Please provide a drug name or upload an image.")
    else:
        # Split input into drug names using comma and newline as delimiters
        drug_names = [name.strip() for name in re.split(r"[,\n]+", drug_name_input) if name.strip()]
        placeholder = st.empty()  # Placeholder for incremental updates
        results_markdown = ""
        
        # Use ThreadPoolExecutor to fetch data concurrently for each drug
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(drug_names)) as executor:
            future_to_drug = {executor.submit(fetch_drug_data, drug): drug for drug in drug_names}
            for future in concurrent.futures.as_completed(future_to_drug):
                result = future.result()
                if "error" in result:
                    results_markdown += f"### {result['drug_name']}\n**Error:** {result['error']}\n\n"
                else:
                    md_text = f"### {result['drug_name']}\n"
                    # Display RxNav classification data
                    for category, items in result["rxnav"]["classes"].items():
                        if items:
                            md_text += f"- **{category}:** {', '.join(items)}\n"
                    # Display FDA data fields with enhanced text formatting
                    for field in FDA_FIELDS:
                        if field in result["fda"]:
                            field_value = result["fda"][field]
                            if field_value and field_value != "No data available":
                                # If field_value is a list, join the items first
                                if isinstance(field_value, list):
                                    combined_text = "\n".join(field_value)
                                else:
                                    combined_text = field_value
                                # Format the text for enhanced readability
                                formatted_field = format_text(combined_text)
                                md_text += f"<details><summary>{field.replace('_', ' ').capitalize()}</summary>\n"
                                md_text += formatted_field
                                md_text += "\n</details>\n"
                    md_text += "\n"
                    results_markdown += md_text
                # Update UI incrementally as each drug's result is appended
                placeholder.markdown(results_markdown, unsafe_allow_html=True)
#Create a LLM Object
llm  = ChatGroq(
                model           =   "llama3-70b-8192",
                temperature     =   0
                )

st.write("### **GenAI_Chatbot                  ... info not necessarily from FDA** ###")
# If the Global variable is not created, create it once.
# if 'chat_history' not in st.session_state:
    # st.session_state.chat_history = []

# Parse through the Global and print the Chat History
# for chats in st.session_state.chat_history:
   # with st.chat_message(chats["role"]):
       # st.markdown(chats['content'])


# st.title("AI Chatbot")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recording' not in st.session_state:
    st.session_state.recording = False

# Function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Press the button again to stop recording.")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            return f"Error: {str(e)}"

# Voice input button
st.button(
    "🎤 Push to Talk" if not st.session_state.recording else "⏹️ Stop Recording",
    on_click=lambda: toggle_recording(),
    key="voice_button"
)

def toggle_recording():
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        user_input = speech_to_text()
        if user_input and not user_input.startswith("Error"):
            process_user_input(user_input)
    st.session_state.recording = False

# Render the chat history (each message is rendered once)
for chats in st.session_state.chat_history:
    with st.chat_message(chats["role"]):
        st.markdown(chats["content"])

# Function to process user input
def process_user_input(user_input):
    if not user_input.strip():
        return
    # Append user input to chat history
    st.session_state.chat_history.append({"role": "User", "content": user_input})
    
    # Prepare prompt for the LLM
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant who answers user queries accurately."
    }
    human_msg = {
        "role": "user",
        "content": user_input
    }
    prompt = [system_msg, human_msg]
    
    with st.spinner("Thinking..."):
        llm_response = llm.invoke(prompt).content

    # Translate the response to Tamil
    response_tamil = GoogleTranslator(source="auto", target="ta").translate(llm_response)
    combined_response = f"{llm_response}\n\n**Tamil Translation:** {response_tamil}"
    
    # Append assistant response (combined text) to chat history
    st.session_state.chat_history.append({
        "role": "Assistant", 
        "content": combined_response
    })

# Fallback text input
user_text_input = st.chat_input("Type your message...", key="chat_input_1")
if user_text_input:
    process_user_input(user_text_input)