import streamlit as st
import requests
import random
import pytesseract
from PIL import Image
import io
import re
import concurrent.futures
import json

# -------------------------
# Google Authentication (Render)
# -------------------------
try:
    google_auth_required = st.secrets.get("google_auth", False)
except Exception:
    google_auth_required = False

if google_auth_required:
    st.error("Google authentication is required. Please log in using your Google account.")
    st.stop()

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
FDA_FIELDS = ["purpose", "adverse_reactions", "drug_and_or_laboratory_test_interactions", "drug_interactions",
    "ask_doctor", "ask_doctor_or_pharmacist", "do_not_use", "information_for_patients",
    "instructions_for_use", "other_safety_information", "patient_medication_information",
    "spl_medguide", "spl_patient_package_insert", "stop_use", "when_using", "boxed_warning",
    "general_precautions", "precautions", "user_safety_warnings", "warnings", "contraindications",
    "geriatric_use", "labor_and_delivery", "mechanism_of_action", "nursing_mothers", "overdosage",
    "pediatric_use", "pregnancy", "pregnancy_or_breast_feeding", "safe_handling_warning",
    "use_in_specific_populations"]

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
st.title("VelanAI_Khel: Essential Med Info")
st.write("### **Why did the Chicken cross the road?!**")
st.write(f"**{random.choice(jokes)}**")

# Option to input drug name manually
drug_name_input = st.text_input("Enter drug name(s) (comma-separated)")

# Option to upload an image containing drug label information
uploaded_image = st.file_uploader("Or upload an image for OCR extraction", type=["png", "jpg", "jpeg"])

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
                    # Display FDA data fields
                    for field in FDA_FIELDS:
                        if field in result["fda"]:
                            field_value = result["fda"][field]
                            if field_value and field_value != "No data available":
                                md_text += f"<details><summary>{field.replace('_', ' ').capitalize()}</summary>"
                                if isinstance(field_value, list):
                                    md_text += "<br>".join(field_value)
                                else:
                                    md_text += field_value
                                md_text += "</details>\n"
                    md_text += "\n"
                    results_markdown += md_text
                # Update UI incrementally as each drug's result is appended
                placeholder.markdown(results_markdown, unsafe_allow_html=True)
