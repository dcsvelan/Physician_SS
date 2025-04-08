# Initialize Redis cache
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# FDA API endpoint
FDA_API_URL = "https://api.fda.gov/drug/label.json"

def get_drug_info(drug_name):
    """Fetch drug label information from FDA API with Redis caching."""
    cached_data = redis_client.get(drug_name.lower())
    if cached_data:
        return json.loads(cached_data)
    
    params = {"search": f"openfda.brand_name:{drug_name}", "limit": 1}
    response = requests.get(FDA_API_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            redis_client.setex(drug_name.lower(), 86400, json.dumps(data))  # Cache for 24 hours
            return data
    return None

def extract_drug_name(user_input):
    """Extract potential drug names from user input using regex."""
    drug_pattern = r"\b[A-Z][a-z]+\b"  # Simple heuristic for capitalized words
    matches = re.findall(drug_pattern, user_input)
    return matches[0] if matches else None

def chatbot_response(user_input):
    """Generate chatbot response based on user query."""
    drug_name = extract_drug_name(user_input)
    if drug_name:
        drug_info = get_drug_info(drug_name)
        if drug_info:
            result = drug_info['results'][0]
            response_text = f"**Information about {drug_name}:**\n\n"
            response_text += f"- **Purpose:** {result.get('purpose', 'Not available')}\n"
            response_text += f"- **Dosage:** {result.get('dosage_and_administration', 'Not available')}\n"
            response_text += f"- **Warnings:** {result.get('warnings', 'Not available')}\n"
            return response_text
        return f"Sorry, no FDA label information found for **{drug_name}**."
    return "I couldn't identify a drug name in your query. Please specify a drug name."

# Streamlit UI
st.title("Drug Information Chatbot 💊")
st.write("Ask about a drug, and I'll fetch information from the FDA database.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Type your question here...")
if user_query:
    # Append user input
    st.session_state["messages"].append({"role": "user", "content": user_query})
    
    # Process query and get chatbot response
    response = chatbot_response(user_query)

    # Append chatbot response
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Display chat history
    with st.chat_message("assistant"):
        st.markdown(response)