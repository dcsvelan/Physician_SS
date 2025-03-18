# VelanAI_Khel: Essential Med Info

This is a Streamlit application that fetches FDA drug label information and RxNav classifications. It uses Google Authentication (configured via Render secrets) and caches FDA responses in Redis for fast, persistent access.

## Features

- **FDA Data Fetching:** Retrieves FDA drug labels using the FDA API.
- **RxNav Classifications:** Fetches drug class information from the RxNav API.
- **Google Authentication:** (Configured via secrets) to protect your app.
- **Redis Caching:** Caches FDA responses for 1 hour to improve performance.
- **OCR Extraction:** Extracts drug names from uploaded images using Tesseract.

## File Structure

