import os
import io
import json
from google.cloud import vision
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from PIL import Image
import streamlit as st

# Initialize Google APIs
def load_google_services():
    credentials_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    
    # Initialize Vision and Drive clients
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    drive_service = build('drive', 'v3', credentials=credentials)
    return vision_client, drive_service

# Retrieve image files from a Google Drive folder
def get_images_from_drive(drive_service, folder_id):
    results = drive_service.files().list(q=f"'{folder_id}' in parents and mimeType contains 'image/'").execute()
    items = results.get('files', [])
    return items

# Download and analyze images
def analyze_and_label_images(vision_client, drive_service, items, output_folder_id):
    for item in items:
        request = drive_service.files().get_media(fileId=item['id'])
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        downloader.next_chunk()
        file_stream.seek(0)
        
        # Vision API image analysis
        image = vision.Image(content=file_stream.read())
        response = vision_client.label_detection(image=image)
        labels = [label.description for label in response.label_annotations]
        
        # Display results in Streamlit
        label_text = ', '.join(labels)
        st.write(f"Image: {item['name']}, Labels: {label_text}")
        
        # Save labeled image back to Google Drive (optional)
        output_file_metadata = {'name': f"Labeled_{item['name']}", 'parents': [output_folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(file_stream.read()), mimetype='image/jpeg')
        drive_service.files().create(body=output_file_metadata, media_body=media).execute()

# Streamlit app structure
st.title("Automated Image Analysis and Labeling with Google Vision API")

# Google Drive folder input
folder_id = st.text_input("Enter the Google Drive Folder ID:")

if folder_id:
    vision_client, drive_service = load_google_services()
    items = get_images_from_drive(drive_service, folder_id)
    st.write(f"Found {len(items)} images in the folder.")
    
    if st.button("Analyze Images"):
        output_folder_id = st.text_input("Enter the Output Google Drive Folder ID:")
        analyze_and_label_images(vision_client, drive_service, items, output_folder_id)
        st.write("Image analysis and labeling completed!")



