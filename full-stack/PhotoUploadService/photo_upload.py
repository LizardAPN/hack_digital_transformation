from fastapi import FastAPI, File, UploadFile, Cookie, HTTPException, Response, Depends
from pydantic import BaseModel
import psycopg2
import os
import uuid
import boto3
from botocore.exceptions import ClientError
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
AWS_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_KEY")
AWS_ENDPOINT_URL = os.getenv("S3_ENDPOINT")
S3_BUCKET_NAME = os.getenv("S3_BUCKET")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=AWS_ENDPOINT_URL,
)

class PhotoResponse(BaseModel):
    id: int
    owner_id: int
    photo_url: str
    created_at: str

class UserPhotosResponse(BaseModel):
    photos: List[PhotoResponse]

def check_single_quote(*args):
    """
    Check if any of the arguments contain single quotes
    """
    for arg in args:
        if isinstance(arg, str) and "'" in arg:
            raise HTTPException(status_code=400, detail="Invalid input: contains single quote")

def get_user_id_from_session(session_token: str) -> int:
    """
    Get user ID from session token
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="No session token provided")
    
    check_single_quote(session_token)
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE session_token = %s", (session_token,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user:
            return user[0]  # Return user ID
        else:
            raise HTTPException(status_code=401, detail="Invalid session token")
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/")
def health():
    return {"Status": "ok"}

@app.post("/api/photo_upload")
async def upload_photo(
    file: UploadFile = File(...),
    session_token: str = Cookie(None)
):
    """
    Upload a photo file to S3 and save metadata to database
    """
    # Authenticate user
    owner_id = get_user_id_from_session(session_token)
    
    # Validate file
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate unique filename
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    
    try:
        # Upload file to S3
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET_NAME,
            unique_filename,
            ExtraArgs={'ContentType': file.content_type}
        )
        
        # Generate S3 URL
        photo_url = unique_filename
        
        # Save metadata to database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute(
            "INSERT INTO user_photos (owner_id, photo_url) VALUES (%s, %s) RETURNING id, created_at",
            (owner_id, photo_url)
        )
        
        result = cur.fetchone()
        photo_id = result[0]
        created_at = result[1]
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "id": photo_id,
            "photo_url": photo_url,
            "created_at": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
        }
        
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file to S3")
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/photos", response_model=UserPhotosResponse)
def get_user_photos(session_token: str = Cookie(None)):
    """
    Fetch all photos for the authenticated user
    """
    # Authenticate user
    owner_id = get_user_id_from_session(session_token)
    
    try:
        # Fetch photos from database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute(
            "SELECT id, owner_id, photo_url, created_at FROM user_photos WHERE owner_id = %s ORDER BY created_at DESC",
            (owner_id,)
        )
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Format response
        photos = []
        for row in rows:
            photos.append({
                "id": row[0],
                "owner_id": row[1],
                "photo_url": f"https://storage.yandexcloud.net/{S3_BUCKET_NAME}/{row[2]}",
                "created_at": row[3].isoformat() if hasattr(row[3], 'isoformat') else str(row[3])
            })
        
        return {"photos": photos}
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")