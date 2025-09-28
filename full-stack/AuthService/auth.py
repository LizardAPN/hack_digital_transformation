from fastapi import FastAPI, Cookie, HTTPException, Response
from pydantic import BaseModel
import psycopg2
import os
import uuid
import hashlib

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    name: str
    password: str

def check_single_quote(*args):
    """
    Check if any of the arguments contain single quotes
    """
    for arg in args:
        if isinstance(arg, str) and "'" in arg:
            raise HTTPException(status_code=400, detail="Invalid input: contains single quote")

@app.get("/")
def health():
    return {"Status": "ok"}

@app.get("/api/auth")
def auth(session_token: str = Cookie(None)):
    """
    make a request to DATABASE_URL and check if the user has session token
    if user does not have session token, return 401
    if user has session token, return 200
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="No session token provided")
    check_single_quote(session_token)
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE session_token = %s", (session_token,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user:
            return Response(status_code=200)
        else:
            raise HTTPException(status_code=401, detail="Invalid session token")
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error")
    
@app.post("/api/auth")
def auth(session_token: str = Cookie(None)):
    """
    make a request to DATABASE_URL and check if the user has session token
    if user does not have session token, return 401
    if user has session token, return 200
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="No session token provided")
    check_single_quote(session_token)
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE session_token = %s", (session_token,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        
        if user:
            return Response(status_code=200)
        else:
            raise HTTPException(status_code=401, detail="Invalid session token")
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/api/register")
def register(user_data: UserRegister):
    """
    make a request to DATABASE_URL and register the user
    return the session token in cookie and status 200
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        check_single_quote(user_data.email)
        cur = conn.cursor()
        
        # Check if user already exists
        cur.execute("SELECT * FROM users WHERE email = %s", (user_data.email,))
        existing_user = cur.fetchone()
        
        if existing_user:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash the password
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        check_single_quote(user_data.name, user_data.email)
        # Generate session token
        session_token = str(uuid.uuid4()).replace('-', '')[:64]
        
        # Insert new user
        cur.execute("INSERT INTO users (name, email, password_hash, session_token) VALUES (%s, %s, %s, %s)", 
                   (user_data.name, user_data.email, password_hash, session_token))
        conn.commit()
        
        cur.close()
        conn.close()
        
        # Create response with cookie
        response = Response(status_code=200)
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        return response
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/api/login")
def login(user_data: UserLogin):
    """
    make a request to DATABASE_URL and check user credentials
    return the session token in cookie and status 200
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        check_single_quote(user_data.name)
        
        # Hash the provided password
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        # Check user credentials
        cur.execute("SELECT * FROM users WHERE name = %s AND password_hash = %s", 
                   (user_data.name, password_hash))
        user = cur.fetchone()
        
        if not user:
            cur.close()
            conn.close()
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate new session token
        session_token = str(uuid.uuid4()).replace('-', '')[:64]
        
        # Update user's session token
        cur.execute("UPDATE users SET session_token = %s WHERE name = %s", 
                   (session_token, user_data.name))
        conn.commit()
        
        cur.close()
        conn.close()
        
        # Create response with cookie
        response = Response(status_code=200)
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        return response
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Database error")
