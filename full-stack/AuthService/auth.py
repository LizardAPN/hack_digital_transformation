from fastapi import FastAPI, Cookie, HTTPException, Response
from pydantic import BaseModel
import psycopg2
import os
import uuid
import hashlib

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")

class UserRegister(BaseModel):
    """Модель данных для регистрации пользователя"""
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    """Модель данных для входа пользователя"""
    name: str
    password: str

def check_single_quote(*args):
    """
    Проверка наличия одинарных кавычек в аргументах для предотвращения SQL-инъекций
    
    Параметры
    ----------
    *args : tuple
        Аргументы для проверки
    """
    for arg in args:
        if isinstance(arg, str) and "'" in arg:
            raise HTTPException(status_code=400, detail="Недопустимый ввод: содержит одинарную кавычку")

@app.get("/")
def health():
    """Проверка состояния сервиса"""
    return {"Status": "ok"}

@app.get("/api/auth")
def auth_get(session_token: str = Cookie(None)):
    """
    Проверка аутентификации пользователя по токену сессии для GET запросов
    
    Параметры
    ----------
    session_token : str, optional
        Токен сессии пользователя из cookie
        
    Возвращает
    -------
    Response
        Ответ со статусом 200 если пользователь аутентифицирован, 401 если нет
        
    Выбрасывает
    ------
    HTTPException
        При ошибке базы данных или отсутствии токена сессии
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="Токен сессии не предоставлен")
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
            raise HTTPException(status_code=401, detail="Недействительный токен сессии")
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
    
@app.post("/api/auth")
def auth_post(session_token: str = Cookie(None)):
    """
    Проверка аутентификации пользователя по токену сессии для POST запросов
    
    Параметры
    ----------
    session_token : str, optional
        Токен сессии пользователя из cookie
        
    Возвращает
    -------
    Response
        Ответ со статусом 200 если пользователь аутентифицирован, 401 если нет
        
    Выбрасывает
    ------
    HTTPException
        При ошибке базы данных или отсутствии токена сессии
    """
    if not session_token:
        raise HTTPException(status_code=401, detail="Токен сессии не предоставлен")
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
            raise HTTPException(status_code=401, detail="Недействительный токен сессии")
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Ошибка базы данных")

@app.post("/api/register")
def register(user_data: UserRegister):
    """
    Регистрация нового пользователя в системе
    
    Параметры
    ----------
    user_data : UserRegister
        Данные пользователя для регистрации
        
    Возвращает
    -------
    Response
        Ответ со статусом 200 и токеном сессии в cookie при успешной регистрации
        
    Выбрасывает
    ------
    HTTPException
        При ошибке базы данных или если пользователь уже существует
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        check_single_quote(user_data.email)
        cur = conn.cursor()
        
        # Проверка, существует ли пользователь с таким email
        cur.execute("SELECT * FROM users WHERE email = %s", (user_data.email,))
        existing_user = cur.fetchone()
        
        if existing_user:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Пользователь уже существует")
        
        # Хеширование пароля
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        check_single_quote(user_data.name, user_data.email)
        # Генерация токена сессии
        session_token = str(uuid.uuid4()).replace('-', '')[:64]
        
        # Вставка нового пользователя
        cur.execute("INSERT INTO users (name, email, password_hash, session_token) VALUES (%s, %s, %s, %s)", 
                   (user_data.name, user_data.email, password_hash, session_token))
        conn.commit()
        
        cur.close()
        conn.close()
        
        # Создание ответа с cookie
        response = Response(status_code=200)
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        return response
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Ошибка базы данных")

@app.post("/api/login")
def login(user_data: UserLogin):
    """
    Аутентификация пользователя по имени и паролю
    
    Параметры
    ----------
    user_data : UserLogin
        Данные пользователя для входа
        
    Возвращает
    -------
    Response
        Ответ со статусом 200 и токеном сессии в cookie при успешной аутентификации
        
    Выбрасывает
    ------
    HTTPException
        При ошибке базы данных или неверных учетных данных
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        check_single_quote(user_data.name)
        
        # Хеширование предоставленного пароля
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        # Проверка учетных данных пользователя
        cur.execute("SELECT * FROM users WHERE name = %s AND password_hash = %s", 
                   (user_data.name, password_hash))
        user = cur.fetchone()
        
        if not user:
            cur.close()
            conn.close()
            raise HTTPException(status_code=401, detail="Неверные учетные данные")
        
        # Генерация нового токена сессии
        session_token = str(uuid.uuid4()).replace('-', '')[:64]
        
        # Обновление токена сессии пользователя
        cur.execute("UPDATE users SET session_token = %s WHERE name = %s", 
                   (session_token, user_data.name))
        conn.commit()
        
        cur.close()
        conn.close()
        
        # Создание ответа с cookie
        response = Response(status_code=200)
        response.set_cookie(key="session_token", value=session_token, httponly=True)
        return response
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail="Ошибка базы данных")
