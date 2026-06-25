import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from datetime import datetime, timedelta
from tes0 import ai, bd
from database import init_db, SessionLocal, User, Token, ChatHistory

# 初始化数据库
init_db()

app = FastAPI(title="RAG Agent 知识库问答系统", description="带登录和文档管理的问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 默认管理员账号
def init_default_admin():
    db = SessionLocal()
    try:
        # 检查是否已有管理员
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin_user = User(username="admin", password="admin123", role="admin")
            db.add(admin_user)
            db.commit()
            print("默认管理员账号已创建: admin / admin123")
        
        # 检查是否已有测试用户
        user = db.query(User).filter(User.username == "user").first()
        if not user:
            normal_user = User(username="user", password="123456", role="user")
            db.add(normal_user)
            db.commit()
            print("默认用户账号已创建: user / 123456")
    finally:
        db.close()

init_default_admin()


class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

class ChatMessage(BaseModel):
    role: str
    content: str

class LoginRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class RegisterRequest(BaseModel):
    username: str
    password: str


def create_token(username: str, role: str) -> str:
    token = str(uuid.uuid4())
    db = SessionLocal()
    try:
        db_token = Token(
            token=token,
            username=username,
            role=role,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        db.add(db_token)
        db.commit()
    finally:
        db.close()
    return token

def verify_token(token: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        db_token = db.query(Token).filter(Token.token == token).first()
        if not db_token:
            return None
        
        if datetime.utcnow() > db_token.expires_at:
            db.delete(db_token)
            db.commit()
            return None
        
        return {
            "username": db_token.username,
            "role": db_token.role
        }
    finally:
        db.close()

from fastapi import Header

def get_token_from_header(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith('Bearer '):
        return authorization[7:]
    return None

def get_current_user(token: Optional[str] = None):
    if not token:
        return None
    return verify_token(token)

def get_answer(question: str, username: str) -> str:
    question = question.strip()
    ai_instance = ai()
    answer = ai_instance.yong(question)
    
    # 保存对话到数据库
    db = SessionLocal()
    try:
        user_msg = ChatHistory(username=username, role="user", content=question)
        assistant_msg = ChatHistory(username=username, role="assistant", content=answer)
        db.add(user_msg)
        db.add(assistant_msg)
        db.commit()
        
        # 自动清理超过 100 条的旧记录（每个用户最多保留 100 条）
        MAX_HISTORY = 100
        records = db.query(ChatHistory).filter(ChatHistory.username == username).order_by(ChatHistory.created_at).all()
        if len(records) > MAX_HISTORY:
            records_to_delete = records[:len(records) - MAX_HISTORY]
            for record in records_to_delete:
                db.delete(record)
            db.commit()
    finally:
        db.close()
    
    return answer

@app.get("/")
async def root():
    return RedirectResponse(url="/static/login.html")

@app.post("/api/register")
async def register(request: RegisterRequest):
    db = SessionLocal()
    try:
        # 检查用户名是否已存在
        existing_user = db.query(User).filter(User.username == request.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="用户名已存在")
        
        # 创建新用户
        new_user = User(username=request.username, password=request.password, role="user")
        db.add(new_user)
        db.commit()
        
        return {"message": "注册成功"}
    finally:
        db.close()

@app.post("/api/login")
async def login(request: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == request.username).first()
        
        if not user or user.password != request.password:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        
        token = create_token(user.username, user.role)
        return {"access_token": token, "token_type": "bearer", "role": user.role}
    finally:
        db.close()

@app.get("/api/me")
async def get_me(authorization: Optional[str] = Header(None)):
    token = get_token_from_header(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    return user

@app.post("/ask", response_model=AnswerResponse)
async def ask(question_request: QuestionRequest, authorization: Optional[str] = Header(None)):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
        
    try:
        question = question_request.question
        
        if question == "清空":
            # 清空该用户的对话历史
            db = SessionLocal()
            try:
                db.query(ChatHistory).filter(ChatHistory.username == user["username"]).delete()
                db.commit()
            finally:
                db.close()
            return {"answer": "对话历史已清空"}
        
        answer = get_answer(question, user["username"])
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clear-history")
async def clear_history(authorization: Optional[str] = Header(None)):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    
    db = SessionLocal()
    try:
        db.query(ChatHistory).filter(ChatHistory.username == user["username"]).delete()
        db.commit()
    finally:
        db.close()
    
    return {"message": "对话历史已清空"}

@app.get("/history", response_model=List[ChatMessage])
async def get_history(authorization: Optional[str] = Header(None)):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    
    db = SessionLocal()
    try:
        messages = db.query(ChatHistory).filter(
            ChatHistory.username == user["username"]
        ).order_by(ChatHistory.created_at).all()
        
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    finally:
        db.close()

DOCUMENTS_DIR = "./knowledge_base"
FENCIQI_DIR = "./fenci"

@app.get("/api/documents")
async def get_documents(authorization: Optional[str] = Header(None)):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    
    documents = []
    if os.path.exists(DOCUMENTS_DIR):
        for filename in os.listdir(DOCUMENTS_DIR):
            if not filename.startswith('~$'):
                filepath = os.path.join(DOCUMENTS_DIR, filename)
                if os.path.isfile(filepath):
                    documents.append({
                        "filename": filename,
                        "size": os.path.getsize(filepath),
                        "mtime": os.path.getmtime(filepath)
                    })
    return {"documents": documents}

@app.get("/ap")
async def get_documents(authorization: Optional[str] = Header(None)):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    
    documents = []
    if os.path.exists(FENCIQI_DIR):
        for filename in os.listdir(FENCIQI_DIR):
            if not filename.startswith('~$'):
                filepath = os.path.join(FENCIQI_DIR, filename)
                if os.path.isfile(filepath):
                    documents.append({
                        "filename": filename,
                        "size": os.path.getsize(filepath),
                        "mtime": os.path.getmtime(filepath)
                    })
    return {"documents": documents}
@app.post("/upwendang")
async def up(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    
  
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".txt":
        raise HTTPException(status_code=400, detail="只允许上传TXT文件")
    
    os.makedirs(FENCIQI_DIR, exist_ok=True)
    
    file_path = os.path.join(FENCIQI_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    duixiang = bd()
    duixiang.rufenciwendang(file.filename)
    return {"message": f"文件 {file.filename} 上传成功"}

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    
    allowed_extensions = [".pdf", ".docx", ".doc"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="只允许上传PDF和DOCX文件")
    
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    duixiang = bd()
    duixiang.qie(file_path)
    return {"message": f"文件 {file.filename} 上传成功"}
@app.delete("/a/{filename}")
async def d(
    filename: str,
    authorization: Optional[str] = Header(None)
):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    
    
    xiang = bd()
    xiang.fencishanwen(filename)
    return {"message": f"文件 {filename} 删除成功"}
@app.delete("/api/documents/{filename}")
async def delete_document(
    filename: str,
    authorization: Optional[str] = Header(None)
):
    token = get_token_from_header(authorization)
    user = get_current_user(token)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="权限不足")
    
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    xiang = bd()
    xiang.shan(file_path)

    os.remove(file_path)
    return {"message": f"文件 {filename} 删除成功"}

@app.post("/api/logout")
async def logout(token: Optional[str] = None):
    if token:
        db = SessionLocal()
        try:
            db_token = db.query(Token).filter(Token.token == token).first()
            if db_token:
                db.delete(db_token)
                db.commit()
        finally:
            db.close()
    return {"message": "退出成功"}

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9090)
