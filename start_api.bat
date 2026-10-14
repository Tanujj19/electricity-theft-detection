@echo off
echo Starting FastAPI server on http://localhost:8000
echo API docs available at http://localhost:8000/docs
C:\Users\HP\AppData\Local\Programs\Python\Python310\Scripts\uvicorn.exe api.main:app --reload --port 8000
