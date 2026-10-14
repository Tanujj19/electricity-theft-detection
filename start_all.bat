@echo off
echo Starting Electricity Theft Detection System...
echo.
echo [1/2] Starting FastAPI on http://localhost:8000
start "FastAPI Server" cmd /k "cd /d C:\Users\HP\Documents\Electric Theft Detection\electricity-theft-detection && C:\Users\HP\AppData\Local\Programs\Python\Python310\Scripts\uvicorn.exe api.main:app --reload --port 8000"

echo Waiting for API to start...
timeout /t 4 /nobreak >nul

echo [2/2] Starting Streamlit on http://localhost:8501
start "Streamlit UI" cmd /k "cd /d C:\Users\HP\Documents\Electric Theft Detection\electricity-theft-detection && C:\Users\HP\AppData\Local\Programs\Python\Python310\Scripts\streamlit.exe run app/app.py"

echo.
echo Both services started!
echo API:      http://localhost:8000/docs
echo UI:       http://localhost:8501
echo.
pause
