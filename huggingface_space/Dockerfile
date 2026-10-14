FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# HuggingFace Spaces requires the app to run on port 7860
EXPOSE 7860

# Launch Gradio app (Gradio auto-detects HF_SPACE env and uses correct port)
CMD ["python", "app.py"]
