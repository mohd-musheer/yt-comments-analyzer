🎬 YouTube Comment Sentiment Analysis (Transformer + Quantized Model)

An AI-powered web application that analyzes YouTube comments in real time and generates sentiment insights using a dynamically quantized Transformer model.

This project combines NLP, FastAPI, YouTube Data API, interactive UI, and Dockerized deployment into a full-stack AI application.

:globe_with_meridians: Live App: https://youtube-comments-sentiment-analysis-ai.onrender.com
:link: GitHub: https://github.com/mohd-musheer/youtube-comments-sentiment-analysis
:whale: Docker: https://hub.docker.com/r/mohdmusheer/youtube-comments-sentiment-analysis


🚀 Project Overview

This system allows users to:

Paste a YouTube video URL (or select from demo videos)

Fetch up to 100 / 200 / 500 comments

Analyze comments using a Quantized Transformer model

Generate:

✅ Sentiment classification (Positive / Negative)

📊 Sentiment distribution charts

🎯 Overall confidence score

🖼️ Video title and thumbnail

View results in a modern interactive dashboard

🧠 What Makes This Project Unique

🔥 Uses Transformer-based model (MiniLM)

⚡ Dynamically quantized with PyTorch for faster CPU inference

🎯 Real-time YouTube comment fetching

🎨 Modern purple-yellow themed UI

📊 Interactive visual analytics (Pie & Bar charts)

🐳 Fully Dockerized backend

🏗️ System Architecture

User → Web UI → FastAPI Backend
↓
YouTube Data API v3 (comments + metadata)
↓
Quantized Transformer Model (MiniLM)
↓
Sentiment Prediction (Softmax Probabilities)
↓
Aggregation & Visualization (Charts)

🤖 Machine Learning Model

Base Model: MiniLM Transformer

Framework: PyTorch + Transformers

Optimization: Dynamic Quantization (INT8)

Inference Mode: CPU

Task: Binary Sentiment Classification

Positive

Negative

Output: Label + Confidence Score (%)

Quantization significantly reduces memory footprint and speeds up inference, making it suitable for lightweight deployments.

🛠️ Tech Stack
Backend

Python

FastAPI

PyTorch

HuggingFace Transformers

Google YouTube Data API v3

python-dotenv

Frontend

HTML

CSS (Purple-Yellow theme)

JavaScript

Chart.js

DevOps

Docker

Uvicorn ASGI Server

🐳 Docker Usage
Pull Image
docker pull mohdmusheer/yt-comment-analyser

Run Container
docker run -p 8000:8000 \
-e YOUTUBE_API_KEY=YOUR_API_KEY \
mohdmusheer/yt-comment-analyser


Then open:

http://localhost:8000

⚙️ Local Setup (Without Docker)
git clone https://github.com/mohd-musheer/youtube-comments-sentiment-analysis.git
cd youtube-comments-sentiment-analysis

pip install -r requirements.txt

uvicorn api:app --host 0.0.0.0 --port 8000 --reload


Make sure to set your environment variable:

Windows
setx YOUTUBE_API_KEY "YOUR_API_KEY"

Mac/Linux
export YOUTUBE_API_KEY="YOUR_API_KEY"

📊 Features

Accepts full YouTube URL or Video ID

Demo video selection (10 preloaded videos)

Interactive loading animation

Full-screen analytics view

Positive vs Negative breakdown

Confidence percentage display

Video metadata display (Title + Thumbnail)

🔐 Environment Variable

The application requires:

YOUTUBE_API_KEY


Get your API key from:
Google Cloud Console → Enable YouTube Data API v3 → Create API Key

📌 Use Cases

Social media sentiment research

YouTube community analysis

NLP demonstrations

AI portfolio projects

Academic presentations

Hackathons

👥 Team & Contribution

Developed collaboratively by Group 2.

This project demonstrates:

NLP model integration

Transformer optimization

API design

Full-stack AI system deployment

Docker containerization

⚠️ Notes

English comments only (as supported by model training)

Requires active YouTube API key

Performance depends on selected comment limit

📈 Future Improvements

Multi-class sentiment support

Emotion detection

Real-time streaming analysis

ONNX Runtime optimization

Cloud-native microservice deployment

Authentication & user history

🎉 Project Status

✔ Model trained and quantized
✔ Backend API implemented
✔ Interactive UI completed
✔ Docker image built and published

This project is deployment-ready and portfolio-ready.