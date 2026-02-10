# ---------- BASE IMAGE ----------
FROM python:3.10-slim

# ---------- ENV ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- WORKDIR ----------
WORKDIR /app

# ---------- SYSTEM DEPS ----------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- INSTALL PYTHON DEPS ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- COPY PROJECT ----------
COPY . .

# ---------- EXPOSE ----------
EXPOSE 8000

# ---------- RUN ----------
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
