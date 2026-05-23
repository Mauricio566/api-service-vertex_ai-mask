FROM python:3.10-slim

WORKDIR /app

# dependencias sistema (opencv)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*


# copiar requirements
COPY requirements.txt .

# instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# copiar código y modelo
COPY . .

# puerto
ENV PORT=8080

# correr API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]