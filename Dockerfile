FROM python:3.11-slim

WORKDIR /code

# System libraries needed by Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# If you're baking the trained weights into the image, make sure
# app/model/ripeness_model.pt exists before building. Otherwise set
# the MODEL_URL environment variable at deploy time and the app will
# download it on startup (see app/model_utils.py).

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
