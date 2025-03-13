FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl gcc g++ make libffi-dev libssl-dev \
    && apt-get clean

RUN pip install --upgrade pip

RUN curl -sSL https://github.com/jwilder/dockerize/releases/download/v0.6.1/dockerize-linux-amd64-v0.6.1.tar.gz | tar -xz -C /usr/local/bin

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "main.py"]
