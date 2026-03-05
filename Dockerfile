FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

# Проверяем что код проходит линтеры при сборке
RUN ruff check . && mypy --strict .

CMD ["python", "main.py"]
