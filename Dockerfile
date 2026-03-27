FROM python:3.12-slim

# system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install uv
RUN pip install uv

# virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .

# expose chainlit port
EXPOSE 8000

# Start chainlit
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]