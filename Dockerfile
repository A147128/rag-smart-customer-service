FROM python:3.11-slim

WORKDIR /app

# ???????jieba???
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8501

# ???????
VOLUME ["/app/data"]

CMD ["streamlit", "run", "app_qa.py", "--server.port=8501", "--server.address=0.0.0.0"]
