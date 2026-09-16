FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY xianyu_agent ./xianyu_agent
COPY prompts ./prompts

RUN pip install --no-cache-dir -r requirements.txt

# 数据目录（对话库、会话快照）挂载点
VOLUME ["/app/data"]

CMD ["python", "-m", "xianyu_agent.cli"]
