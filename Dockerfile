FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential python3.9 python3-pip python3-dev ffmpeg sox tzdata cuda-toolkit-12-3 && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY . /app
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir tensorboardX

RUN python3 /app/src/download_models.py

EXPOSE 8000

CMD ["python3", "src/webui.py", "--listen-host", "0.0.0.0", "--listen-port", "8000", "--listen"]
