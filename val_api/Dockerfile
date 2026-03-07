FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    flex \
    bison \
    git \
    python3.10 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN git clone https://github.com/KCL-Planning/VAL.git /tmp/VAL && \
    cd /tmp/VAL && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/VAL

ENV VAL=/usr/local
ENV PATH=$VAL/bin:$PATH

RUN pip3 install --no-cache-dir fastapi uvicorn[standard]

WORKDIR /app
COPY main.py /app/main.py

RUN echo "VAL binary check:" && ls -la $VAL/bin/validate

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
