FROM python:3.9.9-buster
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY ["requirements.txt", "run.sh", "./"] /app/
RUN pip install -r requirements.txt && chmod +x run.sh
COPY . .
ENTRYPOINT bash ./run.sh