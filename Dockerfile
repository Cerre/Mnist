FROM pytorch/pytorch

WORKDIR /app
COPY . ./
RUN apt-get update && apt-get install -y python3 python3-pip sudo && \
    pip install -r requirements.txt