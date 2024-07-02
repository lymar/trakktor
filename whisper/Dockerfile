ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

LABEL org.opencontainers.image.source https://github.com/lymar/trakktor
LABEL org.opencontainers.image.licenses=BSD-3-Clause

RUN apt update && apt install -y ffmpeg python3 python3-pip && \
    pip install -U openai-whisper awscli && \
    apt autoremove -y && apt clean -y

ARG WHISPER_MODEL
ENV WHISPER_MODEL=${WHISPER_MODEL}

RUN mkdir /whisper_models && \
    python3 -c "import whisper; print(whisper._download(whisper._MODELS['${WHISPER_MODEL}'], '/whisper_models', False))"

COPY ./main.sh /main.sh

CMD ["/bin/bash", "/main.sh"]
