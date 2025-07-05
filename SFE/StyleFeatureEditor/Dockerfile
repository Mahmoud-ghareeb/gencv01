
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    git-lfs \
    cmake \
    build-essential \
    wget \
    unzip \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python


RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip -q && \
    unzip ninja-linux.zip -d /usr/local/bin/ > /dev/null && \
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force > /dev/null && \
    rm ninja-linux.zip


RUN python3 -m pip install --upgrade pip


WORKDIR /workspace


RUN git clone https://github.com/ControlGenAI/StyleFeatureEditor.git && \
    cd StyleFeatureEditor


WORKDIR /workspace/StyleFeatureEditor

COPY app.py .

RUN pip install gradio opencv-python==4.11.0.86


COPY env_install.sh .
RUN chmod +x env_install.sh && bash env_install.sh


RUN git clone https://huggingface.co/AIRI-Institute/StyleFeatureEditor pretrained_models_repo && \
    cd pretrained_models_repo && \
    git lfs pull && \
    cd .. && \
    mv pretrained_models_repo/pretrained_models . && \
    rm -rf pretrained_models_repo


EXPOSE 7860


CMD ["python", "app.py"]