# Pull Base Image
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set Working Directory
RUN mkdir /usr/src/kod
WORKDIR /usr/src/kod

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /usr/src/kod/requirements.txt

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Indian
RUN apt-get update && apt-get install -y \ 
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    python-dev \
    libopencv-dev \
    libpng-dev \ 
    libtiff-dev \
    libgtk2.0-dev \ 
    python-numpy \ 
    python-pycurl \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    qt5-default \
    libvtk6-dev \ 
    zlib1g-dev 

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install jupyter

COPY . /usr/src/kod/