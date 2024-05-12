# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

# Install prerequesites
RUN apt-get update && apt-get install -y software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.10 python3-pip openssh-client git htop nano sudo curl task-spooler
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10


ARG DOXA_PACKAGE_CACHEBUST=1
COPY doxa-requirements.txt extra-requirements.txt ./
RUN pip install --no-cache-dir install -r doxa-requirements.txt \
    && pip --no-cache-dir install setuptools --upgrade --force
ARG EXTRA_PACKAGE_CACHEBUST=1
RUN pip --no-cache-dir install -r extra-requirements.txt

WORKDIR /root

CMD ["bash"]