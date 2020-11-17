FROM python:3.7-stretch

RUN apt-get update -y && apt-get install -y libzbar-dev bash gcc git libc-dev curl wget vim nano \
  && apt-get install -y ranger caca-utils highlight atool w3m poppler-utils mediainfo \
  && apt-get install -y iputils-ping telnet openssh-client net-tools \
  man unzip vim-tiny bc openssh-server thrift-compiler netcat sudo \
  && apt-get install -y build-essential \
  && apt-get autoremove -y \
  && apt-get clean

RUN apt-get update && apt-get install -y libgl1-mesa-glx 'ffmpeg' 'libsm6' 'libxext6'

RUN pip install --no-cache-dir jupyterlab

COPY container/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
  && rm -rf requirements.txt

RUN pip install torch torchvision tensorboardX

#RUN mkdir -p /main
COPY ./application /main

COPY container/jupyter_run.sh /jupyter_run.sh
RUN chmod a+rwx /jupyter_run.sh

WORKDIR /main
