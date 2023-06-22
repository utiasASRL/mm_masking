#FROM ubuntu:22.04
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

CMD ["/bin/bash"]

ARG GROUPID=0
ARG USERID=0
ARG USERNAME=root
ARG HOMEDIR=/root

RUN if [ ${GROUPID} -ne 0 ]; then addgroup --gid ${GROUPID} ${USERNAME}; fi \
  && if [ ${USERID} -ne 0 ]; then adduser --disabled-password --gecos '' --uid ${USERID} --gid ${GROUPID} ${USERNAME}; fi

# Default number of threads for make build
ARG NUMPROC=12

ENV DEBIAN_FRONTEND=noninteractive

## Switch to specified user to create directories
USER ${USERID}:${GROUPID}

ENV ROOTDIR=$(pwd)

## Switch to root to install dependencies
USER 0:0

## Dependencies
RUN apt update && apt upgrade -q -y
RUN apt update && apt install -q -y cmake git build-essential lsb-release curl gnupg2 wget
RUN apt update && apt install -q -y libboost-all-dev libomp-dev
RUN apt update && apt install -q -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
RUN apt update && apt install -q -y freeglut3-dev
RUN apt update && apt install -q -y python3 python3-distutils python3-pip
RUN apt update && apt install -q -y libeigen3-dev
RUN apt update && apt install -q -y libsqlite3-dev sqlite3

RUN apt update && apt install -q -y cmake libsqlite3-dev sqlite3 libtiff-dev libcurl4-openssl-dev

ENV LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## Install ROS2
# UTF-8
RUN apt install -q -y locales \
  && locale-gen en_US en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
# Add ROS2 key and install from Debian packages
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt update && apt install -q -y ros-galactic-desktop \
  && apt install ros-galactic-sensor-msgs-py

## Install misc dependencies
RUN apt update && apt install -q -y \
  tmux \
  nodejs npm protobuf-compiler \
  libboost-all-dev libomp-dev \
  libpcl-dev \
  libcanberra-gtk-module libcanberra-gtk3-module \
  libbluetooth-dev libcwiid-dev \
  python3-colcon-common-extensions \
  virtualenv \
  texlive-latex-extra \
  clang-format \
  htop

# Install vim
RUN apt update && apt install -q -y vim

RUN apt install python3.8-venv

# Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Set the PATH environment variable
#ENV PATH /opt/conda/bin:$PATH

# Copy the environment.yml file to the container
#COPY environment.yml /root/environment.yml
#RUN conda env create -f /root/environment.yml
#ENV PATH /opt/conda/envs/mmm_env/bin:$PATH
#RUN conda init bash
# Change ownership of the environment directory to the current user
#RUN chown -R ${USERID}:${GROUPID} /opt/conda/envs/mmm_env

## Switch to specified user
USER ${USERID}:${GROUPID}