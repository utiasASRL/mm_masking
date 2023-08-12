#FROM ubuntu:22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG GROUPID=0
ARG USERID=0
ARG USERNAME=root
ARG HOMEDIR=/root

RUN if [ ${GROUPID} -ne 0 ]; then addgroup --gid ${GROUPID} ${USERNAME}; fi \
  && if [ ${USERID} -ne 0 ]; then adduser --disabled-password --gecos '' --uid ${USERID} --gid ${GROUPID} ${USERNAME}; fi

# Default number of threads for make build
ARG NUMPROC=12

ENV DEBIAN_FRONTEND=noninteractive

## Switch to root to install dependencies
USER 0:0

## Dependencies
RUN apt update && apt upgrade -q -y
RUN apt update && apt install -q -y cmake git build-essential lsb-release curl gnupg2
RUN apt update && apt install -q -y libboost-all-dev libomp-dev
RUN apt update && apt install -q -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
RUN apt update && apt install -q -y freeglut3-dev
RUN apt update && apt install -q -y python3 python3-distutils python3-pip
RUN apt update && apt install -q -y libeigen3-dev

## Install PROJ (8.2.0) (this is for graph_map_server in vtr_navigation)
RUN apt update && apt install -q -y cmake libsqlite3-dev sqlite3 libtiff-dev libcurl4-openssl-dev
RUN mkdir -p ${HOMEDIR}/proj && cd ${HOMEDIR}/proj \
  && git clone https://github.com/OSGeo/PROJ.git . && git checkout 8.2.0 \
  && mkdir -p ${HOMEDIR}/proj/build && cd ${HOMEDIR}/proj/build \
  && cmake .. && cmake --build . -j${NUMPROC} --target install
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
  && apt update && apt install -q -y ros-humble-desktop

## Install VTR specific ROS2 dependencies
RUN apt update && apt install -q -y \
  ros-humble-xacro \
  ros-humble-vision-opencv \
  ros-humble-perception-pcl ros-humble-pcl-ros \
  ros-humble-rmw-cyclonedds-cpp

RUN mkdir -p ${HOMEDIR}/.matplotcpp && cd ${HOMEDIR}/.matplotcpp \
  && git clone https://github.com/lava/matplotlib-cpp.git . \
  && mkdir -p ${HOMEDIR}/.matplotcpp/build && cd ${HOMEDIR}/.matplotcpp/build \
  && cmake .. && cmake --build . -j${NUMPROC} --target install

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
  htop \
  wget

# Install aws dependencies for boreas dataset installation
RUN apt update && apt upgrade -q -y zip unzip
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
  unzip awscliv2.zip && \
  ./aws/install

# Install vim
RUN apt update && apt install -q -y vim

# Install this for some potential cuda bugs
RUN apt install nvidia-modprobe
RUN apt update && apt install

## Install python dependencies
RUN pip3 install \
  tmuxp \
  pyproj \
  zmq \
  flask \
  flask_socketio \
  eventlet \
  python-socketio \
  python-socketio[client] \
  websocket-client

RUN apt install -q -y doxygen

# Set up entrypoint
COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

USER ${USERID}:${GROUPID}

# Set up env variables
ENV ROOTDIR=$(pwd)