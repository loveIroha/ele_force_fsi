################################################################################
# Spack Dockerfile
################################################################################
# Step 1 : Create a base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder
ENV TZ Asia/Shanghai
ENV LANG zh_CN.UTF-8

# 设置root用户密码为root
RUN echo 'root:root' |chpasswd  
 # 设置非交互式安装        
ARG DEBIAN_FRONTEND=noninteractive     

# Install SSH
RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd
RUN mkdir -p /root/.ssh/
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Step 2 : Install GCC
FROM builder AS release
RUN apt-get install -y gcc g++ gfortran make cmake git 
RUN apt-get install -y xz-utils bzip2 zip
RUN cd /root && wget https://github.com/spack/spack/releases/download/v0.21.1/spack-0.21.1.tar.gz && tar -zxvf spack-0.21.1.tar.gz
RUN rm /root/spack-0.21.1.tar.gz
RUN mv /root/spack-0.21.1 /root/spack 

FROM release AS npuheartlala
RUN . ~/spack/share/spack/setup-env.sh && spack external find 
RUN cat /root/.spack/packages.yaml
RUN . ~/spack/share/spack/setup-env.sh && spack env create fenics_kokkos && spack env activate fenics_kokkos -p
RUN cat /root/.spack/packages.yaml
ADD https://github.com/shaoyaoqian/docker-computing-images/releases/download/v0.04/packages.yaml /root/.spack/packages.yaml
RUN cat /root/.spack/packages.yaml

# Step 3 : Install FEniCS
FROM npuheartlala AS lahalaha
WORKDIR /root/spack
ADD https://github.com/shaoyaoqian/docker-computing-images/releases/download/v0.04/spack.yaml /root/spack/var/spack/environments/fenics_kokkos/spack.yaml
RUN . ~/spack/share/spack/setup-env.sh && spack env activate fenics_kokkos -p && spack concretize -f && spack install && spack clean --all


# # Step 4 : Install 暴露22端口
EXPOSE 22
#################################################################################
# Spack and Kokkos Dockerfile
#################################################################################
# FROM mapengfei/npuheart:v0.2.2
# WORKDIR /root/spack
# RUN . share/spack/setup-env.sh && spack env activate fenics_kokkos -p && spack concretize -f
# RUN . share/spack/setup-env.sh && spack env activate fenics_kokkos -p && spack install
