# this Dockerfile is for torchtext smoke test, it will be created periodically via CI system
# if you need to do it locally, follow below steps once you have Docker installed
# to test the build use : docker build . -t torchtext/smoketest
# to upload the Dockerfile use build_and_push.sh script

FROM ubuntu:latest

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 sox libsox-dev libsox-fmt-all \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -c conda-forge gcc \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH


RUN conda create -y --name python3.7 python=3.7
RUN conda create -y --name python3.8 python=3.8
RUN conda create -y --name python3.9 python=3.9
RUN conda create -y --name python3.10 python=3.10

SHELL [ "/bin/bash", "-c" ]
RUN echo "source /usr/local/etc/profile.d/conda.sh" >> ~/.bashrc
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python3.7 && conda install -y numpy
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python3.8 && conda install -y numpy
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python3.9 && conda install -y numpy
RUN source /usr/local/etc/profile.d/conda.sh && conda activate python3.10 && conda install -y numpy
CMD [ "/bin/bash"]
