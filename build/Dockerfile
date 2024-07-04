FROM continuumio/miniconda3

RUN mkdir /root/scratch
RUN apt-get update && apt-get install -y build-essential
RUN apt update
RUN apt install -y git

COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda init
# Activate the environment, and make sure it's activated:
#RUN conda activate myenv
SHELL ["conda", "run", "-n", "harmonic", "/bin/bash", "-c"]
RUN git clone https://github.com/saparkison/lie_learn.git
WORKDIR lie_learn
CMD  ["python", "setup.py", "install"]
WORKDIR /app

