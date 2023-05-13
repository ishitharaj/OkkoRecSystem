FROM python:3.9-slim-bullseye
ENV HOME /home

WORKDIR ${HOME}
ENV PYTHONPATH ${HOME}

COPY . .

RUN apt-get update && apt-get install -y gcc
RUN pip3 install -r requirements.txt

#ENTRYPOINT python3 api/flaskapi.py
ENTRYPOINT python3 api/faaastapi.py
