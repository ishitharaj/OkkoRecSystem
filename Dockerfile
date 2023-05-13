FROM python:3.9-slim-bullseye
ENV HOME /home

WORKDIR ${HOME}
ENV PYTHONPATH ${HOME}

COPY . .

RUN apt-get update && apt-get install -y gcc
RUN pip3 install -r requirements.txt

# ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT python3 api/api.py


# http://127.0.0.1:5000/index?id=646321