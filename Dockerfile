# use an official Python image
FROM python:3.10-slim-buster

# install c++ compilers
RUN apt-get update && \
    apt-get -y install gcc

# set variable for working directory
ARG work_dir=/app
WORKDIR $work_dir

# copy the poetry config to the container
COPY pyproject.toml $work_dir/

# install poetry
RUN pip install poetry Cmake

# install dependencies
RUN poetry export --with=dev --without-hashes --output requirements.txt \
    && pip3 install --no-cache-dir -r requirements.txt

# workaround to install lightfm
RUN pip install lightfm==1.17 --no-use-pep517

# copy the rest of the app code to the container
COPY . $work_dir

# expose the port used by the microservice
EXPOSE 8080

# load the models and start the app
CMD ["python", "api/flaskapi.py"]

# docker build --platform=linux/amd64 . -t recsys_api
# docker run -p 8080:8080 recsys_api
# http://127.0.0.1:8080/get_recommendation?user_id=790772
# http://127.0.0.1:8080/get_recommendation?user_id=228319880
# http://127.0.0.1:8080/get_recommendation?user_id=202914785




## Option 2
# FROM python:3.9-slim-bullseye
# ENV HOME /home

# WORKDIR ${HOME}
# ENV PYTHONPATH ${HOME}

# COPY . .

# RUN apt-get update && apt-get install -y gcc
# RUN pip3 install -r requirements.txt

# ENTRYPOINT python3 api/flaskapi.py
# #ENTRYPOINT python3 api/faaastapi.py
