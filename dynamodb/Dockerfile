FROM python:3-slim-buster

ADD requirements.txt /tmp/requirements.txt 
RUN pip install -r /tmp/requirements.txt 
COPY . /dyn
WORKDIR /dyn

ENTRYPOINT ["python3", "dynamo.py"]