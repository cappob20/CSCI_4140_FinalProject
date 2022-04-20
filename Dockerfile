FROM tensorflow/tensorflow:2.8.0
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

COPY requirements.txt .

RUN pip install -r requirements.txt
