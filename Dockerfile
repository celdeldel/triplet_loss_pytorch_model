FROM nvcr.io/nvidia/pytorch:19.07-py3

RUN mkdir -p /dir_src

WORKDIR /dir_src




ADD requirements.txt /dir_src


RUN pip install -r requirements.txt


ENV NAME class1

CMD [ "python3", "train.py" ]
