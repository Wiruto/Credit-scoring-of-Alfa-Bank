FROM python:latest

WORKDIR /usr/src/app

COPY ./ ./

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./server.py" ]


