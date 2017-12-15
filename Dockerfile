FROM python:3.6.3

WORKDIR /diploma

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD python main.py > log.txt 2>&1