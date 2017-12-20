FROM python:3.6.3

WORKDIR /diploma

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src ./src
COPY SPH2_031612.csv .
COPY requirements.txt .

CMD python -u src/main.py