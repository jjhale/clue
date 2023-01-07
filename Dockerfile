FROM python:3.11-slim

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip --no-cache-dir \
  && pip install -r requirements.txt --no-cache-dir \
  && rm requirements.txt

# Copy the source code:
COPY . /src
WORKDIR /src
ENV PYTHONPATH "${PYTHONPATH}:/src/"
