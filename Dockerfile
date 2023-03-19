FROM python:3.9

COPY requirements.txt requirements.txt
# keep a pip cache locally so you don't have to hit the web each time
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip \
  && pip install -r requirements.txt \
  && rm requirements.txt

# Copy the source code:
COPY . /src
WORKDIR /src
ENV PYTHONPATH "${PYTHONPATH}:/src/"
