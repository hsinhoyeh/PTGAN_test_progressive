FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir kserve
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
