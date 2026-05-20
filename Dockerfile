FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ .

CMD ["main.handler"]
