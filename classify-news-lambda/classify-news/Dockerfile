FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN python3.10 -m pip install -r requirements.txt -t /opt/python/lib/python3.10/site-packages


WORKDIR /var/task
COPY app.py ./
COPY engine ./engine

# RUN find /var/task -type f -exec chmod 644 {} +
# RUN find /var/task -type d -exec chmod 755 {} +

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
