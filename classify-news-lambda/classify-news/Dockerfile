FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ./

RUN pip install --upgrade pip
# Lambda requires packages to be installed in /opt/python
RUN python3.10 -m pip install -r requirements.txt -t /opt/python/lib/python3.10/site-packages


WORKDIR /var/task
COPY app.py ./
COPY engine ./engine

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
