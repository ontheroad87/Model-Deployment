# Install libraries
FROM python:3.8-slim-buster
RUN pip install numpy==1.23.5
RUN pip install pandas==1.5.2
RUN pip install joblib==1.2.0
RUN pip install scikit_learn==1.2.0

# Copy model and http_server code
ARG MODEL_PATH

COPY $MODEL_PATH .
COPY http_server.py .

# Run http_server for scoring
CMD ["python", "./http_server.py", "8888"]