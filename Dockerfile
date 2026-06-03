# use a lightweight official python image
FROM python:3.10-slim

# set the working directory inside the container
WORKDIR /app

# copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your api script and the trained model into the container
COPY main.py .
COPY xgboost_worldcup_model.pkl .

# expose the port that Google Cloud Run expects (8080)
EXPOSE 8080

# command to run the FastAPI server using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]