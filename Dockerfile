# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
# This includes the app script, data files, and image directories.
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable to tell Streamlit to run on port 8501
ENV STREAMLIT_SERVER_PORT 8501
ENV STREAMLIT_SERVER_HEADLESS true


# Run the app when the container launches
CMD ["streamlit", "run", "cpte_app3.py"] 