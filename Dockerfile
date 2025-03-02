# Use official Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt ./
COPY exoplanet_model.pkl ./
COPY scaler.pkl ./
COPY spacee.jpg ./
COPY app.py ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
