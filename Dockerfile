FROM python:3.12-slim

# Copy the entire project
COPY . MLOPS_PROJECT/

# Install the package with dependencies
RUN pip install ./MLOPS_PROJECT

# Set the working directory
WORKDIR /MLOPS_PROJECT 

# Expose the port gunicorn will listen on
EXPOSE 5001

# Run gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:5001", "app.main:app"]