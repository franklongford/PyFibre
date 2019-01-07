# Use an official Python runtime as a parent image
FROM python:3.6.8

RUN ls

# Set the working directory to /PyFibre
WORKDIR /PyFibre

# Copy the current directory contents into the container at /PyFibre
ADD . /PyFibre

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run src/gui.py when the container launches
ENTRYPOINT ["python", "src/gui.py"]

CMD input
