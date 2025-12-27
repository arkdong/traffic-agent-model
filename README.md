Authors:

- Sietse van de griend - 12652776
- Hamid Ahmadi - 14473380
- Adam Dong - 13654675
- Sebastian Gielens - 14657287

This project models traffic on a micro level using agents to simulate the
decision-making of many individual drivers.

## Requirements
We use Python 3.10 along with the following libaries:

- NumPy
- Matplotlib
- Pandas

These may be installed manually with pip or through Anaconda. To automatically
install them in a `venv` you can use the following commands:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Clone the GitHub repository
To be able to run the files locally, you first have to clone the github repositiory
as follows: git clone https://github.com/Laika404/ComputationalProject.git
Then, because we have created sub-directories for each component(e.g. test for unit tests, src for the main code, data for the
data generated, images for the validation), if you want to for example run the AgentTest.py file which is 
in the folder test, you have to run: python -m test.AgentTest from the root directory. 

## Running/plotting
To run the model along with the code that plots the resulting data, run the
following command:

```sh
python -m src.model
```
