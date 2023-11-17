
# Logistic Regression with gradient descent

- project summary.
- Installation and set ups.


### Project summary
The project requires us to build a logistic learner model to classify the data into binary classe.In the first part of the task we built a logistic learner with batch descent using numpy and math libraries only.The objective is to analyze the impact of diffrent learning rates in the convergence of model.The second task is to implement a logistic regression learner with 3 gradient descents(stochastic gradient descent, mini-batch and batch gradient descent).Here the focus is to compare the computational time it takes to implement each and evaluate the accuracy of predictions.

### Set up and running script

Create a python environment 
```
python -m venv dts_logistic
```
Run the following code to activate the virtaual environment
```
 source dts_logistic/Scripts/activate

```
Install the required packages and libraries
```
pip instal -r requirements.txt

```



Other specification I worked with
- python 3.9
- VSCODE editor 1.82.2
- windows 10
- terminal: Gitbash

## Run script
Open `logistic.py` script on your preferred editor
- Activate the virtual environment.
- On your working directory create a folder to store the output and change the path `plt.savefig('output/5000_iterations/logistic_regression_loss_plot.jpg')`
- NB:Take note also of input data path and provide the appropriate path where you have the two csv files.