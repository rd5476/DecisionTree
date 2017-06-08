Readme:

Foundations of Intelligent system
Author :Ritvik Joshi,Rahul Dashora,Amruta Deshpande.

Inputs:
train_data.csv
test_data.csv

Execution files:
trainMLP
executeMLP
trainDT
executeDT

Steps:

1. MLP

	1.Run the trainMLP.py file
	2.Enter train_data.csv on the terminal when asked'Enter the filename'
	3.The terminal displays two 2 plots
 		a) Plot of SSE vs Epoch 
  		b) Plot of the training data set.
		c) 5 weight files are generated through this
	4.Now run the executeMLP.py file 
	5.Enter test_data.csv in the command line when asked 'Enter the file name'.
		(Weight files must be in the same directory)
	6.The terminal displays
		a)Confusion Matrix 
		b)Recognition Rate 
		c)Profit
		d)Plot for test sample classification region for different epochs.

2.Decision Tree:
	1. Run the trainDT.py file
	2. Input :Enter the train_data.csv file on the terminal line when asked'Enter the filename'.
	3. Output:
	   The command line displays following contents of Decision Tree before prunning and after prunning
  		a) Total number of nodes
  		b) Total number of leaf nodes
  		c) For each node display Max path, Min path and the average path. 
	4. Run executeDt.py file
	5.Input: 
	  Enter test_data.csv file in the command line when asked for 'Enter the file name'.
	  Enter normal_threshold.csv in the command line when asked for 'Enter normal tree threshold file'
	6.Output:
	  The terminal displays following:
		a)Confusion Matrix
		b)Recognition Rate
		c)Profit earned
	8.Input: Enter chi_squared_threshold.csv in the command line when asked for'Enter pruned tree threshold file'.
	9.Output :
	 The terminal displays following:
		a)Confusion Matrix
		b)Recognition Rate
		c)Profit earned