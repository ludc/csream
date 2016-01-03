# Description

The software implements the 3 models (M-REAM, B-REAM and D-REAM) described in the article [waiting for publication]

##Dependencies

* torch 7
  * nn
  * nngraph
  * cutorch and cunn (for CUDA implementation)
  * csv, lfs and gnuplot (for pareto front computation)

## Input

The software is based on libSVM files and produces classifiers. Each model needs three different input files:
* A training file
* A validation file
* A test file

A each iteration, the model computes the performance over the three files. The Pareto front is then computed using the validation file, and plotted on the test file. The three files have the same name but the extensions are .train, .test and .validation. For safety, the index of the first features is 1. Examples of uci files are given in the repository

### Common parameters

* **dataset**: the dataset name (i.e the filename without extension)
* **N**: the size of the latent space
* **maxEpoch**: the number of training iterations
* **evaluationEpoch**: the number of iterations between two model evaluations e.g 10 means that the model will be evaluated every 10 training iterations
* **size_minibatch**: the number of examples in each minibatch - 1 means that the model is learned following a classical SGD procedure
* **device**: the GPU device 
  * 0 means that the model will be learned on the CPU
* **uniform**: the variance of the model's parameters initialization
* **learningRate**: the learnin rate 
* **verbose**: 
  * false = the output of the model ill be saved in a CSV file in the 'logPath' directory.
  * true = the model will output on the console
* **logPath**: the directory of the log file. Note that the filename is automaticcally defined in such a way that two experiments will generate two files with two different names. Note also that all the parameters are saved in the log file.
* **costs**: the cost model
  * no = all features have a weight of 1
  * linear = feature 1 as a cost of 1/n, feature 2 has a cost of 2/n, ... where n is the total number of features
  * weigth1:weigth2:weigth3:... = feature 1 has weight1, feature 2 has weight2, ....
* **cell**: the dynamic cell
  * gru = gru cell
  * rnn = rnn cell
  * add = additive cell (lstm without gate...)
* **size**: the number of steps of the model

## M-REAM 

In M-REAM, the model selects one features at each timestep. It means that the 'size' parameter is the max number of selected features. M-REAM does not need extra parameters.


TODO: remove extra parameters in the mream script


## B-REAM

In B-REAM, the model samples many features at each time step. 

* **bias**: the initial bias of the selection policy. High bias means that the model will tend to select a large number of features at each timestep (at the beginning)
* **l1**: the initial l_1 regularization coefficient allowing to produce sparse models. In the classical learning schema, l1 wil be increased during training in order to generate different levels of sparsity
  * **maxL1** is the maximum L1 value at the end of the learning process (when iteration == maxEpoch)
  * **burninEpoch** is the number of iterations where the value of l1 will be fixed. Then l1 will incread in a linear way to maxL1

## D-REAM

D-ream has the same parameters than M-REAM (but learns without sampling, and is thus faster during training)



#Interpretation of the results

In order to compute global performance of the models, we provide the **compute_pareto_front.lua** script. The principle is the following:
* The script reads all the output CSV files in a particular directory
* The script only kept the lines that follow the **filters** argument such as *N=5:learningRate=0.01* 
* The script then computes the pareto front of the resulting values:
  * the pareto front is computed on the validation set (using the *cost_validation* and *accuracy_validation* columns)
  * the pareto front is then drawn (and saved) on the testing set (using the *cost_test* and *accuracy_test* columns)
* One front is drawn saved for each value of the **by** column. For example the parameters *--by size* allows one to output one curve for each *size* value in the experiment files

# Examples

Here is a learning example that will generate outputs in the 'log/' subdirectory using the 'abalone' dataset with one third as training set, one third as validation set and one thrid as testing set. Many experiments will be launched, and the pareto curves will be drawn on the resulting files

... To be continued


