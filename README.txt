Q1. Independent Bayesian Networks

Execute: python IBN.py <training-dataset> <validation-dataset> <testing-dataset>

example: python IBN.py small-10-datasets/nltcs.ts.data small-10-datasets/nltcs.valid.data small-10-datasets/nltcs.test.data

Output : Avg. Log Likelihood: -9.233597557239834

(This will load the data from the training-dataset and the validation-dataset and then trains an Independent Bayesian network on the combined dataset. Then it loads the testing-dataset and outputs the Log Likelihood of the testing data.)
-------------------------------------------------------------------------------------------------------------------------------------
Q2. Tree Bayesian networks

Execute: python TBN.py <training-dataset> <validation-dataset> <testing-dataset>

example: python TBN.py small-10-datasets/msnbc.ts.data small-10-datasets/msnbc.valid.data small-10-datasets/msnbc.test.data

Output : Avg. Log Likelihood: -6.540076033854932

(This will load the data from the training-dataset and the validation-dataset and then trains an Tree Bayesian network (Chow Liu Tree) on the combined dataset. Then it loads the testing-dataset and outputs the Log Likelihood of the testing data.)
-------------------------------------------------------------------------------------------------------------------------------------
Q3. Mixtures of Tree Bayesian networks using EM

Execute: python MT.py <k> <training-dataset> <validation-dataset> <testing-dataset>

k : Number of mixture components (Chow liu trees)

example: python MT.py 2 small-10-datasets/nltcs.ts.data small-10-datasets/nltcs.valid.data small-10-datasets/nltcs.test.data

Output : 

k=2
LL: -6.417757660440885
LL: -6.757496566333628
LL: -6.757481219496182
LL: -6.395641187080139
LL: -6.75738776958848
LL: -6.757593958953666
LL: -6.75738776958848
LL: -6.757524971615047
LL: -6.757496573770411
LL: -6.75738776958848
Mean=-6.68731554464554, Std=0.14039520207540865, Best=-6.395641187080139

(This will load the data from the training-dataset and the validation-dataset and then train k Tree Bayesian networks (Chow Liu Trees) on the combined dataset using 100 iterations of the EM algorithm. Then it will load the testing-dataset and print the Log Likelihood of the testing data on k chow liu trees multiplied by their respective probabilities. It will repeat the entire process 10 times with random initializations and then output the Mean, Standard Deviation and the Best Log Likelihood score on the mixture of trees out of all the 10 iterations.)
-------------------------------------------------------------------------------------------------------------------------------------
Q4. Mixtures of Tree Bayesian networks using Random Forests

Execute: python BRF.py <k> <r> <training-dataset> <validation-dataset> <testing-dataset>

k : Number of boot-strapped datasets

r : Percentage of total edges to be removed

example: python BRF.py 2 0.05 small-10-datasets/nltcs.ts.data small-10-datasets/nltcs.valid.data small-10-datasets/nltcs.test.data

Output : k=12 edges_removed=6 bs_M=-6.786748908173334 wt_M=-6.7854330253688335 wt_S=0.009684072274244677

(This will load the data from the training-dataset and the validation-dataset. It will then create k boot-strapped datasets and train k Tree Bayesian networks (Chow Liu Trees) on individual boot-strapped datasets. While creating the Chow Liu trees it will delete r% (percent) edges from the fully connected network before making the Maximum Spanning Tree. Then it will load the testing-dataset and print two Log Likelihoods of the testing data on k chow liu trees. The first (base) Log likelihood will assign equal weightage (1/k) to all the trees. The second method will assign the trees different normalized weightage based on their log likelihood score on the validation dataset. It will repeat the entire process 10 times with random initializations and then output the Avg. (1/k) Mean, Weighted Mean and Standard Deviation of the Log Likelihood score on the random forest of trees of all the 10 iterations.)
-------------------------------------------------------------------------------------------------------------------------------------