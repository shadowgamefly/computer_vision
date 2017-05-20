Project 3:
----------
Assume you have conda installed, to run this project:
~~~ Bash
conda env create -f environment.yaml
source activate project
~~~
and then run one of the following python script, `MNIST_CNN.py`, `MNIST_MLP.py`, `fine_tune.py`, `classification_utility.py`.

You may also view the output of classification_utility by following:
~~~ Bash
jupyter classification_utility.ipynb
~~~

#### MNIST with a side of MLP
* Q #1: the test set accuracy is 92.62%

* Q #2: the normalized confusion metrics

[[  9.7653e-01   0.0000e+00   2.0408e-03   2.0408e-03   0.0000e+00
    7.1429e-03   8.1633e-03   3.0612e-03   1.0204e-03   0.0000e+00]
 [  0.0000e+00   9.7533e-01   4.4053e-03   1.7621e-03   0.0000e+00
    8.8106e-04   3.5242e-03   1.7621e-03   1.2335e-02   0.0000e+00]
 [  2.9070e-03   5.8140e-03   9.0407e-01   1.7442e-02   6.7829e-03
    3.8760e-03   1.1628e-02   9.6899e-03   3.4884e-02   2.9070e-03]
 [  1.9802e-03   0.0000e+00   1.7822e-02   9.1881e-01   9.9010e-04
    2.4752e-02   1.9802e-03   9.9010e-03   1.6832e-02   6.9307e-03]
 [  2.0367e-03   1.0183e-03   6.1100e-03   1.0183e-03   9.3585e-01
    0.0000e+00   7.1283e-03   6.1100e-03   1.0183e-02   3.0550e-02]
 [  8.9686e-03   2.2422e-03   2.2422e-03   3.8117e-02   1.0090e-02
    8.8117e-01   1.4574e-02   7.8475e-03   2.9148e-02   5.6054e-03]
 [  9.3946e-03   3.1315e-03   8.3507e-03   1.0438e-03   7.3069e-03
    1.5658e-02   9.5198e-01   2.0877e-03   1.0438e-03   0.0000e+00]
 [  9.7276e-04   5.8366e-03   2.3346e-02   3.8911e-03   7.7821e-03
    0.0000e+00   0.0000e+00   9.2996e-01   1.9455e-03   2.6265e-02]
 [  6.1602e-03   6.1602e-03   6.1602e-03   2.6694e-02   9.2402e-03
    2.7721e-02   9.2402e-03   1.2320e-02   8.8706e-01   9.2402e-03]
 [  9.9108e-03   6.9376e-03   9.9108e-04   1.0902e-02   2.6759e-02
    5.9465e-03   0.0000e+00   2.2795e-02   3.9643e-03   9.1179e-01]]

* Q #3: 0 has the highest entry, and 5 has the lowest entry

* Q #4: for 5 it is most oftern confused with 3 and 8

* Q #5: the reason for confusion shall be the similarity between those numbers in shape, and similarly 0 has the highest accuracy simply because it has a unique topological structure

* Q #6: the test set accuracy is 95.18%

* Q #7: the test set accuracy is 98.19%

* Q #8: the new confusion matrix after normalization is as follows

[[  9.9286e-01   1.0204e-03   1.0204e-03   1.0204e-03   0.0000e+00
    0.0000e+00   2.0408e-03   0.0000e+00   1.0204e-03   1.0204e-03]
 [  8.8106e-04   9.8855e-01   1.7621e-03   1.7621e-03   0.0000e+00
    8.8106e-04   2.6432e-03   0.0000e+00   2.6432e-03   8.8106e-04]
 [  1.9380e-03   9.6899e-04   9.7771e-01   8.7209e-03   9.6899e-04
    9.6899e-04   1.9380e-03   2.9070e-03   3.8760e-03   0.0000e+00]
 [  0.0000e+00   0.0000e+00   1.9802e-03   9.8812e-01   0.0000e+00
    2.9703e-03   0.0000e+00   9.9010e-04   2.9703e-03   2.9703e-03]
 [  1.0183e-03   0.0000e+00   2.0367e-03   1.0183e-03   9.8065e-01
    0.0000e+00   4.0733e-03   2.0367e-03   0.0000e+00   9.1650e-03]
 [  2.2422e-03   0.0000e+00   0.0000e+00   8.9686e-03   0.0000e+00
    9.8094e-01   2.2422e-03   0.0000e+00   2.2422e-03   3.3632e-03]
 [  3.1315e-03   2.0877e-03   1.0438e-03   1.0438e-03   3.1315e-03
    7.3069e-03   9.8017e-01   0.0000e+00   2.0877e-03   0.0000e+00]
 [  1.9455e-03   3.8911e-03   8.7549e-03   1.9455e-03   0.0000e+00
    0.0000e+00   0.0000e+00   9.7179e-01   3.8911e-03   7.7821e-03]
 [  1.0267e-03   0.0000e+00   1.0267e-03   7.1869e-03   2.0534e-03
    5.1335e-03   1.0267e-03   2.0534e-03   9.7433e-01   6.1602e-03]
 [  1.9822e-03   1.9822e-03   0.0000e+00   2.9732e-03   5.9465e-03
    2.9732e-03   0.0000e+00   9.9108e-04   0.0000e+00   9.8315e-01]]

* Q #9: The last network works the best, because it added more layers and each layer is more complicated than the first two

* Q #10: the test set accuracy is 98.04%. There is no significant change in performance here.

#### MNIST garnished with a CNN

* Q #11:
   **All the following testings are under MNIST dataset with 60000 traincases, and 10000 testcases. Each timing is regarding one single epoch through traincases, and the accuracy is gained through 12 epochs.**

  * A: test accuracy: 0.9453 average time per epoch: 10s

  * B: test accuracy: 0.9803 average time per epoch: 34s

  * C: test accuracy: 0.9783 average time per epoch: 34s

#### Finely-tuned Cats and Dogs
* Q #12: it only has accuracy of 0.4958, as the main problem is the lack of training samples and low epoch times

* Q #13: For one epoch, the accuracy we get is around 87%, for two epochs the accuracy goes up to around 90.5%
