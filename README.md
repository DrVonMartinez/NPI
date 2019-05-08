# NPI
CIS 700 NPI
This is an implementation of the Neural Programmer Interpreter by Scott Reed and Nando de Freitas [1]. 
I chose to try and implement the Card Matching. This problem has been previously been implemented with the GALIS architecture by Sylvester and Reggia[2]. The basic problem has a table with m columns and n rows of cards. There are exactly two cards that have a matching pixel pattern on one side. All cards have a uniform backing and the table is assumed to have a uniform shading. The table shading and the face down patterns are both unique (so that the network can actually find pairs). For simplicity, both approaches guaranteed that the card matching had a solvable solution.

This implementation uses primarily pytorch with python 3. The other modules used are matplotlib for plotting error, stats for providing a linear regression to the plot (which should allow for better readability). The generation of the training and testing data make use of the module random (which allows for sequential creation, which is then randomized).
The parameters that can be adjusted are:
(m,n) which adjusts the number of cards on the table (m rows, n columns).
(a,b) which adjusts the number of pixels on each card (which are created in card.py as a uniform distribution between 0.1 and 0.9)
(alpha) which controls the threshold for completion of the current function the NPI is running
(beta, sigma) which are the hidden layers for the encoder.
(train_iters) number of training examples for each of the functions. Currently, the number of iterations that net.train() will go through is 7*train_iters (as there are 7 functions in M_prog)
(num_test) number of testing examples

[1] https://arxiv.org/pdf/1511.0629.pdf
[2] http://www.jsylvest.com/papers/2016-sylvester-reggia-nn.pdf
