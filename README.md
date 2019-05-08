# NPI
CIS 700 NPI
This is an implementation of the Neural Programmer Interpreter by Scott Reed and Nando de Freitas [1]. 
I chose to try and implement the Card Matching. This problem has been previously been implemented with the GALIS architecture by Sylvester and Reggia[2]. The basic problem has a table with m columns and n rows of cards. There are exactly two cards that have a matching pixel pattern on one side. All cards have a uniform backing and the table is assumed to have a uniform shading. The table shading and the face down patterns are both unique (so that the network can actually find pairs). For simplicity, both approaches guaranteed that the card matching had a solvable solution.

[1] https://arxiv.org/pdf/1511.0629.pdf
[2] http://www.jsylvest.com/papers/2016-sylvester-reggia-nn.pdf
