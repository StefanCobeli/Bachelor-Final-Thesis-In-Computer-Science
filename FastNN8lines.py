# -*- coding: utf-8 -*-

#https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1#.8wnabb1et
 #9 lines of Python code

#As part of my quest to learn about AI, I set myself the goal of building a simple neural network in Python. To ensure I truly understand it, I had to build it from scratch without using a neural network library. Thanks to an excellent blog post by Andrew Trask I achieved my goal. Here it is in just 9 lines of code:
   
from numpy import exp, array, random, dot

training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
#>>> print training_set_inputs
#[[0 0 1]
 #[1 1 1]
 #[1 0 1]
 #[0 1 1]]

# Modeled Problem: Given the input as a series of lines,
# the output needs to be the value of the leftmost input column. 
# [1 0 0] --> ?
# Therefore the answer in the ‘?’ should be 1.

training_set_outputs = array([[0, 1, 1, 0]]).T   #Transpusa -- aici vectorul transpus
#>>> print training_set_outputs
#[[0]
 #[1]
 #[1]
 #[0]]

random.seed(1) # fără asta se generează de fiecare dată alte nr. aleatoare!
synaptic_weights = 2 * random.random((3, 1)) - 1 # O coloană de 3 randoms din (-1,1)
#>>> print synaptic_weights
#[[ 0.39207751]
 #[-0.43069824]
 #[ 0.26934855]]
M=5 # number of training repeats merge chiar și cu asta!!! aproape întotdeuna??? cu M=4 nu totdeauna
# se ia ca soluție cel mai aproape întreg de răspunsul oferit de program
#Number of training repeats=5
#[ 0.7405836]
M=100000 #This was the original value
#Number of training repeats=10000
# Output: [ 0.99993704] M=10000
#Number of training repeats=100
# Output: [ 0.99025576]
#Number of training repeats=100000  # Running time: 3s 
#[ 0.99999391]
print "Number of training repeats=%s"%M
for iteration in xrange(M):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    #product of matrices
    #>>> print dot(training_set_inputs, synaptic_weights)
   #[[ 0.26934855]
   #[ 0.23072782]
   #[ 0.66142606]
   #[-0.16134969]]
   #>>> print output
   #[[ 0.56693297]
   #[ 0.55742742]
   #[ 0.65958066]
   #[ 0.45974986]]
   
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))

#cristi ~/SAGE_SCRIPTS/NEURAL_NETWORK $ python FastNN8lines.py 
#Number of training repeats=100
#[ 0.99025576]

#cristi ~/SAGE_SCRIPTS/NEURAL_NETWORK $ python FastNN8lines.py 
#Number of training repeats=1000
#[ 0.99929937]

#Outputs, cu 10000 steps
#cristi ~/SAGE_SCRIPTS/NEURAL_NETWORK $ python FastNN8lines.py 
#[ 0.99993704]
##Outputs, cu 100000 steps
#cristi ~/SAGE_SCRIPTS/NEURAL_NETWORK $ python FastNN8lines.py 
#[ 0.99999391]


#>>> TT=[1,2,3,4,5]
#>>> print TT-1
#Traceback (most recent call last):
  #File "<stdin>", line 1, in <module>
#TypeError: unsupported operand type(s) for -: 'list' and 'int'
#>>> print array(TT)-1
#[0 1 2 3 4]

##############################################

#>>> training_set_outputs = array([[0, 1, 1, 0]]).T
#>>> print training_set_outputs
#[[0]
 #[1]
 #[1]
 #[0]]
#>>> training_set_outputs = array([[0, 1, 1, 0]])
#>>> print training_set_outputs
#[[0 1 1 0]]
#>>> 