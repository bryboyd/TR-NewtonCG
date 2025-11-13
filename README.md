**Matrix-Free Trust-Region Newton-Conjugate-Gradient Optimizer in
Deep Learning**

In this project we explored the use of Trust-Region Newton-Conjugate-Gradient (TRNCG) (specifically CG Steihaug)
as an optimizer in training a neural network. Specifically, we leveraged the forward and backpropagation process of
the neural network—via reverse-mode automatic differentiation (AD)—to perform matrix-free computations of the
action of the Hessian matrix on a vector. We compared the results of this approach with the results of a stochastic
gradient descent optimizer (Adam) in approximating a variety of “elementary” mathematical functions (e.g., linear,
quadratic, trigonometric), as well as in approximating a solution to the forward heat problem with the use of a Physics-
Informed Neural Network (PINN). 
