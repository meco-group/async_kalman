import filterpy.common as common
from filterpy.common import van_loan_discretization
import numpy as np

np.set_printoptions(formatter={'all':lambda x: "%0.5e" % x})


A = np.matrix([[1  , 0.1],[0.2, 1.1]])
B = np.matrix([[3],[4.0]])
Q = np.matrix([[2  , 0.1],[0.1, 3]])

Ad, Qd = van_loan_discretization(A, np.linalg.cholesky(Q), 0.1)

print Ad
print np.linalg.inv(A)*(Ad-np.eye(2))*B
print Qd
