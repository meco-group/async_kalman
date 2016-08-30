import filterpy.common as common
from filterpy.common import van_loan_discretization
from pykalman.sqrt import CholeskyKalmanFilter
from pykalman.sqrt.cholesky import _filter_predict, _filter_correct
import numpy as np

np.set_printoptions(formatter={'all':lambda x: "%0.5e" % x})


A = np.matrix([[1  , 0.1],[0.2, 1.1]])
B = np.matrix([[3],[4.0]])
Q = np.matrix([[2  , 0.1],[0.1, 3]])

Ad, Qd = van_loan_discretization(A, np.linalg.cholesky(Q), 0.1)
Bd = np.linalg.inv(A)*(Ad-np.eye(2))*B

print Ad
print Bd
print Qd

C = np.matrix([[0.3  , 4],[1, 2]])
D = np.matrix([[0.7],[0.13]])

S  = np.matrix([[3, 0], [2,7 ]])
R  = np.matrix([[2, 0.1], [0.1, 2]])
x0 = np.matrix([[1],[0.1]])
u = 2

z = np.matrix([[1],[0.2]])

kf = CholeskyKalmanFilter(
    transition_matrices=Ad,
    observation_matrices=C,
    transition_covariance=Qd,
    observation_covariance=R,
    transition_offsets=np.array(Bd*u).squeeze(),
    observation_offsets=np.array(D*u).squeeze(),
    initial_state_mean=np.array(x0).squeeze(),
    initial_state_covariance=S*S.T
)

(xp, Sp) = _filter_predict(Ad, np.linalg.cholesky(Qd),
                    np.array(Bd*u).squeeze(), np.array(x0).squeeze(),
                    S)

Sp= np.matrix(Sp)
print np.matrix(xp).T
print Sp

(xp, Sp) = _filter_correct(np.array(C), np.array(np.linalg.cholesky(R)),
                    np.array(D*u).squeeze(), np.array(x0).squeeze(),
                    np.array(S), np.array(z).squeeze())

Sp= np.matrix(Sp)
print np.matrix(xp).T
print Sp*Sp.T

(xp, Sp) = _filter_correct(np.array(C[0,:]), np.array(np.linalg.cholesky(R[:1,:1])),
                    np.array(D*u).squeeze()[:-1], np.array(x0).squeeze(),
                    np.array(S), np.array(z).squeeze()[:-1])

Sp= np.matrix(Sp)
print np.matrix(xp).T
print Sp*Sp.T

print "KinematicKalmanFilter"

A = np.matrix([[0  , 1],[0, 0]])
B = np.matrix([[],[]])
Q = np.matrix([[0.1]])

Ad, Qd = van_loan_discretization(A, np.linalg.cholesky(Q), 0.1)

C = np.matrix([[1.0,0.0]])
D = np.matrix([[]])

P  = np.matrix([[1, 0], [0,1 ]])
R  = np.matrix([[0.1]])
x0 = np.matrix([[0],[0]])

kf = CholeskyKalmanFilter(
    transition_matrices=Ad,
    observation_matrices=C,
    transition_covariance=Qd,
    observation_covariance=R,
    transition_offsets=np.array([0,0]),
    observation_offsets=np.array([0]),
    initial_state_mean=np.array(x0).squeeze(),
    initial_state_covariance=P
)

for x,P in zip(*kf.filter(np.matrix([[2],[2],[2],[2],[2],[2],[2],[2],[2]]))):
    print x
    print P
