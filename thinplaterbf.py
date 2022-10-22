import itertools
import math as m
import numpy as np
from typing import Union
from numpy import linalg as la
from numba import njit, prange
from scipy.io import loadmat
from scipy.special import xlogy
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

# Calculate Hadamard product (thin plate RBF kernel) with Numba parallelization
@njit(parallel=True)
def hhproduct(k, grid, n, p, q):
    T = np.zeros((n, n))
    for i in prange(n):
        for j in range(i+1):
            ki = k - grid[i]
            kj = k - grid[j]
            di = np.sqrt(np.dot(ki, ki))
            dj = np.sqrt(np.dot(kj, kj))
            xxip = (k - grid[i])[p]
            xxjp = (k - grid[j])[p]
            if p==q:
                T[i, j] = (2*m.log(di)+2*(xxip/di)**2+1)*(2*m.log(dj)+2*(xxjp/dj)**2+1)
            else:
                xxiq = (k - grid[i])[q]
                xxjq = (k - grid[j])[q]
                T[i, j] = 4*xxip*xxiq*xxjp*xxjq/di**2/dj**2
    return T

# Multidimensional RBF approximation with thin plate smoothing
def rbf(
        points: np.ndarray,
        regularizer: Union[str, float] = 'default',
        constrained: str = 'false',
        step: float = 0.001,
        precision: float = 0.001
        ):
    '''
    Arguments:
        points (np.ndarray):        array of multidimensional points with corresponding values
                                    arranged as [[x1, x2, .., xd, val], ...]
        regularizer (str, float):   regularizer value as float or as 'default' = (d+1)/n**(1/d)
                                    where n - number of points, d - number of dimensions
        constrained (str):          restricts values at boundary points to the original ones,
                                    'false' or 'true'
        step (float):               gradient ascent step size for contrained approximation case
                                    (optimization problem)
        precision (float):          desired precision value at constrained boundary points

    Returns:
        approximation:              approximating function   
    '''

    # Calculates length of an input vector
    def sprod(vec):
        return np.sqrt(np.dot(vec, vec))

    # Adds midpoints to an input array
    def middle(arr):
        mid = np.vstack([arr[1:], arr[:-1]]).mean(axis=0)
        return np.union1d(arr, mid)

    # Thin plate RBF kernel
    def thinp(r):
        return xlogy(r*r, r)

    # Construct integration steps and subgrid 
    def subgrid(grid):
        n = len(grid)
        d = len(grid[0])
        ugrid = np.empty(d, dtype=object)
        delta = np.empty(d, dtype=object)
        
        for i in range(d):
            ugrid[i] = np.unique(grid.T[i])
            ugrid[i] = middle(ugrid[i])
            delta[i] = ugrid[i]
            ugrid[i] = np.vstack([ugrid[i][1:], ugrid[i][:-1]]).mean(axis=0)
            delta[i] = np.diff(delta[i])

        sbgrid = np.array(list(itertools.product(*ugrid)))
        deltas = np.array(list(itertools.product(*delta)))
        deltas = np.prod(deltas, axis=1)

        return np.asarray(sbgrid), np.asarray(deltas)

    # Rescale a given domain
    def rescale(points):
        oldgrid = np.delete(points, -1, axis=1)
        d = len(oldgrid[0])
        mx = np.amax(oldgrid, axis=0)
        mn = np.amin(oldgrid, axis=0)
        scales = np.ones(d+1)
        
        for i in range(d):
            scales[i] = (len(np.unique(oldgrid.T[i]))-1)/(mx[i]-mn[i])
            
        points[:] = points[:]*scales
        scales = np.delete(scales, -1)
        
        return points, scales

    # Compose initial matrices
    points, scales = rescale(points)
    n = len(points)
    d = len(points[0])-1
    grid = np.delete(points, -1, axis=1)    
    Phi = [[thinp(sprod(np.subtract(points[i, 0:-1], points[j, 0:-1]))) \
            for j in range(n)] for i in range(n)]
    P = np.ones((n, d+1))
    P[:, 0:-1] = points[:, 0:-1]
    F = points[:, -1]
    M = np.zeros((n+d+1, n+d+1))
    M[0:n, 0:n] = Phi     
    M[0:n, n:n+d+1] = P
    M[n:n+d+1, 0:n] = np.transpose(P)
    v = np.zeros(n+d+1)
    v[0:n] = F

    # No regularization
    if regularizer==0.0:
        Q = np.zeros((n+d+1, n+d+1))
        sol = np.linalg.solve(M, v)
        
    # Default regularization
    else:
        if regularizer=='default':
            regularizer = (d+1)/n**(1/d)

        # Compose integration matrix Q
        Q = np.zeros((n, n))    
        intgrid, deltas = subgrid(grid) 
        lenintgrid = len(intgrid)

        for p in range(d):
            for q in range(d-p):
                integral = np.zeros((n, n))
                for k in range(lenintgrid):
                    T = hhproduct(intgrid[k], grid, n, p, q)
                    integral = integral + T*deltas[k]
                Q = Q + integral*(2-(p==q))

        Q = np.tril(Q) + np.triu(Q.T, 1)
        Q = np.append(Q, np.zeros((d+1, n)), axis=0)
        Q = np.append(Q, np.zeros((n+d+1, d+1)), axis=1)

    # Not constrained
    if (constrained=='false')&(regularizer!=0.0):
        sol = np.linalg.solve(M.T @ M + regularizer*Q, M.T @ v)

    # Constrained
    elif (constrained=='true')&(regularizer!=0):
        if  d==1:
            Mc = M[[0,n-1]]
            vc = v[[0,n-1]]
        else:
            indices = np.unique(Delaunay(grid).convex_hull.flatten())
            Mc = M[indices]
            vc = v[indices]

        # Solve optimization problem to find constrained solution
        def solution(mu):
            return np.linalg.solve(M.T @ M + regularizer*Q + mu*(Mc.T @ Mc), M.T @ v + mu*(Mc.T @ vc))
        def dldm(w):
            return np.linalg.norm(Mc @ w - vc)**2

        # Initial guess
        mu = 0.
        sol = solution(mu)
        error0 = dldm(sol)
        
        while error0 > precision:
            mu = mu + step*error0
            sol = solution(mu)
            error1 = dldm(sol)
            ratio = error1/error0
            if ratio > precision:
                step = step*(1+ratio)**4
            error0 = error1
            print(error0)
            
    # Form an approximating function 
    lam, b, a = sol[0:n], sol[n:n+d], sol[n+d]

    def approximation(x):
        return sum(lam[i]*thinp(np.linalg.norm(np.subtract(x*scales, points[i, 0:-1]))) \
                   for i in range(n)) + np.dot(b, x*scales) + a

    return approximation
