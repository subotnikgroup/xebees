#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:50:08 2025

@author: cakhan
"""

"""
ERF for a diatomic with charge-weighted theta function
"""

import numpy as np
import matplotlib.pyplot as plt

M1 = 1
M2 = 1
mu = M1+M2/(M1+M2)
g = 1 #charge ratio between nuclei

NRgrid = 200
Nthetagrid = 100
thetagrid = np.linspace(-np.pi,np.pi,Nthetagrid)
Rgrid = np.linspace(3,8,NRgrid)

Rmesh,thetamesh = np.meshgrid(Rgrid,thetagrid)
X1 = -mu/M1*np.array([Rmesh*np.cos(thetamesh), Rmesh*np.sin(thetamesh)])
X2 = mu/M2*np.array([Rmesh*np.cos(thetamesh), Rmesh*np.sin(thetamesh)])
"""
Helper functions - give the matrix elements and vectors necessary to calculate the ERF at a given nuclear grid point (X1,Y1,X2,Y2) = (R,\theta).
"""

def get_theta(myRind,sigma =1):
    R = Rgrid[myRind]
    Z = np.exp(-R**2/sigma**2) + g*np.exp(-R**2/sigma**2)
    thetafunc1 = np.exp(-R**2/sigma**2)/Z
    thetafunc2 = g*np.exp(-R**2/sigma**2)/Z
    return thetafunc1, thetafunc2
def get_zeta(myRind,beta):
    zeta11 = M1
    zeta22 = M2
    zeta21 = M2*np.exp(-Rgrid[myRind]**2/beta**2)
    zeta12 = M1*np.exp(-Rgrid[myRind]**2/beta**2)
    return np.array([[zeta11, zeta12],[zeta21, zeta22]])
def get_X0s(myRind,mythetaind, beta):
    zeta = get_zeta(myRind,beta)
    myX1 = X1[:,mythetaind,myRind]
    myX2 = X2[:,mythetaind,myRind]
    numerator1 = zeta[0,0]*myX1 + zeta[0,1]*myX2
    denominator1 = zeta[0,0]+zeta[0,1]
    X10 = numerator1/denominator1
    numerator2 = zeta[1,0]*myX1 + zeta[1,1]*myX2
    denominator2 = zeta[1,0] + zeta[1,1]
    X20 = numerator2/denominator2
    return X10, X20
"""
NB: It turns out, explicit calculation of the K matrices isn't required when the molecule lives in a plane. 
def get_Ks(myRind,mythetaind,beta=100): #FOR REFERENCE : General form of K's, adapted from Z. Tao, T. Qiu, et al., 2024, 
    zeta = get_zeta(myRind,beta)
    X10,X20 = get_X0s(myRind,mythetaind,beta)
    myX1 = X1[:,mythetaind,myRind]
    myX2 = X2[:,mythetaind,myRind]
    X1out = np.outer(myX1,myX1)
    X2out = np.outer(myX2,myX2)
    X1in = np.dot(myX1,myX1)
    X2in = np.dot(myX2,myX2)
    X10out = np.outer(X10,X10)
    X20out = np.outer(X20,X20)
    X10in = np.dot(X10,X10)
    X20in = np.dot(X20,X20)
    myK1 = zeta[0,0]*(X1out - X10out - (X1in - X10in)*np.eye(2)) + zeta[1,0]*(X2out - X10out - (X2in - X10in)*np.eye(2))
    myK2 = zeta[1,0]*(X1out - X20out - (X1in - X20in)*np.eye(2)) + zeta[1,1]*(X2out - X20out - (X2in - X20in)*np.eye(2))
    return myK1, myK2

def get_lin_Ks(myRind,mythetaind,beta=100): #Special form of the K matrix for a linear molecule, adapted from T. Qiu, et al., J. Chem. Phys. 160, 124102ff (2024). 
    zeta = get_zeta(myRind,beta)
    myX1 = X1[:,mythetaind,myRind]
    myX2 = X2[:,mythetaind,myRind]
    molaxis = myX2 - myX1 #the bond axis, which is the eigenvector of the K's corresponding to eigenvalue 0.
    proj = np.outer(molaxis,molaxis)
    myX10,myX20 = get_X0s(myRind,mythetaind,beta)
    X1in = np.dot(myX1,myX1)
    X2in = np.dot(myX2,myX2)
    X10in = np.dot(myX10,myX10)
    X20in = np.dot(myX20,myX20)
    myK1 = -zeta[0,0]*((X1in - X10in)*np.eye(2) - proj) - zeta[1,0]*((X2in - X10in)*np.eye(2) - proj)
    myK2 = -zeta[0,1]*((X1in - X20in)*np.eye(2) - proj) - zeta[1,1]*((X2in - X20in)*np.eye(2) - proj)
    return myK1, myK2
# print(get_lin_Ks(100,50))
"""
Nrgrid = 20
rgrid = np.linspace(0.1,9,Nrgrid)
Nphigrid = 10
phigrid = np.linspace(-np.pi,np.pi,Nphigrid)
phimesh,rmesh = np.meshgrid(phigrid,rgrid)
Nsubtot = Nrgrid*Nphigrid
m = 1/1836
dr = (rgrid[Nrgrid-1] - rgrid[0])/Nrgrid
dr = (phigrid[Nphigrid-1] - phigrid[0])/Nphigrid
Ptheta = 1 #nuclear rotation is a quantized parameter
#Param's of the interaction potential
D = 60
d = 0.95
a = 2.52
c = 1 #dimensionless bias factor - 0.707 in Scherrer et al.
A = 2.32e5
betapot = 3.15
C = 2.31e4
def H1(myRind):    
    Te1 = (8*m*rmesh**2)**-1
    R = Rgrid[myRind]*0.529
    r = rmesh*0.529
    rho = np.sqrt(R**2+r**2-2*R*r*np.cos(phimesh))
    V = D*(np.exp(-2*a*(r-d))+1-2*np.exp(-a*(r-d)) + c**2*(np.exp(-2*a/c*(rho-d))-2*np.exp(-a/c*(rho-d)))) + A*np.exp(-betapot*R) - C/R**6
    V = V*0.00159362
    H1 = -Te1+V
    H1 = H1.reshape(Nsubtot)
    return H1
def pe(vec): #returns the radial and angular components of the electronic momentum acting on an input vector. 
    gridvec = vec.reshape(Nrgrid,Nphigrid) #Reshape the vector to perform the derivatives more efficiently
    dummyvect = rmesh**-1*gridvec
    #Term below - 1/r d/dphi
    derphi = 4/5*np.concatenate((dummyvect[0:,1:],np.zeros([Nrgrid,1])),axis=1) -4/5*np.concatenate((np.zeros([Nrgrid,1]),dummyvect[0:,0:Nphigrid-1]),axis = 1)- 1/5*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/5*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:Nphigrid-2]),axis = 1) + 4/105*np.concatenate((dummyvect[0:,3:],np.zeros([Nrgrid,3])),axis=1) -4/105*np.concatenate((np.zeros([Nrgrid,3]),dummyvect[0:,0:Nphigrid-3]),axis = 1)-1/280*np.concatenate((dummyvect[0:,4:],np.zeros([Nrgrid,4])),axis=1) +1/280*np.concatenate((np.zeros([Nrgrid,4]),dummyvect[0:,0:Nphigrid-4]),axis = 1)
    derr = 4/5*np.concatenate((gridvec[1:,0:],np.zeros([1,Nphigrid])),axis=0) -4/5*np.concatenate((np.zeros([1,Nphigrid]),gridvec[0:Nrgrid-1,:]),axis = 0)- 1/5*np.concatenate((gridvec[2:,0:],np.zeros([2,Nphigrid])),axis=0) +1/5*np.concatenate((np.zeros([2,Nphigrid]),gridvec[0:Nrgrid-2,0:]),axis = 0) + 4/105*np.concatenate((gridvec[3:,0:],np.zeros([3,Nphigrid])),axis=0) -4/105*np.concatenate((np.zeros([3,Nphigrid]),gridvec[0:Nrgrid-3,0:]),axis = 0)-1/280*np.concatenate((gridvec[4:,0:],np.zeros([4,Nphigrid])),axis=0) +1/280*np.concatenate((np.zeros([4,Nphigrid]),gridvec[0:Nrgrid-4,0:]), axis = 0)
    p_r = -1j*derr
    p_phi = -1j*derphi
    p_r = p_r.reshape(Nsubtot)
    p_phi = p_phi.reshape(Nsubtot)
    return p_r, p_phi 
# testvec = np.ones(Nsubtot)
# print(pe(testvec)) 
def get_theta_op(myRind,sigma =1):
    R = Rgrid[myRind]
    rho1 = np.sqrt((mu/M1*R)**2+rmesh**2+2*mu/M1*R*rmesh*np.cos(phimesh))
    rho2 = np.sqrt((mu/M2*R)**2+rmesh**2-2*mu/M2*R*rmesh*np.cos(phimesh))
    Z = np.exp(-rho1**2/sigma**2) + g*np.exp(-rho2**2/sigma**2)
    thetafunc1 = np.exp(-rho1**2/sigma**2)/Z
    thetafunc2 = g*np.exp(-rho2**2/sigma**2)/Z
    return thetafunc1, thetafunc2
def J_op(vec,myRind, mythetaind, sigma=1):
    theta1, theta2 = get_theta_op(myRind,sigma)
    theta1list = theta1.reshape(Nsubtot)
    theta2list = theta2.reshape(Nsubtot)
    theta1_vec = theta1list*vec
    theta2_vec = theta2list*vec
    x1 = rmesh*np.cos(phimesh) - X1[0,mythetaind,myRind]
    y1 = rmesh*np.sin(phimesh) - X1[1,mythetaind,myRind]
    x2 = rmesh*np.cos(phimesh) - X2[0,mythetaind,myRind]
    y2 = rmesh*np.sin(phimesh) - X2[1,mythetaind,myRind]
    x1list = x1.reshape(Nsubtot)
    x2list = x2.reshape(Nsubtot)
    y1list = y1.reshape(Nsubtot)
    y2list = y2.reshape(Nsubtot)
    pe_r,pe_phi = pe(vec) #gives radial components of the electronic momentum acting on a vector 
    philist = phimesh.reshape(Nsubtot)
    pe_x = pe_r*np.cos(philist) - pe_phi*np.sin(philist) #converting p_e\ket{psi} to cartesian basis
    pe_y = pe_r*np.sin(philist) + pe_phi*np.cos(philist)
    pe_theta1_r, pe_theta1_phi = pe(theta1_vec)
    pe_theta1_x = pe_theta1_r*np.cos(philist) - pe_theta1_phi*np.sin(philist)
    pe_theta1_y = pe_theta1_r*np.sin(philist) + pe_theta1_phi*np.cos(philist)
    pe_theta2_r, pe_theta2_phi = pe(theta2_vec)
    pe_theta2_x = pe_theta2_r*np.cos(philist) - pe_theta2_phi*np.sin(philist)
    pe_theta2_y = pe_theta2_r*np.sin(philist) + pe_theta2_phi*np.cos(philist)
    J1 = x1list*(theta1list*pe_y + pe_theta1_y) - y1list*(theta1list*pe_x +pe_theta1_x)
    J2 = x2list*(theta2list*pe_y + pe_theta2_y) - y2list*(theta2list*pe_x +pe_theta2_x)
    return J1,J2
def gamma_op(vec,myRind,mythetaind, beta = 1, sigma = 1):
    J1,J2 = J_op(vec,myRind,mythetaind,sigma)
    zeta = get_zeta(myRind,beta)
    myX10, myX20 = get_X0s(myRind,mythetaind, beta)
    myX1 = X1[:,mythetaind,myRind]
    myX2 = X2[:,mythetaind,myRind]
    relX11 = myX1 - myX10
    relX12 = myX1 - myX20
    relX21 = myX2 - myX10
    relX22 = myX2 - myX20
    K1J1  = (zeta[0,0]*np.linalg.norm(relX11)**2)**-1 + (zeta[1,0]*np.linalg.norm(relX21)**2)*J1
    K2J2  = (zeta[0,1]*np.linalg.norm(relX12)**2)**-1 + (zeta[1,1]*np.linalg.norm(relX22)**2)*J2
    gamma1 = zeta[0,0]*np.linalg.norm(relX11)*K1J1 + zeta[0,1]*np.linalg.norm(relX12)*K2J2
    gamma2 = zeta[1,0]*np.linalg.norm(relX21)*K1J1 + zeta[1,1]*np.linalg.norm(relX22)*K2J2
    return gamma1, gamma2
def Te(vec): #subroutine to evaluate the electronic kinetic energy term on a vector in polar coordinates
    # Position-diagonal terms
    if len(vec) != Nsubtot:
        return print('Incorect vector length - must conform to grid choice!')
    gridvec = vec.reshape(Nrgrid,Nphigrid) #Reshape the vector to perform the derivatives more efficiently
    #Derivative terms (stencil)
    dummyvect = rmesh**-2*gridvec
    ddphi = -205/72*dummyvect + 8/5*np.concatenate((gridvec[0:,1:],np.zeros([Nrgrid,1])),axis=1) +8/5*np.concatenate((np.zeros([Nrgrid,1]),gridvec[0:,0:Nphigrid-1]),axis = 1)- 1/5*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) -1/5*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:Nphigrid-2]),axis = 1) + 8/315*np.concatenate((dummyvect[0:,3:],np.zeros([Nrgrid,3])),axis=1) +8/315*np.concatenate((np.zeros([Nrgrid,3]),dummyvect[0:,0:Nphigrid-3]),axis = 1)-1/560*np.concatenate((dummyvect[0:,4:],np.zeros([Nrgrid,4])),axis=1) -1/560*np.concatenate((np.zeros([Nrgrid,4]),dummyvect[0:,0:Nphigrid-4]),axis = 1)
    ddr = -205/72*gridvec + 8/5*np.concatenate((gridvec[1:,0:],np.zeros([1,Nphigrid])),axis=0) +8/5*np.concatenate((np.zeros([1,Nphigrid]),gridvec[0:Nrgrid-1,0:]),axis = 0)- 1/5*np.concatenate((gridvec[2:,0:],np.zeros([2,Nphigrid])),axis=0) -1/5*np.concatenate((np.zeros([2,Nphigrid]),gridvec[0:Nrgrid-2,0:]),axis = 0) + 8/315*np.concatenate((gridvec[3:,0:],np.zeros([3,Nphigrid])),axis=0) +8/315*np.concatenate((np.zeros([3,Nphigrid]),gridvec[0:Nrgrid-3,0:]),axis = 0)-1/560*np.concatenate((gridvec[4:,0:],np.zeros([4,Nphigrid])),axis=0) -1/560*np.concatenate((np.zeros([4,Nphigrid]),gridvec[0:Nrgrid-4,0:]),axis = 0)
    derphi1 = 4/5*np.concatenate((gridvec[0:,1:],np.zeros([Nrgrid,1])),axis=1) -4/5*np.concatenate((np.zeros([Nrgrid,1]),gridvec[0:,0:Nphigrid-1]),axis = 1)- 1/5*np.concatenate((gridvec[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/5*np.concatenate((np.zeros([Nrgrid,2]),gridvec[0:,0:Nphigrid-2]),axis = 1) + 4/105*np.concatenate((gridvec[0:,3:],np.zeros([Nrgrid,3])),axis=1) -4/105*np.concatenate((np.zeros([Nrgrid,3]),gridvec[0:,0:Nphigrid-3]),axis = 1)-1/280*np.concatenate((gridvec[0:,4:],np.zeros([Nrgrid,4])),axis=1) +1/280*np.concatenate((np.zeros([Nrgrid,4]),gridvec[0:,0:Nphigrid-4]),axis = 1)
    derphi2 = 4/5*np.concatenate((dummyvect[0:,1:],np.zeros([Nrgrid,1])),axis=1) -4/5*np.concatenate((np.zeros([Nrgrid,1]),dummyvect[0:,0:Nphigrid-1]),axis = 1)- 1/5*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/5*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:Nphigrid-2]),axis = 1) + 4/105*np.concatenate((dummyvect[0:,3:],np.zeros([Nrgrid,3])),axis=1) -4/105*np.concatenate((np.zeros([Nrgrid,3]),dummyvect[0:,0:Nphigrid-3]),axis = 1)-1/280*np.concatenate((dummyvect[0:,4:],np.zeros([Nrgrid,4])),axis=1) +1/280*np.concatenate((np.zeros([Nrgrid,4]),dummyvect[0:,0:Nphigrid-4]),axis = 1)
    # # Less accurate stencils if desired 
    # ddr = -5/2*gridvec + 4/3*np.concatenate((gridvec[1:,0:],np.zeros([1,Nphigrid])),axis=0) +4/3*np.concatenate((np.zeros([1,Nphigrid]),gridvec[:-1,0:]),axis = 0)- 1/12*np.concatenate((gridvec[2:,0:],np.zeros([2,Nphigrid])),axis=0) -1/12*np.concatenate((np.zeros([2,Nphigrid]),gridvec[:-2,0:]),axis = 0)
    # ddphi = -5/2*dummyvect + 4/3*np.concatenate((dummyvect[0:,1:],np.zeros([Nrgrid,1])),axis=1) +4/3*np.concatenate((np.zeros([Nrgrid,1]),dummyvect[0:,0:Nphigrid-1]),axis = 1)- 1/12*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) -1/12*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:Nphigrid-2]),axis = 1)
    # derphi1 = 2/3*np.concatenate((gridvec[0:,1:],np.zeros([Nrgrid,1])),axis=1)-2/3*np.concatenate((np.zeros([Nrgrid,1]),gridvec[0:,:-1]),axis = 1)- 1/12*np.concatenate((gridvec[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/12*np.concatenate((np.zeros([Nrgrid,2]),gridvec[0:,0:-2]),axis = 1)
    # derphi2 = 2/3*np.concatenate((dummyvect[0:,1:],np.zeros([Nrgrid,1])),axis=1)-2/3*np.concatenate((np.zeros([Nrgrid,1]),dummyvect[0:,:-1]),axis = 1)- 1/12*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/12*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:-2]),axis = 1)
    ddphiterm = -ddphi/dphi**2/2/m
    ddrterm = -ddr/dr**2/2/m
    #Construct the diagonal of the Hamiltonian (which is needed for the Davidson routine)
    return myvec
