#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:50:38 2024

@author: cakhan
"""

"""
Born-Oppenheimer for a diatomic molecule in a magnetic field
"""

from pyscf import lib
import numpy as np
import time
import matplotlib.pyplot as plt

MASS = 10 #nucleus mass (M_H = 1)
MAG = 0.01 #magnetic field strength
ROT = 1 #nuclear rotational state 

"""
----- The routine -----

The PySCF module linalg.helper.davidson is used for all matrix diagonalization steps in both the B.O and exact calculations. 

Parameters - Section 1 - System size (grid size and spacing) were converged for the given form of V, the interaction. They can be changed.
Born-Oppenheimer Calculation - Section 2 - The subroutine elham(vec, nucpos) outputs the matrix vector product newvec_j = \sum_i H_ji vec_i for the electronic Hamiltonian at
    a given nuclear configuration. The initial guess for the Born-Oppenheimer ground state is a Gaussian in phi and r, whose widths were set to minimize <H_el> at the equilibrium 
    nuclear configuration
Exact Calculation - Section 3 - Takes the Born-Oppenheimer electronic ground state \psi(r,phi; R_min) x \chi_0(R) as the guess, where R_min is the minimum of the adiabat. 

"""
#Section 1 - System size param's 
manifold = 2 #No. of Born Oppenheimer states to calculate if you choose to use Davidson. NB - convergence beyond 3 e'states can be tricky!!!!!
#Param's of the potential
D = 60
d = 0.95
a = 2.52
c = 1 #dimensionless bias factor - 0.707 in Scherrer et al.
A = 2.32e5
beta = 3.15
C = 2.31e4
#Construct coordinate grids
Rmin = 3.9
Rmax = 7.3
dR = 0.02
Rgrid = np.arange(Rmin,Rmax,dR)
NRgrid = len(Rgrid)
Masslist = 1836*np.array([5])
m = 1836
dr = 0.02
rmax = 7.5
rmin = 0.05
rgrid = np.arange(rmin,rmax,dr)
dphi = 0.02
phigrid = np.arange(-np.pi/2,np.pi/2,dphi)
[elphi,elr] = np.meshgrid(phigrid,rgrid)
Nrgrid = len(rgrid)
Nphigrid = len(phigrid)
Nsubtot = Nrgrid*Nphigrid
rmesh = elr.reshape(Nsubtot)
isqrmesh = elr**-2
isqrmesh = isqrmesh.reshape(Nsubtot)
B = MAG
thetanum = ROT
thst = thetanum*2*np.pi
M = MASS*1836

#Section 2 -  Born-Oppenheimer calcualtion via approximate (PySCF Davidson) diagonalization of the electronic Hamiltonian
savevecs = np.zeros([Nsubtot,manifold,NRgrid],dtype=complex) #Remember - in a magnetic field, the wave function can be complex-valued!!!!!
gsadiabat = np.zeros(NRgrid) #pre-allocate the adiabats
esadiabat = np.zeros(NRgrid)
start = time.time()
#Routine to diagonalize the electronic Hamiltonian via Davidson - Vide eqn. 20 and 28 of the Overleaf for the terms invovled.
def H1(nucpos):    
    Te1 = (8*m*elr**2)**-1
    qterms = 0.25*B**2*elr**2/(2*m)
    R = Rgrid[nucpos]*0.529
    r = elr*0.529
    rho = np.sqrt(R**2+r**2-2*R*r*np.cos(elphi))
    V = D*(np.exp(-2*a*(r-d))+1-2*np.exp(-a*(r-d)) + c**2*(np.exp(-2*a/c*(rho-d))-2*np.exp(-a/c*(rho-d)))) + A*np.exp(-beta*R) - C/R**6
    V = V*0.00159362
    Rterms = B**2*Rgrid[nucpos]**2/8/M- thst**2/Rgrid[nucpos]**2/2/M - 1/8/Rgrid[nucpos]**2/M
    H1 = -Te1+qterms+V +Rterms
    H1 = H1.reshape(Nsubtot)
    return H1
def diagelham(nucpos):
    diagham= H1(nucpos)+5/2*(1/dr**2/2/m*np.ones(Nsubtot) + isqrmesh/dphi**2/2/m) - thst**2*isqrmesh/2/m 
    return diagham
def elham(vec,nucpos):
    # Position-diagonal terms
    # if len(vec) != Nsubtot:
    #     return print('Incorect vector length - must conform to grid choice!')
    myH1 = H1(nucpos)
    vec1 = vec*myH1 #elementwise list multiplication - the same as multiplying the vector by the diagonal of the Hamiltonian
    gridvec = vec.reshape(Nrgrid,Nphigrid) #Reshape the vector to perform the derivatives more efficiently
    #Derivative terms (stencil)
    dummyvect = elr**-2*gridvec
    # ddphi = -205/72*dummyvect + 8/5*np.concatenate((gridvec[0:,1:],np.zeros([Nrgrid,1])),axis=1) +8/5*np.concatenate((np.zeros([Nrgrid,1]),gridvec[0:,0:Nphigrid-1]),axis = 1)- 1/5*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) -1/5*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:Nphigrid-2]),axis = 1) + 8/315*np.concatenate((dummyvect[0:,3:],np.zeros([Nrgrid,3])),axis=1) +8/315*np.concatenate((np.zeros([Nrgrid,3]),dummyvect[0:,0:Nphigrid-3]),axis = 1)-1/560*np.concatenate((dummyvect[0:,4:],np.zeros([Nrgrid,4])),axis=1) -1/560*np.concatenate((np.zeros([Nrgrid,4]),dummyvect[0:,0:Nphigrid-4]),axis = 1)
    # ddr = -205/72*gridvec + 8/5*np.concatenate((gridvec[1:,0:],np.zeros([1,Nphigrid])),axis=0) +8/5*np.concatenate((np.zeros([1,Nphigrid]),gridvec[0:Nrgrid-1,0:]),axis = 0)- 1/5*np.concatenate((gridvec[2:,0:],np.zeros([2,Nphigrid])),axis=0) -1/5*np.concatenate((np.zeros([2,Nphigrid]),gridvec[0:Nrgrid-2,0:]),axis = 0) + 8/315*np.concatenate((gridvec[3:,0:],np.zeros([3,Nphigrid])),axis=0) +8/315*np.concatenate((np.zeros([3,Nphigrid]),gridvec[0:Nrgrid-3,0:]),axis = 0)-1/560*np.concatenate((gridvec[4:,0:],np.zeros([4,Nphigrid])),axis=0) -1/560*np.concatenate((np.zeros([4,Nphigrid]),gridvec[0:Nrgrid-4,0:]),axis = 0)
    # derphi1 = 4/5*np.concatenate((gridvec[0:,1:],np.zeros([Nrgrid,1])),axis=1) -4/5*np.concatenate((np.zeros([Nrgrid,1]),gridvec[0:,0:Nphigrid-1]),axis = 1)- 1/5*np.concatenate((gridvec[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/5*np.concatenate((np.zeros([Nrgrid,2]),gridvec[0:,0:Nphigrid-2]),axis = 1) + 4/105*np.concatenate((gridvec[0:,3:],np.zeros([Nrgrid,3])),axis=1) -4/105*np.concatenate((np.zeros([Nrgrid,3]),gridvec[0:,0:Nphigrid-3]),axis = 1)-1/280*np.concatenate((gridvec[0:,4:],np.zeros([Nrgrid,4])),axis=1) +1/280*np.concatenate((np.zeros([Nrgrid,4]),gridvec[0:,0:Nphigrid-4]),axis = 1)
    # derphi2 = 4/5*np.concatenate((dummyvect[0:,1:],np.zeros([Nrgrid,1])),axis=1) -4/5*np.concatenate((np.zeros([Nrgrid,1]),dummyvect[0:,0:Nphigrid-1]),axis = 1)- 1/5*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/5*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:Nphigrid-2]),axis = 1) + 4/105*np.concatenate((dummyvect[0:,3:],np.zeros([Nrgrid,3])),axis=1) -4/105*np.concatenate((np.zeros([Nrgrid,3]),dummyvect[0:,0:Nphigrid-3]),axis = 1)-1/280*np.concatenate((dummyvect[0:,4:],np.zeros([Nrgrid,4])),axis=1) +1/280*np.concatenate((np.zeros([Nrgrid,4]),dummyvect[0:,0:Nphigrid-4]),axis = 1)
    # # Less accurate stencils if desired 
    ddr = -5/2*gridvec + 4/3*np.concatenate((gridvec[1:,0:],np.zeros([1,Nphigrid])),axis=0) +4/3*np.concatenate((np.zeros([1,Nphigrid]),gridvec[:-1,0:]),axis = 0)- 1/12*np.concatenate((gridvec[2:,0:],np.zeros([2,Nphigrid])),axis=0) -1/12*np.concatenate((np.zeros([2,Nphigrid]),gridvec[:-2,0:]),axis = 0)
    ddphi = -5/2*dummyvect + 4/3*np.concatenate((dummyvect[0:,1:],np.zeros([Nrgrid,1])),axis=1) +4/3*np.concatenate((np.zeros([Nrgrid,1]),dummyvect[0:,0:Nphigrid-1]),axis = 1)- 1/12*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) -1/12*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:Nphigrid-2]),axis = 1)
    derphi1 = 2/3*np.concatenate((gridvec[0:,1:],np.zeros([Nrgrid,1])),axis=1)-2/3*np.concatenate((np.zeros([Nrgrid,1]),gridvec[0:,:-1]),axis = 1)- 1/12*np.concatenate((gridvec[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/12*np.concatenate((np.zeros([Nrgrid,2]),gridvec[0:,0:-2]),axis = 1)
    derphi2 = 2/3*np.concatenate((dummyvect[0:,1:],np.zeros([Nrgrid,1])),axis=1)-2/3*np.concatenate((np.zeros([Nrgrid,1]),dummyvect[0:,:-1]),axis = 1)- 1/12*np.concatenate((dummyvect[0:,2:],np.zeros([Nrgrid,2])),axis=1) +1/12*np.concatenate((np.zeros([Nrgrid,2]),dummyvect[0:,0:-2]),axis = 1)
    ddphiterm = -ddphi/dphi**2/2/m
    ddrterm = -ddr/dr**2/2/m
    derphi1term = 1j*B/2/m*derphi1/dphi
    derphi2term = -1j*thst/m*derphi2/dphi
    derivterms = ddrterm+derphi1term+derphi2term+ddphiterm
    derivterms = derivterms.reshape(Nsubtot)
    thetaterm = -thst**2/2/m*isqrmesh*vec
    myvec = vec1+thetaterm+derivterms 
    #Construct the diagonal of the Hamiltonian (which is needed for the Davidson routine)
    return myvec
for k in range(NRgrid):
    aop = lambda x :elham(x,k)
    mydiag = diagelham(k)
    r0 = Rgrid[k]/2
    guesssig = 0.007
    phisig = 0.0084
    guess = np.exp(-(elr-r0)**2/guesssig) + np.exp(-elphi**2/phisig)
    guess = guess/np.linalg.norm(guess)
    guess = guess.reshape(Nsubtot)
    evals,evecs = lib.davidson(aop,guess,mydiag,nroots = manifold,follow_state = True)
    # Anop = linalg.LinearOperator((Nsubtot,Nsubtot),matvec=aop)
    # evals,evecs = linalg.eigs(Anop,k=3,which='SM')
    # idx = np.argsort(evals)
    # evals = np.sort(evals)
    # evecs = np.array(evecs)
    # evecs = evecs[:,idx]
    esadiabat[k] = evals[1]
    gsadiabat[k] = evals[0]
    for level in range(manifold):
         savevecs[:,level,k] = evecs[level] #save the eigenvectors to construct our guess in the exact calculation
plt.plot(Rgrid,gsadiabat)
plt.xlabel('Nuclear position')
plt.ylabel('Energy (a.u.)')
plt.show()
dderiv_R = np.zeros([NRgrid,NRgrid])
for i in range(NRgrid):
    dderiv_R[i,i] = -205/72
for i in range(NRgrid-1):
    dderiv_R[i,i+1] = 8/5
    dderiv_R[i+1,i] = 8/5
for i in range(NRgrid-2):
    dderiv_R[i,i+2] = -1/5
    dderiv_R[i+2,i] = -1/5
for i in range(NRgrid-3):
    dderiv_R[i,i+3] = 8/315
    dderiv_R[i+3,i] = 8/315
for i in range(NRgrid-4):
    dderiv_R[i,i+4] = -1/560
    dderiv_R[i+4,i] = -1/560
dderiv_R = dderiv_R/dR**2
isqRgrid = 1/Rgrid**2
isqRmat = np.diag(isqRgrid)
theguess = np.zeros(Nsubtot*NRgrid,dtype='complex')
Tn = (-dderiv_R)/2/M  #- Blist[field]*thst/2*(1/m-1/M)
# adiabat = adiabat - np.amin(adiabat)
# adiabat[field,:] = adiabat[field,:] - np.min(adiabat[field,:])
HBO = Tn + np.diag(gsadiabat)
[BOvals,BOvecs] = np.linalg.eig(HBO)
idx = np.argsort(BOvals)
BOvals = np.sort(BOvals)
print('k='+str(thetanum) +', B = '+str(B) +', Mass ratio = '+str(M/m)+' 2 lowest BO eigenenergies:'+str(BOvals[1])+' , '+str(BOvals[0]))
BOvecs = BOvecs[:,idx]
BOtrans = BOvals[1]-BOvals[0]
idxmin = np.argmin(gsadiabat)
elgs = savevecs[:,0,idxmin]
eles = savevecs[:,1,idxmin]
mygs = np.kron(BOvecs[:,0],elgs)
myes = np.kron(BOvecs[:,1],elgs)
elgsgrid = elgs.reshape(Nrgrid,Nphigrid)
plt.contourf(phigrid,rgrid,elgsgrid**2)
plt.xlabel('phi')
plt.ylabel('r (a.u.)')
plt.show()
elesgrid = eles.reshape(Nrgrid,Nphigrid)
plt.contourf(phigrid,rgrid,elesgrid**2)
plt.xlabel('phi')
plt.ylabel('r (a.u.)')
plt.show()
print('First vibrational excitation:'+str(BOtrans))
end = time.time()
print('Walltime = '+ str(end-start))
rexpectgs = np.dot(elgs,rmesh*elgs)
rexpectes = np.dot(eles,rmesh*eles)
BOgs = BOvecs[:,0]
Rexpect = np.dot(BOgs.conj(),Rgrid*BOgs)
print('<r> = ' + str(rexpectgs) + ', <R> = ' + str(Rexpect))
#%%

'''
Exact routine - Davidson diagonalization of the full Hamiltonian using the Born-Oppenheimer wave functions as a guess. 
'''
start = time.time()
[elr,elR,elphi] = np.meshgrid(rgrid,Rgrid,phigrid)
Ntot = Nrgrid*Nphigrid*NRgrid
rmesh = elr.reshape(Ntot)
guesslist = [mygs,myes]

Tn1 = (8*M*elR**2)**-1
Te1 = (8*m*elr**2)**-1
qterms = 0.25*B**2*(elR**2/(2*M) + elr**2/(2*m))
Rbohr = elR*0.529 #temporarily convert the lengths to angstroms for ease of calculating the potential
rbohr = elr*0.529
rho = np.sqrt(Rbohr**2+rbohr**2-2*np.multiply(rbohr,Rbohr)*np.cos(elphi))
V = D*(np.exp(-2*a*(rbohr-d))+1-2*np.exp(-a*(rbohr-d)) + c**2*(np.exp(-2*a/c*(rho-d))-2*np.exp(-a/c*(rho-d)))) + A*np.exp(-beta*Rbohr) - C/Rbohr**6
V = V*0.00159362
H2 = -Tn1-Te1+qterms+V
thetaterms = -thst**2*(1/2/M/elR**2 + 1/2/m/elr**2)
H2list = H2.reshape(Ntot)
thetaterms = thetaterms.reshape(Ntot)
myexdiag = thetaterms+H2list + 205/72*(1/rmesh**2/dphi**2/2/m + 1/dr**2/2/m)+ +5/2*(1/dR**2/2/M)
def myham(vec):
    # Position-diagonal terms
    if len(vec) != Ntot:
        return print('Incorect vector length - must conform to grid choice!')
    gridvec = vec.reshape(NRgrid,Nrgrid,Nphigrid) #Reshape the vector to perform the derivatives more efficiently
    #Derivative terms (stencil)
    dummyvect = elr**-2*gridvec
    # ddphi = -205/72*dummyvect + 8/5*np.concatenate((gridvec[0:,0:,1:],np.zeros([NRgrid,Nrgrid,1])),axis=2) +8/5*np.concatenate((np.zeros([NRgrid,Nrgrid,1]),gridvec[0:,0:,0:Nphigrid-1]),axis = 2)- 1/5*np.concatenate((dummyvect[0:,0:,2:],np.zeros([NRgrid,Nrgrid,2])),axis=2) -1/5*np.concatenate((np.zeros([NRgrid,Nrgrid,2]),dummyvect[0:,0:,0:Nphigrid-2]),axis = 2) + 8/315*np.concatenate((dummyvect[0:,0:,3:],np.zeros([NRgrid,Nrgrid,3])),axis=2) +8/315*np.concatenate((np.zeros([NRgrid,Nrgrid,3]),dummyvect[0:,0:,0:Nphigrid-3]),axis = 2)-1/560*np.concatenate((dummyvect[0:,0:,4:],np.zeros([NRgrid,Nrgrid,4])),axis=2) -1/560*np.concatenate((np.zeros([NRgrid,Nrgrid,4]),dummyvect[0:,0:,0:Nphigrid-4]),axis = 2)
    # ddr = -205/72*gridvec + 8/5*np.concatenate((gridvec[0:,1:,0:],np.zeros([NRgrid,1,Nphigrid])),axis=1) +8/5*np.concatenate((np.zeros([NRgrid,1,Nphigrid]),gridvec[0:,0:Nrgrid-1,0:]),axis = 1)- 1/5*np.concatenate((gridvec[0:,2:,0:],np.zeros([NRgrid,2,Nphigrid])),axis=1) -1/5*np.concatenate((np.zeros([NRgrid,2,Nphigrid]),gridvec[0:,0:Nrgrid-2,0:]),axis = 1) + 8/315*np.concatenate((gridvec[0:,3:,0:],np.zeros([NRgrid,3,Nphigrid])),axis=1) +8/315*np.concatenate((np.zeros([NRgrid,3,Nphigrid]),gridvec[0:,0:Nrgrid-3,0:]),axis = 1)-1/560*np.concatenate((gridvec[0:,4:,0:],np.zeros([NRgrid,4,Nphigrid])),axis=1) -1/560*np.concatenate((np.zeros([NRgrid,4,Nphigrid]),gridvec[0:,0:Nrgrid-4,0:]),axis = 1)
    ddR = -205/72*gridvec + 8/5*np.concatenate((gridvec[1:,0:,0:],np.zeros([1,Nrgrid,Nphigrid])),axis=0) +8/5*np.concatenate((np.zeros([1,Nrgrid,Nphigrid]),gridvec[0:NRgrid-1,0:,0:]),axis = 0)- 1/5*np.concatenate((gridvec[2:,0:,0:],np.zeros([2,Nrgrid,Nphigrid])),axis=0) -1/5*np.concatenate((np.zeros([2,Nrgrid,Nphigrid]),gridvec[0:NRgrid-2,0:,0:]),axis = 0) + 8/315*np.concatenate((gridvec[3:,0:,0:],np.zeros([3,Nrgrid,Nphigrid])),axis=0) +8/315*np.concatenate((np.zeros([3,Nrgrid,Nphigrid]),gridvec[0:NRgrid-3,0:,0:]),axis = 0)-1/560*np.concatenate((gridvec[4:,0:,0:],np.zeros([4,Nrgrid,Nphigrid])),axis=0) -1/560*np.concatenate((np.zeros([4,Nrgrid,Nphigrid]),gridvec[0:NRgrid-4,0:,0:]),axis = 0)
    # derphi1 = 4/5*np.concatenate((gridvec[0:,0:,1:],np.zeros([NRgrid,Nrgrid,1])),axis=2) -4/5*np.concatenate((np.zeros([NRgrid,Nrgrid,1]),gridvec[0:,0:,0:Nphigrid-1]),axis = 2)- 1/5*np.concatenate((gridvec[0:,0:,2:],np.zeros([NRgrid,Nrgrid,2])),axis=2) +1/5*np.concatenate((np.zeros([NRgrid,Nrgrid,2]),gridvec[0:,0:,0:Nphigrid-2]),axis = 2) + 4/105*np.concatenate((gridvec[0:,0:,3:],np.zeros([NRgrid,Nrgrid,3])),axis=2) -4/105*np.concatenate((np.zeros([NRgrid,Nrgrid,3]),gridvec[0:,0:,0:Nphigrid-3]),axis = 2)-1/280*np.concatenate((gridvec[0:,0:,4:],np.zeros([NRgrid,Nrgrid,4])),axis=2) +1/280*np.concatenate((np.zeros([NRgrid,Nrgrid,4]),gridvec[0:,0:,0:Nphigrid-4]),axis = 2)
    # derphi2 = 4/5*np.concatenate((dummyvect[0:,0:,1:],np.zeros([NRgrid,Nrgrid,1])),axis=2) -4/5*np.concatenate((np.zeros([NRgrid,Nrgrid,1]),dummyvect[0:,0:,0:Nphigrid-1]),axis = 2)- 1/5*np.concatenate((dummyvect[0:,0:,2:],np.zeros([NRgrid,Nrgrid,2])),axis=2) +1/5*np.concatenate((np.zeros([NRgrid,Nrgrid,2]),dummyvect[0:,0:,0:Nphigrid-2]),axis = 2) + 4/105*np.concatenate((dummyvect[0:,0:,3:],np.zeros([NRgrid,Nrgrid,3])),axis=2) -4/105*np.concatenate((np.zeros([NRgrid,Nrgrid,3]),dummyvect[0:,0:,0:Nphigrid-3]),axis = 2)-1/280*np.concatenate((dummyvect[0:,0:,4:],np.zeros([NRgrid,Nrgrid,4])),axis=2) +1/280*np.concatenate((np.zeros([NRgrid,Nrgrid,4]),dummyvect[0:,0:,0:Nphigrid-4]),axis = 2)
    # # Less accurate stencils if desired 
    ddr = -5/2*gridvec + 4/3*np.concatenate((gridvec[0:,1:,0:],np.zeros([NRgrid,1,Nphigrid])),axis=1) +4/3*np.concatenate((np.zeros([NRgrid,1,Nphigrid]),gridvec[0:,:-1,0:]),axis = 1)- 1/12*np.concatenate((gridvec[0:,2:,0:],np.zeros([NRgrid,2,Nphigrid])),axis=1) -1/12*np.concatenate((np.zeros([NRgrid,2,Nphigrid]),gridvec[0:,:-2,0:]),axis = 1)
    # ddR = -5/2*gridvec + 4/3*np.concatenate((gridvec[1:,0:,0:],np.zeros([1,Nrgrid,Nphigrid])),axis=0) +4/3*np.concatenate((np.zeros([1,Nrgrid,Nphigrid]),gridvec[:-1,0:,0:]),axis = 0)- 1/12*np.concatenate((gridvec[2:,0:,0:],np.zeros([2,Nrgrid,Nphigrid])),axis=0) -1/12*np.concatenate((np.zeros([2,Nrgrid,Nphigrid]),gridvec[:-2,0:,0:]),axis = 0)
    ddphi = -5/2*dummyvect + 4/3*np.concatenate((dummyvect[0:,0:,1:],np.zeros([NRgrid,Nrgrid,1])),axis=2) +4/3*np.concatenate((np.zeros([NRgrid,Nrgrid,1]),dummyvect[0:,0:,0:Nphigrid-1]),axis = 2)- 1/12*np.concatenate((dummyvect[0:,0:,2:],np.zeros([NRgrid,Nrgrid,2])),axis=2) -1/12*np.concatenate((np.zeros([NRgrid,Nrgrid,2]),dummyvect[0:,0:,0:Nphigrid-2]),axis = 2)
    derphi1 = 2/3*np.concatenate((gridvec[0:,0:,1:],np.zeros([NRgrid,Nrgrid,1])),axis=2) -2/3*np.concatenate((np.zeros([NRgrid,Nrgrid,1]),gridvec[0:,0:,0:Nphigrid-1]),axis = 2)- 1/12*np.concatenate((gridvec[0:,0:,2:],np.zeros([NRgrid,Nrgrid,2])),axis=2) +1/12*np.concatenate((np.zeros([NRgrid,Nrgrid,2]),gridvec[0:,0:,0:Nphigrid-2]),axis = 2)
    derphi2 = 2/3*np.concatenate((dummyvect[0:,0:,1:],np.zeros([NRgrid,Nrgrid,1])),axis=2) -2/3*np.concatenate((np.zeros([NRgrid,Nrgrid,1]),dummyvect[0:,0:,0:Nphigrid-1]),axis = 2)- 1/12*np.concatenate((dummyvect[0:,0:,2:],np.zeros([NRgrid,Nrgrid,2])),axis=2) +1/12*np.concatenate((np.zeros([NRgrid,Nrgrid,2]),dummyvect[0:,0:,0:Nphigrid-2]),axis = 2)
    ddphiterm = -ddphi/dphi**2/2/m
    ddrterm = -ddr/dr**2/2/m
    ddRterm = -ddR/dR**2/2/M
    derphi1term = 1j*B/2/m*derphi1/dphi
    derphi2term = -1j*thst/m*derphi2/dphi
    derivterms = ddrterm+ddRterm+ddphiterm+derphi1term+derphi2term
    derivterms = derivterms.reshape(Ntot)
    diagterms = (thetaterms+H2list)*vec
    myvec = diagterms+derivterms
    return myvec
start = time.time()
aop = lambda x: myham(x)
exvals,exvecs = lib.davidson(aop,guesslist,myexdiag,nroots = manifold)
end = time.time()
extrans = exvals[1]-exvals[0]
print('Energies:'+ str(exvals[1])+' , '+str(exvals[0])+' Trans = ' + str(extrans) +', Walltime = '+str(end-start))
exgs = exvecs[0]
rmesh = elr.reshape(Ntot)
rexpectexgs = exgs.conj().T@(rmesh*exgs)
Rmesh = elR.reshape(Ntot)
Rexpectexgs = exgs.conj().T@(Rmesh*exgs)
print('<r> = '+ str(rexpectexgs) + '<R> = ' + str(Rexpectexgs))
print('The error = '+str(extrans-BOtrans))