def V_2Dfcm(self, R_amu, r_amu, gamma):
      R = R_amu / ANGSTROM_TO_BOHR
      r = r_amu / ANGSTROM_TO_BOHR

      D, d, a, c = 60, 0.95, 2.52, 1
      A, B, C = 2.32e5, 3.15, 2.31e4


      mu12 = self.M_1*self.M_2/(self.M_1+self.M_2)
      aa = np.sqrt(self.mu/mu12) # factor of 'a' for lab and scaled coordinates

      kappa2 = r*R*np.cos(gamma)
      r1e = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/self.M_1)**2 - 2*kappa2*mu12/self.M_1)
      re2 = np.sqrt((aa*r)**2 + (R/aa)**2*(mu12/self.M_2)**2 + 2*kappa2*mu12/self.M_2)

      D2 = self.g_2 * D * (    np.exp(-2*a * (re2-d))
                               - 2*np.exp(  -a * (re2-d))
                               + 1)
      D1 = self.g_1 * D * c**2 * (    np.exp(-(2*a/c) * (r1e-d))
                                      - 2*np.exp(-(  a/c) * (r1e-d)))
      dv = 1
      G
      V1 = -G*self.g_1/(r1e**2 + dv**2)**0.5
      V2 = -G*self.g_2/(re2**2 + dv**2)**0.5
      VN = G*self.g_1*self.g_2/((R/aa)**2 + dv**2)**0.5

      # return KCALMOLE_TO_HARTREE * (D1 + D2 + A*np.exp(-B*R/aa) - C/(R/aa)**6)
      return (V1+V2+VN) #G*np.exp(-2*B*R/aa))
