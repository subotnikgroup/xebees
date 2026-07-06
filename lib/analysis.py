import xp
from hamiltonian import KE_ColbertMiller_zero_inf

def get_wfc_proj(evecs, H):
    symb = ['s','p','d'] + [chr(c) for c in range(ord('f'), ord('z')+1)]
    J = H.J

    ev = evecs.reshape((-1,) + H.shape)
    prj = xp.sum(ev**2, axis=(1,2,3))
    sgn = xp.sign(xp.sum(ev, axis=(1,2,3)))

    d = sgn[:,J:].copy()*prj[: ,J:].copy() # 0..J
    #d[:, 1:] += prj[:, :J][:, ::-1]  # -J..-1 with flip along 2nd axis (J) # not supported by cupynumeric
    d[:, 1:] += xp.fliplr(prj[:, :J]) # -J..-1 with flip along 2nd axis (J)

    symbols = list(map(lambda x: symb[int(xp.argmax(x))], d))
    return symbols, prj

def get_wfc_Om_proj_wS(evecs, H):
    #### symbols are currently meaningless!!!! Need to rethink how we want to describe Om for half int J!
    symb = ['s','p','d'] + [chr(c) for c in range(ord('f'), ord('z')+1)]
    J = int(H.J+0.5)

    ev = evecs.reshape((-1,) + H.shape)
    prj = xp.sum(xp.abs(ev)**2, axis=(1,2,3,4))

    d = prj[: ,J:].copy() # 1/2..J
    d += xp.fliplr(prj[:, :J]) # -J..-1/2 with flip along 2nd axis (J)

    symbols = list(map(lambda x: symb[int(xp.argmax(x))], d))
    return symbols, prj

def get_jls_expectations(evecs, H):
    evecs = evecs.reshape((-1,) + H.shape)
    kappa = H.sg[:None]*(2*H.j[:,None]+1)
    dR = H.R[1]-H.R[0]
    dr = H.r[1]-H.r[0]
    ## l^2 = kappa(kappa+1)
    ll1 = kappa*(kappa+1)
    el2 = xp.einsum('BRrjsO, js, BRrjsO -> B', evecs, ll1, evecs, optimize=True)
    
    ## j2 = (kappa^2 - 1/4)
    jj1 = (kappa**2-0.25)
    ej2 = xp.einsum('BRrjsO, js, BRrjsO -> B', evecs, jj1, evecs, optimize=True)

    ## l_z built in recursion relation
    vlz = H.apply_Lz(evecs)
    elz = xp.einsum('BRrjsO, BRrjso -> B', evecs, vlz, optimize=True)

    ## s_z built in recusion relation
    vsz = H.apply_Sz(evecs)
    esz = xp.einsum('BRrjsO, BRrjso -> B', evecs, vsz, optimize=True)

    ## j_z = Om
    ejz = xp.einsum('BRrjsO, O, BRrjsO -> B', evecs, H.Om, evecs)

    return el2, ej2, elz, ejz, esz

def get_spin_expectations(evecs, H):
    '''Per-state electron-spin expectations <s_x>, <s_y>, <s_z> in the body frame
    (x,y perpendicular to, z along, the internuclear axis), using the operators
    H.s_x, H.s_y (Omega<->Omega+/-1) and H.s_z (diagonal in Omega).

    Equations (l = j + sigma; spinor spherical harmonics
    |j,sigma,Omega> = c_up|l,Omega-1/2>|up> + c_dn|l,Omega+1/2>|dn>):
        <s_a> = sum_{n,n'} c_n* <n| s_a |n'> c_n',   n = (R,r,j,sigma,Omega)
        s_z = (1/2)(s_+ s_- diag),  s_x = (s_+ + s_-)/2,  s_y = (s_+ - s_-)/(2i).
    s_x,s_y flip the spin so they connect Omega' = Omega +/- 1.

    N.B.  For the real eigenvectors of the real, time-reversal-symmetric H,
    <s_y> = 0 identically and <s_x>,<s_z> are basis-dependent within a Kramers
    doublet.  To resolve the spin texture, diagonalize {s_x,s_y,s_z} inside each
    degenerate multiplet (pass the doublet as the columns of `evecs`). '''
    ev = evecs.reshape((-1,) + H.shape)
    vsx = H.apply_Sx(ev).reshape(ev.shape)
    vsy = H.apply_Sy(ev).reshape(ev.shape)
    vsz = H.apply_Sz(ev).reshape(ev.shape)
    esx = xp.einsum('BRrjsO, BRrjsO -> B', xp.conj(ev), vsx, optimize=True)
    esy = xp.einsum('BRrjsO, BRrjsO -> B', xp.conj(ev), vsy, optimize=True)
    esz = xp.einsum('BRrjsO, BRrjsO -> B', xp.conj(ev), vsz, optimize=True)
    return esx.real, esy.real, esz.real

def get_p01_radial(evecs,H):
    dR = H.R_lab[1]-H.R_lab[0]
    dr = H.r_lab[1]-H.r_lab[0]
    # dr = H.r[1]-H.r[0]
    ddr1 = KE_ColbertMiller_zero_inf(H.r.size, dr, order=1, bare=True)
    ddR1 = KE_ColbertMiller_zero_inf(H.R.size, dR, order=1, bare=True)
    wfc0 = evecs[0].reshape(H.shape)
    wfc1 = evecs[1].reshape(H.shape)
    wfc2 = evecs[2].reshape(H.shape)

    if len(H.shape)==3: #identify a 2D wfc 
        dg = H.g[1]-H.g[0]
        p01_r = -1j*xp.einsum('Rrg, rv, Rvg -> ', xp.conj(wfc0), ddr1-xp.diag(1/H.r/2), wfc1)
        p01_z = -1j*xp.einsum('Rrg, rv,g, Rvg -> ', xp.conj(wfc0), ddr1-xp.diag(1/H.r/2), xp.cos(H.g), wfc1)*dg
        P01_R = -1j*xp.einsum('Rrg, RV, Vrg ->', xp.conj(wfc0), ddR1, wfc1)

    elif len(H.shape)==4: #identify a 3D wfc without spin
        dg =  H.g[1]-H.g[0]
        p01_r = -1j*xp.einsum('RrjO, rv, RvjO ->', wfc0, ddr1- xp.diag(1/H.r), wfc1)
        p01_z = -1j*xp.einsum('RrjO, g, Ojkg, rv, RvkO->', 
                          wfc0, xp.sin(H.g)*xp.cos(H.g), H.Pjk, ddr1-xp.diag(1/H.r), wfc1, optimize=True)*dg
        P01_R = -1j*xp.einsum('RrjO, RV, VrjO ->', wfc0, ddR1- xp.diag(1/H.R), wfc1)
        
    elif len(H.shape)==5: #identify a 3D wfc with spin
        # note the 2nd excited state is the first vibration we want to check against
        p01_r = -1j*xp.einsum('RrjsO, rv, RvjsO ->', wfc0, ddr1-xp.diag(1/H.r), wfc2)
        p01_z = 0 ### not implemented yet
        P01_R = -1j*xp.einsum('RrjsO, RV, VrjsO ->', wfc0, ddR1- xp.diag(1/H.R), wfc2)

    return p01_z, p01_r, P01_R

