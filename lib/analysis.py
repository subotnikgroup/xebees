import xp
from hamiltonian import KE_ColbertMiller_zero_inf

def get_wfc_proj(evecs, H):
    symb = ['s','p','d'] + [chr(c) for c in range(ord('f'), ord('z')+1)]
    J = H.J

    ev = evecs.reshape((-1,) + H.shape)
    prj = xp.sum(ev**2, axis=(1,2,3))

    d = prj[: ,J:].copy() # 0..J
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

    return p01_z, p01_r, P01_R

