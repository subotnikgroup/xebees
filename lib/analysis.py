import xp

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
    el2 = xp.einsum('BRrjsO, js, BRrjsO -> B', evecs, ll1, evecs, optimize=True)*dR*dr
    
    ## j2 = (kappa^2 - 1/4)
    jj1 = (kappa**2-0.25)
    ej2 = xp.einsum('BRrjsO, js, BRrjsO -> B', evecs, jj1, evecs, optimize=True)*dR*dr

    ## l_z built in recursion relation
    vlz = H.apply_Lz(evecs)
    elz = xp.einsum('BRrjsO, BRrjso -> B', evecs, vlz, optimize=True)*dR*dr

    ## s_z built in recusion relation
    vsz = H.apply_Sz(evecs)
    esz = xp.einsum('BRrjsO, BRrjso -> B', evecs, vsz, optimize=True)*dR*dr

    ## j_z = Om
    ejz = xp.einsum('BRrjsO, O, BRrjsO -> B', evecs, H.Om, evecs)*dR*dr

    return el2, ej2, elz, ejz, esz

