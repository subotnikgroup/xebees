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
    J = int(H.J-0.5)

    ev = evecs.reshape((-1,) + H.shape)
    prj = xp.sum(ev**2, axis=(1,2,3,4))
    if J==0:
        symbols = ['s' for _,_ in enumerate(evecs)]
        return symbols, prj
    
    d = prj[: ,J:].copy() # 0..J
    #d[:, 1:] += prj[:, :J][:, ::-1]  # -J..-1 with flip along 2nd axis (J) # not supported by cupynumeric
    d[:, 1:] += xp.fliplr(prj[:, :J]) # -J..-1 with flip along 2nd axis (J)

    symbols = list(map(lambda x: symb[int(xp.argmax(x))], d))
    return symbols, prj


