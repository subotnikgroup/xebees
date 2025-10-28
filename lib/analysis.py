import xp

def get_wfc_proj(evecs, H):
    char = ['s','p','d','f','g']
    J = H.J
    if len(evecs.shape)==1:
        proj = xp.sum((evecs.reshape(H.shape))**2, axis=(0,1,2))
        flat = xp.asarray([xp.round(proj[i]+proj[-i-1],1) for i in range(J+1)])
        return char[int(xp.argmax(xp.flip(flat)))],proj
    else:
        chars = []
        projs = []
        for e,w in enumerate(evecs):
            proj = xp.sum((w.reshape(H.shape))**2, axis=(0,1,2))
            flat = xp.asarray([xp.round(proj[i]+proj[-i-1],1) for i in range(J+1)])
            projs.append(proj)
            chars.append(char[int(xp.argmax(xp.flip(flat)))])
        return chars, xp.array(projs)
        