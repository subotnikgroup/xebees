import matplotlib.pyplot as plt
import matplotlib.animation
import numpy
import xp
from matplotlib.widgets import Slider
import matplotlib as mpl

def fromgpu(tensor):
    return tensor.get() if hasattr(tensor, 'get') else tensor

def plot_psi3D_fixedPsi(wfc, H, levels=None, psi=0, Ngamma=150, scale='linear', save=None, figsize=(5,6)):
    '''plot polar slice of exact wfc, shape (NR,Nr,Nj,NOm) as a function of R for fixed psi'''
    from scipy.special import sph_harm_y
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize)
    # for the potential plotting
    g     = fromgpu(H.g)
    r_lab = fromgpu(H.r_lab)
    Vgrid = fromgpu(H.Vgrid)

    if len(wfc.shape)==1:
        assert wfc.size%H.size==0
        #assert number block is integer number of wfcs or H mismatch has occured
        wfc = xp.reshape(wfc,H.shape)
    

    wfc = wfc[:,:,:,:]/fromgpu(H.r[None,:,None,None]) # rescale the radial wfc for 3D
    wfc = wfc[:,:,:,:]/fromgpu(H.R[:,None,None,None])
    
    # rotate out of the sph harm basis to sph grid
    j = fromgpu(H.j)
    Om = fromgpu(H.Om)
    theta = xp.linspace(0, xp.pi, Ngamma, endpoint=True)
    psi = xp.array([psi,psi+xp.pi])
    Yj, Yo, Yt, Yp = xp.meshgrid(j,Om,theta,psi)
    Yjo = sph_harm_y(Yj,Yo,Yt,Yp)
    wfc_sph = xp.einsum('ojtp,Rrjo->Rrtp', Yjo,wfc)

    # plot potential on grid of (r, gamma)
    if levels is None:
        if H.args.potential=='erf_coulomb':
            levels = xp.linspace(xp.max(Vgrid[:,:,:])-2,
                                xp.max(Vgrid[:,:,:]), 7) # 2 a.u. ~50 eV range
        if H.args.potential=='borgis':
            levels = xp.linspace(xp.min(Vgrid[:,:,:]),
                                xp.min(Vgrid[:,:,:])+0.5, 16) # 0.15 a.u. ~4 eV range
        else:
            levels = xp.linspace(xp.min(Vgrid[:,:,:]), xp.max(Vgrid[:,:,:]), 7)
    
    print("contour levels", levels)
    ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels, cmap='binary_r')
    ax.contour(*xp.meshgrid(g*-1, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels, cmap='binary_r')
    cs = ax.contourf(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels, cmap='binary_r') # dummy contour with filled patches
    ax.set_title("Plotting polar slices of the sphere")

    
    if scale == 'linear':
        cmap = 'seismic'
        limit = xp.max(xp.abs(wfc_sph.real))
        toplimit = limit
        lowlimit = -limit
    elif scale == 'log':
        cmap = 'Blues'
        limit = xp.log10(xp.max(xp.abs(wfc_sph)**2))
        toplimit = limit
        lowlimit = limit - 6  # 6 orders of magnitude
    else:
        raise RuntimeError(f"scale must be either linear or log, not `{scale}`!")

    print("wfc limits", lowlimit, toplimit)

    if scale=='linear':
        pc1 = ax.pcolormesh(theta, r_lab, wfc_sph[0,:,:,0].real,
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        pc2 = ax.pcolormesh(theta*-1, r_lab, wfc_sph[0,:,:,1].real,
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        cbar2 = fig.colorbar(pc1, orientation='vertical', fraction=0.1)
        cbar2.set_label("wfc heatmap: amplitude ")
    elif scale=='log':
        pc1 = ax.pcolormesh(theta, r_lab, xp.log10(xp.abs(wfc_sph[0,:,:,0])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        pc2 = ax.pcolormesh(theta*-1, r_lab, xp.log10(xp.abs(wfc_sph[0,:,:,1])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        cbar2 = fig.colorbar(pc1, orientation='vertical', fraction=0.1)
        cbar2.set_label("wfc heatmap: Log[density] ")
    fig.tight_layout()

    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("potential contours: E / a.u.")
    formatter = mpl.ticker.FormatStrFormatter('%.1f')
    cbar.ax.xaxis.set_major_formatter(formatter)

    def update_R(val):
        ax.cla()
        ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[val,:,:].T, levels=levels, cmap='binary_r')
        ax.contour(*xp.meshgrid(g*-1, r_lab, indexing='ij'), Vgrid[val,:,:].T, levels=levels, cmap='binary_r')

        if scale=='linear':
            pc1 = ax.pcolormesh(theta, r_lab, wfc_sph[val,:,:,0].real,
                          cmap=cmap, edgecolor='face',
                          antialiased=True,
                          vmin=lowlimit, vmax=toplimit, alpha=0.5)
            pc2 = ax.pcolormesh(theta*-1, r_lab, wfc_sph[val,:,:,1].real,
                          cmap=cmap, edgecolor='face',
                          antialiased=True,
                          vmin=lowlimit, vmax=toplimit, alpha=0.5)
        elif scale=='log':
             pc1 = ax.pcolormesh(theta, r_lab, xp.log10(xp.abs(wfc_sph[val,:,:,0])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
             pc2 = ax.pcolormesh(theta*-1, r_lab, xp.log10(xp.abs(wfc_sph[val,:,:,1])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        ax.text(xp.pi/2,xp.max(r_lab)*.75,r"$\psi$={:.2f}$\pi$".format(psi[0]/xp.pi), ha='center')
        ax.text(xp.pi/2,xp.max(r_lab)*.85,r"R={:.2f}$a_0$".format(H.R_lab[val]), ha='center')
    
    ani = mpl.animation.FuncAnimation(fig, update_R, frames=H.R_lab.size)
    
    if save!= None:
        writer = mpl.animation.PillowWriter(fps=10, metadata=dict(artist='xebees 3D code'))
        ani.save(save+'.gif', writer=writer)
    return ani

def plot_psi3D_BO(wfc, H, levels=None, iR=20, Ngamma=150, Npsi=10, scale='linear', save=None, figsize=(5,6)):
    ''' plot a BO wfc, wfc slice, shape (Nr,Nj,NOm), as function of psi'''
    from scipy.special import sph_harm_y
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize)
    # for the potential plotting
    g     = fromgpu(H.g)
    r_lab = fromgpu(H.r_lab)
    Vgrid = fromgpu(H.Vgrid)

    if len(wfc.shape)==1:
        assert wfc.size%H.size==0
        #assert number block is integer number of wfcs or H mismatch has occured
        wfc = xp.reshape(wfc,H.shape)[iR,:,:,:]
    elif wfc.shape==H.shape:
        wfc = wfc[iR,:,:,:]

    wfc = wfc[:,:,:]/fromgpu(H.r[:,None,None]) # rescale the radial wfc for 3D

    # rotate out of the sph harm basis to sph grid
    j = fromgpu(H.j)
    Om = fromgpu(H.Om)
    theta = xp.linspace(0, xp.pi, Ngamma, endpoint=True)
    phi = xp.linspace(0, 2 * xp.pi, Npsi)
    Yj, Yo, Yt, Yp = xp.meshgrid(j,Om,theta,phi)
    Yjo = sph_harm_y(Yj,Yo,Yt,Yp)
    wfc_sph = xp.einsum('ojtp,rjo->rtp', Yjo,wfc)

    # plot potential on grid of (r, gamma)
    if levels is None:
        if H.args.potential=='erf_coulomb':
            levels = xp.linspace(xp.max(Vgrid[iR,:,:])-2,
                                xp.max(Vgrid[iR,:,:]), 7) # 2 a.u. ~50 eV range
        if H.args.potential=='borgis':
            levels = xp.linspace(xp.min(Vgrid[iR,:,:]),
                                xp.min(Vgrid[iR,:,:])+0.5, 16) # 0.15 a.u. ~4 eV range
        else:
            levels = xp.linspace(xp.min(Vgrid[iR,:,:]), xp.max(Vgrid[iR,:,:]), 7)
    
    print("contour levels", levels)
    ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[iR,:,:].T, levels=levels, cmap='binary_r')
    ax.contour(*xp.meshgrid(g*-1, r_lab, indexing='ij'), Vgrid[iR,:,:].T, levels=levels, cmap='binary_r')
    cs = ax.contourf(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[iR,:,:].T, levels=levels, cmap='binary_r') # dummy contour with filled patches
    ax.set_title("Plotting polar slices of the sphere")

    
    if scale == 'linear':
        cmap = 'seismic'
        limit = xp.max(xp.abs(wfc_sph.real))
        toplimit = limit
        lowlimit = -limit
    elif scale == 'log':
        cmap = 'Blues'
        limit = xp.log10(xp.max(xp.abs(wfc_sph)))
        toplimit = limit
        lowlimit = limit - 6  # 6 orders of magnitude
    else:
        raise RuntimeError(f"scale must be either linear or log, not `{scale}`!")

    print("wfc limits", lowlimit, toplimit)

    if scale=='linear':
        pc1 = ax.pcolormesh(theta, r_lab, wfc_sph[:,:,0].real,
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        pc2 = ax.pcolormesh(theta*-1, r_lab, wfc_sph[:,:,Npsi//2].real,
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        cbar2 = fig.colorbar(pc1, orientation='vertical', fraction=0.1)
        cbar2.set_label("wfc heatmap: amplitude ")
    elif scale=='log':
        pc1 = ax.pcolormesh(theta, r_lab, xp.log10(xp.abs(wfc_sph[:,:,0])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        pc2 = ax.pcolormesh(theta*-1, r_lab, xp.log10(xp.abs(wfc_sph[:,:,Npsi//2])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        cbar2 = fig.colorbar(pc1, orientation='vertical', fraction=0.1)
        cbar2.set_label("wfc heatmap: Log[density] ")
    fig.tight_layout()

    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("potential contours: E / a.u.")
    formatter = mpl.ticker.FormatStrFormatter('%.1f')
    cbar.ax.xaxis.set_major_formatter(formatter)

    def update_psi(val):
        ax.cla()
        ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[iR,:,:].T, levels=levels, cmap='binary_r')
        ax.contour(*xp.meshgrid(g*-1, r_lab, indexing='ij'), Vgrid[iR,:,:].T, levels=levels, cmap='binary_r')

        if scale=='linear':
            pc1 = ax.pcolormesh(theta, r_lab, wfc_sph[:,:,0+val].real,
                          cmap=cmap, edgecolor='face',
                          antialiased=True,
                          vmin=lowlimit, vmax=toplimit, alpha=0.5)
            pc2 = ax.pcolormesh(theta*-1, r_lab, wfc_sph[:,:,Npsi//2+val].real,
                          cmap=cmap, edgecolor='face',
                          antialiased=True,
                          vmin=lowlimit, vmax=toplimit, alpha=0.5)
        elif scale=='log':
             pc1 = ax.pcolormesh(theta, r_lab, xp.log10(xp.abs(wfc_sph[:,:,0])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
             pc2 = ax.pcolormesh(theta*-1, r_lab, xp.log10(xp.abs(wfc_sph[:,:,Npsi//2])**2),
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        ax.text(xp.pi/2,xp.max(r_lab)*.75,r"$\psi$={:.2f}$\pi$".format(phi[val]/xp.pi), ha='center')
        ax.text(xp.pi/2,xp.max(r_lab)*.85,r"$\psi$={:.2f}$\pi$".format(H.R_lab[iR]), ha='center')

    ani = mpl.animation.FuncAnimation(fig, update_psi, frames=phi.size//2)
    
    if save!= None:
        writer = mpl.animation.PillowWriter(fps=10, metadata=dict(artist='xebees 3D code'))
        ani.save(save+'.gif', writer=writer)
    return ani

def plot_psi3D_fixedPsi_multi(wfcs, H, levels=None, psi=0, Ngamma=150, scale='linear', imag=False, save=None, figsize=(9,9)):
    '''plot polar slice of exact wfc, shape (NR,Nr,Nj,NOm) as a function of R for fixed psi'''
    from scipy.special import sph_harm_y
    # for the potential plotting
    g     = fromgpu(H.g)
    r_lab = fromgpu(H.r_lab)
    Vgrid = fromgpu(H.Vgrid)

    if len(wfcs.shape)==2:
        nwfc = wfcs.shape[0]
        wfcs = xp.reshape(wfcs, (nwfc,)+H.shape)
    elif len(wfcs.shape)==5:
        nwfc = wfcs.shape[0]
        assert wfcs.shape[1:]==H.shape
    elif len(wfcs.shape)==1:
        assert wfcs.size%H.size==0
        #assert number block is integer number of wfcs or H mismatch has occured
        nwfc = wfcs.size//H.size
        wfcs = xp.reshape(wfcs, (nwfc,)+H.shape)
    else:
        raise RuntimeError("unable to determine how many wfcs or wfcs correct shape!")
    w = int(numpy.ceil(numpy.sqrt(nwfc)))
    h = int(numpy.ceil(nwfc/w))

    fig, axs = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize, nrows=h, ncols=w)
    axs = axs.flatten()
    fig.subplots_adjust(hspace=0.3)
    
    wfcs = wfcs[:,:,:,:,:]/fromgpu(H.r[None,None,:,None,None]) # rescale the radial wfc for 3D
    wfcs = wfcs[:,:,:,:,:]/fromgpu(H.R[None,:,None,None,None])
    
    # rotate out of the sph harm basis to sph grid
    j = fromgpu(H.j)
    Om = fromgpu(H.Om)
    theta = xp.linspace(0, xp.pi, Ngamma, endpoint=True)
    psi = xp.array([psi,psi+xp.pi])
    # phi = xp.linspace(0, 2 * xp.pi, Npsi)
    Yj, Yo, Yt, Yp = xp.meshgrid(j,Om,theta,psi)
    Yjo = sph_harm_y(Yj,Yo,Yt,Yp)
    wfcs_sph = xp.einsum('ojtp,nRrjo->nRrtp', Yjo,wfcs)

    # plot potential on grid of (r, gamma)
    if levels is None:
        if H.args.potential=='erf_coulomb':
            levels = xp.linspace(xp.max(Vgrid[:,:,:])-2,
                                xp.max(Vgrid[:,:,:]), 7) # 2 a.u. ~50 eV range
        if H.args.potential=='borgis':
            levels = xp.linspace(xp.min(Vgrid[:,:,:]),
                                xp.min(Vgrid[:,:,:])+0.5, 16) # 0.15 a.u. ~4 eV range
        else:
            levels = xp.linspace(xp.min(Vgrid[:,:,:]), xp.max(Vgrid[:,:,:]), 7)
    
    print("contour levels", levels)
    for state, (wfc, ax) in enumerate(zip(wfcs, axs)):
        ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels, cmap='binary_r')
        ax.contour(*xp.meshgrid(g*-1, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels, cmap='binary_r')
    
    cs = ax.contourf(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels, cmap='binary_r') # dummy contour with filled patches
    cbar = fig.colorbar(cs, ax=axs, orientation='horizontal', fraction=0.04)
    cs.remove()
    cbar.set_label("potential contours: E / a.u.", fontsize=18)
    formatter = mpl.ticker.FormatStrFormatter('%.1f')
    cbar.ax.xaxis.set_major_formatter(formatter)

    #plot real or imag part
    plotted = xp.copy(wfcs_sph.real)
    if imag: plotted = xp.copy(wfcs_sph.imag)
    if scale == 'log': plotted = xp.log10(xp.abs(wfcs_sph)**2)
        
    print("plotted shape",plotted.shape)
    
    if scale == 'linear':
        cmap = 'seismic'
        limit = xp.max(xp.abs(plotted))
        # limit = 1e-4
        toplimit = limit
        lowlimit = -limit
    elif scale == 'log':
        cmap = 'Blues'
        limit = xp.log10(xp.max(xp.abs(wfcs_sph)))
        toplimit = limit
        lowlimit = limit - 10  # 4 orders of magnitude
    else:
        raise RuntimeError(f"scale must be either linear or log, not `{scale}`!")
    
    if imag and limit < 1e-8:
        raise RuntimeError(f"trying to plot imag part of real wfc!")
        
    print("wfc limits", lowlimit, toplimit)
    
    for state, (wfc, ax) in enumerate(zip(plotted, axs)):
        pc1 = ax.pcolormesh(theta, r_lab, wfc[0,:,:,0],
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
        pc2 = ax.pcolormesh(theta*-1, r_lab, wfc[0,:,:,1],
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
    cbar2 = fig.colorbar(pc1, ax=axs, orientation='vertical', fraction=0.1)
    
    if scale=='linear':
        cbar2.set_label("wfc heatmap: amplitude ", fontsize=18)
    elif scale=='log':
        cbar2.set_label("wfc heatmap: Log10[density]", fontsize=18)
    
    def update_R(val):
        for state, (wfc, ax) in enumerate(zip(plotted, axs)):
            ax.cla()
            ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[val,:,:].T, levels=levels, cmap='binary_r')
            ax.contour(*xp.meshgrid(g*-1, r_lab, indexing='ij'), Vgrid[val,:,:].T, levels=levels, cmap='binary_r')
            ax.set_title("State {}".format(state), fontsize=14)
             
            pc1 = ax.pcolormesh(theta, r_lab, wfc[val,:,:,0],
                          cmap=cmap, edgecolor='face',
                          antialiased=True,
                          vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
            pc2 = ax.pcolormesh(theta*-1, r_lab, wfc[val,:,:,1],
                          cmap=cmap, edgecolor='face',
                          antialiased=True,
                          vmin=lowlimit, vmax=toplimit, alpha=0.5)
        fig.suptitle(r"R={:.2f}$a_0$, $\psi$={:.2f}$\pi$".format(H.R_lab[val], psi[0]/xp.pi), fontsize=18)

    ani = mpl.animation.FuncAnimation(fig, update_R, frames=H.R_lab.size)
    
    if save!= None:
        writer = mpl.animation.PillowWriter(fps=10, metadata=dict(artist='xebees 3D code'))
        ani.save(save+'.gif', writer=writer)
    return ani

# use like IPython.display(plotpotential2D(H)))
def plotpotential2D(H, levels=None):
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    g     = fromgpu(H.g)
    r_lab = fromgpu(H.r_lab)
    Vgrid = fromgpu(H.Vgrid)

    if levels is None:
        levels = numpy.linspace(xp.min(Vgrid),
                                xp.min(Vgrid) + 0.15, 16) # 0.15 a.u. ~4 eV range

    ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels)

    cs = ax.contourf(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels) # dummy contour with filled patches
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("E / a.u.")

    # wavefunctions as many polar slices at R
    def animate(t):
        ax.cla()
        ax.contour(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[t,:,:].T, levels=levels)
        ax.text(numpy.pi/2,numpy.max(r_lab)*.75,f"R={H.R_lab[t]:0.03}a₀", ha='center')

        ax.grid(axis='x')
        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀"])

    return  matplotlib.animation.FuncAnimation(fig, animate, frames=H.shape[0])


def plotpotential2D_ps(H, levels=None):
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    g     = fromgpu(H.g)
    r_lab = fromgpu(H.r_lab)
    Vgrid = fromgpu(H.Vgrid)

    if levels is None:
        levels = numpy.linspace(xp.min(Vgrid),
                                xp.min(Vgrid) + 0.15, 16) # 0.15 a.u. ~4 eV range

    ax.contour(*xp.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels)

    cs = ax.contourf(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels) # dummy contour with filled patches
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("E / a.u.")

    # wavefunctions as many polar slices at R
    def animate(t):
        ax.cla()
        ax.contour(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[t,:,:].T, levels=levels)
        ax.text(numpy.pi/2,numpy.max(r_lab)*.75,f"R={H.R_lab[t]:0.03}a₀", ha='center')

        ax.grid(axis='x')
        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀"])

    return  matplotlib.animation.FuncAnimation(fig, animate, frames=H.shape[0])


def plot3D_test1(H, Ng,Np,levels=None):
    fig, ax = plt.subplots(1,subplot_kw=dict(projection='polar'))
     
    g     = fromgpu(H.g)
    p     = fromgpu(H.p)
    r     = fromgpu(H.r)
    R     = fromgpu(H.R)
    Vgrid = fromgpu(H.Vgrid)
    #plt.close(fig)
    #g = xp.linspace(0, 2*xp.pi, Ng, endpoint=False)
    #p = xp.linspace(0, 2*xp.pi, Np, endpoint=False)
    
    Vdisplay = Vgrid[0,:,0,:].T

    if levels is None:
        levels = numpy.linspace(xp.min(Vgrid),
                                xp.min(Vgrid) + 0.15, 16) # 0.15 a.u. ~4 eV range

    print("p",p.shape)
    print("g",g.shape)
    print("V",Vdisplay.shape)
    ax.contour(*xp.meshgrid(p, r, indexing='ij'), Vdisplay, levels=levels)
    
    cs = ax.contourf(*numpy.meshgrid(p, r, indexing='ij'), Vdisplay, levels=levels) # dummy contour with filled patches
    #print("here?")
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("E / a.u.")

    ax_R = plt.axes([0.2, 0.20, 0.65, 0.03])
    #ax_p = plt.axes([0.2, 0.175, 0.65, 0.03])
    ax_g = plt.axes([0.2, 0.15, 0.65, 0.03])

    slider_R = Slider(ax_R,"R",0,H.shape[0]-1,valstep=1)
    slider_g = Slider(ax_g,"g",0,H.shape[3]-1,valstep=1)

    # wavefunctions as many polar slices at R
    def update_R(val):
        idx_R = int(slider_R.val)
        idx_g = int(slider_g.val)

        ax.cla()
        Vdisplay = Vgrid[idx_R,:,:,idx_g].T
        contour = ax.contour(*numpy.meshgrid(p, r, indexing='ij'), Vdisplay, levels=levels)        
        ax.grid(axis='x')
        slider_R.valtext.set_text(f"{R[idx_R]:.3f}")
        
        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        limit=ax.get_yticks()[-1]
        ax.set_yticks([limit], [f"r={limit}a₀,γ fix"])
        
        fig.canvas.draw_idle()
        
    def update_g(val):
        idx_R = int(slider_R.val)
        idx_g = int(slider_g.val)

        ax.cla()
        Vdisplay = Vgrid[idx_R,:,:,idx_g].T
        contour = ax.contour(*numpy.meshgrid(p, r, indexing='ij'), Vdisplay, levels=levels)        
        ax.grid(axis='x')
        slider_g.valtext.set_text(f"{g[idx_g]/xp.pi:.3f} π")

        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀,ψ fix"])
        fig.canvas.draw_idle()

    slider_R.on_changed(update_R)
    slider_g.on_changed(update_g)
    plt.show()
 
        #return matplotlib.animation.FuncAnimation(fig, animate, frames=H.shape[0])



def plotpsi2D(psi, H, levels=None, scale='linear'):
    g     = fromgpu(H.g)
    r_lab = fromgpu(H.r_lab)
    R_lab = fromgpu(H.R_lab)
    Vgrid = fromgpu(H.Vgrid)

    if levels is None:
        levels = numpy.linspace(numpy.min(Vgrid),
                                numpy.min(Vgrid) + 0.15, 16) # 0.15 a.u. ~4 eV range
        levels = levels[-2:]
    else:
        levels = fromgpu(levels)

    psi = numpy.copy(
        fromgpu(psi.reshape(H.shape)) / fromgpu(xp.sqrt(H.R_grid*H.r_grid))
    )

    if numpy.iscomplexobj(psi) or scale == 'log':
        psi = numpy.abs(psi*psi.conj())

    if scale == 'log':
        psi = numpy.log10(psi)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.contour(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels)
    cs = ax.contourf(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[0,:,:].T, levels=levels) # dummy contour with filled patches
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("E / a.u.")

    if scale == 'linear':
        cmap = 'seismic'
        limit = numpy.max(numpy.abs(psi))
        toplimit = limit
        lowlimit = -limit
    elif scale == 'log':
        cmap = 'Blues'
        limit = numpy.max(psi)
        toplimit = limit
        lowlimit = limit - 6  # 6 orders of magnitude
    else:
        raise RuntimeError(f"scale must be either linear or log, not `{scale}`!")

    # wavefunctions as many polar slices at R
    def animate(t):
        ax.cla()
        ax.contour(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[t,:,:].T, levels=levels)
        ax.pcolormesh(g, r_lab, psi[t,:,:],
                      cmap=cmap, edgecolor='face',
                      antialiased=True,
                      vmin=lowlimit, vmax=toplimit)#, shading='gouraud')

        ax.text(numpy.pi/2,numpy.max(r_lab)*.75,f"R={R_lab[t]:0.03}a₀", ha='center')

        ax.grid(axis='x')
        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀"])

    return  matplotlib.animation.FuncAnimation(fig, animate, frames=H.shape[0])

def plotpsi2D_multi(psis, H, levels=None, scale='linear'):
    g     = fromgpu(H.g)
    r_lab = fromgpu(H.r_lab)
    R_lab = fromgpu(H.R_lab)
    Vgrid = fromgpu(H.Vgrid)

    if levels is None:
        levels = numpy.linspace(numpy.min(Vgrid),
                                numpy.min(Vgrid) + 0.15, 16) # 0.15 a.u. ~4 eV range
        levels = levels[-2:]
    else:
        levels = fromgpu(levels)

    N = len(psis)
    w = int(numpy.ceil(numpy.sqrt(N)))
    h = int(numpy.ceil(N/w))

    psis = numpy.copy(fromgpu(psis.reshape((N,) + H.shape)) / fromgpu(xp.sqrt(H.R_grid*H.r_grid)))

    
    if numpy.iscomplexobj(psis) or scale == 'log':
        psis = numpy.abs(psis*psis.conj())

    if scale == 'log':
        psis = numpy.log10(psis)


    fig, axs = plt.subplots(subplot_kw=dict(projection='polar'),
                            nrows=h, ncols=w)
    axs = axs.flatten()

    # wavefunctions as many polar slices at R
    def animate(t):
        for state, (psi, ax) in enumerate(zip(psis, axs)):
            ax.cla()
            ax.contour(*numpy.meshgrid(g, r_lab, indexing='ij'), Vgrid[t,:,:].T, levels=levels)
            if scale == 'linear':
                cmap = 'seismic'
                limit = numpy.max(numpy.abs(psi))
                toplimit = limit
                lowlimit = -limit
            elif scale == 'log':
                cmap = 'Blues'
                limit = numpy.max(psi)
                toplimit = limit
                lowlimit = limit - 6  # 6 orders of magnitude

            ax.pcolormesh(g, r_lab, psi[t,:,:], cmap=cmap, edgecolor='face', antialiased=True, vmin=lowlimit, vmax=toplimit)#, shading='gouraud')
            ax.text( numpy.pi/2, numpy.max(r_lab)*.75, f"R={R_lab[t]:0.03}a₀", ha='center')
            ax.text(-numpy.pi/2, numpy.max(r_lab)*.75, f"state {state}", ha='center')

            ax.grid(axis='x')
            locs=ax.get_xticks()
            labels = [f'{th/numpy.pi:.03}π' for th in locs]
            ax.set_xticks(locs, labels)
            [limit]=ax.get_yticks()[-1:]
            ax.set_yticks([limit], [f"r={limit}a₀"])
        for ax in axs[N:]:
            ax.set_axis_off()
        fig.subplots_adjust(hspace=1, wspace=1)

    return  matplotlib.animation.FuncAnimation(fig, animate, frames=H.shape[0])

