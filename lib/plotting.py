import matplotlib.pyplot as plt
import matplotlib.animation
import numpy
import xp
from matplotlib.widgets import Slider

def fromgpu(tensor):
    return tensor.get() if hasattr(tensor, 'get') else tensor

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
    ax_p = plt.axes([0.2, 0.175, 0.65, 0.03])
    ax_g = plt.axes([0.2, 0.15, 0.65, 0.03])

    slider_R = Slider(ax_R,"R",0,H.shape[0]-1,valstep=1)
    slider_p = Slider(ax_p,"p",0,H.shape[2]-1,valstep=1)
    slider_g = Slider(ax_g,"g",0,H.shape[3]-1,valstep=1)

    # wavefunctions as many polar slices at R
    def update_R(val):
        idx_R = int(slider_R.val)
        idx_p = int(slider_p.val)
        idx_g = int(slider_g.val)

        ax.cla()
        Vdisplay = Vgrid[idx_R,:,:,idx_g].T
        contour = ax.contour(*numpy.meshgrid(g, r, indexing='ij'), Vdisplay, levels=levels)        
        ax.grid(axis='x')
        slider_R.valtext.set_text(f"{R[idx_R]:.3f}")
        
        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        limit=ax.get_yticks()[-1]
        ax.set_yticks([limit], [f"r={limit}a₀,γ fix"])
        
        fig.canvas.draw_idle()

    def update_p(val):
        idx_R = int(slider_R.val)
        idx_p = int(slider_p.val)
        idx_g = int(slider_g.val)

        ax.cla()
        Vdisplay = Vgrid[idx_R,:,idx_p,:].T
        contour = ax.contour(*numpy.meshgrid(p, r, indexing='ij'), Vdisplay, levels=levels)       
        ax.grid(axis='x')
        slider_p.valtext.set_text(f"{p[idx_p]/xp.pi:.3f} π")

        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀,γ fix"])
        fig.canvas.draw_idle()
        
    def update_g(val):
        idx_R = int(slider_R.val)
        idx_p = int(slider_p.val)
        idx_g = int(slider_g.val)

        ax.cla()
        Vdisplay = Vgrid[idx_R,:,:,idx_g].T
        contour = ax.contour(*numpy.meshgrid(g, r, indexing='ij'), Vdisplay, levels=levels)        
        ax.grid(axis='x')
        slider_g.valtext.set_text(f"{g[idx_g]/xp.pi:.3f} π")

        locs=ax.get_xticks()
        labels = [f'{th/numpy.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀,ψ fix"])
        fig.canvas.draw_idle()

    slider_R.on_changed(update_R)
    slider_p.on_changed(update_p)
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

