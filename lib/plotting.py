import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

# use like IPython.display(plotpotential2D(H)))
def plotpotential2D(H, levels=None):
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    if levels is None:
        levels = np.linspace(np.min(H.Vgrid),
                             np.min(H.Vgrid) + 0.15, 16) # 0.15 a.u. ~4 eV range

    ax.contour(*np.meshgrid(H.g,H.r, indexing='ij'), H.Vgrid[0,:,:].T, levels=levels)
    cs = ax.contourf(*np.meshgrid(H.g,H.r, indexing='ij'), H.Vgrid[0,:,:].T, levels=levels) # dummy contour with filled patches
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("E / a.u.")

    # wavefunctions as many polar slices at R 
    def animate(t):
        ax.cla()
        ax.contour(*np.meshgrid(H.g,H.r, indexing='ij'), H.Vgrid[t,:,:].T, levels=levels)
        ax.text(np.pi/2,6,f"R={H.R[t]:0.03}a₀", ha='center')
    
        ax.grid(axis='x')
        locs=ax.get_xticks()
        labels = [f'{th/np.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀"])
    
    return  matplotlib.animation.FuncAnimation(fig, animate, frames=H.shape[0])


def plotpsi2D(psi, H, levels=None, scale=1):
    if levels is None:
        levels = np.linspace(np.min(H.Vgrid),
                             np.min(H.Vgrid) + 0.15, 16) # 0.15 a.u. ~4 eV range
        levels = levels[-2:]
    
    psi = np.copy(psi).reshape(H.shape) / np.sqrt(H.R_grid*H.r_grid)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    ax.contour(*np.meshgrid(H.g,H.r, indexing='ij'), H.Vgrid[0,:,:].T, levels=levels)
    cs = ax.contourf(*np.meshgrid(H.g,H.r, indexing='ij'), H.Vgrid[0,:,:].T, levels=levels) # dummy contour with filled patches
    cbar = fig.colorbar(cs, orientation='horizontal', fraction=0.05)
    cs.remove()
    cbar.set_label("E / a.u.")

    # wavefunctions as many polar slices at R 
    def animate(t):
        ax.cla()
        ax.contour(*np.meshgrid(H.g,H.r, indexing='ij'), H.Vgrid[t,:,:].T, levels=levels)
        limit = np.max(np.abs(psi))/scale
        ax.pcolormesh(H.g, H.r, psi[t,:,:], cmap='seismic', edgecolor='face', antialiased=True, vmin=-limit, vmax=limit)#, shading='gouraud')
        ax.text(np.pi/2,6,f"R={H.R[t]:0.03}a₀", ha='center')

        #ax.set_rmax(5)

        ax.grid(axis='x')
        locs=ax.get_xticks()
        labels = [f'{th/np.pi:.03}π' for th in locs]
        ax.set_xticks(locs, labels)
        [limit]=ax.get_yticks()[-1:]
        ax.set_yticks([limit], [f"r={limit}a₀"])

    return  matplotlib.animation.FuncAnimation(fig, animate, frames=H.shape[0])


# # plot many slices, but this time just the radial components
# fig, ax = plt.subplots()
# def animate(t):
#     ax.cla()
#     #for c in ax.collections: c.remove
#     ax.contour(*np.meshgrid(H.R, H.r, indexing='ij'), H.Vgrid[:,:,t], levels=levels)
#     ax.set_xlabel("R / a.u.")
#     ax.set_ylabel("r / a.u.")
#     ax.text(4,8,f"γ={H.g[t]/np.pi:0.03}π")

# ani=matplotlib.animation.FuncAnimation(fig, animate, frames=args.Ng)
# display(ani)
