import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import numpy as np
def makefigure(x, y1, y2, y3, y4, info, fs_label=16, fs_tick=14):
    
    label1 = info['label1']['label']
    label2 = info['label2']['label']
    label3 = info['label3']['label']
    label4 = info['label4']['label']
    
    ylabel1 = label1 + info['label1']['units']
    ylabel2 = info['label2']['units']
    
    fig = plt.figure()
#     host = AA.Axes(fig, [0.1, 0.1, 0.8, 0.8]) 
#     fig.add_axes(host)
    host = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
    par1 = host.twinx()

    host.set_xlabel("Time [s]",fontsize=fs_label)
    host.set_ylabel(ylabel1,fontsize=fs_label)
    par1.set_ylabel(ylabel2,fontsize=fs_label)

    p1 = host.plot(x, y1,'k-', label=label1)
    p2 = par1.plot(x, y2,'b-', label=label2)
    p3 = par1.plot(x, y3,'g-', label=label3)
    p4 = par1.plot(x, y4,'-',color='r',label=label4)

    host.set_yticks([1000,1500,2000,2500])
    loc_x = info['loc_x']
    loc_y = info['loc_y']
    
    par1.set_xlim(info['xlim'])

    # added these three lines
    lns = p1+p2+p3+p4
    labs = [l.get_label() for l in lns]
    host.legend(lns, labs, bbox_to_anchor=(loc_x, loc_y),fontsize=fs_tick,frameon=False)
    
    inset = fig.add_axes([info['inset_x1'], info['inset_y1'], info['inset_x2'], info['inset_y2']]) 
    inset.plot(x, y1,'k.-')
    inset.set_yticks([])
    par2 = inset.twinx()
    par2.set_yticks([])
    par2.plot(x, y2,'b.-')
    par2.plot(x, y3,'g.-')
    par2.plot(x, y4,'.-',color='r')
    inset.set_xlim(info['xlim2'])
    for tick in host.xaxis.get_major_ticks()+host.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs_tick)
    for tick in inset.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs_tick-2)
    for tick in par1.yaxis.get_major_ticks():
        tick.label2.set_fontsize(fs_tick)

    return

def plotResults(x, y, z, name_title , x_label_name='1000/T ',y_label_name='mass fraction CO',\
               xtickvalues=None,ytickvalues=None,cticks=None,cticklabels=None,fs_label=22,fs_tick=20):
    N = 5
    lista = np.logspace(-2,-1,N).tolist()
    for i in [-1,0,1]:
        lista += np.logspace(i,i+1,N).tolist()[1:]
    fig, ax2 = plt.subplots(figsize=(10,8), nrows=1)
    ax2.tricontour(x, y, z, levels=lista, linewidths=0.5,  colors='k',norm=colors.LogNorm())
    cntr2 = ax2.tricontourf(x, y, z,  levels=lista, cmap="RdBu_r",norm=colors.LogNorm())
    if cticks is not None:
        cbar = fig.colorbar(cntr2, ax=ax2, ticks=cticks)    
    else:
        cbar = fig.colorbar(cntr2, ax=ax2)
    if cticklabels is not None:
        cbar.ax.set_yticklabels(cticklabels) 
    cbar.ax.tick_params(labelsize=fs_label)
    cbar.ax.set_ylabel(name_title,fontsize=fs_label)
    cbar.ax.tick_params(labelsize=fs_label)
    ax2.set_title(name_title,fontsize=fs_label)
    plt.subplots_adjust(hspace=0.5)
    ax2.set_xlabel(x_label_name,fontsize=fs_label)
    ax2.set_ylabel(y_label_name,fontsize=fs_label) 
    if xtickvalues is not None:
        ax2.set_xticks(xtickvalues)
    if ytickvalues is not None:
        ax2.set_yticks(ytickvalues)
    for tick in ax2.xaxis.get_major_ticks()+ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs_tick)
    return  
