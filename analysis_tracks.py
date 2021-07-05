import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logging.info("Importing modules")

import glob
import sys
import os
os.environ['NUMEXPR_MAX_THREADS'] = str(1)
import subprocess
import tempfile
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import compress
from scipy import stats

from helpers.track_tools import distances_accelerations, sinuosities_max
from helpers.FTLE import test_trajectory, get_traj, get_ftle

def load_data(path):
    scale=pd.read_table(path+'/scale',header=0)
    sc=scale["real_distance"][0]/scale["imaris_distance"][0]

    fl=next(os.walk(path))[1]
    fl=list(compress(fl, list(map(lambda x: x!=".ipynb_checkpoints",fl))))

    logging.info("Loading tracks, vectors and mouth positions")

    files = [f for f in glob.glob(path + "/**/all_tracks.csv", recursive=True)]
    adata=pd.concat(list(map(pd.read_csv,files)),keys=list(range(0,len(files))))

    files = [f for f in glob.glob(path + "/**/*mouths_Position.csv", recursive=True)]
    mdata=pd.concat(list(map(lambda x: pd.read_csv(x,skiprows=3),files)),keys=list(range(0,len(files))))
    mdata["Position X"]=mdata["Position X"]*sc
    mdata["Position Y"]=mdata["Position Y"]*sc

    files = [f for f in glob.glob(path + "/**/all_maxspeeds.csv", recursive=True)]
    msdata=pd.concat(list(map(pd.read_csv,files)),keys=list(range(0,len(files))))

    files = [f for f in glob.glob(path + "/**/all_meanspeeds.csv", recursive=True)]
    usdata=pd.concat(list(map(pd.read_csv,files)),keys=list(range(0,len(files))))

    logging.info("Plotting track speeds")
    cmp=list(map(lambda x: pd.DataFrame({"maxspeed":msdata.loc[x]["Value"],"replicate":x}),range(0,len(files))))
    plt.subplot(2,1,1)
    compa=pd.concat(cmp)
    snsplot = sns.violinplot(x="replicate", y="maxspeed", data=compa)
    #fig = snsplot.get_figure()
    cmp=list(map(lambda x: pd.DataFrame({"mean track speed":usdata.loc[x]["Value"],"replicate":x}),range(0,len(files))))
    plt.subplot(2,1,2)
    compa=pd.concat(cmp)
    snsplot = sns.violinplot(x="replicate", y="mean track speed", data=compa)
    #fig = snsplot.get_figure()
    plt.savefig(path+"/tracksspeed.png",dpi=300)
    plt.close()


    imgs=glob.glob(path + "/**/*.tiff", recursive=True)
    adata["turning"]=False
    adata["closestmouth_d"]=0
    
    return adata, mdata, imgs, fl, sc
    
def process_tracks(rep):
    allparents=np.unique(adata.loc[rep]["Parent"])
    allparents=allparents[~np.isnan(allparents)]
    mouthpos=mdata.loc[rep].loc[:,["Position X","Position Y"]]/sc/355
    
    ldata=list(map(lambda x: adata.loc[rep][["Position X","Position Y","Time"]].loc[adata.loc[rep].index[adata.loc[rep]["Parent"].isin([x],)]], 
                         [par for par in allparents]))
    
    
    logging.info("replicate "+str(rep)+": Making grid vector field")
    with tempfile.TemporaryDirectory() as tmp:
        logging.info("replicate "+str(rep)+": Data temporary saved in "+tmp)
        init=[min(adata.loc[rep]["Position X"]),max(adata.loc[rep]["Position X"]),
              min(adata.loc[rep]["Position Y"]),max(adata.loc[rep]["Position Y"]),
              1,max(adata.loc[rep]["Time"])]
        with open(tmp+'/tracks', 'w') as f:
            f.writelines(["%s " % item  for item in init])
        for ld in ldata:
            ld.to_csv(tmp+'/tracks', mode='a', header=False,sep=" ",index=False)
            pd.DataFrame([0,0,0]).transpose().to_csv(tmp+'/tracks',header=False,sep=" ",mode="a",index=False)

        proc = subprocess.Popen("helpers/vfkm/src/vfkm "+tmp+"/tracks 30 1 0.5 "+tmp, stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        #print(out.decode('ascii'))
        
        proc = subprocess.Popen("ls "+tmp, stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        #print(out.decode('ascii'))

        files_c = [f for f in glob.glob(tmp+"/*curves_r_*", recursive=True)]
        files_v = [f for f in glob.glob(tmp+"/*vf_r_*", recursive=True)]
        curves=pd.read_table(files_c[0],header=None,sep=" ")
        vect=pd.read_table(files_v[0],header=None,sep=" ",skiprows=1)/sc/355    
    
    gs=int(np.sqrt(vect.shape[0]))

    a=np.linspace(min(pd.concat(ldata)["Position X"]),max(pd.concat(ldata)["Position X"]),gs)/sc/355
    b=np.linspace(min(pd.concat(ldata)["Position Y"]),max(pd.concat(ldata)["Position Y"]),gs)/sc/355
    out = np.stack([each.ravel(order='C') for each in np.meshgrid(a, b)])
    
    u=vect[0].values.reshape(gs,gs)
    v=vect[1].values.reshape(gs,gs)
    speed = np.sqrt(u**2 + v**2)
    X,Y = np.meshgrid(a, b)
    
    map_img = mpimg.imread(imgs[rep]) 
    
    fig = plt.figure(figsize=(10,6))
    
    
    ax1 = fig.add_subplot(2,2,1)
    
    ax1.set_aspect('equal')
    ax1.imshow(map_img,origin='lower')
    speed = np.sqrt(u**2 + v**2)
    ax1.streamplot(X, Y, u, v,color=speed, density=2, linewidth=1, 
                   arrowsize=1, arrowstyle='->', cmap="autumn")
    ax1.axis("off")
    ax1.scatter(mouthpos["Position X"],mouthpos["Position Y"],zorder=3,c="white")

    
    logging.info("replicate "+str(rep)+": Calculating FTLE")
    traj_x, traj_y = get_traj(X, Y, u, v, 20000, 5)
    ftle = get_ftle(traj_x, traj_y, X, Y, 20000)
    
    ax2 = fig.add_subplot(2,2,2)
    
    ax2.set_aspect('equal')
    contf = ax2.contourf(X, Y, ftle, extend='both',cmap="jet")
    ax2.streamplot(X, Y, u, v, density=2, linewidth=1, 
                   arrowsize=1, arrowstyle='->', color='k')
    ax2.scatter(mouthpos["Position X"],mouthpos["Position Y"],zorder=3,c="white")
    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(contf,cax=cax)
    cbar.set_label('$FTLE$', fontsize=12)
    plt.xticks([]),plt.yticks([])
    
    logging.info("replicate "+str(rep)+": Calculating speeds and sinuosities")
    
    ldata=list(map(distances_accelerations,ldata))
    ldata = list(map(sinuosities_max, ldata))

    compil=pd.concat(ldata)
    
    compil.reset_index(drop=True,inplace=True)
    torem=[]
    for col in compil.columns:
        torem=torem+np.argwhere(np.isnan(compil[col]).values).tolist()
        
    compil.drop(np.unique(np.array(torem)),inplace=True) 
    
    logging.info("replicate "+str(rep)+": Making plots")
    ret = stats.binned_statistic_2d(compil["Position X"], 
                                    compil["Position Y"], 
                                    (compil["acc"]), bins=[30,30])
    stat=ret.statistic
    stat=stat/3.25
    perc=np.nanpercentile(stat,[5,95])
    stat[stat<perc[0]]=perc[0]
    stat[stat>perc[1]]=perc[1]
    

    ax3 = fig.add_subplot(2,2,3)
    ax3.set_aspect('equal')
    ctf=ax3.contourf(X,Y,stat,extend='both',cmap="RdBu_r", 
                     vmin=-max(np.abs(perc)), vmax=max(np.abs(perc)))
    ax3.streamplot(X, Y, u, v, density=2, linewidth=1, 
                   arrowsize=1, arrowstyle='->', color='k')
    ax3.scatter(mouthpos["Position X"],mouthpos["Position Y"],zorder=3,c="green")

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(ctf,cax=cax)
    cbar.set_label('acceleration $(\mu m \cdot s^{-1})$\n(5th and 95th percentiles)', fontsize=12)
    
    ret = stats.binned_statistic_2d(compil["Position X"], compil["Position Y"], 
                                    (compil["sinuosity"]), bins=[30,30])

    stat=ret.statistic
    perc=np.nanpercentile(stat,[5,95])
    stat[stat<perc[0]]=perc[0]
    stat[stat>perc[1]]=perc[1]
    
    ax4 = fig.add_subplot(2,2,4)
    ax4.set_aspect('equal')
    vmax=1.5
    if np.nanmin(stat[:])>vmax:
        vmax=np.nanmin(stat[:])+1
        
    #print(np.nanmin(stat[:]))
    #print(np.nanmax(stat[:]))
    ctf=ax4.contourf(X,Y,stat,extend='both',cmap="Reds",vmin=1,vmax=vmax)
    ax4.streamplot(X, Y, u, v, density=2, linewidth=1, 
                   arrowsize=1, arrowstyle='->', color='k')
    ax4.scatter(mouthpos["Position X"],mouthpos["Position Y"],zorder=3,c="green")

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(ctf,cax=cax)
    cbar.set_label('$sinuosity$', fontsize=12)
    #
    
    plt.tight_layout()
    fig.savefig(path+"/"+fl[rep]+'/REPORT.png',dpi=600)
    
    logging.info("replicate "+str(rep)+": Done!")
    
if __name__ == "__main__":
    path=sys.argv[1]
    adata, mdata, imgs, fl, sc = load_data(path)
    logging.info("Running parallel analysis")
    with Pool(len(imgs)) as p:
        p.map(process_tracks, range(len(imgs)))