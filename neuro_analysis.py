#!/usr/bin/env python
import matplotlib
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from functools import partial
import glob
import sys
import logging
import multiprocessing
from p_tqdm import p_map
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

import os
import utils
numcores=int(multiprocessing.cpu_count()/2)
os.environ['NUMEXPR_MAX_THREADS'] = str(numcores)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def summarize(df_ls,path):
    nm=df_ls["filename"].values[0]
    adata, mdata, msdata, usdata, map_img, sc = utils.import_dat(path+"/"+nm)
    ldata=utils.fetch_track(adata)
    X,Y,u,v=utils.grid_vector_field(adata,ldata)
    return [X,Y,u,v,msdata.loc[0]["Value"],usdata.loc[0]["Value"],map_img,sc,ldata]

def compute_angle(uv):    
    #vec=[u[i],v[i]]
    base=[0,1]
    unit_vector_1 = base / np.linalg.norm(base)
    unit_vector_2 = uv / np.linalg.norm(uv)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    return math.degrees(angle)


def plot_neuro(res,title):
    allX=list(map(lambda x: x[0],res))
    allY=list(map(lambda x: x[1],res))
    allu=list(map(lambda x: x[2],res))
    allv=list(map(lambda x: x[3],res))
    allus=list(map(lambda x: x[4],res))
    allms=list(map(lambda x: x[5],res))
    allimg=list(map(lambda x: (x[6]),res))
    allsc=list(map(lambda x: (x[7]),res))
    allldat=list(map(lambda x: (x[8]),res))
    fig = plt.figure(figsize=(20,8),dpi=300)
    grid = plt.GridSpec(6, 4)

    #logging.info("plot single trajectories and streamplot")
    for i in range(0,4):
        ax1 = fig.add_subplot(grid[0:2,i])
        ax1.set_aspect('equal')
        for ldat in (allldat[i]):
          ax1.plot(ldat[["Position X"]],ldat[["Position Y"]],linewidth=.5)
          ax1.axis("off")

        ax1 = fig.add_subplot(grid[2:4,i])
        ax1.set_aspect('equal')
        
        ax1.imshow(allimg[i][0],origin='lower')
        speed = np.sqrt(allu[i]**2 + allv[i]**2)
        ax1.streamplot(allX[i]/allsc[i]/355, allY[i]/allsc[i]/355, allu[i]/allsc[i]/355, allv[i]/allsc[i]/355,color=speed, density=2, linewidth=1, 
         arrowsize=1, arrowstyle='->', cmap="autumn")
        ax1.axis("off")


    #logging.info("track speed analysis")    
    allus=pd.concat(allus,keys=["before\n(0 to 5min)","before\n(5 to 10min)","after\n(0 to 5min)","after\n(5 to 10min)"],axis=1)
    #from scipy.stats import ranksums as test
    from scipy.stats import median_test as test

    tt=list(map(lambda x: allus[x].dropna(), allus.columns))

    pvalues=[test(tt[0],tt[1])[1]]
    pvalues=pvalues+[test(tt[1],tt[2])[1]]
    pvalues=pvalues+[test(tt[2],tt[3])[1]]
    pvalues=pvalues+[test(tt[0],tt[3])[1]]

    from statsmodels.stats.multitest import multipletests
    pval_corrected=multipletests(pvalues)[1]


    import seaborn as sns
    from statannot import add_stat_annotation
    ax1 = fig.add_subplot(grid[4,0])
    ax=sns.violinplot(data=pd.melt(allus),y="value",x="variable")
    plt.ylabel("Mean track speed")
    plt.xlabel(None)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) #

    test_results = add_stat_annotation(ax, data=pd.melt(allus),y="value",x="variable",
                                       box_pairs=[("before\n(0 to 5min)", "before\n(5 to 10min)"),
                                                  ("before\n(5 to 10min)", "after\n(0 to 5min)"),
                                                  ("after\n(0 to 5min)", "after\n(5 to 10min)"),
                                                  ("before\n(0 to 5min)", "after\n(5 to 10min)")],
    perform_stat_test=False, pvalues=pval_corrected)


    allms=pd.concat(allms,keys=["before\n(0 to 5min)","before\n(5 to 10min)","after\n(0 to 5min)","after\n(5 to 10min)"],axis=1)

    tt=list(map(lambda x: allms[x].dropna(), allms.columns))

    pvalues=[test(tt[0],tt[1])[1]]
    pvalues=pvalues+[test(tt[1],tt[2])[1]]
    pvalues=pvalues+[test(tt[2],tt[3])[1]]
    pvalues=pvalues+[test(tt[0],tt[3])[1]]

    from statsmodels.stats.multitest import multipletests
    pval_corrected=multipletests(pvalues)[1]


    import seaborn as sns
    from statannot import add_stat_annotation
    ax1 = fig.add_subplot(grid[5,0])
    ax=sns.violinplot(data=pd.melt(allms),y="value",x="variable")
    plt.ylabel("Max")
    plt.xlabel(None)

    test_results = add_stat_annotation(ax, data=pd.melt(allms),y="value",x="variable",
    box_pairs=[("before\n(0 to 5min)", "before\n(5 to 10min)"),
               ("before\n(5 to 10min)","after\n(0 to 5min)"),
               ("after\n(0 to 5min)", "after\n(5 to 10min)"),
               ("before\n(0 to 5min)", "after\n(5 to 10min)")],
    perform_stat_test=False, pvalues=pval_corrected)


    #logging.info("computing angles")
    angles=[]

    for ti in range(4):
        u=allu[ti].flatten().tolist()
        v=allv[ti].flatten().tolist()

        uv=list(map(lambda x, y:[x,y], u, v))
        angles=angles+[np.array(list(map(compute_angle,uv))).reshape(30,30)]

    ax = fig.add_subplot(grid[4:,1])
    contf = ax.contourf(allX[0], allY[0], np.abs(angles[1]-angles[0]), extend='both')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(contf,cax=cax)
    #cbar.set_label('$FTLE$', fontsize=12)
    plt.xticks([]),plt.yticks([])

    ax = fig.add_subplot(grid[4:,2])
    contf = ax.contourf(allX[0], allY[0], np.abs(angles[2]-angles[1]), extend='both')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(contf,cax=cax)
    #cbar.set_label('$FTLE$', fontsize=12)
    plt.xticks([]),plt.yticks([])

    ax = fig.add_subplot(grid[4:,3])
    contf = ax.contourf(allX[0], allY[0], np.abs(angles[3]-angles[2]), extend='both')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(contf,cax=cax)
    #cbar.set_label('$FTLE$', fontsize=12)
    plt.xticks([]),plt.yticks([])

    plt.tight_layout()
    plt.savefig(title)



if __name__ == "__main__":
    path = "Neuro_Data/Pavona_d/"
    df=pd.DataFrame({"filename":next(os.walk(path))[1]})

    df[['Date','Specie','Cond','treatment','timepoint']]=df.filename.str.split("_",expand=True)
    dates=np.unique(df[["Date"]])
    dct={0:["before","1"],1:["before","2"],2:["after","1"],3:["after","2"]}

    tolinearize=[]

    for date in dates:
        conds=np.unique(df.loc[(df.Date==date),"Cond"])
        for cond in conds:
            tolinearize=tolinearize+[[date,cond]]

    def runall(dc):
        df_s=df.loc[(df.Date==dc[0]) & (df.Cond==dc[1]),:]
        
        df_l=list(map(lambda x: df_s.loc[(df_s.treatment==dct[x][0])&(df_s.timepoint==dct[x][1]),:],range(4)))
        fun=partial(summarize,path=path)
        res=list(map(fun,df_l))
        plot_neuro(res,dc[0]+"_"+dc[1])

    p_map(runall,tolinearize)
