import pandas as pd
import os
import glob
from itertools import compress
import numpy as np
from p_tqdm import p_map
import multiprocessing
import matplotlib.image as mpimg 

import tempfile

def import_dat(path):
    scale=pd.read_table(path+'/scale',header=0)
    sc=scale["real_distance"][0]/scale["imaris_distance"][0]

    fl=next(os.walk(path))[1]
    fl=list(compress(fl, list(map(lambda x: x!=".ipynb_checkpoints",fl))))


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
    
    files = [f for f in glob.glob(path + "/**/*.tiff", recursive=True)]
    map_img = list(map(lambda x: mpimg.imread(x),files))
    
    return adata, mdata, msdata, usdata, map_img, sc


def fetch_track(adata,rep=0):
    allparents=np.unique(adata.loc[rep]["Parent"])
    allparents=allparents[~np.isnan(allparents)]
    
    ldata=list(map(lambda x: 
                     adata.loc[rep][["Position X",
                                     "Position Y",
                                     "Time"]].loc[adata.loc[rep].index[adata.loc[rep]["Parent"].isin([x],)]],[par for par in allparents]))
    return ldata

def grid_vector_field_DEP(adata,ldata,rep=0):
    files = glob.glob('clustertracks/*')
    for f in files:
        os.remove(f)

    init=[min(adata.loc[rep]["Position X"])-1,max(adata.loc[rep]["Position X"])+1,
          min(adata.loc[rep]["Position Y"])-1,max(adata.loc[rep]["Position Y"])+1,
          1,max(adata.loc[rep]["Time"])]
    with open('tracks', 'w') as f:
        f.writelines(["%s " % item  for item in init])
    for ld in ldata:
        ld.to_csv('tracks', mode='a', header=False,sep=" ",index=False)
        pd.DataFrame([0,0,0]).transpose().to_csv('tracks',header=False,sep=" ",mode="a",index=False)


    import subprocess

    proc = subprocess.Popen("vfkm/src/vfkm tracks 30 1 0.5 clustertracks", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    print(out.decode('ascii'))
    
    files_c = [f for f in glob.glob("clustertracks/*curves_r_*", recursive=True)]
    files_v = [f for f in glob.glob("clustertracks/*vf_r_*", recursive=True)]
    curves=pd.read_table(files_c[0],header=None,sep=" ")
    vect=pd.read_table(files_v[0],header=None,sep=" ",skiprows=1)
    
    files = glob.glob('clustertracks/*')
    for f in files:
        os.remove(f)

    gs=int(np.sqrt(vect.shape[0]))

    a=np.linspace(min(pd.concat(ldata)["Position X"]),max(pd.concat(ldata)["Position X"]),gs)
    b=np.linspace(min(pd.concat(ldata)["Position Y"]),max(pd.concat(ldata)["Position Y"]),gs)
    out = np.stack([each.ravel(order='C') for each in np.meshgrid(a, b)])
    
    u=vect[0].values.reshape(gs,gs)
    v=vect[1].values.reshape(gs,gs)
    speed = np.sqrt(u**2 + v**2)
    X,Y = np.meshgrid(a, b)
    
    return X,Y,u,v


def grid_vector_field(adata,ldata,rep=0,verbose=False):
    with tempfile.TemporaryDirectory() as directory:
        init=[min(adata.loc[rep]["Position X"])-1,max(adata.loc[rep]["Position X"])+1,
              min(adata.loc[rep]["Position Y"])-1,max(adata.loc[rep]["Position Y"])+1,
              1,max(adata.loc[rep]["Time"])]
        with open(directory+'/tracks', 'w') as f:
            f.writelines(["%s " % item  for item in init])
        for ld in ldata:
            ld.to_csv(directory+'/tracks', mode='a', header=False,sep=" ",index=False)
            pd.DataFrame([0,0,0]).transpose().to_csv(directory+'/tracks',header=False,sep=" ",mode="a",index=False)


        import subprocess

        proc = subprocess.Popen("helpers/vfkm/src/vfkm "+directory+"/tracks 30 1 0.5 "+directory, stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        
        if verbose:
            print(out.decode('ascii'))

        files_c = [f for f in glob.glob(directory+"/*curves_r_*", recursive=True)]
        files_v = [f for f in glob.glob(directory+"/*vf_r_*", recursive=True)]
        curves=pd.read_table(files_c[0],header=None,sep=" ")
        vect=pd.read_table(files_v[0],header=None,sep=" ",skiprows=1)

        gs=int(np.sqrt(vect.shape[0]))

        a=np.linspace(min(pd.concat(ldata)["Position X"]),max(pd.concat(ldata)["Position X"]),gs)
        b=np.linspace(min(pd.concat(ldata)["Position Y"]),max(pd.concat(ldata)["Position Y"]),gs)
        out = np.stack([each.ravel(order='C') for each in np.meshgrid(a, b)])

        u=vect[0].values.reshape(gs,gs)
        v=vect[1].values.reshape(gs,gs)
        speed = np.sqrt(u**2 + v**2)
        X,Y = np.meshgrid(a, b)

        return X,Y,u,v