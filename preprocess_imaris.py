import glob
import pandas as pd
from p_tqdm import p_map
import sys
patterns=["/**/*_tracks_Position*","/**/*_tracks_Speed*",
         "/**/*_Track_Speed_Max*","/**/*_Track_Speed_Mean*"]

def gather_inaris(fls):
    dataset=pd.read_csv(fls[0],skiprows=3)
    for fl in fls[1:]:
        dataset_temp=pd.read_csv(fl,skiprows=3)
        if any(dataset.columns=="Parent"):
            dataset_temp.Parent=dataset_temp.Parent+max(dataset.Parent)
        dataset_temp.ID=max(dataset.ID)+1
        dataset=dataset.append(dataset_temp)
    if any(dataset.columns=="Value"):
        dataset.loc[:,["Value"]]=dataset.loc[:,["Value"]]*sc
    else:
        dataset.loc[:,["Position X","Position Y"]]=dataset.loc[:,["Position X","Position Y"]]*sc
        
    return dataset

def imaris_to_csv(fold):
    filesets=list(map(lambda x: glob.glob(fold+x),patterns))
    datasets=list(map(gather_inaris,filesets))
    data_tracks, data_speeds, data_maxspeeds, data_meanspeeds = datasets[0], datasets[1], datasets[2], datasets[3]
    data_tracks.to_csv(fold+"all_tracks.csv")
    data_speeds.to_csv(fold+"all_speeds.csv")
    data_maxspeeds.to_csv(fold+"all_maxspeeds.csv")
    data_meanspeeds.to_csv(fold+"all_meanspeeds.csv")
    
    
path=sys.argv[1]
scale=pd.read_table(path+"/scale")
sc=(scale["real_distance"]/scale["imaris_distance"])[0]
folds=glob.glob(path+"/*/")

p_map(imaris_to_csv,folds)
