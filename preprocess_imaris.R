library(readr)
library(pbmcapply)
library(parallel)
library(logging)
options(warn=-1)
basicConfig()

gatherdat=function(fil){
    dataset <- read_csv(fil[1],skip = 3,col_types=cols())
    
    for (f in fil[-1]){
        dataset_temp <- read_csv(f,skip = 3,col_types=cols())
        if (any(grepl("Parent",colnames(dataset)))){
            dataset_temp$Parent=dataset_temp$Parent+max(dataset$Parent)
        }
        dataset_temp$ID=max(dataset$ID)+1
        dataset=rbind(dataset,dataset_temp)
    }
    
    return(dataset)
}

args = commandArgs(trailingOnly=TRUE)

l=list.dirs(args[1],recursive=F)
l=l[!grepl(".ipynb_checkpoints",l)]

scale=read.table(paste0(args[1],"/scale"),header=TRUE)
sc=scale$real_distance/scale$imaris_distance


for (d in l){
    loginfo(paste0("Processing ",d))
    
    allfiles=list(list.files(d,recursive=T,full.names=T,pattern="_tracks_Position"),
              list.files(d,recursive=T,full.names=T,pattern="_tracks_Speed"),
              list.files(d,recursive=T,full.names=T,pattern="_Track_Speed_Max"),
              list.files(d,recursive=T,full.names=T,pattern="_Track_Speed_Mean"))

    loginfo(paste0("Gathering and rescaling all tracks and speeds using ",length(allfiles)," cores"))
    
    datasets=mclapply(allfiles,gatherdat,mc.cores=length(allfiles))
    
    alltracks=datasets[[1]]
    alltracks_tocsv=alltracks;alltracks_tocsv[,1:2]=alltracks_tocsv[,1:2]*sc
    allspeeds=datasets[[2]];allspeeds[,1]=allspeeds[,1]*sc
    allmaxspeeds=datasets[[3]];allmaxspeeds[,1]=allmaxspeeds[,1]*sc
    allmeanspeeds=datasets[[4]];allmeanspeeds[,1]=allmeanspeeds[,1]*sc
    
    write.csv(allspeeds,paste0(d,"/all_speeds.csv"))
    write.csv(allmaxspeeds,paste0(d,"/all_maxspeeds.csv"))
    write.csv(allmeanspeeds,paste0(d,"/all_meanspeeds.csv"))
    
    write.csv(alltracks_tocsv,paste0(d,"/all_tracks.csv"))
    
}


