library(readr)
library(pbmcapply)
library(parallel)
library(logging)
options(warn=-1)
basicConfig()

buildvec=function(i){
    vect <- vector()
    for (j in alltracks[alltracks$Time==i,]$Parent){
    one=alltracks[alltracks$Time==i & alltracks$Parent==j,1:2]
    one=one[!is.na(one[,1]),]
    if (dim(one)[1]!=0 & !is.na(j)){

      plusone=alltracks[alltracks$Time==i+1 & alltracks$Parent==j,1:2]
      plusone=plusone[!is.na(plusone[,1]),]
      if (dim(plusone)[1]==0){}
      else{vect=rbind(vect,cbind(one,one-plusone))}
    }
    }
    return(vect)
}

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
    
    loginfo(paste0("Building vector data on ",
                   max(alltracks$Time)-min(alltracks$Time),
                   " timepoints using ",detectCores()/2," cores"))
    pb = txtProgressBar(min = min(alltracks$Time), max = max(alltracks$Time), initial = min(alltracks$Time)) 
    vect=do.call(rbind,pbmclapply(min(alltracks$Time):max(alltracks$Time),buildvec,
                                  mc.cores=detectCores()/2,
                                  ignore.interactive=TRUE))
    write.csv(vect,paste0(d,"/all_vectors.csv"))
}


