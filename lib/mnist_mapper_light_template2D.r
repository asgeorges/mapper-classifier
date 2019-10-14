#! /usr/bin/Rscript

library(networkD3)
library(devtools)
library(TDAmapper)
library(fastcluster)
library(igraph)
require(fastcluster) 

#Data
votes=read.table('../data_temp/stamp_layers.txt')
label_name = scan('../data_temp/stamp_names.txt',what="", sep="\n")

#Metric
dist1=dist(votes,method="manhattan")

#PCA
Eigenfunction1=read.table('../data_temp/PCACOMP1_temp_stamp.txt')
Eigenfunction2=read.table('../data_temp/PCACOMP2_temp_stamp.txt')

# 1D Mapper
m2 <- mapper2D(distance_matrix = dist1, filter_values = list(Eigenfunction1, Eigenfunction2), num_intervals = c(N2AINT,N2BINT), percent_overlap = 50, num_bins_when_clustering = NBIN)
g2 <- graph.adjacency(m2$adjacency, mode="undirected")
MapperNodes2 <- mapperVertices(m2, label_name)
MapperLinks2 <- mapperEdges(m2)
write.csv2(MapperNodes2, file = "../data_temp/stamp_nodes.txt", eol = "\n")

net2=forceNetwork(Nodes = MapperNodes2, Links = MapperLinks2, fontSize = 10, 
             Source = "Linksource", Target = "Linktarget",
             Value = "Linkvalue", NodeID = "Nodename",
             Group = "Nodegroup", opacity = 0.8, 
             linkDistance = 50, charge = -500,
             Nodesize = "Nodesize")

saveNetwork(net2, file="../results/stamp_2D.html", selfcontained = FALSE)
