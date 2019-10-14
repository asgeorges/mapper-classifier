#! /usr/bin/Rscript

library(networkD3)
library(devtools)
library(TDAmapper)
library(fastcluster)
library(igraph)
require(fastcluster) 

#Data
votes=read.table('../data_temp/trueexamples_in_10_10_PROJ2_layers.txt')
label_name = scan('../data_temp/trueexamples_in_10_10_PROJ2_names.txt',what="", sep="\n")

#Metric
dist1=dist(votes,method="euclidean") #formerly manhattan

#PCA
Eigenfunction=read.table('../data_temp/temp_trueexamples_in_10_10_PROJ2.txt')


# 1D Mapper
m1 <- mapper1D(distance_matrix = dist1, filter_values = Eigenfunction, num_intervals = 10, percent_overlap = 33, num_bins_when_clustering = 10)
g1 <- graph.adjacency(m1$adjacency, mode="undirected")
MapperNodes1 <- mapperVertices(m1, label_name)
MapperLinks1 <- mapperEdges(m1)
write.csv2(MapperNodes1, file = "../data_temp/trueexamples_in_10_10_PROJ2_nodes.txt", eol = "\n")

#net1=forceNetwork(Nodes = MapperNodes1, Links = MapperLinks1, fontSize = 10, 
#             Source = "Linksource", Target = "Linktarget",
#             Value = "Linkvalue", NodeID = "Nodename",
#             Group = "Nodegroup", opacity = 0.8, 
#             linkDistance = 50, charge = -500,
#             Nodesize = "Nodesize")

#saveNetwork(net1, file="../results/trueexamples_in_10_10_PROJ2_1D.html", selfcontained = FALSE)
