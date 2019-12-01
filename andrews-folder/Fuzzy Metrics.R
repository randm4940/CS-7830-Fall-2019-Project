# R code for project

#setup work space\n
getwd()
setwd("/Users/andrewyoung/Desktop/Project")
getwd()
#setup libraries
library("lattice")
library("ggplot2")
library("caret")
library("pROC")

library(ppclust)
library(factoextra)
library(dplyr)
library(cluster)
library(fclust)
library(psych)
library(e1071)


heart = read.csv("processedHeart.csv", header = F, sep = ",")

for(seedNum in 1:10){
for (cent in 2:9) {

	#generate seperator
	trainSize <- floor(0.8 * nrow(heart))
	set.seed(seedNum)
	index <- sample(seq_len(nrow(heart)), size = trainSize)

	#seperate training and test sets
	trainSet <- heart[index, ]
	testSet <- heart[-index, ]


	features <- trainSet[, -22]
	class <- trainSet[, 22]

	#pairs(features, class)
	#cor(features, class)
#res.fcm <- fcm(features, centers = cent)
#summary(res.fcm)
#slot <- which.max(row)


	result <- cmeans(features, centers = cent, iter.max = 1000, verbose = FALSE, method = "cmeans")

	assignment <- rep(0, nrow(result$centers))
	TP <- 0
	FP <- 0
	FN <- 0
	TN <- 0

	#create membership matrix
	member <- data.frame(matrix(0, nrow = nrow(testSet), ncol = nrow(result$centers)))

	#find membership matrix for test set
	for (i in 1:nrow(testSet)) {

		minDistance <- -1
		distance <- rep(0, nrow(result$centers))

		#find distances
		for (row in 1:nrow(result$centers)) {

			distance[row] <- 0

			for (col in 1:ncol(result$centers)) {
				distance[row] <- distance[row] + (testSet[i, col] - result$centers[row, col])^2
			}

			distance[row] <- sqrt(distance[row])
		}

		#find sum distance
		sum <- 0
		for (row in 1:length(distance)) {
			sum <- sum + distance[row]
		}

		#find membership
		for (row in 1:length(distance)) {
			member[i, row] <- distance[row]/sum
		}
	}

	#sort each cluster into 1 or 0 assignment groups
	for (i in 1:nrow(result$centers)) {
		#find max count	
		count_0 <- 0
		count_1 <- 0
		for (j in 1:nrow(testSet)) {
			#test if row is in cluster
			if (i == which.max(member[j, ])) {
				if (testSet[j, 22] == 0) {
					count_0 <- count_0 + 1
				} else {
					count_1 <- count_1 + 1
				}
			}
		}
		if (count_0 >= count_1) {
			assignment[i] <- 1
		} else {
			assignment[i] <- 2
		}
	}


	#find confusion matrix
	for (i in 1:nrow(result$centers)) {
		#find accuracy of each cluster
		for (j in 1:nrow(testSet)) {
			#test if row is in cluster
			if (i == which.max(member[j, ])) {
				#find which slot in confusion matrix row goes
				if ((assignment[i] == 2) && (assignment[i] == (testSet[j, 22] + 1))) { #TP
					TP <- TP + 1
				} else if ((assignment[i] == 2) && (assignment[i] != (testSet[j, 22] + 1))) { #FP
					FP <- FP + 1
				} else if ((assignment[i] != 2) && (assignment[i] == (testSet[j, 22] + 1))) { #FN
					FN <- FN + 1
				} else { #TN
					TN <- TN + 1
				}
			}
		}
	}

	#find info for when threshold is unbiased
	true_TP <- TP
	true_FP <- FP
	true_FN <- FN
	true_TN <- TN
	true_acc <- (TP + TN)/(TP + TN + FP + FN)
	true_sensitivity <- (TP/(TP[1] + FN))
	true_specificity <- (TN/(TN[1] + FP))
	true_precision <- (TP/(TP + FP))
	true_F1 <- ((TP * 2)/((TP * 2) + FP + FN))


	print(paste("Seed Number", seedNum))
	print(paste("Number of Clusters", cent))
	print(paste("TP", true_TP))
	print(paste("FP", true_FP))
	print(paste("FN", true_FN))
	print(paste("TN", true_TN))
	print(paste("acc", true_acc))
	print(paste("sensitivity", true_sensitivity))
	print(paste("specificity", true_specificity))
	print(paste("precision", true_precision))
	print(paste("F1", true_F1))
	print("")
}
}
