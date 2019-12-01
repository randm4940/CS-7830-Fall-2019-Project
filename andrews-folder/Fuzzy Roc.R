# R code for project

#setup work space
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

seedNum <- 7
for (cent in 3:3) {

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

	threshold <- 0.01
	rocX <- c(1)
	rocY <- c(1)

	while (threshold <= 1) {

		assignment <- rep(0, nrow(result$centers))
		countLimit <- rep(0, nrow(result$centers))
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
				countLimit[i] <- (count_0/(count_0 + count_1))
			} else {
				assignment[i] <- 2
				countLimit[i] <- (count_1/(count_0 + count_1))
			}
		}


		#find confusion matrix
		for (i in 1:nrow(result$centers)) {
			#find accuracy of each cluster
			for (j in 1:nrow(testSet)) {
				#test if row is in cluster
				if (i == which.max(member[j, ])) {
					#test if membership passes threshold
					if (countLimit[i] >= threshold) {
						centroid <- assignment[i]
					} else {
						if (assignment[i] == 1) {
							centroid <- 2
						} else {
							centroid <- 1
						}
					}
					#find which slot in confusion matrix row goes
					if ((assignment[i] == centroid) && (centroid == (testSet[j, 22] + 1))) { #TP
						TP <- TP + 1
					} else if ((assignment[i] == centroid) && (centroid != (testSet[j, 22] + 1))) { #FP
						FP <- FP + 1
					} else if ((assignment[i] != centroid) && (centroid == (testSet[j, 22] + 1))) { #FN
						FN <- FN + 1
					} else { #TN
						TN <- TN + 1
					}
				}
			}
		}




		#find info for roc curve
		sensitivity <- (TP/(TP + FN))
		specificity <- (TN/(TN + FP))

		#add points to roc curves
		rocX <- append(rocX, (1 - specificity))
		rocY <- append(rocY, sensitivity)
		threshold <- threshold + 0.01
	}

	#print roc
	print(paste("Number of Clusters", cent))
	plot(rocX, rocY, type = "l", main = "ROC Curve", ylab = "True Positive Rate", xlab = "False Positive Rate")
	text(rocX, rocY, paste("(",round(rocX,3),",",round(rocY, 3),")"), cex=.85, pos = 3)
}
