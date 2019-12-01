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


#heart data input
heart = read.csv("processedHeart.csv", header = F, sep = ",")

#display info on inputed data
summary(heart)

#remove rows with na
heart <- na.omit(heart)

#display info on updated data
summary(heart)


for (i in 1:5) {

	#Separate into test and training sets
	train_size <- floor(0.8 * nrow(heart))
	set.seed(i)
	index <- sample(seq_len(nrow(heart)), size = train_size)
	training <- heart[index, ]
	testing <- heart[-index, ]

	#Extract features and Classes
	heart_features <- training[, -22]
	heart_class <- as.factor(training[, 22])

	print(paste("Round", i))

	#Train Random Forest Classifier 
	
	time <- proc.time()
	trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
	set.seed(i)
	rf <- train(x = heart_features, y = heart_class, method = "rf", trControl = trctrl)
	rf_time_storage <- (proc.time() - time)

	#Test Random Forest Classifier 
	
	time <- proc.time()
	rf_predict_1 <- predict(rf, testing[, -22], type = "prob")
	rf_predict_2 <- predict(rf, testing[, -22])
	rf_roc_results <- roc(testing[, 22], rf_predict_1 $"1")
	plot(rf_roc_results, print.thres = "best", print.thres.best.method = "closest.topleft")
	rf2_time_storage <- (proc.time() - time)

	#Display Random Forest Classifier Test Results
	
	print(rf)
	print(confusionMatrix(rf_predict_2, as.factor(testing[, 22])))

	#Train K Nearest Neighbor Classifier
	
	time <- proc.time()
	trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
	set.seed(i)
	knn <- train(x = heart_features, y = heart_class, method = "knn", trControl = trctrl)
	knn_time_storage <- (proc.time() - time)

	#Testing K Nearest Neighbor Classifier
	
	time <- proc.time()
	knn_predict_1 = predict(knn, testing[, -22], type = "prob")
	knn_predict_2 = predict(knn, testing[, -22])
	knn_roc_results <- roc(testing[, 22], knn_predict_1 $"1")
	plot(knn_roc_results, print.thres = "best", print.thres.best.method = "closest.topleft")
	knn2_time_storage <- (proc.time() - time)

	#Display K Nearest Neighbor Test Results
	
	print(knn)
	print(confusionMatrix(knn_predict_2, as.factor(testing[, 22])))

}
