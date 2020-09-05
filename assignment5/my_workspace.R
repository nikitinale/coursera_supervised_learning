library(ggplot2)
library(rpart)
library(randomForest)

data1 = read.csv("seaflow_21min.csv", header=TRUE)
data1 = data3
head(data1)
summary(data1)
dim(data1)
spli_vec = rbinom(72343, 1, 0.5)
data1["split"] = spli_vec
data2 = split(data1, data1["split"])

ggplot(data1, aes(x = chl_small, y = pe, colour=pop)) +
  geom_point()

data_train=data2[["0"]]
data_test=data2[["1"]]
summary(data_test)

# Decission Tree
form = formula(pop ~ fsc_small + fsc_perp + fsc_big + pe + chl_big + chl_small)
mymodel = rpart(form, method="class", data=data_train)
printcp(mymodel)
plotcp(mymodel)
summary(mymodel)
plot(mymodel, uniform=TRUE,
     main="Classification Tree for PE")
text(mymodel, use.n=TRUE, all=TRUE, cex=.8)
print(mymodel)

predict_test = predict(mymodel, data_test, type = "class")
predict_test = matrix(predict_test)
accur = predict_test == data_test$pop # ["pop"]
accuracy = sum(accur)/length(accur)
print(accuracy)
table(pred=predict_test, true=data_test$pop)

# Random Forest
mymodel2 = randomForest(form, data=data_train)
predict_test2 = predict(mymodel2, data_test)
predict_test2 = matrix(predict_test2)
accur = predict_test2 == data_test["pop"]
accuracy = sum(accur)/length(accur)
print(accuracy)
importance(mymodel2)
table(pred=predict_test2, true=data_test$pop)

# SVM
library(e1071)
mymodel3 = svm(form, data=data_train)
predict_test3 = predict(mymodel3, data_test)
predict_test3 = matrix(predict_test3)
accur = predict_test3 == data_test["pop"]
accuracy = sum(accur)/length(accur)
print(accuracy)
table(pred=predict_test3, true=data_test$pop)

ggplot(data1, aes(x = fsc_big, y = pop)) +
  geom_point()

data3 = data1[data1$file_id != 208, ]

ggplot(data3, aes(x = time, y = file_id, colour=pop)) +
  geom_point()


hist(data1["fsc_small"])
plot(vector(data1["fsc_small"]), vector(data1["pop"]))

data_train = data1[data1["split"]==0]
data_test
sample(data1, size=100)



ggplot(airquality, aes(x = Day, y = Ozone)) +
  geom_point()
airquality
airquality.heads
airquality["Wind"]
airquality[1]

ggplot(airquality, aes(x = Month, y = Temp)) +
  geom_point()
