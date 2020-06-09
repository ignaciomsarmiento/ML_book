##########################################################
#  Author: Ignacio Sarmiento-Barbieri (ignaciomsarmiento at gmail.com)
# please do not cite or circulate without permission
##########################################################

#Clean the workspace
rm(list=ls())
cat("\014")


require("stringr")
require("dplyr")
require("McSpatial")

#house data, dsp puedo agregar otras charateristics si estan cerca de un parque, distance from a rail line, el stop, el line
data(matchdata)
house<-matchdata
house$age <- house$year - house$yrbuilt

set.seed(20170313)
house$holdout <- as.logical(1:nrow(house) %in% sample(nrow(house), nrow(house) - 2704))
test<-house[house$holdout==T,]
train<-house[house$holdout==F,]
train$holdout<-NULL
test$holdout<-NULL
getwd()
save(train,test,file="house_data.rda")

house$lassofolds <- as.factor(ceiling(10 * sample(nrow(house)) / nrow(house)))



stargazer::stargazer(train,type="text")


model1<-lm(lnprice~1,data=train)
summary(model1)
model2<-lm(lnprice~bedrooms,data=house)
summary(model2)

model3<-lm(lnprice~bedrooms+bathrooms,data=house)
summary(model3)


test$model1<-predict(model1,newdata=test)
test$model2<-predict(model2,newdata=test)
test$model3<-predict(model3,newdata=test)


mean(test$lnprice)
mean(train$lnprice)

mean(test$model1)
