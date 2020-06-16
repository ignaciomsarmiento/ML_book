
# Lab 3: Regression for prediction

## Introduction {#sec:introduction}

In this lab we are going to focus on predicting house prices with the tools described in the chapter. The interest in predicting house prices is not new, but it's has proven to be a quite challenging problem where machine learning may have some interesting input. For example the  data science competitions company \href{https://www.kaggle.com}{Kaggle} is hosting a competition desined to improve \href{https://www.kaggle.com/c/zillow-prize-1}{Zillow's Home Value} estimates with a prize of one million dollars.
 
We consider the data set included in the  \texttt{McSpatial package} \citep{mcmspatial} for \texttt{R}. The data includes sales prices, structural characteristics and geo-location of single-family homes on the Far North Side of Chicago sold in 1995 and 2005. We randomly divided the sample into a trianing and a testing sample.




```r
load("house_data.rda")
ls()
```

```
## [1] "test"  "train"
```


```r
stargazer::stargazer(train, header=FALSE, type='latex')
```

```
## 
## \begin{table}[!htbp] \centering 
##   \caption{} 
##   \label{} 
## \begin{tabular}{@{\extracolsep{5pt}}lccccccc} 
## \\[-1.8ex]\hline 
## \hline \\[-1.8ex] 
## Statistic & \multicolumn{1}{c}{N} & \multicolumn{1}{c}{Mean} & \multicolumn{1}{c}{St. Dev.} & \multicolumn{1}{c}{Min} & \multicolumn{1}{c}{Pctl(25)} & \multicolumn{1}{c}{Pctl(75)} & \multicolumn{1}{c}{Max} \\ 
## \hline \\[-1.8ex] 
## year & 2,704 & 1,999.963 & 5.001 & 1,995 & 1,995 & 2,005 & 2,005 \\ 
## lnland & 2,704 & 8.305 & 0.386 & 6.633 & 8.221 & 8.509 & 10.076 \\ 
## lnbldg & 2,704 & 7.178 & 0.287 & 6.174 & 6.985 & 7.364 & 8.356 \\ 
## rooms & 2,704 & 5.780 & 1.229 & 2 & 5 & 6 & 12 \\ 
## bedrooms & 2,704 & 3.017 & 0.744 & 1 & 3 & 3 & 7 \\ 
## bathrooms & 2,704 & 1.418 & 0.514 & 1.000 & 1.000 & 1.500 & 5.000 \\ 
## centair & 2,704 & 0.350 & 0.477 & 0 & 0 & 1 & 1 \\ 
## fireplace & 2,704 & 0.159 & 0.365 & 0 & 0 & 0 & 1 \\ 
## brick & 2,704 & 0.670 & 0.470 & 0 & 0 & 1 & 1 \\ 
## garage1 & 2,704 & 0.308 & 0.462 & 0 & 0 & 1 & 1 \\ 
## garage2 & 2,704 & 0.449 & 0.498 & 0 & 0 & 1 & 1 \\ 
## dcbd & 2,704 & 9.697 & 1.719 & 5.245 & 8.459 & 10.975 & 13.587 \\ 
## rr & 2,704 & 0.161 & 0.368 & 0 & 0 & 0 & 1 \\ 
## yrbuilt & 2,704 & 1,935.040 & 21.704 & 1,876 & 1,919.8 & 1,952 & 1,991 \\ 
## latitude & 2,704 & 41.987 & 0.015 & 41.956 & 41.975 & 41.997 & 42.022 \\ 
## longitude & 2,704 & $-$87.745 & 0.050 & $-$87.834 & $-$87.790 & $-$87.699 & $-$87.647 \\ 
## lnprice & 2,704 & 12.405 & 0.524 & 10.166 & 11.951 & 12.835 & 13.825 \\ 
## age & 2,704 & 64.923 & 22.325 & 4 & 48 & 82 & 129 \\ 
## \hline \\[-1.8ex] 
## \end{tabular} 
## \end{table}
```

## Linear Regression
### MSE and regression

The objective then is to be able to get the best prediction of house prices. We begin by using a simple model with no covariates, just a constant


```r
model1<-lm(lnprice~1,data=train)
summary(model1)
```

```
## 
## Call:
## lm(formula = lnprice ~ 1, data = train)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -2.23948 -0.45415  0.01787  0.42935  1.42013 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 12.40533    0.01008    1230   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.5244 on 2703 degrees of freedom
```

In this case our prediction for the log price is the average train sample average


$$
\hat{y}=\hat{\beta_1}=\frac{\sum y_i}{n}=m
$$


```r
coef(model1)
```

```
## (Intercept) 
##    12.40533
```

```r
mean(train$lnprice)
```

```
## [1] 12.40533
```

But we are concernded on predicting well our of sample, so we need to evaluate our model in the testing data 


```r
test$model1<-predict(model1,newdata = test)
with(test,mean((lnprice-model1)^2))
```

```
## [1] 0.2857339
```

Then the $MSE=E(y-\hat{y})=E(y-m)=$\Sexpr{with(test,mean((lnprice-model1)^2))}. This is our starting point, then the question is how can we improve it.

### Complexity

To improve our prediction we can start adding variables and thus \textit{building} $f$. The standard approach to build $f$ would be using a hedonic house price function derived directly from the theory of hedonic pricing ([@rosen1974hedonic]). In its basic form the hedonic price function is linear in the explanatory characteristics

$$
y=\beta_1+\beta_2 x_2 + \dots + \beta_K x_k +u
 $$

where $y$ is ussually the log of the sale price, and $x_1  \dots x_k$ are attributes of the house, like  structural characteristics and it's location. So estimating an hedonic price function seems a good idea to start with. 
However, the theory says little on what are the relevant attributes of the house. So we are going to procede with one foot in the theory and one foot in the data, to guide us in building $f$.

We begin by showing that the simple inclusion of a single covariate reduces the MSE with respect to the \textit{naive} model that used the sample mean.


```r
model2<-lm(lnprice~bedrooms,data=train)
test$model2<-predict(model2,newdata = test)
with(test,mean((lnprice-model2)^2))
```

```
## [1] 0.2836981
```

What about if we include more variables? 


```r
model3<-lm(lnprice~bedrooms+bathrooms+centair+fireplace+brick+age,data=train)
test$model3<-predict(model3,newdata = test)
with(test,mean((lnprice-model3)^2))
```

```
## [1] 0.255731
```



```r
model4<-lm(lnprice~bedrooms+bathrooms+centair+fireplace+brick+poly(age,2),data=train)
test$model4<-predict(model4,newdata = test)
```



Note that the MSE is once more reduced. What about if we include some non linear variables, like $age$ and $age^2$?. Then the MSE for model 3 goes from  \Sexpr{with(test,mean((lnprice-model3)^2))} to \Sexpr{with(test,mean((lnprice-model4)^2))}. In this case the MSE gets slightly worse, showing how we are subject to the bias/variance trade off.




### Non-linear models


In practice empirical researches have recognized that hedonic house functions are likely to be nonlinear in structural characteristics, specially when it comes to continuous measures of location variables such as distance from the city center [@mcmillen2010estimation].



* Idea Show examples of other models
  

It is clear that we have a wide *pallete* of models to choose from. But the question is how do we choose from. And this is where supervised machine learning can give some guidance.

### Comparison on the models




