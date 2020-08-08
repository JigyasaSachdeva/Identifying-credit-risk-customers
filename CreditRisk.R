# Jigyasa Sachdeva
# Data Mining- IDS 572
#Identifying customers with a potential credit risk
#-------------------------------------------------------------------------------

#Reading file
Bank.data <- read.csv("~/Desktop/Bank.data.csv")
View(Bank.data)




#Data Pre-processing

#To check for null values
library(funModeling)
df_status(Bank.data)
#Saving.account has 18.3% null values
#Checking.account has 39.4% null values
#Since both are factor variables, imputation with mean and median cannot occur
#Imputing 183 and 394 observations with mode would make the final model bias

#Hence removing all null values from the data:
b <- na.omit(Bank.data)
#left with 522 observations

#Checking null proportion again to be sure: 
df_status(b)
#No null values confirmed

#Checking structure of the data
str(b)
#ID is an integer variable which would not be needed in model
b$ID <- NULL
#Removed ID column
#Age, credit.amount and duration are integer variables
#Checking outliers
out <- boxplot.stats(b$Age)$out #18 observations
out1 <- boxplot.stats(b$Credit.amount)$out #36 observations
out2 <- boxplot.stats(b$Duration)$out #8 observations
#Since outlier values are very less removing them from the data:

#Age:
b_out <- ifelse(b$Age %in% out, NA, b$Age)
b_out <- na.omit(b_out) #Down to 504 observations
b1 <- b[b$Age %in% b_out, ] #Keeping only the non Null values
rm(b) #Removing previous dataframe

#Credit.amount:
b_out1 <- ifelse(b1$Credit.amount %in% out1, NA, b1$Credit.amount)
b_out1 <- na.omit(b_out1) #Down to 469 observations
b2 <- b1[b1$Credit.amount %in% b_out1, ] #Keeping only the non Null values
rm(b1) #Removing previous dataframe

#Duration:
b_out2 <- ifelse(b2$Duration %in% out2, NA, b2$Duration)
b_out2 <- na.omit(b_out2) #Down to 464 observations
b3 <- b2[b2$Duration %in% b_out2, ] #Keeping only the non Null values
rm(b2) #Removing previous dataframe

#Final prepared dataset: b3

data <- b3 #using data 
rm(b3) #Removing b3
str(data)
#Job is an integer, should be a factor
data$Job <- as.factor(data$Job)
str(data$Job) #factor

str(data)
#Target is a factor variable, given as integer
data$Target <- as.factor(data$Target)
str(data$Target) #Done




#Univariate Analysis

#1
#Checking 5 number summary for credit amount:
summary(data$Credit.amount)
#Output:
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#276    1275    2150    2631    3592    7966 

#After removing outliers and null values:
#Credit amount ranges from 276 to 7966
#The median is 2150 which is less than the mean 2631
#hence credit amount is right skewed

#2
#Duration and credit amount
cor.test(data$Duration, data$Credit.amount)

#Output:
#Pearson's product-moment correlation
#data:  data$Duration and data$Credit.amount
#t = 16.963, df = 462, p-value < 2.2e-16
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
# 0.5600590 0.6726068
#sample estimates:
      cor 
#0.6195065 

#Correlation is 61%, and hence, yes they are correlated

#Possible solutions:
#1- If using a model which is not robust to correlation: 
#Removing either variable depending on which provides better prediction with respect to target
#2- If using a model which is robust to correlation (decision tree, randome forest):
# We can use both
#3- Normalizing the variable





#Bivariate Analysis
     
#3
#Distribution of credit amount for good and bad instances
boxplot(data$Credit.amount, data$Target, 
              col = c("coral", "steelblue"),
              main= "Credit amoiunt v/s Target variable",
              xlab= "Target",
              ylab= "Credit amount")    
      
#4
#Table of housing types v/s Target
t <- table(data$Housing, data$Target, 
      dnn= c("Housing Types", "Target"))
t
#               Target
#Housing Types   0   1
#         free  26  22
#         own  117 199
#         rent  50  50

#5
#The missing values have been handled during data pre-processing already

#6
#The outliers have also been taken care of during data pre-processing





#Data Modeling


#7
#Splitting data in train and test: 70 and 30
index <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
#1 and 2 repeated 70 and 30% of the time as indexes
training_data <- data[index ==1, ]
test_data <- data[index ==2,]
#1 and 2 index assigned to train and test


#8 
data1 <- data
options(scipen=99) #Removing scientific notation
rm(data)
#Forward logistic regression:
null = glm(Target~1, data= training_data, family = "binomial") 
full = glm(Target~., data= training_data, family = "binomial") # includes all the variables
# We can perform forward selection using the command:
#Forward selection
fwd_mod <- step(null, scope=list(lower=null, upper=full), direction="forward")
fwd_mod
summary(fwd_mod)
#The summary shows that the null deviance is 429.11 
#Which is the error of prediction without any variable (common classifier)
#The residual deviance shows that this error has reduced to 358.35
#Even though the model isn't that good on fit, we attain the important variables


#9
#AIC measure is 376.35
#AIC is the -2log(likelihood) + 2P
#It is the measure for tarde off between goodness of fit and number of variables in the model


#10
#The important variables can be seen from the table with p-values being significant
summary(fwd_mod)
#Output: 
#Estimate Std. Error z value      Pr(>|z|)    
#(Intercept)               -0.2296977  0.5071452  -0.453       0.65060    
#Duration                  -0.0996812  0.0173532  -5.744 0.00000000923 ***
#Credit.amount              0.0004376  0.0001041   4.202 0.00002642983 ***
#Saving.accountsmoderate    0.1429962  0.3934134   0.363       0.71625    
#Saving.accountsquite rich  2.2418300  1.0785095   2.079       0.03765 *  
#Saving.accountsrich        2.0182686  1.0630446   1.899       0.05762 .  
#Checking.accountmoderate   0.5505850  0.2760249   1.995       0.04608 *  
#Checking.accountrich       1.2530453  0.4610921   2.718       0.00658 ** 
#Age                        0.0256031  0.0129014   1.985       0.04720 * 

#Varibles that are used in the mode: Duration, Credit.amount, Savings.account, Checking.account, Age
#Variables with significant p-value: Duration > Credit.amount > Checking.account> Savings.account ( a few categpries ) > Age



#11
#C5 decision tree
#Decision tree using Information gain
library(rpart)
#As required to run the rpart function
library(rpart)
rpart_mod <- rpart(Target~., 
                    data = training_data, 
                    control = rpart.control
                    (minsplit= 20, 
                    minbucket = 7, 
                    cp = 0.01), 
                    parms = list(split = "information"),
                    method = "class")

#minsplit: minimum number of observations required to split further in the model
#minbucket: minimum number of observations that should be in the leaf node to do the required split
#minsplit and minbucket are used to pre-prune the tree
#We can tune the number of observations required for further split (keeping in mind the required support)

#cp: trade off between error and size of the tree
#Equation: min(error(tree) - cp|tree|)
#We need to minimize this trade off to avoid underfitting (when the error is hig)
#As well as avoid overfitting (when the tree size is high)

#To find optimal values: we perform parameter tuning (a part of cross validation)
#But before that finding cp values and behaviour:
printcp(rpart_mod)
#We can see that as the number of split sincrease, xerror decreases
#Finding optimal cut off from the plot:
plotcp(rpart_mod)
#The minimum relative error is seen at cp = 0.011
opt <- which.min(rpart_mod$cptable[ ,"xerror"]) 
cp1 <- rpart_mod$cptable[opt, "CP"]
cp1 #0.01


#Performing parameter tuning for the rest of the parameters:
#Considering concern as maximizing accuracy:

mins <- seq(from = 1, to = 50, by=1)
#Assigning these values to minsplit to be iterated through the function:
tree <- function(maxd)
{
  rpart_mod <- rpart(Target~., 
                     data = training_data, 
                     control = rpart.control
                     (minsplit= mins, 
                      cp = 0.01), 
                     parms = list(split = "information"),
                     method = "class") 
  predict_rpart <- predict(rpart_mod, newdata = test_data, type = "class")
  library(caret)
  conf <- confusionMatrix(predict_rpart, test_data$Target)
  return(conf$overall[1])
}

lapply(mins, tree)
#Almost similar, lower accuracy at the large values
#Selecting 20 as minsplit (default)



minb <- seq(from = 1, to = 20, by=1)
#Assigning these values to minsplit to be iterated through the function:
tree <- function(maxd)
{
  rpart_mod <- rpart(Target~., 
                     data = training_data, 
                     control = rpart.control
                     (minsplit= 20, 
                      minbucket = minb,
                      cp = 0.01), 
                     parms = list(split = "information"),
                     method = "class")
  predict_rpart <- predict(rpart_mod, newdata = test_data, type = "class")
  library(caret)
  conf <- confusionMatrix(predict_rpart, test_data$Target)
  return(conf$overall[1])
}

lapply(minb, tree)
#Similar values, hence keeping default



#12
#After parameter tuning: 
rpart_mod <- rpart(Target~., 
                   data = training_data, 
                   control = rpart.control
                   (minsplit = 20,
                    minbucket =7,
                     cp = 0.010), 
                   parms = list(split = "information"),
                   method = "class")

rpart_mod
#produces a bunch of rules
#Finding decision rules on the basis of maximum support and confidence
library(rpart.plot)
rpart.plot(rpart_mod)
#Plotting to visualize better

#4) Age< 31.5 30 5 0 (0.83333333 0.16666667) *
#Confidence: 83.3%
#Support = 30
#Rule:
#If (Duration>= 27.5) & (Age <31.5); Then '0': Bad

#51) Duration< 11.5 44   8 1 (0.18181818 0.81818182) *
#Support= 44 (Number of observations)
#Confidence = 0.818
#decision rule:

#If (Duration < 28) & (Savings.account = little.moderate) & (Credit amount < 5011) & (Purpose ! = education) & (Duration< 11.5)
#Then: 1: Good


#13
#Important variables: (in the prder of most important to least)
#Duration
#Savings.account
#Age
#Credit.amount

#Can be seen from the plot: from root node to below


#14
#SVM Model
library(e1071)
svm.model <- svm(Target ~ ., data = training_data, cost = 100, gamma = 1,
                 type = "C-classification" , kernel = "linear", probability =T)
summary(svm.model)
#Support vectors are 212

#cross validation to picking gamma:
g1 <- seq(0,1, by = 0.1)
#Changing the gamma parameter
for(g in g1)
{
svm.model <- svm(Target ~ ., data = training_data, cost = 100, gamma = g,
                 type = "C-classification" , kernel = "linear", probability =T)
#type is c-classification because factors
#kernel is kept linear
#probability true to compute later
#cost is 100 static
#gammea is g and will be tuned in the loop
pred <- predict(svm.model, newdata = test_data)
pred <- as.factor(pred)
c <- confusionMatrix(pred, test_data$Target)
print(c$byClass[1])
#byClass[1] is accuracy and printing that for each iterated value of gamma here
}

#Same accuracy for all gamma:
#Choosing 1: as sensitivity is same for all values of gamma passed

 

#15

fwd_mod <- step(null, scope=list(lower=null, upper=full), direction="forward")
predicted <- predict(fwd_mod, newdata = test_data)
class <- as.factor(ifelse(predicted>0.5, 1, 0))

accuracy <- function(predicted, actual)
{
  conf <- confusionMatrix(predicted, actual)
  acc <- (conf$table[1,1]+conf$table[2,2]) /(conf$table[1,1]+conf$table[1,2] + conf$table[2,1] + conf$table[2,2])
  #accuracy is the sum of diagonal divided by the total
  return(acc)
}
accuracy(class, test_data$Target)
#0.5533333 


sensitive <- function(predicted, actual)
{
  conf <- confusionMatrix(predicted, actual)
  sens <- conf$table[1,1]/(conf$table[1,1]+conf$table[2,1])
  #sensitivity is true positive/true positive+false negative
  return(sens)
}
sensitive(class, test_data$Target)
#0.6551724 


prec <- function(predicted, actual)
{
  conf <- confusionMatrix(predicted, actual)
  prec <- conf$table[1,1]/(conf$table[1,1]+conf$table[1,2])
  #precision is true true positive/true positive+false positive
  return(prec)
}
prec(class, test_data$Target)
#0.4470588


#16
#Since Bad credit risk class is more important than Good credit risk class
#Because someone with a bad credit risk identified as good is more harmful than losing a potential customer
#But that doesn;t mean that the other class isn't imprtant

#hence using Cost based method where I have assigned penalty = 7 to bad class missclasification and 3 to good class misclassification
p1=7
p2=3


#Logistic regression
fwd_mod <- step(null, scope=list(lower=null, upper=full), direction="forward")
predicted <- predict(fwd_mod, newdata = test_data)
class <- as.factor(ifelse(predicted>0.5, 1, 0))
#p10
false_negative <- confusionMatrix(class, test_data$Target, positive = '1')$table[1,2]
#p01
false_positive <- confusionMatrix(class, test_data$Target, positive = '1')$table[2,1]
cost <- p1*false_negative + p2*false_positive
#Cost is computed as respective penalties multiplied to false positive and false negatives
#389


#Decision Tree
rpart_mod <- rpart(Target~., #all variables
                   data = training_data, #train data
                   control = rpart.control 
                   (minsplit = 20, 
                     minbucket =7,
                     cp = 0.010), #keeping cp value as 0.10 (optimal as found)
                   parms = list(split = "information"),
                   method = "class")
predict_rpart <- predict(rpart_mod, newdata = test_data, type = "class")
#p10
false_negative <- confusionMatrix(predict_rpart, test_data$Target, positive = '1')$table[1,2]
#p01
false_positive <- confusionMatrix(predict_rpart, test_data$Target, positive = '1')$table[2,1]
cost <- p1*false_negative + p2*false_positive
#425



#SVM
svm.model <- svm(Target ~ ., data = training_data, cost = 100, gamma = 1,
                 type = "C-classification" , kernel = "linear", probability =T)
pred <- predict(svm.model, newdata = test_data)
pred <- as.factor(pred)
#p10
false_negative <- confusionMatrix(pred, test_data$Target, positive = '1')$table[1,2]
#p01
false_positive <- confusionMatrix(pred, test_data$Target, positive = '1')$table[2,1]
cost <- p1*false_negative + p2*false_positive
#401

#hence out of the three models, on the basis of cost: the best method is logistic regression



#17

library(ROCR)

# Logistic Regression 
fwd_mod <- step(null, scope=list(lower=null, upper=full), direction="forward")
predicted <- predict(fwd_mod, newdata = test_data)
pred <- predict(fwd_mod, test_data, type="response")
#predicting probabilities of the model with the tesgt data 
pref <- prediction(pred,test_data$Target)
#calculating perf using the prediction formula 
# ROC Curve 
eval <- performance(pref,"tpr","fpr")
plot(eval,main="Evaluating a Model through ROC Curve",col=1,lwd=2) 


# Decision Tree
rpart_mod <- rpart(Target~., 
                   data = training_data, 
                   control = rpart.control
                   (minsplit = 20,
                     minbucket =7,
                     cp = 0.010), 
                   parms = list(split = "information"),
                   method = "class")
predict_rpart <- predict(rpart_mod, newdata = test_data, type = "prob")
p <- predict_rpart[,2]
pref1 <- prediction(predict_rpart[,2], test_data$Target)
eval1 <- performance(pref1, "tpr", "fpr")
plot(eval1, col=3, add=TRUE,lwd=2)

legend(0.6, 0.6, c("Logistic","Decision Trees"),1:2)
#Logistic is better than decision tree
#But more or less the same

# SVM
svm.model <- svm(Target ~ ., data = training_data, cost = 100, gamma = 1,
                 type = "C-classification" , kernel = "linear", probability =T)
pred1 <- predict(svm.model, newdata = test_data, probability = TRUE)
str(pred1)
pred1
p <- pred1$probabilities # If this works:
#Couldn't compute probabilities: else would have put it here:
pref1 <- prediction(p, test_data$Target)
eval1 <- performance(pred1, "tpr", "fpr")
plot(eval1, col=2, add=TRUE,lwd=2)


















