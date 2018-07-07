###################################################################################################
## Exploratory and Descriptive Analysis on Loan Data Set
###################################################################################################

setwd("C:/StudyMaterial/ABI/ABI_Project/loan-data")

#read the dataset in R
loan_dataset <- read.csv("LoanDataSet.csv",header=TRUE,na.strings=c("","n/a","NA"),stringsAsFactors = FALSE)

#Data Summary
dim(loan_dataset)

names(loan_dataset)

#missing data
colnames(loan_dataset)[apply(is.na(loan_dataset), 2, any)]
library(VIM)
par(mfrow=c(2,2))
a <- aggr(loan_dataset)
a
summary(a)

#missing data imputation using MICE
#install.packages("mice")
library(mice)
# use mice with predictive mean matching
loan_dataset_imputed <- mice(loan_dataset,m=5,maxit=50,meth='cart',seed=500)
summary(loan_dataset_imputed)
str(loan_dataset_imputed)

#Histogram using ggplot2
library(ggplot2)
ggplot(loan_dataset, aes(x = loan_dataset$loan_status)) +
  geom_histogram(stat = "count")

#Normalising the loan status
loan_dataset$loan_status=ifelse(loan_dataset$loan_status=="Fully Paid" | loan_dataset$loan_status=="Does not meet the credit policy. Status:Fully Paid","PAID",loan_dataset$loan_status)
loan_dataset$loan_status=ifelse(loan_dataset$loan_status=="Charged Off"|loan_dataset$loan_status=="Default"|loan_dataset$loan_status=="Late (31-120 days)"|loan_dataset$loan_status=="Late (16-30 days)"|loan_dataset$loan_status=="Does not meet the credit policy. Status:Charged Off","DEFAULT",loan_dataset$loan_status)

unique(loan_dataset$loan_status)
table(loan_dataset$loan_status)
typeof(loan_dataset$loan_status)

#Dropping emp_title and memberId since irrevalent field
names(loan_dataset)
loan_dataset=loan_dataset[,-9]
loan_dataset=loan_dataset[,-1]
dim(loan_dataset)

#Convert relevant fields to FACTORS
loan_dataset$loan_status = as.factor(loan_dataset$loan_status)
loan_dataset$emp_length = as.factor(loan_dataset$emp_length)
loan_dataset$home_ownership = as.factor(loan_dataset$home_ownership)
loan_dataset$verification_status = as.factor(loan_dataset$verification_status)
loan_dataset$pymnt_plan = as.factor(loan_dataset$pymnt_plan)
loan_dataset$purpose = as.factor(loan_dataset$purpose)
loan_dataset$grade = as.factor(loan_dataset$grade)


#for emp_length
str(loan_dataset$emp_length)
levels(loan_dataset$emp_length)

#for application_type
unique(loan_dataset$application_type)
table(loan_dataset$application_type)

#for loan_status
unique(loan_dataset$loan_status)
table(loan_dataset$loan_status)
levels(loan_dataset$loan_status)

#subset the data to select records which are relevant for our model
loan_paid=subset(x = loan_dataset,loan_dataset$loan_status=="PAID")
loan_default=subset(x = loan_dataset,loan_dataset$loan_status=="DEFAULT")

loan_paid$loan_status = droplevels(loan_paid$loan_status)
loan_default$loan_status = droplevels(loan_default$loan_status)

table(loan_paid$loan_status)
table(droplevels(loan_default$loan_status))

dim(loan_paid)
dim(loan_default)

########################### Creating subsets from PAID part ###############################

subset_paid=sample(x = nrow(loan_paid),size = round(0.8*nrow(loan_paid))) 

train_paid=loan_paid[subset_paid,]   #80% Train Data
dim(train_paid)

validation_paid=loan_paid[-subset_paid,]  #20% Validation Set
dim(validation_paid)

########################### Creating subsets from DEFAULT part ###############################

subset_default=sample(x = nrow(loan_default),size = round(0.8*nrow(loan_default)))  

train_default=loan_default[subset_default,]   #80% Train Data
dim(train_default)

validation_default=loan_default[-subset_default,]  #20% Validation Set
dim(validation_default)

################# Merge both sets of Data to obtain Validation Set ###########################

training_dataSet=rbind(train_paid,train_default)
dim(training_dataSet)

validation_set=rbind(validation_paid,validation_default)
dim(validation_set)

##############################################################################################
################################ Logistic Regression #########################################
##############################################################################################

names(training_dataSet)
dim(training_dataSet)
table(training_dataSet$loan_status)
training_dataSet$loan_status = factor(training_dataSet$loan_status)

str(training_dataSet)

logistic_loanData=glm(as.factor(training_dataSet$loan_status) ~ training_dataSet$out_prncp+
                         training_dataSet$total_rec_prncp+training_dataSet$loan_amnt+
                         training_dataSet$annual_inc+training_dataSet$int_rate+
                         training_dataSet$funded_amnt+
                         training_dataSet$last_pymnt_amnt
                       ,family = binomial,data=training_dataSet)

summary(logistic_loanData)

##############################################################################################
################################ DECISION TREE ###############################################
##############################################################################################

#Using TREE Package
library("ISLR")
#install.packages("tree")
library("tree")

dim(training_dataSet)
table(training_dataSet$loan_status)
table(validation_set$loan_status)
names(validation_set)
names(training_dataSet)

tree.loanStatus = tree(training_dataSet$loan_status ~ ., data = training_dataSet)

str(tree.loanStatus)
summary(tree.loanStatus)

par(mfrow=c(1,1))
plot(tree.loanStatus)
text(tree.loanStatus, pretty = 0)

######################################################################
########### Applying the decision tree on validation set #############
######################################################################

tree.pred = predict(tree.loanStatus, validation_set, type = "class")
with(validation_set, table(tree.pred, validation_set$loan_status))

## This tree was grown to full depth, and might be too variable. We now use CV to prune it.

cv.loanStatus = cv.tree(tree.loanStatus, FUN = prune.misclass)
cv.loanStatus
plot(cv.loanStatus)

prune.loanStatus = prune.misclass(tree.loanStatus, best = 5)
plot(prune.loanStatus)
text(prune.loanStatus, pretty = 0)
summary(prune.loanStatus)

#Applying the pruned tree on validation set
tree.pruned_pred = predict(prune.loanStatus, validation_set, type = "class")
with(validation_set, table(tree.pruned_pred, validation_set$loan_status))

################################################################################
############################# Using RPART Package ##############################
################################################################################

library(rpart)
training_tree_dataSet = training_dataSet[-c(12,32,30)]

fit_decision=rpart(formula =  loan_status~.,method ="class",data =  training_tree_dataSet,control = rpart.control(minsplit = 10,xval=20,maxsurrogate = 10),parms =list(split="gini"))
summary(fit_decision)

t_pred = predict(fit_decision,training_tree_dataSet,type="class")
t = training_tree_dataSet$loan_status
accuracy = sum(t_pred == t)/length(t)
print(accuracy)

plot(fit_decision)
text(fit_decision,pretty = 0)

#################################################################################