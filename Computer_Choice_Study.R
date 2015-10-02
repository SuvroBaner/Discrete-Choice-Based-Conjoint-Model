############################################## Computer Choice Study ########################################

# In 1998 Microsoft introduced a new operating system. Computer manufacturers were interested in making predictions
# about the personal computer marketplace. To help manufacturers understand the market for personal computers,
# we conducted a computer choice study involving the following six Attributes and their Levels.

### Attribute ############### Level Code ################### Level Description ###############
# Brand                       Apple             Manufacturer : Apple
#                             Compaq            Manufacturer : Compaq
#                             Dell              Manufacturer : Dell
#                             Gateway           Manufacturer : Gateway
#                             HP                Manufacturer : HP
#                             IBM               Manufacturer : IBM
#                             Sony              Manufacturer : Sony
#                             Sun               Manufacturer : Sun Microsystems
# Compatibility               1                 65 % Compatible
#                             2                 70 % Compatible
#                             3                 75 % Compatible
#                             4                 80 % Compatible
#                             5                 85 % Compatible
#                             6                 90 % Compatible
#                             7                 95 % Compatible
#                             8                 100 % Compatible
# Performance                 1                 Just as fast
#                             2                 Twice as fast
#                             3                 Three times as fast
#                             4                 Four times as fast
# Reliability                 1                 As likely to fail
#                             2                 Less likely to fail
# Learn                       1                 4 hours to learn
#                             2                 8 hours to learn
#                             3                 12 hours to learn
#                             4                 16 hours to learn
#                             5                 20 hours to learn
#                             6                 24 hours to learn
#                             7                 28 hours to learn
#                             8                 32 hours to learn
# Price                       1                 $ 1,000
#                             2                 $ 1,250
#                             3                 $ 1,500
#                             4                 $ 1,750
#                             5                 $ 2,000
#                             6                 $ 2,250
#                             7                 $ 2,500
#                             8                 $ 2,750

# The computer choice study was a nationwide study. We identified people who expressed an interest in buying 
# a new personal computer within the next year. Consumers volunteering for the study were sent questionnaire
# booklets, answer sheets, and postage-paid return mail envelopes. Each respondent received $ 25 for participating 
# in the study. 
# The survey consisted of 16 pages(i.e. Choice sets), with each page showing a choice set of 4 product profiles.
# For each choice set, survey participants were asked first to select the computer they most preferred,
# and second, to indicate whether or not they would actually buy that computer.

# Each product profile, defined in terms of its attributes, is associated with the consumer's binary response
# (0 = not chosen, 1 = chosen) and (0 = not buy, 1 = buy)

########################### Hierarchical Bayes Model ########################################

### Prediction Problem : To predict individual choice.

### Model Train & Test regimen : In this context we build predictive models on 12 choice sets and test on 4 choice sets. 
# Train choice sets: 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14 and 16
# Test choice sets : 3, 7, 11, and 15

# So, in total with 16 choice sets of four each, we have 16 x 4 = 64 product profiles for each individual in the study.
# Training data includes 12 x 4 = 48 product profiles i.e 48 rows for each individual,
# Test data includes 4 x 4 = 16 product profiles i.e. 16 rows for each individual.

install.packages("ChoiceModelR")  # Estimates coefficients of a Hierarchical Bayes multinomial logit model
install.packages("caret") # for confusion matrix function

library(ChoiceModelR)
library(caret)

# Read in the data from the case study of Consumer Choice.

complete.data.frame = read.csv("computer_choice_study.csv")

# id : Consumer's ID (i.e. distinct consumers who responded)
length(unique(complete.data.frame$id))  # Total 224 consumers responded answering for all the 16 Choice Sets.

# profile : value from 1 to 64, representing 64 different product profiles given to each consumer for the survey.

# setid : value from 1 to 16, i.e 16 different Choice sets given to each consumer for the survey.
# Note: each choice set has 4 product profiles.

# position : How the product profiles are alligned in each Choice Set (or each page of the survey)

# choice : 0 or 1 for each Choice set. (One product profile has to be chosen as the best in that Choice Set)

# buy : 0 or 1 for each Choice set (Consumer may not select any.)

# Other variables are explained above

dim(complete.data.frame)  # 14,336 rows and 12 columns

print.digits = 2

# user-defined function for printing conjoint measures.

if (print.digits == 2)
  pretty.print = function(x) {sprintf("%1.2f", round(x, digits = 2))}

if (print.digits == 3)
  pretty.print = function(x) {sprintf("%1.3f", round(x, digits = 3))}

# Note: sprintf() is a wrapper for the C function sprintf

# Set up sum contrasts for effects coding

options(contrasts = c("contr.sum", "contr.poly"))


# Employ a training-and-test regimen across survey sets/items

unique.set.ids = unique(complete.data.frame$setid)

test.set.ids = c("3", "7", "11", "15")  # select four sets/items

training.set.ids = setdiff(unique.set.ids, test.set.ids)  # set operation (difference)

training.data.frame = subset(complete.data.frame, subset = (setid %in% training.set.ids))
dim(training.data.frame)  # 10,752 rows and 12 columns

test.data.frame = subset(complete.data.frame, subset = (setid %in% test.set.ids))
dim(test.data.frame)  # 3,584 rows and 12 columns

UniqueID = unique(training.data.frame$id)  # as stated earlier, there are 224 unique consumers in this study.

# initializing with 0 value, below is a matrix of 224 rows and 13 columns.

cc.priors = matrix(0, nrow = length(UniqueID), ncol = 13)

colnames(cc.priors) = c("A1B1", "A1B2", "A1B3", "A1B4", "A1B5", "A1B6", "A1B7", "A1B8",
                        "A2B1", "A3B1", "A4B1", "A5B1", "A6B1")
# here A : Attributes (total 6 attributes, so A1:A6)
# here B : Brands (total 8 brands, so B1:B8)

# The actual names are as follows-

AB.names = c("Apple", "Compaq", "Dell", "Gateway", "HP", "IBM", "Sony", "Sun",
             "Compatibility", "Performance", "Reliability", "Learning", "Price")



### Hierarchical Bayes Part-worth extimation ###

# Set up run parameters for the Markov Chain Monte Carlo (MCMC) to estimate the HB model.

# To obtain useful conjoint measures for individuals, we employ HB methods with the following constraints:-
# Other than "Brand" there are constraints upon the signs of attribute coefficients.
# For the Computer Choice Study, prior to fitting the model to the data, other than "Brand" we'll have
# "Compatibility", "Performance", and "Reliability" with the POSITIVE coefficients, whereas
# "Learn" and "Price" have negative coefficients.

# Using aggregate Beta estimates to get started.
truebetas = cc.priors

cc.xcoding = c(0, 1, 1, 1, 1, 1)  # the first variable is 'Brand' which is categorical and the others are continuous.

cc.attlevels = c(8, 8, 4, 2, 8, 8) # number of attribute levels each attribute has (also explained in the begining)

##### Below are the CONSTRAINTS denoted by c1:c6 

# BRAND : no constraint for order on brand, so 8 x 8 matrix of zeros.
c1 = matrix(0, ncol = 8, nrow = 8)

# COMPATIBILITY : ordered higher numbers are better , (continuous attributes have 1x1 matrix representation)
c2 = matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# PERFORMANCE : ordered higher numbers are better
c3 = matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# RELIABILITY : ordered higher numbers are better
c4 = matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# LEARNING : higher the learning time, lower the value to the consumer.
c5 = matrix(-1, ncol = 1, nrow = 1, byrow = TRUE)

# PRICE : higher the price, lower the value to the customer.
c6 = matrix(-1, ncol = 1, nrow = 1, byrow = TRUE)


cc.constraints = list(c1, c2, c3, c4, c5, c6)

### For MCMC to run, set the run parameters-
cc.mcmc = list(R = 10000, use = 2000)
# run parameters = 10000 total iterations with estimates based on last 2000

# run options
cc.options = list(none = FALSE, save = TRUE, keep = 1)

### Set up the data frame for analysis-

# Redefine set ids so they are a complete set 1-12 as needed for HB functions.

training.data.frame$newsetid = training.data.frame$setid

training.data.frame$newsetid = ifelse((training.data.frame$newsetid == 16), 3, training.data.frame$newsetid)

training.data.frame$newsetid = ifelse((training.data.frame$newsetid == 14), 7, training.data.frame$newsetid)

training.data.frame$newsetid = ifelse((training.data.frame$newsetid == 13), 11, training.data.frame$newsetid)


UnitID = training.data.frame$id
Set = as.integer(training.data.frame$newsetid)
Alt = as.integer(training.data.frame$position)
X_1 = as.integer(training.data.frame$brand)  # categories by brand
X_2 = as.integer(training.data.frame$compat) # integer values 1 to 8 for compatibility
X_3 = as.integer(training.data.frame$perform) # interger values 1 to 4 for performance
X_4 = as.integer(training.data.frame$reliab) # integer values 1 to 2 for reliability
X_5 = as.integer(training.data.frame$learn) # integer values 1 to 8 for learning
X_6 = as.integer(training.data.frame$price) # integer values 1 to 8 for price
y = as.numeric(training.data.frame$choice)  # using special response coding, i.e. 0 or 1
# Here the dependent variable "y" which is individual choice is a discrete variable.

cc.data = data.frame(UnitID, Set, Alt, X_1, X_2, X_3, X_4, X_5, X_6, y)
dim(cc.data)  # 10,752 rows and 10 columns

### Now starts the MCMC estimation , we'll use thge Choice Modeling in R ###

set.seed(9999)

out = choicemodelr(data = cc.data,
                   xcoding = cc.xcoding,
                   mcmc = cc.mcmc,
                   options = cc.options,
                   constraints = cc.constraints)


# Let us understand the Hierarchical Bayes model in this context.
# Although the choicemodelr does a multinomial logit model (MNL) but I'll also cover the linear model as well.

###### Linear HB Model-

# Yj = 1 if Zi > 0
# Yj = 0 if Zi <= 0

# Here Y = Individual consumer choice (discrete choice)
# j = all the unique consumer , here they are 224 in numbers.
# Z = Utility or value
# i = list of all 8 brands in context, i.e. i = ['Apple'...'Sun']

# Zi = B1*Brand + B2*Compat + B3*Perform + B4*Reliab + B5*Learn + B6*Price

# Note the combination of these two models form the Hierarchical Bayes model.
# The first one is the within unit and the second one gives the heterogenity.

###### Multinomial Logit HB Model-

# Pr(Yj = 1 | Zi) = exp(B1*Brand + B2*Compat + B3*Perform + B4*Reliab + B5*Learn + B6*Price) / 
#                   1 + exp(B1*Brand + B2*Compat + B3*Perform + B4*Reliab + B5*Learn + B6*Price)

# Pr(Yj = 0 | Zi) = exp(B1*Brand + B2*Compat + B3*Perform + B4*Reliab + B5*Learn + B6*Price) /
#                   1 + exp(B1*Brand + B2*Compat + B3*Perform + B4*Reliab + B5*Learn + B6*Price)

# Note our job is to estimate the posterior distribution i.e all the Beta coefficients of Brand, Compat ... Price
# and by estimating that we'll compute the probability of Yj i.e. probability of discrete choice.

# Note: While estimating the logit model the probability obtained will be the probability of logit and not Pr(Y = 1 | Z)
# So, the logit looks like 
# log(Pr(Y = 1 | Z) / (1 - Pr(Y = 1 | Z))) = B1*Brand + B2*Compat + B3*Perform + B4*Reliab + B5*Learn + B6*Price
# So, you can see that the logit is linear in the predictors (Z's) i.e. Brand, Compat ... Price


# As the posterior probability distribution of the HB model is difficult to calculate analytically,
# so, we'll create a Markov Chain of 10,000 iterations where the posterior distribution would reach an equilibrium state,
# and using Monte Carlo method, we would estimate the coefficients. We in short call it as MCMC.

# So, we've estimated the Betas using the MCMC method. Now let's gather them to make the inference.

# 'out' provides a list for the posterior parameter estimates for the runs sampled (use = 2000)


# the MCMC beta parameter estimates are traced on the screen as it runs

# Individual PART-WORTHS estimates are provided in the output file RBetas.csv
# and the final estimates are printed to RBetas.csv with colums labeled as follows-
# A1B1 = first attribute first level
# A1B2 = first attribute second level
# ...
# A2B1 = second attribute first level
# ...

# There is also a graphic output, During model estimation, estimates of mu 
# (mean of model coefficients from the distribution of heterogeneity) are plotted in the graphics window.

# Now after the MCMC estimation is done, start collecting the estimates.

# Gather data from HB posterior parameter distributions, 
# as we had imposed constraints on all continuous parameters, so we use 'betadraw.c' instead of 'betadraw'

names(out)
# [1] "betadraw"   "betadraw.c" "compdraw"   "loglike" 

dim(out$betadraw.c) # 224  12  2000, i.e 224 unique consumers, 12 predictors of the HB model and 2,000 MCMC sample estimates.

posterior.mean = matrix(0, nrow = dim(out$betadraw.c)[1],
                        ncol = dim(out$betadraw.c)[2])  # 224 rows and 12 columns

posterior.sd = matrix(0, nrow = dim(out$betadraw.c)[1],
                      ncol = dim(out$betadraw.c)[2])

for (index.row in 1:dim(out$betadraw.c)[1])  # rows, i.e. customer's unit id 
{
  for (index.col in 1:dim(out$betadraw.c)[2])  # columns, all the predictors , will soon explain
    {
      posterior.mean[index.row, index.col] = mean(out$betadraw.c[index.row, index.col, ])
  
      posterior.sd[index.row, index.col] = sd(out$betadraw.c[index.row, index.col, ])
    }
}


# Note this posterior.mean and posterior.sd matrices have 224 rows i.e each row for each individual.
# Lets understand the values well.
# mean(out$betadraw.c[1, 1, ]) = -14.16646
# this is for the first consumer, first brand (Apple) and last 2000 samples of Beta estimates (posterior distribution).
# We are calculating the mean of the last 2000 samples from the total 10000 samples (iterations) as
# we are simulating this process with the intention that the posterior distribution will go to an equilibrium.


# HB program uses effects coding for categorical variables and mean-centers continuous variables 
# across the levels appearing in the data.
# working with data for one respondent at a time we compute predicted choices for both the training
# and test choice sets.

create.design.matrix = function(input.data.frame.row)
{
  xdesign.row = numeric(12)
  
  if (input.data.frame.row$brand == "Apple")
    xdesign.row[1:7] = c(1, 0, 0, 0, 0, 0, 0)
  
  if (input.data.frame.row$brand == "Compaq")
    xdesign.row[1:7] = c(0, 1, 0, 0, 0, 0, 0)
  
  if (input.data.frame.row$brand == "Dell")
    xdesign.row[1:7] = c(0, 0, 1, 0, 0, 0, 0)
  
  if (input.data.frame.row$brand == "Gateway")
    xdesign.row[1:7] = c(0, 0, 0, 1, 0, 0, 0)
  
  if (input.data.frame.row$brand == "HP")
    xdesign.row[1:7] = c(0, 0, 0, 0, 1, 0, 0)
  
  if (input.data.frame.row$brand == "IBM")
    xdesign.row[1:7] = c(0, 0, 0, 0, 0, 1, 0)
  
  if (input.data.frame.row$brand == "Sony")
    xdesign.row[1:7] = c(0, 0, 0, 0, 0, 0, 1)
  
  if (input.data.frame.row$brand == "Sun")
    xdesign.row[1:7] = c(-1, -1, -1, -1, -1, -1, -1)
  
  xdesign.row[8] = input.data.frame.row$compat -4.5
  xdesign.row[9] = input.data.frame.row$perform -2.5
  xdesign.row[10] = input.data.frame.row$reliab -1.5
  xdesign.row[11] = input.data.frame.row$learn -4.5
  xdesign.row[12] = input.data.frame.row$price -4.5
  
  t(as.matrix(xdesign.row)) # return row of design matrix
}

### Evaluate performance in the training set

training.choice.utility = NULL  # initialize utility vector

# work with one row of respondent training data frame at a time
# create choice predictions using the individual part-worths

list.of.ids = unique(training.data.frame$id)

for (index.for.id in seq(along = list.of.ids)) # iterate thorugh all the 224 customers
{
  this.id.part.worths = posterior.mean[index.for.id, ]  # gather part-worths for each customer, starting from the 1st
  this.id.data.frame = subset(training.data.frame,
                              subset = (id == list.of.ids[index.for.id]))  # pick the data frame info for the 1st customer, the cust numbers are not sequential
  
  for (index.for.profile in 1:nrow(this.id.data.frame))  # iterate through all the 4 * 12 = 48 product profiles
  {
    temp = nrow(this.id.data.frame)  # just to check that all the 48 product proifiles are iterated.
    training.choice.utility = c(training.choice.utility,
                                create.design.matrix(this.id.data.frame[index.for.profile, ]) %*%
                                  this.id.part.worths)  # matrix multiplication of all the 12 attributes parth values.
  }
}

length(training.choice.utility) # 10,752
# This is exactly the number of rows for the training data set. i.e (4 * 12 * 224) = 10,752
# So, the vector 'training.choice.utility' has the Utility value (addition of all partworths of attributes)
# of each product profile. We added it because, the logit is linear in the predictor's coefficients.
# Note: Consumer has chosen 1 from 4 product profiles, in the same way from all product profiles.
# So, we had to compute the utility of all product profiles and understand why the customer had chosen one which he/she did.

# Now, we have to process this logit prob for all the product profiles to prob of individual choosing one.
# But before that let's verify our theory for more satisfaction.

# This is from the posterior mean matrix-


# Apple       Compaq      Dell        Gateway     HP          IBM         Sony
#-14.16646134  -6.3585370	7.991975737	7.43793231	9.45217471	18.17678074	-11.63756260	


# Compat        Perform       Reliab        Learn         Price
# 6.732032e-01	9.703769e+00	2.769425e-01	-0.0003804059	-1.695590660

# Note "Sun" brand is missing, but it is actually not as we have enforced contrasts coding so that all the part worths sum to zero
# within attribute. So, Sun would be 10.8963
sum(posterior.mean[1, 1:7])  # Sun would be -10.8963

create.design.matrix(this.id.data.frame[1, ])
#[,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12]
#[1,]   -1   -1   -1   -1   -1   -1   -1 -3.5  1.5   0.5   3.5   3.5

# Further lets investigate the utility values as well.
-10.8963*1 + 6.732032e-01*(-3.5) + 9.703769e+00*1.5 + 2.769425e-01*0.5 + (-0.0003804059)*3.5  + (-1.695590660)*3.5


# Utility of the 1st product profile is -4.494285


training.choice.utility[1] # -4.494287

# So, we have verified to our satisfaction that the training.choice.utility is the Logit Prob of Utility.
# and the value matches to perfection.

training.choice.utility

source("R_utility_program_2.R")


# choice.set.predictor() is an user-defined function which takes the Logit Probablility (i.e Choice Utility)
# for all product profiles, group them into sets of 4 (i.e 1 Choice set) and labels the largest probability as 1
# and rest as 0 and converts 1 into "YES" and 0 into "NO"

training.predicted.choice = choice.set.predictor(training.choice.utility)

training.actual.choice = factor(training.data.frame$choice, levels = c(0, 1), labels = c("NO", "YES"))

# look for sensitivity > 0.25 for four-profile choice sets.

training.set.performance = confusionMatrix(data = training.predicted.choice,
                                           reference = training.actual.choice, positive = "YES")

# Report choice prediction sensitivity for training data

cat("\n\nTraining choice set sensitivity = ",
    sprintf("%1.1f", training.set.performance$byClass[1]*100), " Percent", sep = "")

#### Training choice set sensitivity = 93.7 Percent

### Now evaluate peformance in the test set-

test.choice.utility = NULL  # initialize utility vector

# work with one row of respondent test data frame at a time.
# create choice prediction using the individual part-worths.

list.of.ids = unique(test.data.frame$id)

for (index.for.id in seq(along = list.of.ids))
{
  this.id.part.worths = posterior.mean[index.for.id, ]
  this.id.data.frame = subset(test.data.frame,
                              subset = (id == list.of.ids[index.for.id]))
  
  for (index.for.profile in 1:nrow(this.id.data.frame))
  {
    test.choice.utility = c(test.choice.utility,
                            create.design.matrix(this.id.data.frame[index.for.profile, ]) %*%
                              this.id.part.worths)
  }
}

test.predicted.choice = choice.set.predictor(test.choice.utility)

test.actual.choice = factor(test.data.frame$choice, levels = c(0, 1), labels = c("NO", "YES"))

# look for sensitivity > 0.25 for four-profile choice sets.

test.set.performance = confusionMatrix(data = test.predicted.choice,
                                       reference = test.actual.choice, positive = "YES")

# report choice prediction sensitivity for test data-

cat("\n\nTest choice set sensitivity = ",
    sprintf("%1.1f", test.set.performance$byClass[1]*100), "Percent", sep = "")


# Test choice set sensitivity = 52.6Percent

# Note: Here as a criterion of evaluation, we use the percentage of choices predicted,
# noting that by chance alone we would expect to predict 25 % of the choices correctly
# as each choice item contains 4 product alternatives.
# The percentage of choices correctly predicted is sensitivity or true positive rate in a binary classification problem.

# An HB model fit to the computer choice data provides training set sensitivity of 93.7 %
# and test set sensitivity of 52.6 %
