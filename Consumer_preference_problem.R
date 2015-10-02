############################################## Computer Choice Study ########################################


############### Analyzing Consumer Preferences and Building a Market Simulation ########################


#### After having demonstrated the predictive power of the HB model we'll now return 
#### to the complete set of 16 choice sets to obtain individual-level part-worths for further analysis.

#### We'll try to address Market Response, Brand Loyalty, Price Sensitivity and Feature focus which are key 
#### aspects to consider in determining PRICING POLICY.

################## The Problem Context #########################

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



install.packages("lattice")   # lattice graphics
install.packages("vcd")  # graphics package with mosaic plots
install.packages("ggplot2") # graphics
install.packages("ChoiceModelR")  # for Hierarchical Bayes estimation
install.packages("caret")  # for confusion matrix


library(lattice)
library(vcd)
library(ggplot2)
library(ChoiceModelR)
library(caret)


# load split-plotting utilities
source("R_utility_program_3.R")

# load market simulation utilities
source("R_utility_program_2.R")


# Read in the data from a case study in Computer Choice.

complete.data.frame = read.csv("computer_choice_study.csv")

# In the previous research work, we had estimated the Hierarchical Bayes part-worths using test and train sets.
# Here, we will be using the complete data from the computer choice study.

working.data.frame = complete.data.frame

dim(working.data.frame)  # 14,336 rows and 12 columns 

# User-defined function for plotting descriptive attribute names

effect.name.map = function(effect.name)
{
  if(effect.name == "brand") return("Manufacturer/Brand")
  if(effect.name == "compat") return("Compatibility with Windows 95")
  if(effect.name == "perform") return("Performance")
  if(effect.name == "reliab") return("Reliability")
  if(effect.name == "learn") return("Learning Time (4 to 32 hours)")
  if(effect.name == "price") return("Price ($1,000 to $2,750)")
}

print.digits = 2

# user-defined function for printing conjoint measures

if (print.digits == 2)
  ptretty.print = function(x) {sprintf("%1.2f", round(x, digits = 2))}

if (print.digits == 3)
  pretty.print = function(x) {sprintf("%1.3f", round(x, digits = 3))}


# Set up sum contrasts for effects coding

options(contrasts = c("contr.sum", "contr.poly"))


UniqueID = unique(working.data.frame$id)

# set up zero priors

cc.priors = matrix(0, nrow = length(UniqueID), ncol = 13)

colnames(cc.priors) = c("A1B1", "A1B2", "A1B3", "A1B4", "A1B5", "A1B6", "A1B7", "A1B8",
                        "A2B1", "A3B1", "A4B1", "A5B1", "A6B1")

# The actual names are as follows :

AB.names = c("Apple", "Compaq", "Dell", "Gateway", "HP", "IBM", "Sony", "Sun",
             "Compatibility", "Performance", "Reliability", "Learning", "Price")


# Now before we go further let me outline the model which we are studying.
# The model is called Hierarchical Bayes Multinomial Logit Model.

# Pr(Yj = 1 | Zi ) = exp(B1*Brand + B2*compat + B3*perform + B4*reliab + B5*learn + B6*price) /
#                     1 + exp(B1*Brand + B2*compat + B3*perform + B4*reliab + B5*learn + B6*price)

# Pr(Yj = 0 | Zi ) = exp(B1*Brand + B2*compat + B3*perform + B4*reliab + B5*learn + B6*price) /
#                     1 + exp(B1*Brand + B2*compat + B3*perform + B4*reliab + B5*learn + B6*price)

# Here Yj = Consumer choice (discrete choice) for a brand, where 'j' each consumer in the study.
# Zi = Utility of a brand, where i denotes all attributes for the specific brand chosen (j)
# i = ['brand', 'compat', 'perform', 'reliab', 'learn', 'price']

# Our job is to estimate the posterior distribution of all the attributes coefficients i.e B1, B2 ..., B6
# and come up with the value of Z which is the utility (which is sum of all the part-worths, i.e attribute*levels)
# and come up with prediction of discrete choice, Y for each individual in the test set.

# We'll use Markov Chain Monte Carlo (MCMC) method to estimate the model parameters.


# Set up run parameters for the MCMC

# using aggregate beta estimates to get started

truebetas = cc.priors

cc.xcoding = c(0, 1, 1, 1, 1, 1)  # first variable is categorical and others continuous

cc.attlevels = c(8, 8, 4, 2, 8, 8) # the number of levels for each attributes

# Below are the constraints for each attributes denoted by c1:c6

# Brand : no constraints for order on brand, so a null constraint matrix as per the function arguments
c1 = matrix(0, ncol = 8, nrow = 8)

# Compatibility: Higher the better. Continuous attributes have 1x1 matrix representation as per the function arguments.
c2 = matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# Performance : Higher the better
c3 = matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# Reliability : Higher the better
c4 = matrix(1, ncol = 1, nrow = 1, byrow = TRUE)

# Learning : Lesser time the better , so lower the better
c5 = matrix(-1, ncol = 1, nrow = 1, byrow = TRUE)

# Price : Lower the better
c6 = matrix(-1, ncol = 1, nrow = 1, byrow = TRUE)


cc.constraints = list(c1, c2, c3, c4, c5, c6)


# The below contraints are for the length of run and sampling from the end of the run.
# So, we will do a total of 10,000 iterations with estimates based on the last 2,000 runs.
# Note: It is always better to do a test run using very few iterations.

# Run parameters
cc.mcmc = list(R = 10000, use = 2000)

# Run options
cc.options = list(none = FALSE, save = TRUE) # we are nor modeling any "None" option in teh suvery answers. We want to save the coef estimates.

# Now let's set up the data for analysis.

UnitID = working.data.frame$id
Set = as.integer(working.data.frame$setid)
Alt = as.integer(working.data.frame$position)
X_1 = as.integer(working.data.frame$brand)  # categories by brand
X_2 = as.integer(working.data.frame$compat) # integer values 1 to 8
X_3 = as.integer(working.data.frame$perform) # integer values 1 to 4
X_4 = as.integer(working.data.frame$reliab) # integer values 1 to 2
X_5 = as.integer(working.data.frame$learn) # integer values 1 to 8
X_6 = as.integer(working.data.frame$price) # integer values 1 to 8
y = as.numeric(working.data.frame$choice)  # using 0 or 1 response coding.

cc.data = data.frame(UnitID, Set, Alt, X_1, X_2, X_3, X_4, X_5, X_6, y)

dim(cc.data)  # 14,336 rows and 10 columns

# MCMC estimation begins

set.seed(9999)

out = choicemodelr(data = cc.data,
                   xcoding = cc.xcoding,
                   mcmc = cc.mcmc,
                   options = cc.options,
                   constraints = cc.constraints)


# How did Markov Chain Monte Carlo estimated the posterior distribution of coefficients-

# a) Draw Z (Utility) given Y (Choice) and All Attributes (Brand, Compat, Perform, Reliab, Learn & Price)
# b) Draw all the Coefficients (Attribute Coefs) given the value of Z (Utility)
# c) Repeat 10,000 iterations so that the posterior distribution of the mean of all the coefficients go into an equilibrium.

# Note: This can be also seen in the diagram when the MCMC works.

# Gather data from HB posterior parameter estimates-

# As we had imposed constraints on all continuous parameters so we use betadraw.c

names(out)
# [1] "betadraw"   "betadraw.c" "compdraw"   "loglike"

dim(out$betadraw.c)
# [1] 224  12 200   # 224 unique consumers , 12 features and for each feature last 200 samples of estimates.

posterior.mean = matrix(0, nrow = dim(out$betadraw.c)[1], ncol = dim(out$betadraw.c)[2])

posterior.sd = matrix(0, nrow = dim(out$betadraw.c)[1], ncol = dim(out$betadraw.c)[2])

for (index.row in 1:dim(out$betadraw.c)[1])
{
  for (index.col in 1:dim(out$betadraw.c)[2])
  {
    posterior.mean[index.row, index.col] = mean(out$betadraw.c[index.row, index.col, ])
    # in the first iteration, it takes the mean of the last 200 samples of the 1st feature i.e. brand
    # in the same manner individually all the features are covered and coefficient estimates (beta) are calculated.
    
    posterior.sd[index.row, index.col] = sd(out$betadraw.c[index.row, index.col, ])
    # same way as above it calculates the estimates of coefficients' standard deviation.
  }
}

# Working with data for ONE RESPONDENT at a time we compute predicted choices for the full set of consumer responses.

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
  
  t(as.matrix(xdesign.row))  # return row of design matrix
}


# Evaluate performance in the full set of consumer responses

working.choice.utility = NULL  # initialize utility vector

# Work with one row of respondent data at a time

# Create choice prediction using the individual part-worths

list.of.ids = unique(working.data.frame$id)  # 1, 7, 9, 10 ... unique customer ids

for (index.for.id in seq(along = list.of.ids))  # 1, 2, 3, ...224 i.e. total number of unique customers
{
  # for the 1st customer
  this.id.part.worths = posterior.mean[index.for.id, ]  # 1st row and all columns from posterior.mean
  # this is MCMC coeficients estimate of all the features for the 1st customer (in the first loop)
  
  this.id.data.frame = subset(working.data.frame, subset = (id == list.of.ids[index.for.id]))
  # for this customer (i.e. 1st customer), it fetches the level values for all the features from working.data.frame
  # this data.frame has all the 64 product profiles in 16 choice sets.
  
  for (index.for.profile in 1:nrow(this.id.data.frame))
  {
    working.choice.utility = c(working.choice.utility,
                               create.design.matrix(this.id.data.frame[index.for.profile, ]) %*%
                                 this.id.part.worths)
    # it picks the 1st record i.e. 1st product profile for this customer and pass it to the design matrix function
    # The design matrix receives one row at a time i.e. one product profile at a time for that customer.
    # It then matrix multiplies xdesign.row vector(i.e. transformed level values from working.data.frame) values with 
    # the coefficients estimate from the posterior.mean matrix.
    # This is like B1*Brand + B2*Compat + ... + B6*Price for the 1st product profile.
    # Similarly it does it for the 2nd and goes on until all 64 product profiles are read.
    # working.choice.utility is a vector of 64 different choice utilities.
    # Again, in the next loop it does the same for the second customer and so on.
  }
}

length(working.choice.utility)
# 14,336 i.e. 1 * 64 * 224

# Let's verify for the 1st customer's 1st product profile

working.data.frame[1,]
# id  profile	setid	position	brand	compat	perform	reliab	learn	price	choice	buy
# 1     1	      1	   Top-Left	 Sun	 1	      4	      2	      8	    8	    0	     0

posterior.mean[1,]

#    A1B1             A1B2            A1B3            A1B4            A1B5            A1B6
#[1] -11.98584312836  -4.49235237776   5.14025340185   5.73195749031   4.01104168162  14.92477580676
#     A1B7            A2B1            A3B1            A4B1            A5B1            A6B1
#[7]  -9.36827446797   0.94460950514   7.91098331762   0.58805436664  -0.03784293783  -1.38560771677

product_profile_choice_utility_test = create.design.matrix(working.data.frame[1, ]) %*% posterior.mean[1, ]

print(product_profile_choice_utility_test)  # -0.08926680581

print(working.choice.utility[1])  # -0.08926680581

# So, above calculation proves that my understanding is correct.
# The value -0.08926680581 is the probability of individual consumer choice.
# The reason it is negative as it is the probability of logit and not Pr(Y = 1| Z)
# So, the logit is log(Pr(Y = 1|Z) / (1 - Pr(Y = 1| Z)))

# Let's solve this logit to see what is the discrete choice probability (testing purpose)

discrete.choice.prob.test = exp(product_profile_choice_utility_test) / (1 + exp(product_profile_choice_utility_test))

print(discrete.choice.prob.test)  # 0.4776981061, so now you see it is a postive prob.


# Now let's come back to our problem of predicting the discrete choice for all the respondents

working.predicted.choice = choice.set.predictor(working.choice.utility)  # we call the function from R_utility_program_2.R

length(working.predicted.choice)  # 14336

working.predicted.choice[1:4]
#[1] NO  NO  YES NO 
#Levels: NO YES

# Let's understand how it worked with the 1st 4 probabilities i.e for first 4 product profiles of that customer.

########## Testing the result #########

predicted.choice.test = length(working.choice.utility[1:4])  # length = 4 (1st 4 product profiles)
index.fourth.test =  0  # initialize block-of-four choice set indices
  
    index.first.test = index.fourth.test + 1
    index.second.test = index.fourth.test + 2
    index.third.test = index.fourth.test + 3
    index.fourth.test = index.fourth.test + 4
    this.choice.set.probability.vector.test = 
      c(working.choice.utility[index.first.test],
        working.choice.utility[index.second.test],
        working.choice.utility[index.third.test],
        working.choice.utility[index.fourth.test])
      
      alpha = 1
      if(alpha < 0 || alpha > 1) stop("alpha must be between zero and one")
    response.vector.test <- numeric(length(this.choice.set.probability.vector.test))
    for(k in seq(along=this.choice.set.probability.vector.test))
      if(this.choice.set.probability.vector.test[k] == max(this.choice.set.probability.vector.test)) response.vector.test[k] = 1
  
  # this.choice.set.probability.vector.test : -0.08926680581 -3.94120772161  5.48096759866 -0.30611221497 
  # response.vector.test : 0 0 0 0  (before)
  # response.vector.test : 0 0 1 0  (after)
  
  predicted.choice.test[index.first.test:index.fourth.test] = alpha*(response.vector.test/sum(response.vector.test))  # 0 0 1 0
  
  # predicted.choice.test : 0 0 1 0

  predicted.choice.test = factor(predicted.choice.test, levels = c(0,1), 
                             labels = c("NO","YES"))
  predicted.choice.test

# [1] NO  NO  YES NO 
# Levels: NO YES
############# Testing ends and works perfectly ###################

# going back to the problem again.

working.actual.choice = factor(working.data.frame$choice, levels = c(0, 1), labels = c("NO", "YES"))

working.actual.choice[1:4]
#[1] NO  NO  YES NO 
#Levels: NO YES

# Look, for the 1st Choice set we have correctly predicted the discrete choice.

# Now look for SENSITIVITY > 0.25 for four-profile choice sets.
# Sensitivity is here both actual and predict would have TRUE and consumer choice as YES i.e. TRUE POSITIVE
# As a random guess has a prob of 25 % so we are keeping a threshold more than that.

working.set.performance = confusionMatrix(data = working.predicted.choice,
                                          reference = working.actual.choice, positive = "YES")


#              Reference
#Prediction    NO     YES
#        NO   10360   392
#        YES  392     3192

# Report Choice prediction sensitivity for the full data

cat("\n\nFull data set choice set sensitivity = ",
    sprintf("%1.1f", working.set.performance$byClass[1]*100), " Percent", sep = "")


# Full data set choice set sensitivity = 89.1 Percent
# This result is quite remarkable given the discrete choice is by default noisy.


##### Now we will continue with our analysis of consumer preferences ... ######

# We now build a data frame for the consumers with the full set of eight brands.
# The following is obtained from the posterior.mean of the Beta estimates obtained from the MCMC
# for all the brands and other attributes.

ID = unique(working.data.frame$id)
Apple = posterior.mean[, 1]
Compaq = posterior.mean[, 2]
Dell = posterior.mean[, 3]
Gateway = posterior.mean[, 4]
HP = posterior.mean[, 5]
IBM = posterior.mean[, 6]
Sony = posterior.mean[, 7]
Sun = -1 * (Apple + Compaq + Dell + Gateway + HP + IBM + Sony)
Compatibility = posterior.mean[, 8]
Performance = posterior.mean[, 9]
Reliability = posterior.mean[, 10]
Learning = posterior.mean[, 11]
Price = posterior.mean[, 12]

# Starting with individual-level part-worths...

id.data = data.frame(ID, Apple, Compaq, Dell, Gateway, HP, IBM, Sony, Sun, Compatibility, 
                     Performance, Reliability, Learning, Price)


# Compute Attribute importance values for each attribute. Note it is the range of the levels within an attribute

id.data$brand.range = numeric(nrow(id.data))  # a numeric vector of all 0's of size 224
id.data$compatibility.range = numeric(nrow(id.data))
id.data$performance.range = numeric(nrow(id.data))
id.data$reliability.range = numeric(nrow(id.data))
id.data$learning.range = numeric(nrow(id.data))
id.data$price.range = numeric(nrow(id.data))

id.data$sum.range = numeric(nrow(id.data))  # will be used to calculate the relative attribute importance.

id.data$brand.importance = numeric(nrow(id.data)) # is used to store the relative importance of the attribute
id.data$compatibility.importance = numeric(nrow(id.data))
id.data$performance.importance = numeric(nrow(id.data))
id.data$reliability.importance = numeric(nrow(id.data))
id.data$learning.importance = numeric(nrow(id.data))
id.data$price.importance = numeric(nrow(id.data))


for (id in seq(along = id.data$ID))
{
  # for the 1st consumer, we will now calculate the coefficient estimates' range 
  id.data$brand.range[id] = max(id.data$Apple[id],
                                id.data$Compaq[id],
                                id.data$Dell[id],
                                id.data$Gateway[id],
                                id.data$HP[id],
                                id.data$IBM[id],
                                id.data$Sony[id],
                                id.data$Sun[id]) - min(id.data$Apple[id],
                                                       id.data$Compaq[id],
                                                       id.data$Dell[id],
                                                       id.data$Gateway[id],
                                                       id.data$HP[id],
                                                       id.data$IBM[id],
                                                       id.data$Sony[id],
                                                       id.data$Sun[id])
  
  id.data$compatibility.range[id] = abs(8 * id.data$Compatibility[id]) # multiplying with the number of levels
  id.data$performance.range[id] = abs(4 * id.data$Performance[id])
  id.data$reliability.range[id] = abs(2 * id.data$Reliability[id])
  id.data$learning.range[id] = abs(8 * id.data$Learning[id])
  id.data$price.range[id] = abs(8 * id.data$Price[id])
  
  id.data$sum.range[id] = id.data$brand.range[id] + id.data$compatibility.range[id] +
                          id.data$performance.range[id] + id.data$reliability.range[id] +
                          id.data$learning.range[id] + id.data$price.range[id]
  
  # Now calculate the Relative Importance of each attribute for a given consumer choice.
  
  id.data$brand.importance[id] = id.data$brand.range[id] / id.data$sum.range[id]
  id.data$compatibility.importance[id] = id.data$compatibility.range[id] / id.data$sum.range[id]
  id.data$performance.importance[id] = id.data$performance.range[id] / id.data$sum.range[id]
  id.data$reliability.importance[id] = id.data$reliability.range[id] / id.data$sum.range[id]
  id.data$learning.importance[id] = id.data$learning.range[id] / id.data$sum.range[id]
  id.data$price.importance[id] = id.data$price.range[id] / id.data$sum.range[id]
  
  # Feature importance relates to the most important feature
  
  # Considering product features as not Brand and not Price
  
  id.data$feature.importance[id] = max(id.data$compatibility.importance[id],
                                       id.data$performance.importance[id],
                                       id.data$reliability.importance[id],
                                       id.data$learning.importance[id])
}


# Identify EACH INDIVIDUAL's top brand defining top.brand factor variable

id.data$top.brand = integer(nrow(id.data))

for (id in seq(along = id.data$ID))  # each consumer at a time.
{
  brand.index = 1:8
  
  brand.part.worth = c(id.data$Apple[id], id.data$Compaq[id], id.data$Dell[id], id.data$Gateway[id], id.data$HP[id],
                       id.data$IBM[id], id.data$Sony[id], id.data$Sun[id])
  
  temp.data = data.frame(brand.index, brand.part.worth)
  
  temp.data = temp.data[sort.list(temp.data$brand.part.worth, decreasing = TRUE), ]
  
  id.data$top.brand[id] = temp.data$brand.index[1]
}

id.data$top.brand = factor(id.data$top.brand, levels = 1:8,
                           labels = c("Apple", "Compaq", "Dell", "Gateway", "HP", "IBM", "Sony", "Sun"))


# As the attribute importance is dependent on the number of levels each attribute has, so let's consider
# a relative-value based measure, so let's define an alternative to importance called "Attribute Value".

# Compute "Attribute Value" relative to the consumer group.
# It is a standardized measure, let the "Attribute Value" be mean 50 and sd 10

standardize = function(x)
{
  # standardize x so it has mean zero and standard deviation 1
  (x - mean(x)) / sd(x)
}

compute.value = function(x)
{
  # rescale x so it has the same mean and standard deviation as y
  standardize(x) * 10 + 50
}

id.data$brand.value = compute.value(id.data$brand.range)
id.data$compatibility.value = compute.value(id.data$compatibility.range)
id.data$performance.value = compute.value(id.data$performance.range)
id.data$reliability.value = compute.value(id.data$reliability.range)
id.data$learning.value = compute.value(id.data$learning.range)
id.data$price.value = compute.value(id.data$price.range)

# Identify each individual's top value using computed relative attribute values

id.data$top.attribute = integer(nrow(id.data))

for (id in seq(along = id.data$ID))
{
  attribute.index = 1:6
  
  attribute.value = c(id.data$brand.value[id], id.data$compatibility.value[id],
                      id.data$performance.value[id], id.data$reliability.value[id],
                      id.data$learning.value[id], id.data$price.value[id])
  
  temp.data = data.frame(attribute.index, attribute.value)
  
  temp.data = temp.data[sort.list(temp.data$attribute.value, decreasing = TRUE), ]
  
  id.data$top.attribute[id] = temp.data$attribute.index[1]
}

id.data$top.attribute = factor(id.data$top.attribute, levels = 1:6,
                               labels = c("Brand", "Compatibility", "Performance", "Reliability", "Learninig", "Price"))




# In this part we have mined the original survey with 16 choice sets, we estimated conjoint measures
# (attributes and part-worths) at the INDIVIDUAL LEVEL with an HB model and place consumers into groups 
# based upon their revealed preferences for computer products.

##### Contingency table of Top-Ranked Brands and Most Valued Attributes ######

# ----------------------------------------------------------------------------------
# Top-Ranked #        Most Valued Attribute Relative to Other Consumers
#   Brand    # Brand  Compatibility Performance Reliability Learning  Price   # Total
# -----------------------------------------------------------------------------------
# Apple      #  9       0             4           6           12        7     # 38
# Compaq     #  5       1             4           4            3        2     # 19
# Dell       #  8       3             4           1            6        6     # 28             
# Gateway    # 10       4            10           5            9        4     # 42
# HP         #  7       0             1           5            3        3     # 19
# IBM        #  5       6            10           4           11        2     # 38
# Sony       #  1       0             2           2            1        3     #  9
# Sun        #  2       6             2          10            3        8     # 31
# ------------------------------------------------------------------------------------
# Total      # 47      20            37          37           48       35     # 224
# ---------------------------------------------------------------------------------

# Here Top-Ranked Brands are determined within Individuals and
# the most valued attributes are determined relative to all consumers in the study.

# Here 38 individuals have chosen "Apple" and 42 individuals have purchased "Gateway" brand computers.
# The most valued attribute is learning.

# This data is rendered as a mosiac plot.

# # Mosaic plot of joint frequencies top ranked brand by top value

mosaic( ~ top.brand + top.attribute, data = id.data,
        highlighting = "top.attribute",
        highlighting_fill = c("blue", "white", "green", "lightgray", "magenta", "black"),
        labeling_args = list(set_varnames = c(top.brand = "", top.attribute = ""),
                             rot_labels = c(left = 90, top = 45),
                             pos_labels = c("center", "center"),
                             just_labels = c("left", "center"),
                             offset_labels = c(0.0, 0.0)))

# Mosiac plots provide a convenient visualization of cross-tabulations (contingency tables). 

# The relative heights of rows correspond to the relative row frequencies. 
# Here "Gateway", "IBM" and "Apple" have high relative row frequencies, i.e. Top-Ranked Brand

# The relative width of columns corresponds to the cell frequencies within rows.
# Here for "HP" brand "Brand" attribute is the most important.
# E.G. For "Apple" brand, "Learning" attribute is relatively more important.

############ Let's do another representation of the data #######

# Brand, Price and Product Features are key inputs to models of Consumer Preference and Market Response.

# This alternative representation is called Triplot/Ternary plot with three features identified for each consumer.
# In our case we will use Price, Brand and feature importance(most important feature after excluding Price and Brand) 
# measures to obtain data for three-way plots as the basis for three relative measures that are :
# Brand Loyalty, Price Sensitivity and Feature Focus.

# Note: Triplot/ternary plot are very useful for pricing studies.

id.data$brand.loyalty = numeric(nrow(id.data))
id.data$price.sensitivity = numeric(nrow(id.data))
id.data$feature.focus = numeric(nrow(id.data))

for (id in seq(along = id.data$ID))
{
  # calculating for each consumer at a time
  sum.importances = id.data$brand.importance[id] +
                    id.data$price.importance[id] +
                    id.data$feature.importance[id]
  
  # Brand Loyalty
  id.data$brand.loyalty[id] = id.data$brand.importance[id] / sum.importances
  
  # Price Sensitivity
  id.data$price.sensitivity[id] = id.data$price.importance[id] / sum.importances
  
  # Feature Focus
  id.data$feature.focus[id] = id.data$feature.importance[id] / sum.importances
}

# Ternary (composed of three) model for consumer preference and choice... the plot
# Ternary Diagram : Visualizes compositional, 3-dimensional data in an equilateral triangle.

ternaryplot(id.data[, c("brand.loyalty", "price.sensitivity", "feature.focus")],
            dimnames = c("Brand Loyalty", "Price Sensitivity", "Feature Focus"),
            prop_size = ifelse((id.data$top.brand == "Apple"), 0.8,
                               ifelse((id.data$top.brand == "Dell"), 0.7,
                                      ifelse((id.data$top.brand == "HP"), 0.7, 0.5))),
            
            pch = ifelse((id.data$top.brand == "Apple"), 20, 
                         ifelse((id.data$top.brand == "Dell"), 17,
                                ifelse((id.data$top.brand == "HP"), 15, 1))),
            
            col = ifelse((id.data$top.brand == "Apple"), "red",
                         ifelse((id.data$top.brand == "Dell"), "mediumorchid4",
                                ifelse((id.data$top.brand == "HP"), "blue", "darkblue"))),
            
            grid_color = "#626262",
            
            bg = "#E6E6E6",
            
            dimnames_position = "corner", main = "")

grid_legend(0.725, 0.8, pch = c(20, 17, 15, 1),
            col = c("red", "mediumorchid4", "blue", "darkblue"),
            c("Apple", "Dell", "HP", "Other"), title = "Top-Ranked Brand")

# Let's analyze this Ternary plot:
# a) Brand-loyal consumers fall to the bottom-left vertex.
# b) Price-sensitive consumers fall to the bottom-right vertex.
# c) Featured-focused consumers are closest to the top vertex.

# From the distribution of points across the ternary plot for the Computer Choice study, we can see wide variability
# or heterogenity in consumer preferences. What does this mean for computer suppliers.

# Let's focus on three brands, Apple, Dell and HP and select the subset of consumers for whom 
# one of these brands is the top-ranked brand. These brands are separated in the ternary plot as well.
# Now we'll do another representation of this data-

# We use density plots to examine the "distributions of values" for Brand Loyalty, Price Sensitivity and Feature Focus
# across this subset of consumers. In also shows the degree to which there is overlap in these distributions.

# Comparative Densities plot #

selected.brands = c("Apple", "Dell", "HP")

selected.data = subset(id.data, subset = (top.brand %in% selected.brands))


# Plotting objects for brand.loyalty, price.sensitivity, and feature.focus.
# Create these three objects and then plot them together on one page.

first.object = ggplot(selected.data,
                      aes(x = brand.loyalty, fill = top.brand)) +
  
                      labs(x = "Brand Loyalty",
                           y = "f(x)") +
  
                      theme(axis.title.y = element_text(angle = 0, face = "italic", size = 10)) +
  
                      geom_density(alpha = 0.4) +
  
                      coord_fixed(ratio = 1/15) +
  
                      theme(legend.position = "none") +
  
                      scale_fill_manual(values = c("red", "white", "blue"),
                                        guide = guide_legend(title = NULL)) +
  
                      scale_x_continuous(limits = c(0,1)) +
  
                      scale_y_continuous(limits = c(0, 5))

second.object = ggplot(selected.data,
                       aes(x = price.sensitivity, fill = top.brand)) +
  
                       labs(x = "Price Sensitivity",
                            y = "f(x)") +
  
                       theme(axis.title.y = element_text(angle = 0, face = "italic", size = 10)) +
  
                       geom_density(alpha = 0.4) +
  
                       coord_fixed(ratio = 1/15) +
  
                       theme(legend.position = "none") +
  
                       scale_fill_manual(values = c("red", "white", "blue"),
                                         guide = guide_legend(title = NULL)) +
  
                       scale_x_continuous(limits = c(0, 1)) +
      
                       scale_y_continuous(limits = c(0, 5))

third.object = ggplot(selected.data,
                      aes(x = feature.focus, fill = top.brand)) +
  
                      labs(x = "Feature Focus",
                           y = "f(x)") +
  
                      theme(axis.title.y = element_text(angle = 0, face = "italic", size = 10)) +
  
                      geom_density(alpha = 0.4) +
  
                      coord_fixed(ratio = 1/15) +
    
                      theme(legend.position = "bottom") +
  
                      scale_fill_manual(values = c("red", "white", "blue"),
                                        guide = guide_legend(title = NULL)) +
  
                      scale_x_continuous(limits = c(0, 1)) +
      
                      scale_y_continuous(limits = c(0, 5))

# calling the function from R_utility_program_3.R
three.part.ggplot.print.with.margins(ggfirstplot.object.name = first.object,
                                     ggsecondplot.object.name = second.object,
                                     ggthirdplot.object.name = third.object,
                                     left.margin.pct = 5,
                                     right.margin.pct = 5,
                                     top.margin.pct = 10,
                                     bottom.margin.pct = 9,
                                     first.plot.pct = 25, second.plot.pct = 25, third.plot.pct = 31)

# To interpret the density functions.
# Note the x-axis has the attribute importances and y-axis is the function of that to the consumer i.e Consumer choice.

table(selected.data$top.brand)

# Apple  Compaq    Dell Gateway      HP     IBM    Sony     Sun 
# 38       0      29       0      19       0       0       0

# Here in the selected.data we have only 3 brands which we are studying.

# Consumers (29 respondents) who rated "Dell" as the highest tend to be 
# less price-sensitive and more feature-focused than consumers who rate Apple and HP highest.

# Consumers (19 respondents) who rated "HP" as the highest tend to have
# higher brand-loyalty and less feature-focused than consumers rating Dell or Apple.

# Consumers (38 respondents) who rated "Apple" as the highest tend to be
# more price-sensitive (however this evidence is not very evident)

# A general finding that emerges from a review of these densities is that,
# in terms of the three ternary model measures, there is a considerable overlap
# across consumers rating Apple, Del and HP highest.


################# Brand Switching #######################

# A concern of marketers in many product categories is the extent to which consumers are open to switching
# from one brand to another. Parallel coordinate plots may be used to explore the potential for brand switching.
# It's a part of lattice package 

# The following is the parallel coordinates plots for the brand part-worths.

parallelplot(~selected.data[, c("Apple", "Compaq", "Dell", "Gateway", "HP", "IBM", "Sony", "Sun")] |
               top.brand, selected.data, layout = c(3, 1))


# Parallel Coordinates plot: It shows relationship among many variables. It is like a univariate scatterplot
# of all displayed variables standardized and stacked parallel to one another.
# It show common movements across variables with a line for each observational unit (individual consumer in the computer choice study.)

table(selected.data$top.brand)

#Apple  Compaq    Dell Gateway      HP     IBM    Sony     Sun 
#38       0      29       0      19       0       0       0

# Here the plot display 38 lines for consumers rating Apple as the top brand.
# Here the plot display 29 lines for consumers rating Dell as the top brand.
# Here the plot display 19 lines for consumers rating HP as the top brand.

# These lines shows the individual consumer choice based on the part-worths of all the brands conditioned on the brand which is the top.
# We see that there is considerable variability in these individual's part-worths profiles.

# The more easily interpreted are parallel coordinate plots of mean part-worths

brands.data = aggregate(x = selected.data[, 2:9], by = selected.data["top.brand"], mean)
# this is similar to the group-by clause
# Here Get the values of all the brand part worths (i.e. 2:9) group by "top.brand" and compute the mean.

# top.brand  Apple	Compaq	Dell	Gateway	HP	IBM	Sony	Sun
# 1	Apple	9.19	-1.436	-4.04	0.0134	-2.26	0.0575	0.143	-1.662
# 2	Dell	-5.19	-1.222	8.28	1.4573	-2.50	0.3003	-1.675	0.547
# 3	HP	-3.26	0.994	-2.72	1.9155	6.51	2.5778	-0.882	-5.140

parallelplot(~brands.data[,c("Apple", "Compaq", "Dell", "Gateway", "HP", "IBM", "Sony", "Sun")] |
               top.brand, brands.data, layout = c(3, 1), lwd = 3, col = "mediumorchid4")

# So, this parallel-coordinate plots show the mean part-worths for brands.
# Lines farther to the right show stronger preference for a brand and stronger likelihood of switching to that brand.

# From the plot, "Apple" consumers are most likely to switch to Sony or Sun.
# Dell consumers are most likely to switch to Gateway or Sun.
# HP consumers are most likely to switch to IBM, Gateway, or Compaq

# In this section we described the Consumer preferences and the extent to which consumers are 
# Brand Loyal, Price Sensitive, or Feature-focused and the possibilities of Brand Switching.

# The below section deals with one prime aspect of Choice Models.
# i.e. Ability to predict consumer behavior in the marketplace.

######## Market Simulation ###########

# We will use models of Consumer Preference and Choice to develop Market Simulations.
# exploring a variety of marketplace conditions and evaluating alternative management decisions.
# Market Simulations are also called "What-if analyses".

# Most important for the work of predictive analytics are market simulations constructed from 
# individual-level conjoint measures, as we obtain from the Bayesian methods.

# To demonstrate market simulation with the computer choice study, suppose we are working for Apple computer company,
# and we want to know what price to charge for our computer, given three other competitors in the market:
# Dell, Gateway, and HP.
# In addition, suppose that we have an objective of commanding a 25% share in the market.

# We will describe the competitive products in terms of attributes, creating simulated choice sets for input to the market simulation problem.

### First product in the market is Dell computer.

# Say, it is a high-end system, 100 % compatible with earlier systems, 4 times as fast and less likely to fail.
# It further takes 16 hours to learn and costs $1,750

brand = "Dell"
compat = 8
perform = 4
reliab = 2 
learn = 4
price = 4

dell.competitor = data.frame(brand, compat, perform, reliab, learn, price)


### Second product in the market is Gateway computer.

# Say, it is 90% compatible, twice as fast, and just as likely to fail as earlier systems.
# It further takes 8 hours to learn and costs only $1,250

brand = "Gateway"
compat = 6
perform = 2
reliab = 1
learn = 2
price = 2

gateway.competitor = data.frame(brand, compat, perform, reliab, learn, price)

### Finally we have HP computer in the market

# HP system is 90% compatible, three times as fast, and less likely to fail.
# It takes 8 hours to learn and costs $1,500

brand = "HP"
compat = 6
perform = 3
reliab = 2
learn = 2
price = 3

hp.competitor = data.frame(brand, compat, perform, reliab, learn, price)

### Now suppose, Apple is entering this market with a system that like the
# Dell system is four times as fast as earlier systems and less likely to fail.
# The Apple system is only 85% compatible with prior systems and takes 4 hours to learn.

# We'll allow Apple prices to vary across the full range of prices from the computer choice study,
# defining eight choice sets for Market Simulation.

brand = "Apple"
compat = 5
perform = 4
reliab = 2
learn = 1
price = 1  # $1,000 in the first choice set.

apple1000 = data.frame(brand, compat, perform, reliab, learn, price)

price = 2 # $1,250 in the second choice set.
apple1250 = data.frame(brand, compat, perform, reliab, learn, price)

price = 3 # $1,500 in the third choice set.
apple1500 = data.frame(brand, compat, perform, reliab, learn, price)

price = 4 # $1,750 in the fourth choice set.
apple1750 = data.frame(brand, compat, perform, reliab, learn, price)

price = 5 # $2,000 in the fifth choice set.
apple2000 = data.frame(brand, compat, perform, reliab, learn, price)

price = 6 # $2,250 in the sixth choice set.
apple2250 = data.frame(brand, compat, perform, reliab, learn, price)

price = 7 # $2,500 in the seventh choice set.
apple2500 = data.frame(brand, compat, perform, reliab, learn, price)

price = 8 # $2,750 in the eighth choice set.
apple2750 = data.frame(brand, compat, perform, reliab, learn, price)

# The competitive products are fixed from one choice set to the next.

competition = rbind(dell.competitor, gateway.competitor, hp.competitor)


# Now build the SIMULATION choice sets with Apple varying across choice sets.

simulation.choice.sets = rbind(competition, apple1000,
                               competition, apple1250,
                               competition, apple1500,
                               competition, apple1750,
                               competition, apple2000,
                               competition, apple2250,
                               competition, apple2500,
                               competition, apple2750)

# Now we'll add "set id" to make it inline with the format of the HB models.

setid = NULL

for (index.for.set in 1:8)
{
  setid = c(setid, rep(index.for.set, times = 4))
}

simulation.choice.sets = cbind(setid, simulation.choice.sets)

print(simulation.choice.sets)  # note the price of Apple computers are being changed for each of the 8 choice sets.

# Create the simulation data frame for all individuals in the study.
# by cloning the simulation choice sets for each individual.

simulation.data.frame = NULL

list.of.ids = unique(working.data.frame$id)  # ids from original study i.e 224 unique consumers (respondents)

for (index.for.id in seq(along = list.of.ids))
{
  id = rep(list.of.ids[index.for.id], times = nrow(simulation.choice.sets))  # rep the consumer id for 32 times.
  this.id.data = cbind(data.frame(id), simulation.choice.sets) # adding a column id to the simulation choice set data frame.
  simulation.data.frame = rbind(simulation.data.frame, this.id.data)
}

dim(simulation.data.frame)  # 7168 rows adn 8 columns, i.e 1 * 4 * 8 * 224 = 7168

# Check the structure of the simulation data frame-

print(head(simulation.data.frame))
print(tail(simulation.data.frame))

# Using create.design.matrix function 
# we evaluate the Utility of each product profile in each choice set for each individual in the study.

# Note, now we'll perform the study on the Simulated data and NOT on the respondent's survey data.

list.of.ids = unique(simulation.data.frame$id)

simulation.choice.utility = NULL

for (index.for.id in seq(along = list.of.ids))
{
  # for the 1st consumer
  this.id.part.worths = posterior.mean[index.for.id, ]  # part-worths for the the 1st customer ...
  
  this.id.data.frame = subset(simulation.data.frame,
                              subset = (id == list.of.ids[index.for.id]))
  
  for (index.for.profile in 1:nrow(this.id.data.frame))
  {
    simulation.choice.utility = c(simulation.choice.utility,
                                  create.design.matrix(this.id.data.frame[index.for.profile, ]) %*%
                                    this.id.part.worths)
  }
}


# Use choice.set.predictor function to predict choices in market simulation.

simulation.predicted.choice = choice.set.predictor(simulation.choice.utility)

# Add simulation predictions to simulation data frame for analysis of the results from the market simulation.

simulation.analysis.data.frame = cbind(simulation.data.frame, simulation.predicted.choice)

# Contingency table shows results of market simulation

with(simulation.analysis.data.frame, table(setid, brand, simulation.predicted.choice))

#simulation.predicted.choice = NO

#brand
#setid Dell Gateway  HP Apple
#1  133     204 193   142
#2  126     202 191   153
#3  123     202 187   160
#4  119     199 183   171
#5  119     196 181   176
#6  113     196 180   183
#7  109     195 178   190
#8  109     194 178   191

#, , simulation.predicted.choice = YES

#brand
#setid Dell Gateway  HP Apple
#1   91      20  31    82
#2   98      22  33    71
#3  101      22  37    64
#4  105      25  41    53
#5  105      28  43    48
#6  111      28  44    41
#7  115      29  46    34
#8  115      30  46    33

### Summary table of preference shares

YES.data.frame = subset(simulation.analysis.data.frame,
                        subset = (simulation.predicted.choice == "YES"), select = c("setid", "brand"))

print(with(YES.data.frame, table(setid, brand)))


# Create market share estimates by dividing by number of individuals.

table.work = with(YES.data.frame, as.matrix(table(setid, brand)))

table.work = table.work[, c("Apple", "Dell", "Gateway", "HP")]

table.work = round(100 * table.work / length(list.of.ids), digits = 1)  # percent

Apple.Price = c(1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750)

table.work = cbind(Apple.Price, table.work)

print(table.work)


# Data visualization of Market / Preference share estimates from the simulation.

mosaic.data.frame = YES.data.frame

mosaic.data.frame$setid = factor(mosaic.data.frame$setid, levels = 1:8,
                                 labels = c("$1,000", "$1,250", "$1,500", "$1,750",
                                            "$2,000", "$2,250", "$2,500", "$2,750"))


# Mosaic plots the joint frequencies from the market simulation.
# Length/width of the tiles in each row reflects brand marker share.
# rows relate to Apple prices.

mosaic(~ setid + brand, data = mosaic.data.frame,
       highlighting = "brand",
       highlighting_fill = 
         c("mediumorchid4", "green", "blue", "red"),
       labeling_args = 
         list(set_varnames = c(brand = "", setid = "Price of Apple Computer"),
              rot_labels = c(left = 90, top = 45),
              pos_labels = c("center", "center"),
              just_labels = c("left", "center"),
              offset_labels = c(0.0, 0.0)))

# Obeying the law of demand, higher prices translate into lower market shares for Apple.

#   Apple.Price Apple Dell Gateway   HP
#1        1000  36.6 40.6     8.9 13.8
#2        1250  31.7 43.8     9.8 14.7
#3        1500  28.6 45.1     9.8 16.5
#4        1750  23.7 46.9    11.2 18.3
#5        2000  21.4 46.9    12.5 19.2
#6        2250  18.3 49.6    12.5 19.6
#7        2500  15.2 51.3    12.9 20.5
#8        2750  14.7 51.3    13.4 20.5

# Note our objective for this Market Simulation was also to set the price of Apple computers so that
# it has atleast 25 % of market share.

# From the above table, Apple would need to set its price below $1,750 to capture a 25 % share.

# To understand customers is to predict what they might do when the firm introduces a new product
# or changes the price of a current product. A good model, when paired with a Market Simulation,
# provides predictions to guide management action.

