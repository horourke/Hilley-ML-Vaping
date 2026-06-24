# Hilley et al. - Supervised Machine Learning Prediction of Adolescent Vaping
# Note: Some of this code is adapted from the cv.glmmLasso() function in the "lmmen" package.

library(haven)
library(tidymodels) #"recipes" is loaded by default
library(themis)        
library(glmmLasso)
library(dplyr)
library(mlr3)
library(mlr3pipelines)
library(glmmLasso)
library(pROC)
library(ModelMetrics) 
library(PRROC)
library(data.table)
library(useful)
library(mermboost)
library(shapviz)
library(fastshap)
library(tictoc)

source("G:\\.shortcut-targets-by-id\\1CVvCbC6Mc7F5t1wMb0FrX5T3cNWlSKC4\\Hilley, O'Rourke - ML Adolescent Vaping\\Analyses\\formal_analysis_functions.R")

ays_test3 <- read_sav("G:\\.shortcut-targets-by-id\\1CVvCbC6Mc7F5t1wMb0FrX5T3cNWlSKC4\\Hilley, O'Rourke - ML Adolescent Vaping\\data\\ays_test3.sav")

ays_test3 <- ays_test3 %>%
  select(-gangre,-gangnm,-aocig,-aomarb,-aoalcr, -aomarc, -aomare, -aoecig, -aomar, -aoalc, #delete variables with high missingness
         -id,-year, -format, -eciglif, -ecig30d, -ecig30dp, #delete variables usefulless
         -rcwhit, -rchisp, -rcblac, -rcasin,-rcnatv,-rchwpi,  #delete binary race variables since we have "newrace"
  )

#descriptive results

attr(ays_test3$gender, "labels")
table(ays_test3$gender, useNA = "ifany")

table(ays_test3$grade, useNA = "ifany")

attr(ays_test3$newrace, "labels")
p_newrace <- sort(table(ays_test3$newrace,useNA = "ifany"))
data.frame(
  count = as.integer(p_newrace),
  percent = round(100 * p_newrace / sum(p_newrace), 2)
)

summary(ays_test3$age)

#data cleaning

ays_test3 <- na.omit(ays_test3)

lapply(ays_test3, unique)

ays <- ays_test3 %>%
  mutate(across(where(is.labelled), ~ as.numeric(haven::zap_labels(.x))))%>%
  mutate(
    school  = factor(school),
    newrace = factor(newrace),
    eciglifp = factor(eciglifp)
  )


lapply(ays, unique)
dplyr::glimpse(ays)

unique(ays_test3$eciglifp)
table(ays_test3$eciglifp) #13470/(13470+3078)=0.814

ays %>%
  group_by(school) %>%
  summarise(n = n()) %>%
  arrange(n) %>%
  print(n = Inf)


#outer splitting----
set.seed(123)

# split <- initial_split(ays_test3, prop = 0.8, strata = eciglifp)
# train <- training(split)
# test  <- testing(split)

task_outer <- TaskClassif$new(
  id = "school",
  backend = ays,
  target = "eciglifp"
)

task_outer$set_col_roles("school", roles = c("feature", "group"))

resampling_outer <- rsmp("holdout", ratio = 0.8)

resampling_outer$instantiate(task_outer)

train_indices <- resampling_outer$train_set(1)
test_indices <- resampling_outer$test_set(1)

outer_train_data <- task_outer$data(rows = train_indices)
outer_test_data <- task_outer$data(rows = test_indices)

#check the outcome distribution of the outer_train_data (if a cluster only has one outcome value, oversampling would not be conducted for that cluster in the latter steps)
outer_train_data %>%
  group_by(school) %>%
  summarise(
    n = n(),
    count_0 = sum(eciglifp == 0),
    percent_0 = mean(eciglifp == 0) * 100
  )%>%
  print(n = Inf)  # cluster #327 and #451 only have one outcome (ecilifp = 0)


sort(unique(outer_train_data$school))
sort(unique(outer_test_data$school))

table(outer_train_data$eciglifp) #10808/(10808+2472)=0.814
table(outer_test_data$eciglifp) #2662/(2662+606)=0.815

unique(outer_train_data$school)

# [1] 7   25  27  30  37  39  40  45  54  56  57  59  89  96  97  98  100 102 113 114 116 117 118 127 128 129
# [27] 130 131 138 140 141 144 153 287 288 289 291 305 311 327 341 355 359 360 370 375 386 407 408 409 410 412
# [53] 422 423 440 441 443 451 477 478 487 494 501 507 523 538 543

unique(outer_test_data$school)

#[1] 2   19  99  115 119 142 148 290 292 301 302 385 406 421 493 518 527

#creating k folds----

##creating k folds (training set + validation set) accounting for clusters----

#check if there are overlap school IDs
train_sch <- unique(outer_train_data$school)
valid_sch <- unique(outer_test_data$school)
length(intersect(train_sch, valid_sch))

set.seed(123)

task_k_fold <- TaskClassif$new(
  id = "school",
  backend = outer_train_data,
  target = "eciglifp"
)
task_k_fold$set_col_roles("school", roles = c("feature", "group"))

resampling <- rsmp("cv", folds = 10)
resampling$instantiate(task_k_fold)

i = 1

train_idx <- resampling$train_set(i)
validation_idx <- resampling$test_set(i)

train_data <- task_k_fold$data(rows = train_idx)
validation_data <- task_k_fold$data(rows = validation_idx)

#check if there are overlap school IDs
train_sch <- unique(train_data$school)
valid_sch <- unique(validation_data$school)
length(intersect(train_sch, valid_sch))

##cluster-sensitive oversampling for training set per fold----

for (i in 1:10) {
  
  train_idx <- resampling$train_set(i)
  validation_idx <- resampling$test_set(i)
  
  train_data <- task_k_fold$data(rows = train_idx)
  validation_data <- task_k_fold$data(rows = validation_idx)
  
  set.seed(123)
  
  schools <- unique(train_data$school)
  
  train_data_oversampled <- data.frame()
  
  for (j in schools) {
    school_data <- train_data[train_data$school == j, ]
    
    cases_0 <- school_data[school_data$eciglifp == 0, ]
    cases_1 <- school_data[school_data$eciglifp == 1, ]
    
    #check if a cluster includes obs of both 0 and 1
    if (nrow(cases_1) > 0 && nrow(cases_0) > 0) {
      #check which is the majority (0 or 1)
      if(nrow(cases_0) >  nrow(cases_1)) {
        oversampled_cases <- cases_1[
          sample(nrow(cases_1), nrow(cases_0), replace = TRUE),
        ]
        balanced_data <- rbind(cases_0, oversampled_cases)
      } else if (nrow(cases_0) <  nrow(cases_1)) {
        oversampled_cases <- cases_0[
          sample(nrow(cases_0), nrow(cases_1), replace = TRUE),
        ]
        balanced_data <- rbind(cases_1, oversampled_cases)
      } else {
        balanced_data <- school_data
      }
    } else {
      balanced_data <- school_data
    }
    train_data_oversampled <- rbind(train_data_oversampled, balanced_data)
  }
  train_data_oversampled$fold <-i
  validation_data$fold <-i
  
  assign(paste0("train_data_oversampled_", i), train_data_oversampled)
  assign(paste0("validation_data_", i), validation_data)
}

final_train_data <- dplyr::bind_rows(
  lapply(1:10, function(i) get(paste0("train_data_oversampled_", i)))
)

final_validation_data <- dplyr::bind_rows(
  lapply(1:10, function(i) get(paste0("validation_data_", i)))
)

#check if the cluster-sensitive oversampling works as expected

final_train_data %>%
  group_by(fold) %>%
  summarise(
    n_obs = n(),
    n_unique_schools = n_distinct(school)
  )

# Save training data
fwrite(final_train_data,"G:\\.shortcut-targets-by-id\\1CVvCbC6Mc7F5t1wMb0FrX5T3cNWlSKC4\\Hilley, O'Rourke - ML Adolescent Vaping\\Analyses\\final_train_data.csv")

final_validation_data %>%
  group_by(fold) %>%
  summarise(
    n_obs = n(),
    n_unique_schools = n_distinct(school)
  )

# Save validation data
fwrite(final_validation_data,"G:\\.shortcut-targets-by-id\\1CVvCbC6Mc7F5t1wMb0FrX5T3cNWlSKC4\\Hilley, O'Rourke - ML Adolescent Vaping\\Analyses\\final_validation_data.csv")

final_train_data <- fread("G:\\.shortcut-targets-by-id\\1CVvCbC6Mc7F5t1wMb0FrX5T3cNWlSKC4\\Hilley, O'Rourke - ML Adolescent Vaping\\Analyses\\final_train_data.csv")
final_validation_data <- fread("G:\\.shortcut-targets-by-id\\1CVvCbC6Mc7F5t1wMb0FrX5T3cNWlSKC4\\Hilley, O'Rourke - ML Adolescent Vaping\\Analyses\\final_validation_data.csv")

# Check outcome distribution of the oversampled dataset
table(final_train_data$eciglifp)

final_train_data %>%
  mutate(eciglifp_num = as.numeric(as.character(eciglifp))) %>%
  group_by(fold, school) %>%
  summarise(
    n = n(),
    count_0 = sum(eciglifp_num == 0),
    percent_0 = count_0 / n * 100,
  ) %>%
  arrange(fold, school) %>%
  print(n = Inf)

table(final_validation_data$eciglifp)

final_validation_data %>%
  mutate(eciglifp_num = as.numeric(as.character(eciglifp))) %>%
  group_by(fold, school) %>%
  summarise(
    n = n(),
    count_0 = sum(eciglifp_num == 0),
    percent_0 = count_0 / n * 100,
  ) %>%
  arrange(fold, school) %>%
  print(n = Inf)

#1. Lasso regression (glmmLasso) (hyperparameter tunning)----

## (a) building the search grid for lambda----

d_covariate <- c(
  "gender","grade","age","lunch","rcwhit","rchisp","rcblac","rcasin","rcnatv","rchwpi","newrace",
  "scint","sclern","scdec","scspec","scnotic","scinvl","sctalk","scsafe","sctelp","scprai","scdisc","scbetg","scenj","schat","scbest",
  "scimp","scgrad","scclub","scexwk","volunt","scskip","fclub","fdrgfr","flksch","ftrsch","fcig","fecig","falc","fmar","fdrg",
  "fsldrg","fsusp","fdrop","ffight","fgun","fsteal","farst","fgang","ganglif","gang","rhcig","rhecig","rhalcd","rhbing","rhrx",
  "rhmar","rhmarr","rhdrg","egcig","egalc","egrx","egmar","egdrug","eggun","wcig","wecig","walcr","walcd","wrx","wmar",
  "wdrg","wgun","wsteal","wfight","whurt","wskip","clwkhs","clvol","cldefn","clcig","clalcr","clmar","clgun","phonst","oktake",
  "okcheat","okfght","ignore","oppos","getawy","fmrule","fmyell","pknow","fmrarg","pctalc","fmdrul","pctgun","pctskp","pdecid","mocls",
  "dacls","moshr","dashr","moenj","daenj","phelp","pfun","phmwrk","fmsarg","ptime","pnotic","pproud","pwcig","pwalcd","pwalcr",
  "pwrx","pwmar","pwdrg","pwstel","pwvand","pwfght","bscig","bsalc","bsrx","bsmar","bsdrg","bssus","bsgun","fmdrgp","awcig",
  "awalc","awmar","nhmiss","nhgjob","nhlike","nhtalk","nhout","nhprd","nhbest","nhsafe","akdrnk","akdrg","aksdrg","akstel","cpalc",
  "cprx","cpmar","cpdrg","cpgun"
)

fix <- reformulate(d_covariate, response = "eciglifp") 

fix <- update(fix, . ~ . - newrace + as.factor(newrace)) # in glmmLasso, categorical variable in formula should be written explicitly as "as.factor", such as "as.factor(newrace)"

ays$newrace <-as.factor(ays$newrace) # yx: maybe verbose, but just to make sure factor variables are specified in the dataset; important for "useful::build.x"

lambdas <- buildLambdas(fix = fix,
                        rnd = list(school=~1),
                        data = ays, 
                        nlambdas = 100) 

# > lambdas
# [1] 1.1936922066 1.1816359153 1.1695796240 1.1575233327 1.1454670414 1.1334107502 1.1213544589 1.1092981676 1.0972418763 1.0851855850 1.0731292937 1.0610730024 1.0490167112 1.0369604199
# [15] 1.0249041286 1.0128478373 1.0007915460 0.9887352547 0.9766789634 0.9646226721 0.9525663809 0.9405100896 0.9284537983 0.9163975070 0.9043412157 0.8922849244 0.8802286331 0.8681723419
# [29] 0.8561160506 0.8440597593 0.8320034680 0.8199471767 0.8078908854 0.7958345941 0.7837783029 0.7717220116 0.7596657203 0.7476094290 0.7355531377 0.7234968464 0.7114405551 0.6993842638
# [43] 0.6873279726 0.6752716813 0.6632153900 0.6511590987 0.6391028074 0.6270465161 0.6149902248 0.6029339336 0.5908776423 0.5788213510 0.5667650597 0.5547087684 0.5426524771 0.5305961858
# [57] 0.5185398945 0.5064836033 0.4944273120 0.4823710207 0.4703147294 0.4582584381 0.4462021468 0.4341458555 0.4220895643 0.4100332730 0.3979769817 0.3859206904 0.3738643991 0.3618081078
# [71] 0.3497518165 0.3376955252 0.3256392340 0.3135829427 0.3015266514 0.2894703601 0.2774140688 0.2653577775 0.2533014862 0.2412451950 0.2291889037 0.2171326124 0.2050763211 0.1930200298
# [85] 0.1809637385 0.1689074472 0.1568511559 0.1447948647 0.1327385734 0.1206822821 0.1086259908 0.0965696995 0.0845134082 0.0724571169 0.0604008257 0.0483445344 0.0362882431 0.0242319518
# [99] 0.0121756605 0.0001193692


## (b) k-fold cv----

kfold = length(unique(final_validation_data$fold))

lossVecList <- vector(mode = 'list', length = kfold)
modList_foldk <- vector(mode = 'list', length = kfold)

for(k in 1:kfold) {
  
  validation_set <- final_validation_data %>% dplyr::filter(fold == k)
  train_set <- final_train_data %>% dplyr::filter(fold == k)
  
  validation_set <- validation_set %>%
    mutate(
      school  = factor(school),
      newrace = factor(newrace),
      eciglifp = as.numeric(as.character(eciglifp))
    )
  
  train_set <- train_set %>%
    mutate(
      school  = factor(school),
      newrace = factor(newrace),
      eciglifp = as.numeric(as.character(eciglifp))
    )
  
  # for showing lambda at each iterations
  # message(sprintf('Round: %s\n ', k))
  
  modList_foldk[[k]] <- glmmLasso_MultLambdas(fix = fix,
                                              rnd = list(school=~1),
                                              data = train_set,
                                              family = binomial(link = "logit"),
                                              lambdas = lambdas,
                                              nlambdas = length(lambdas))
  
  
  # Extracting the response variable         
  response_var <- fix[[2]] %>% as.character()
  
  # pulling out actual data for the response variable
  actualDataVector <- validation_set %>% 
    dplyr::pull(response_var)
  
  # predicting values for each of the glmmLasso model (100 lambda) 
  # using matrix form for easier error calculation in loss()
  
  predictionMatrix <- predict.glmmLasso_MultLambdas(
    object = modList_foldk[[k]],
    newdata = validation_set
  )
  
  # employing the loss function in form loss(actual,predicted)
  # using loss function, calculating a list of loss values for each vector 
  # of prediction
  # which comes from a glmmLasso model with a specific lambda 
  # storing loss values for each fold
  
  lossVecList[[k]] <- loss(actual = actualDataVector, predicted = predictionMatrix)
  # each element of this list should be 1 x nlambdas
}

cvLossMatrix <- do.call(what = rbind, args = lossVecList)

cvm = colMeans(cvLossMatrix)

# calculating sd, cv, up, down
cvsd <- apply(cvLossMatrix, 2, stats::sd, na.rm = TRUE)
cvup <- cvm + cvsd
cvlo <- cvm - cvsd

# finding the minimum cvm value in order pull out the lambda.min out of 
# list of lambda
minIndex <- which.min(cvm)    
lambda.min <- lambdas[minIndex]

# finding 1se index by doing vectorized comparison such that cvm <= cvup 
# of minIndex
my1seIndex <- min(which(cvm <= cvup[minIndex]))
lambda.1se <- lambdas[my1seIndex]

chosenLambda <- lambda.1se #need to make a decision here: where to use lambda.1se or lambda.min

## (c) final model----

#outer_train_data (with stratified and upper sampling)

set.seed(123)

schools <- unique(outer_train_data$school)

outer_train_data_oversampled <- data.frame()

for (j in schools) {
  school_data <- outer_train_data[outer_train_data$school == j, ]
  
  cases_0 <- school_data[school_data$eciglifp == 0, ] 
  cases_1 <- school_data[school_data$eciglifp == 1, ]
  
  #check if a cluster includes obs of both 0 and 1
  if (nrow(cases_1) > 0 && nrow(cases_0) > 0) {
    #check which is the majority (0 or 1)
    if(nrow(cases_0) >  nrow(cases_1)) {
      oversampled_cases <- cases_1[
        sample(nrow(cases_1), nrow(cases_0), replace = TRUE),
      ]
      balanced_data <- rbind(cases_0, oversampled_cases)
    } else if (nrow(cases_0) <  nrow(cases_1)) {
      oversampled_cases <- cases_0[
        sample(nrow(cases_0), nrow(cases_1), replace = TRUE),
      ]
      balanced_data <- rbind(cases_1, oversampled_cases)
    } else {
      balanced_data <- school_data
    }
  } else {
    balanced_data <- school_data
  }
  outer_train_data_oversampled <- rbind(outer_train_data_oversampled, balanced_data)
}

outer_train_data_oversampled <- outer_train_data_oversampled %>%
  mutate(
    school  = factor(school),
    newrace = factor(newrace),
    eciglifp = as.numeric(as.character(eciglifp))
  )

str(outer_train_data_oversampled$school)
str(outer_train_data_oversampled$newrace)
str(outer_train_data_oversampled$eciglifp)

# Check outcome distribution of the oversampled dataset

table(outer_train_data_oversampled$eciglifp)

outer_train_data_oversampled %>%
  mutate(eciglifp_num = as.numeric(as.character(eciglifp))) %>%
  group_by(school) %>%
  summarise(
    n = n(),
    count_0 = sum(eciglifp_num == 0),
    percent_0 = count_0 / n * 100,
  ) %>%
  print(n = Inf)

#fitting the best algorithm (with the optimal lambda) to the whole training set
glmmLasso.final <- glmmLasso::glmmLasso(fix = fix,
                                        rnd = list(school=~1),
                                        data = outer_train_data_oversampled,
                                        family = binomial(link = "logit"),
                                        lambda = chosenLambda)


# add control list argument to this to make converge faster form one that 
# create lambda.1se

# mimicking cv.glmnet return objects
return_list_lasso <- list(lambdas=lambdas,
                    cvm=cvm,
                    cvsd=cvsd,
                    cvup=cvup,
                    cvlo=cvlo,
                    glmmLasso.final=glmmLasso.final,
                    lambda.min=lambda.min,
                    lambda.1se=lambda.1se)


class(return_list_lasso) <- 'cv.glmmLasso'

#2. decision tree (glmertree) (hyperparameter tunning)----

## (a) building the search grid ----

grid_glmertree <- expand.grid(
  minsize = c( 100, 150),
  maxdepth = c(1, 2),
  mtry = c(Inf)
) 

## (b) k-fold cv----

d_covariate <- c(
  "gender","grade","age","lunch","rcwhit","rchisp","rcblac","rcasin","rcnatv","rchwpi","newrace",
  "scint","sclern","scdec","scspec","scnotic","scinvl","sctalk","scsafe","sctelp","scprai","scdisc","scbetg","scenj","schat","scbest",
  "scimp","scgrad","scclub","scexwk","volunt","scskip","fclub","fdrgfr","flksch","ftrsch","fcig","fecig","falc","fmar","fdrg",
  "fsldrg","fsusp","fdrop","ffight","fgun","fsteal","farst","fgang","ganglif","gang","rhcig","rhecig","rhalcd","rhbing","rhrx",
  "rhmar","rhmarr","rhdrg","egcig","egalc","egrx","egmar","egdrug","eggun","wcig","wecig","walcr","walcd","wrx","wmar",
  "wdrg","wgun","wsteal","wfight","whurt","wskip","clwkhs","clvol","cldefn","clcig","clalcr","clmar","clgun","phonst","oktake",
  "okcheat","okfght","ignore","oppos","getawy","fmrule","fmyell","pknow","fmrarg","pctalc","fmdrul","pctgun","pctskp","pdecid","mocls",
  "dacls","moshr","dashr","moenj","daenj","phelp","pfun","phmwrk","fmsarg","ptime","pnotic","pproud","pwcig","pwalcd","pwalcr",
  "pwrx","pwmar","pwdrg","pwstel","pwvand","pwfght","bscig","bsalc","bsrx","bsmar","bsdrg","bssus","bsgun","fmdrgp","awcig",
  "awalc","awmar","nhmiss","nhgjob","nhlike","nhtalk","nhout","nhprd","nhbest","nhsafe","akdrnk","akdrg","aksdrg","akstel","cpalc",
  "cprx","cpmar","cpdrg","cpgun")

part_vars <- paste(d_covariate, collapse = " + ")

formula_string <- paste0("eciglifp ~ 1 | school | ", part_vars)

glmertree_formula <- as.formula(formula_string)

kfold <- length(unique(final_validation_data$fold))

return_list_glmertree <- vector("list", nrow(grid_glmertree))

## (c) final model----

tic()
for (h in seq_len(nrow(grid_glmertree))) {
  
  current_minsize    <- grid_glmertree$minsize[h]
  current_maxdepth <- grid_glmertree$maxdepth[h]
  current_mtry <- grid_glmertree$mtry[h]
  current_alpha <- grid_glmertree$alpha[h]
  
  fold_logloss_list <- vector("list", length = kfold)
  
  for(k in 1:kfold) {
    
    validation_set <- final_validation_data %>% dplyr::filter(fold == k)
    train_set <- final_train_data %>% dplyr::filter(fold == k)
    
    validation_set <- validation_set %>%
      mutate(
        school  = factor(school),
        newrace = factor(newrace),
        eciglifp = as.numeric(as.character(eciglifp))
      )
    
    train_set <- train_set %>%
      mutate(
        school  = factor(school),
        newrace = factor(newrace),
        eciglifp = as.numeric(as.character(eciglifp))
      )
    
    train_mod <- glmertree::glmertree(
      formula = glmertree_formula,
      data    = train_set,
      family  = binomial(),
      minsize  = current_minsize,
      maxdepth = current_maxdepth,
      mtry     = current_mtry,
      alpha = current_alpha
    )
    
    eta <- predict(train_mod, newdata = validation_set, type = "link", re.form = NA)
    
    prop <- 1 / (1 + exp(-eta))
    
    # Compute log-loss 
    response_var <- glmertree_formula[[2]] %>% as.character()
    actualDataVector <- validation_set %>% pull(response_var)
    
    logloss <- - mean(actualDataVector * log(prop) + (1 - actualDataVector) * log(1 - prop))
    
    fold_logloss_list[k] <- logloss
    
  }
  
  avg_logloss <- mean(unlist(fold_logloss_list))
  
  return_list_glmertree[[h]] <- list(
    logloss_cv  = avg_logloss,
    fold_logloss = fold_logloss_list,
    minsize = current_minsize,
    maxdepth = current_maxdepth,
    mtry = current_mtry,
    alpha = current_alpha
  )
  
}
toc()

loss_summary <- do.call(rbind, lapply(return_list_glmertree, function(x) {
  data.frame(
    logloss_cv = x$logloss_cv,
    minsize    = x$minsize,
    maxdepth   = x$maxdepth,
    mtry       = x$mtry,
    alpha      = x$alpha
  )
}))

#fwrite(loss_summary, "results/glmertree_loss_summary.csv")

optimal_hp_id <- which.min(loss_summary$logloss_cv)
optimal_hp <- loss_summary[optimal_hp_id, ] #optimal hyperparameter

#3. glmermboost (mermboost) (hyperparameter tuning) ----

## (a) building the search grid for nu and mstop----

grid_glmermboost <- data.frame(
  nu = c(0.10, 0.05),
  mstop = c(2000, 4000)  
)

## (b) k-fold cv----

d_covariate <- c(
  "gender","grade","age","lunch","rcwhit","rchisp","rcblac","rcasin","rcnatv","rchwpi","newrace",
  "scint","sclern","scdec","scspec","scnotic","scinvl","sctalk","scsafe","sctelp","scprai","scdisc","scbetg","scenj","schat","scbest",
  "scimp","scgrad","scclub","scexwk","volunt","scskip","fclub","fdrgfr","flksch","ftrsch","fcig","fecig","falc","fmar","fdrg",
  "fsldrg","fsusp","fdrop","ffight","fgun","fsteal","farst","fgang","ganglif","gang","rhcig","rhecig","rhalcd","rhbing","rhrx",
  "rhmar","rhmarr","rhdrg","egcig","egalc","egrx","egmar","egdrug","eggun","wcig","wecig","walcr","walcd","wrx","wmar",
  "wdrg","wgun","wsteal","wfight","whurt","wskip","clwkhs","clvol","cldefn","clcig","clalcr","clmar","clgun","phonst","oktake",
  "okcheat","okfght","ignore","oppos","getawy","fmrule","fmyell","pknow","fmrarg","pctalc","fmdrul","pctgun","pctskp","pdecid","mocls",
  "dacls","moshr","dashr","moenj","daenj","phelp","pfun","phmwrk","fmsarg","ptime","pnotic","pproud","pwcig","pwalcd","pwalcr",
  "pwrx","pwmar","pwdrg","pwstel","pwvand","pwfght","bscig","bsalc","bsrx","bsmar","bsdrg","bssus","bsgun","fmdrgp","awcig",
  "awalc","awmar","nhmiss","nhgjob","nhlike","nhtalk","nhout","nhprd","nhbest","nhsafe","akdrnk","akdrg","aksdrg","akstel","cpalc",
  "cprx","cpmar","cpdrg","cpgun"
)

formula <- reformulate(d_covariate, response = "eciglifp")

formula_string <- paste(deparse(formula), collapse = "")
formula <- as.formula(paste0(formula_string, " + (1 | school)"))
print(formula)

kfold <- length(unique(final_validation_data$fold))

return_list_glmermboost <- vector("list", nrow(grid_glmermboost))

for (h in seq_len(nrow(grid_glmermboost))) {

current_nu    <- grid_glmermboost$nu[h]
current_mstop <- grid_glmermboost$mstop[h]

# store AIC curve for each fold
fold_aic_list <- vector("list", length = kfold)
  
  for(k in 1:kfold) {
    
    validation_set <- final_validation_data %>% dplyr::filter(fold == k)
    train_set <- final_train_data %>% dplyr::filter(fold == k)
    
    validation_set <- validation_set %>%
      mutate(
        school  = factor(school),
        newrace = factor(newrace),
        eciglifp = as.numeric(as.character(eciglifp))
      )
    
    train_set <- train_set %>%
      mutate(
        school  = factor(school),
        newrace = factor(newrace),
        eciglifp = as.numeric(as.character(eciglifp))
      )
    
    # Identify categorical vars (anything with < some unique levels or explicitly listed)
    categorical_vars <- c( "newrace")
    
    for (v in categorical_vars) {
      train_set[[v]] <- factor(train_set[[v]])
      validation_set[[v]] <- factor(validation_set[[v]], levels = levels(train_set[[v]]))
    }
    
## (c) final model----   
    
    model_k <- glmermboost(
      formula=formula,
      data = train_set, 
      family = binomial(link = "logit"),
      control = boost_control(
        nu = current_nu, 
        mstop = current_mstop
        )
      )
    
    eta <- predict(model_k, newdata = validation_set, type = "link", 
                   aggregate = "cumsum", RE = F)
    
    aic <- c()
    aic_vec <- c()
    
    fam <- binomial()
    
    response_var <- formula[[2]] %>% as.character()
    actualDataVector <- validation_set %>% pull(response_var)
    
    for (j in 1:current_mstop) {
      mu <- fam$linkinv(eta[, j])
      dev <- sum(fam$dev.resids(y = actualDataVector, mu = mu, 
                                wt = 1))
      aic[j] <- fam$aic(y = actualDataVector, n = 1, mu = mu, 
                        wt = 1, dev = dev)
      aic_vec[j] <- aic[j] / 2
    }
    
    fold_aic_list[[k]] <- aic_vec
  }

RISKMAT <- do.call(rbind, fold_aic_list)

# normalize by test-set size:
test_sizes <- sapply(seq_len(kfold), function(k) {
  sum(final_validation_data$fold == k)
})

RISKMAT <- sweep(RISKMAT, 2, test_sizes, "/")

avg_risks <- rowMeans(RISKMAT)

m_opt <- which.min(avg_risks)

return_list_glmermboost[[h]] <- list(
  nu = current_nu,
  max_mstop = current_mstop,
  cv_risks = RISKMAT, 
  avg_risks = avg_risks, 
  m_opt = m_opt)

}

class(return_list_glmermboost) <- "mer_cv"

return_list_glmermboost



# Shap values----

fix_form <- train_mod$fix_formula
df <- train_mod$data
X <- df %>%
  select(-eciglifp,-school,-fold)

pfun <- function(object, newdata) {
  newdata <- as.data.frame(newdata)  # guarantee data.frame
  as.numeric(predict(object, newdata = newdata, type = "response", RE = FALSE))
}

set.seed(123)

shap_values <- fastshap::explain(
  object= train_mod,
  X= X,
  pred_wrapper = pfun,
  nsim = 50,     # larger --> better
  baseline = attr(shap_values, "baseline")   #do we need this line? should we use real baseline?
)

#importance plot
sv_importance(sv)

#waterfall plot
sv_waterfall(sv,row_id = 5)
sv_waterfall(sv,row_id = 5000)

attr(ays_test3$wecig, "labels")
attr(ays_test3$wecig, "label")

attr(ays_test3$wmar, "labels")
attr(ays_test3$wmar, "label")

#force plot
sv_force(sv,row_id = 5000)






