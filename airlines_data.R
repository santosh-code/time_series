library(readr)
airlines <- read.csv(file.choose()) 
View(airlines) # Seasonality 12 months

# Pre Processing
# input t
airlines["t"] <- c(1:96)
View(airlines)

airlines["t_square"] <- airlines["t"] * airlines["t"]
airlines["log_passengers"] <- log(airlines["Passengers"])


# So creating 12 dummy variables
X <- data.frame(outer(rep(month.abb,length = 96), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X) <- month.abb # Assigning month names
View(X)

airlines_passengers <- cbind(airlines, X)
colnames(airlines_passengers)

View(airlines_passengers)
## Pre-procesing completed

attach(airlines_passengers)

# partitioning
train <- airlines_passengers[1:84, ]
test <- airlines_passengers[85:96, ]

########################### LINEAR MODEL #############################

linear_model <- lm(Passengers ~ t, data = train)
summary(linear_model)

linear_pred <- data.frame(predict(linear_model, interval = 'predict', newdata = test))

rmse_linear <- sqrt(mean((test$Passengers - linear_pred$fit)^2, na.rm = T))
rmse_linear

######################### Exponential ############################

expo_model <- lm(log_passengers ~ t, data = train)
summary(expo_model)
expo_pred <- data.frame(predict(expo_model, interval = 'predict', newdata = test))
rmse_expo <- sqrt(mean((test$Passengers - exp(expo_pred$fit))^2, na.rm = T))
rmse_expo

######################### Quadratic ###############################

Quad_model <- lm(Passengers ~ t + t_square, data = train)
summary(Quad_model)
Quad_pred <- data.frame(predict(Quad_model, interval = 'predict', newdata = test))
rmse_Quad <- sqrt(mean((test$Passengers-Quad_pred$fit)^2, na.rm = T))
rmse_Quad

######################### Additive Seasonality #########################

sea_add_model <- lm(Passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov + Dec, data = train)
summary(sea_add_model)
sea_add_pred <- data.frame(predict(sea_add_model, newdata = test, interval = 'predict'))
rmse_sea_add <- sqrt(mean((test$Passengers - sea_add_pred$fit)^2, na.rm = T))
rmse_sea_add


######################## Multiplicative Seasonality #########################

multi_sea_model <- lm(log_passengers ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(multi_sea_model)
multi_sea_pred <- data.frame(predict(multi_sea_model, newdata = test, interval = 'predict'))
rmse_multi_sea <- sqrt(mean((test$Passengers - exp(multi_sea_pred$fit))^2, na.rm = T))
rmse_multi_sea

################### Additive Seasonality with Quadratic Trend #################

Add_sea_Quad_model <- lm(Passengers ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov, data = train)
summary(Add_sea_Quad_model)
Add_sea_Quad_pred <- data.frame(predict(Add_sea_Quad_model, interval = 'predict', newdata = test))
rmse_Add_sea_Quad <- sqrt(mean((test$Passengers - Add_sea_Quad_pred$fit)^2, na.rm=T))
rmse_Add_sea_Quad

# Preparing table on model and it's RMSE values 
table_rmse <- data.frame(c("rmse_linear", "rmse_expo", "rmse_Quad", "rmse_sea_add", "rmse_Add_sea_Quad", "rmse_multi_sea"), c(rmse_linear, rmse_expo, rmse_Quad, rmse_sea_add, rmse_Add_sea_Quad, rmse_multi_sea))
colnames(table_rmse) <- c("model", "RMSE")
View(table_rmse)

# Additive seasonality with Quadratic Trend has least RMSE value

write.csv(airlines_passengers, file = "airlinesPassengers.csv", row.names = F)
getwd()

############### Combining Training & test data to build Additive seasonality using Quadratic Trend ############

Add_sea_Quad_model_final <- lm(Passengers ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov, data = airlines_passengers)
summary(Add_sea_Quad_model_final)