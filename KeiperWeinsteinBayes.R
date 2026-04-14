## Andrea Keiper and Alex Weinstein
## DS4420 Final Project
## April 16, 2026

## MODEL 2: BAYESIAN ML
library(data.table)
library(stringr)
library(brms)
library(ggplot2)
library(patchwork)

## Read data(Wasn't loading with regular read.csv, so using (data.table))
df <- fread("Boston_CrashDetails.csv", skip = 2)
head(df)

## Data Cleaning
unique(df$Vehicle_Configuration)

# 1. Clean vehicles, currently formatted to include ALL vehicles involved in the crash, I want to keep the largest vehicle involved
VEHICLE_RANK <- c(
  "Tractor/triples" = 1,
  "Tractor/doubles" = 1,
  "Tractor/semi-trailer" = 1,
  "Truck tractor (bobtail" = 2,
  "Single-unit truck (3-or-more axles" = 2,
  "Unknown heavy truck, cannot classify" = 2,
  "Single-unit truck (2-axle, 6-tires" = 3,
  "Truck/trailer" = 3,
  "Bus (seats for 16 or more, including driver" = 4,
  "Bus (seats for 9-15 people, including driver" = 4,
  "Motor home/recreational vehicle" = 5,
  "Light truck(van, mini-van, pickup, sport utility" = 6,
  "Passenger car" = 7,
  "Motorcycle" = 8,
  "MOPED" = 9,
  "All Terrain Vehicle (ATV" = 9,
  "Snowmobile" = 9,
  "Low Speed Vehicle" = 9,
  "Registered farm equipment" = 9
)

largest_vehicle <- function(val) {
  # Extract all Vn types; return the one with the lowest rank number (= biggest vehicle)
  if (is.na(val)) {
    return(NA_character_)
  }
  matches <- stringr::str_match_all(as.character(val), "V\\d+:\\((.+?)\\)")[[1]][,2]
  if (length(matches) == 0) {
    return(NA_character_)
  }
  matches <- trimws(matches)
  ranks <- VEHICLE_RANK[matches]
  ranks[is.na(ranks)] <- 99
  matches[which.min(ranks)]
}

df$Vehicle_Type <- sapply(df$Vehicle_Configuration, largest_vehicle)

# Consolidate into broad categories for one-hot encoding
VEHICLE_CATEGORY <- c(
  "Tractor/triples" = "Heavy Truck",
  "Tractor/doubles" = "Heavy Truck",
  "Tractor/semi-trailer" = "Heavy Truck",
  "Truck tractor (bobtail" = "Heavy Truck",
  "Single-unit truck (3-or-more axles" = "Heavy Truck",
  "Unknown heavy truck, cannot classify" = "Heavy Truck",
  "Single-unit truck (2-axle, 6-tires" = "Medium Truck",
  "Truck/trailer" = "Medium Truck",
  "Bus (seats for 16 or more, including driver" = "Bus",
  "Bus (seats for 9-15 people, including driver" = "Bus",
  "Motor home/recreational vehicle" = "Light Truck/Van",
  "Light truck(van, mini-van, pickup, sport utility" = "Light Truck/Van",
  "Passenger car" = "Passenger Car",
  "Motorcycle" = "Motorcycle/Moped",
  "MOPED" = "Motorcycle/Moped",
  "All Terrain Vehicle (ATV" = "Other",
  "Snowmobile" = "Other",
  "Low Speed Vehicle" = "Other",
  "Registered farm equipment" = "Other"
)

df$Largest_Vehicle <- VEHICLE_CATEGORY[df$Vehicle_Type]

sort(table(df$Largest_Vehicle), decreasing = TRUE)

# 2. Clean intersection data
# Currently names the intersection if it's at an intersection, making it binary
df$At_Intersection <- ifelse(df$At_Roadway_Intersection != "", 1, 0)
table(df$At_Intersection)

# 3. Clean weather
# Clean raw weather first
df$Weather_Clean <- ifelse(
  is.na(df$Weather_Condition) |
    df$Weather_Condition %in% c("", "Not Reported", "Unknown"),
  NA_character_,
  df$Weather_Condition
)

# Take first condition(before "/")
df$Weather_Simple <- ifelse(
  is.na(df$Weather_Clean),
  NA_character_,
  trimws(sapply(strsplit(df$Weather_Clean, "/"), `[`, 1))
)

# Keep only top 5 weather categories
top_weather <- names(sort(table(df$Weather_Simple), decreasing = TRUE))[1:5]
df <- df[df$Weather_Simple %in% top_weather, ]

sort(table(df$Weather_Simple), decreasing = TRUE)

## Bayesian Modeling

df <- df[!df$Crash_Severity %in% c("Not Reported", "Unknown"), ]
df <- df[trimws(df$Crash_Severity) != "", ]

df$Crash_Severity <- factor(df$Crash_Severity)

table(df$Crash_Severity)

model_df <- df[, c("Crash_Severity", "Largest_Vehicle", "Weather_Simple",
                   "At_Intersection", "Ambient_Light", "Road_Surface_Condition")]

# drop rows with missing predictors
model_df <- model_df[complete.cases(model_df), ]

# convert to factors for categorical modeling
model_df$Crash_Severity <- droplevels(as.factor(model_df$Crash_Severity))
model_df$Largest_Vehicle <- as.factor(model_df$Largest_Vehicle)
model_df$Weather_Simple <- as.factor(model_df$Weather_Simple)
model_df$At_Intersection <- as.factor(model_df$At_Intersection)
model_df$Ambient_Light <- as.factor(model_df$Ambient_Light)
model_df$Road_Surface_Condition <- as.factor(model_df$Road_Surface_Condition)

set.seed(1)

# split by class
fatal_df <- model_df[model_df$Crash_Severity == "Fatal injury", ]
nonfatal_df <- model_df[model_df$Crash_Severity == "Non-fatal injury", ]
pdo_df <- model_df[model_df$Crash_Severity == "Property damage only (none injured)", ]

# downsample to smallest class (fatal) to balance classes
# avoids class dominance but discards data
n_bal <- nrow(fatal_df)

fatal_sample <- fatal_df[sample(nrow(fatal_df), n_bal), ]
nonfatal_sample <- nonfatal_df[sample(nrow(nonfatal_df), n_bal), ]
pdo_sample <- pdo_df[sample(nrow(pdo_df), n_bal), ]

balanced_df <- rbind(fatal_sample, nonfatal_sample, pdo_sample)
balanced_df <- balanced_df[sample(nrow(balanced_df)), ]

balanced_df$Crash_Severity <- droplevels(factor(balanced_df$Crash_Severity))

# inspect default priors
default_prior(
  Crash_Severity ~ Largest_Vehicle + Weather_Simple + At_Intersection + Ambient_Light + Road_Surface_Condition,
  data = balanced_df,
  family = categorical()
)

# fit simplified model (fewer predictors = better sampling stability)
# less complexity, but more reliable inference
model <- brm(
  Crash_Severity ~ Largest_Vehicle + Weather_Simple + At_Intersection,
  data = balanced_df,
  family = categorical(),
  chains = 4,
  cores = 4,
  iter = 3000,
  warmup = 1500,
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)

summary(model)
plot(model)

# simulated class predictions from posterior (Bayesian equivalent of predictions)
post_preds <- posterior_predict(model)
head(post_preds[,1:10])

# posterior class probabilities across all draws
post_probs <- posterior_epred(model)
dim(post_probs)

# average probabilities per class for each observation
mean_probs <- apply(post_probs, c(2,3), mean)
head(mean_probs)

# final predicted class (highest probability)
pred_class <- colnames(mean_probs)[max.col(mean_probs)]
head(pred_class)

actual <- balanced_df$Crash_Severity

# Model achieved ~43.6% accuracy on a balanced 3-class problem.
# Baseline (random guessing) is ~33%, so this is meaningfully better.
# This is acceptable as a baseline for comparison with the MLP.
accuracy <- mean(pred_class == actual)
accuracy

# confusion matrix
conf_mat <- as.data.frame(table(Predicted = pred_class, Actual = actual))

p <- ggplot(conf_mat, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  scale_x_discrete(labels = c(
    "Fatal injury" = "Fatal",
    "Non-fatal injury" = "Non-Fatal",
    "Property damage only (none injured)" = "Property Damage"
  )) +
  scale_y_discrete(labels = c(
    "Fatal injury" = "Fatal",
    "Non-fatal injury" = "Non-Fatal",
    "Property damage only (none injured)" = "Property Damage"
  )) +
  labs(title = "Confusion Matrix (Bayesian Model)",
       x = "Actual",
       y = "Predicted") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    plot.title = element_text(hjust = 0.5)
  )

p

ggsave("bayesian_confusion_matrix.png", plot = p, width = 6, height = 5)

# effect of vehicle type on crash severity probabilities
ce_vehicle <- conditional_effects(model, "Largest_Vehicle", categorical = TRUE)
p_vehicle <- plot(ce_vehicle, plot = FALSE)[[1]] +
  labs(title = "Vehicle Type") +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

# effect of intersections on severity
ce_intersection <- conditional_effects(model, "At_Intersection", categorical = TRUE)
p_intersection <- plot(ce_intersection, plot = FALSE)[[1]] +
  labs(title = "Intersection") +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

# effect of weather
ce_weather <- conditional_effects(model, "Weather_Simple", categorical = TRUE)
p_weather <- plot(ce_weather, plot = FALSE)[[1]] +
  labs(title = "Weather") +
  scale_x_discrete(labels = c(
    "Clear" = "Clear",
    "Cloudy" = "Cloudy",
    "Rain" = "Rain",
    "Sleet, hail (freezing rain or drizzle)" = "Sleet/Hail",
    "Snow" = "Snow"
  )) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

# combine with one shared legend
p_combined <- (p_weather | p_intersection) / p_vehicle +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

p_combined

ggsave(
  "bayesian_conditional_effects_combined.png",
  plot = p_combined,
  width = 18,
  height = 10
)

# example of a predicted probability distribution for one crash
mean_probs[1, ]

# Bayesian model outputs:
# Accuracy ~43.6% (above 33% baseline for 3 classes)
# Confusion matrix shows moderate performance, especially weaker on fatal cases
# Conditional effects plots show:
#   -Vehicle type has strongest impact (motorcycles are more severe)
#   -Intersections increase severity risk
#   -Weather effects are weaker / less consistent
# These plots provide interpretable insights compared to the MLP model





