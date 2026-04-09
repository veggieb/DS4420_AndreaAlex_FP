## Andrea Keiper and Alex Weinstein
## DS4420 Final Project
## April 16, 2026

## MODEL 2: BAYESION ML
library(data.table)
library(stringr)

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
# Simplify: strip multi-condition combos (e.g. "Clear/Rain" -> "Clear")







## Figure out the distribution 


## use brm and sampling to complete the model, categorical predicting the severity? 
