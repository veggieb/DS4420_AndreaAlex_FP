library(shiny)
library(ggplot2)
library(dplyr)
library(tidyr)

threshold_df  <- read.csv("mlp_threshold_metrics.csv")
loss_df       <- read.csv("mlp_loss_curves.csv")
importance_df <- read.csv("mlp_feature_importance.csv")
vehicle_df      <- read.csv("bayes_vehicle.csv")
intersection_df <- read.csv("bayes_intersection.csv")
weather_df      <- read.csv("bayes_weather.csv")
confusion_df    <- read.csv("bayes_confusion.csv")

# Clean up Bayesian data
colnames(intersection_df)[colnames(intersection_df) == "At_Intersection"] <- "label"
intersection_df$label <- ifelse(intersection_df$label == "1", "At Intersection", "Not at Intersection")

colnames(vehicle_df)[colnames(vehicle_df) == "Largest_Vehicle"] <- "label"
colnames(weather_df)[colnames(weather_df) == "Weather_Simple"]  <- "label"
weather_df$label <- gsub("Sleet, hail.*", "Sleet/Hail", weather_df$label)

# columns we need from conditional effects
ce_cols <- c("label", "cats__", "estimate__")

prep_bayes <- function(df) {
  df <- df[, ce_cols]
  colnames(df) <- c("label", "severity", "probability")
  df$severity <- recode(df$severity,
                        "Fatal injury"= "Fatal",
                        "Non-fatal injury"= "Non-Fatal",
                        "Property damage only (none injured)"= "Property Damage"
  )
  df$severity <- factor(df$severity, levels = c("Fatal", "Non-Fatal", "Property Damage"))
  df
}

veh_long  <- prep_bayes(vehicle_df)
int_long  <- prep_bayes(intersection_df)
wea_long  <- prep_bayes(weather_df)

# Confusion matrix size fix
cm_wide <- confusion_df %>%
  mutate(Actual = recode(Actual,
                         "Fatal injury" = "Fatal",
                         "Non-fatal injury" = "Non-Fatal",
                         "Property damage only (none injured)" = "Prop. Damage"
  ),
  Predicted = recode(Predicted,
                     "Fatal injury"= "Fatal",
                     "Non-fatal injury" = "Non-Fatal",
                     "Property damage only (none injured)" = "Prop. Damage"
  ))

# Shiny app 
ui <- fluidPage(
  titlePanel("Boston Crash Severity: A Machine Learning Analysis"),
  tabsetPanel(
    
    # First page overview
    tabPanel("Overview",
             br(),
             p("DS 4420 · Andrea Keiper & Alex Weinstein · April 2026"),
             br(),
             p("Using crash characteristics from over 74,000 Boston-area crash records (2010–2026),
        this project predicts whether a crash results in fatality or injury and identifies which conditions
        make crashes most dangerous. Two models are compared: a manually implemented
        Multilayer Perceptron (MLP) in Python and a Bayesian categorical logistic regression in R."),
             br(),
             h4("Model 1: Multilayer Perceptron (Python)"),
             p("Binary classifier (injury vs. property damage). 1 hidden layer, 32 nodes, ReLU activation,
        sigmoid output. Trained with gradient descent over 2,000 epochs. Learning rate: 0.01.
        Decision threshold: 0.40."),
             tags$ul(
               tags$li("Test Accuracy: 63.9%"),
               tags$li("F1 Score: 0.291"),
               tags$li("True Positives (injury crashes caught): 1,309"),
               tags$li("Predictors: vehicle type, ambient light, road surface, weather, intersection presence")
             ),
             br(),
             h4("Model 2: Bayesian Categorical Model (R)"),
             p("Multinomial classifier (fatal / non-fatal / property damage). Fit using brms with
        family = categorical(). 4 chains × 3,000 iterations. Balanced dataset by downsampling
        to ~335 records per class."),
             tags$ul(
               tags$li("Accuracy: 43.6%"),
               tags$li("Predictors: vehicle type, weather, intersection presence")
             ),
             br(),
             h4("Key Finding"),
             p("Both models identified vehicle type as the strongest predictor of crash severity.
        The Bayesian model further showed that motorcycle crashes carry a disproportionately
        high probability of fatal outcomes (75.6%), and intersection crashes are meaningfully
        more likely to result in injury."),
             a("Access the Massgov crash open data portal here.", href = "https://apps.crashdata.dot.mass.gov/cdp/extract")
    ),
    
    # Tab 2: MLP
    tabPanel("MLP Model",
             br(),
             h4("Decision Threshold Explorer"),
             p("The model outputs a probability per crash. The threshold determines when that
        probability is classified as 'injury'. Lower thresholds catch more injury crashes
        (higher recall) at the cost of more false alarms."),
             br(),
             sliderInput("threshold", "Decision Threshold",
                         min = 0.05, max = 0.95, value = 0.40, step = 0.01),
             br(),
             fluidRow(
               column(3, wellPanel(
                 h5("Accuracy"),  textOutput("acc")
               )),
               column(3, wellPanel(
                 h5("Precision"), textOutput("prec")
               )),
               column(3, wellPanel(
                 h5("Recall"),    textOutput("rec")
               )),
               column(3, wellPanel(
                 h5("F1 Score"),  textOutput("f1")
               ))
             ),
             br(),
             h4("Confusion Matrix"),
             tableOutput("cm_table"),
             br(),
             h4("Precision, Recall & F1 vs Threshold"),
             plotOutput("threshold_plot", height = "300px"),
             br(),
             h4("Train vs Test Loss (2,000 Epochs)"),
             plotOutput("loss_plot", height = "300px"),
             br(),
             h4("Feature Group Importance"),
             plotOutput("importance_plot", height = "280px")
    ),
    
    # Tab 3: Bayesian
    tabPanel("Bayesian Model",
             br(),
             p("Estimated posterior mean probabilities from the fitted brms model.
        Each bar shows the probability of each severity outcome given the predictor value."),
             br(),
             h4("Effect of Vehicle Type"),
             plotOutput("bayes_vehicle", height = "320px"),
             br(),
             h4("Effect of Intersection Presence"),
             plotOutput("bayes_intersection", height = "200px"),
             br(),
             h4("Effect of Weather"),
             plotOutput("bayes_weather", height = "280px"),
             br(),
             h4("Confusion Matrix: Bayesian Model"),
             p("Balanced dataset: ~335 records per class. Accuracy: 43.6% vs 33% baseline."),
             plotOutput("bayes_cm", height = "320px")
    )
  )
)

# Server
server <- function(input, output) {
  
  # row for chosen threshold
  row <- reactive({
    threshold_df[which.min(abs(threshold_df$threshold - input$threshold)), ]
  })
  
  # Metrics
  output$acc  <- renderText(paste0(round(row()$accuracy * 100, 1), "%"))
  output$prec <- renderText(round(row()$precision, 3))
  output$rec  <- renderText(round(row()$recall, 3))
  output$f1   <- renderText(round(row()$f1, 3))
  
  # Confusion matrix table
  output$cm_table <- renderTable({
    r <- row()
    data.frame(
      ` ` = c("Actual: No Injury", "Actual: Injury"),
      `Pred: No Injury` = c(r$TN, r$FN),
      `Pred: Injury`    = c(r$FP, r$TP),
      check.names = FALSE
    )
  }, striped = FALSE, bordered = TRUE)
  
  # Threshold curves
  output$threshold_plot <- renderPlot({
    df_long <- threshold_df %>%
      select(threshold, precision, recall, f1) %>%
      pivot_longer(-threshold, names_to = "metric", values_to = "value")
    ggplot(df_long, aes(x = threshold, y = value, color = metric)) +
      geom_line(linewidth = 0.9) +
      geom_vline(xintercept = input$threshold, linetype = "dashed", color = "gray40") +
      scale_color_manual(values = c(precision = "forestgreen", recall = "maroon", f1 = "purple")) +
      labs(x = "Threshold", y = "Score", color = NULL) +
      theme_minimal()
  })
  
  # Loss curves
  output$loss_plot <- renderPlot({
    loss_df %>%
      pivot_longer(c(train_loss, test_loss), names_to = "split", values_to = "loss") %>%
      mutate(split = recode(split, train_loss = "Train", test_loss = "Test")) %>%
      ggplot(aes(x = epoch, y = loss, color = split)) +
      geom_line(linewidth = 0.8) +
      scale_color_manual(values = c(Train = "pink", Test = "lightgreen")) +
      labs(x = "Epoch", y = "Avg Cross-Entropy Loss", color = NULL) +
      theme_minimal()
  })
  
  # Feature importance
  output$importance_plot <- renderPlot({
    importance_df %>%
      arrange(importance) %>%
      mutate(feature = factor(feature, levels = feature)) %>%
      ggplot(aes(x = importance, y = feature)) +
      geom_col() +
      geom_text(aes(label = round(importance, 2)), hjust = -0.1, size = 3.5) +
      scale_fill_manual(values = c(
        "Vehicle Type"   = "#4a4a4a",
        "Road Surface"   = "#6b8f71",
        "Ambient Light"  = "#c47c2b",
        "Weather"        = "#7b6ea6",
        "At Intersection"= "#b85450"
      )) +
      xlim(0, max(importance_df$importance) * 1.15) +
      labs(x = "Sum of Avg Absolute W1 Weights", y = NULL) +
      theme_minimal()
  })
  
  # Bayesian bar charts
  bayes_bar <- function(df) {
    ggplot(df, aes(x = label, y = probability, fill = severity)) +
      geom_col(position = "dodge") +
      scale_fill_manual(values = c(Fatal = "darkred", `Non-Fatal` = "darkblue",
                                   `Property Damage` = "grey")) +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
      labs(x = NULL, y = "Estimated Probability", fill = NULL) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 20, hjust = 1))
  }
  
  output$bayes_vehicle      <- renderPlot(bayes_bar(veh_long))
  output$bayes_intersection <- renderPlot(bayes_bar(int_long))
  output$bayes_weather      <- renderPlot(bayes_bar(wea_long))
  
  # Bayesian confusion matrix heatmap
  output$bayes_cm <- renderPlot({
    ggplot(cm_wide, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), size = 5) +
      scale_fill_gradient(low = "lightblue", high = "darkblue") +
      labs(x = "Actual", y = "Predicted", fill = "Count") +
      theme_minimal()
  })
}

shinyApp(ui, server)


