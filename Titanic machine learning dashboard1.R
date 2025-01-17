library(shiny)
library(shinydashboard)
library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(rpart)
library(xgboost)
library(ROCR)
library(ggplot2)

# Load the Titanic dataset
titanic <- read.csv("titanic.csv")
titanic <- titanic %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  mutate(
    Survived = as.factor(Survived),
    Pclass = as.factor(Pclass),
    Sex = as.factor(Sex),
    Embarked = as.factor(Embarked)
  )
titanic <- na.omit(titanic)

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Titanic ML Dashboard",theme),
  dashboardSidebar(
    sidebarMenu(
      menuItem("EDA", tabName = "eda", icon = icon("chart-bar")),
      menuItem("Model Training", tabName = "model", icon = icon("cogs")),
      menuItem("Predictions", tabName = "predict", icon = icon("table"))
    )
  ),
  # Main body
  dashboardBody(
    tabItems(
      # EDA tab
      tabItem(
        tabName = "eda",
        fluidRow(
          box(
            title = "Bar Plot Survived",
            status = "primary",
            plotOutput("barPlotSurvived"),
            width = 6
          ),
          box(
            title = "Age Distribution",
            status = "primary",
            plotOutput("ageDist"),
            width = 6
          )
        ),
        fluidRow(
          box(
            title = "Fare Distribution",
            status = "primary",
            plotOutput("fareDist"),
            width = 6
          ),
          box(
            title = "Pclass Distribution",
            status = "primary",
            plotOutput("pclassDist"),
            width = 6
          )
        )
      ),
      # Model training
      tabItem(
        tabName = "model",
        fluidRow(
          box(
            title = "Model Selection",
            status = "primary",
            width = 4,
            selectInput(
              "model",
              "Choose a Model:",
              choices = c(
                "Logistic Regression",
                "Naive Bayes",
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "SVM"
              )
            )
          ),
          box(
            title = "Train Model",
            status = "primary",
            width = 6,
            actionButton("train", "Train Model")
          ),
          box(
            title = "Metrics",
            status = "primary",
            dataTableOutput("metricsTable"),
            width = 12
          )
        )
      ),
      # Predictions tab
      tabItem(
        tabName = "predict",
        fluidRow(
          box(
            actionButton("predict", "Predict Survival"),
            title = "Input Features",
            status = "primary",
            width = 4,
            sliderInput("age", "Age:", min = min(titanic$Age), max = max(titanic$Age), value = median(titanic$Age)),
            selectInput("pclass", "Pclass:", choices = levels(titanic$Pclass)),
            selectInput("sex", "Sex:", choices = levels(titanic$Sex)),
            sliderInput("fare", "Fare:", min = min(titanic$Fare), max = max(titanic$Fare), value = median(titanic$Fare)),
            numericInput("sibsp", "Siblings/Spouses Aboard:", value = 0),
            numericInput("parch", "Parents/Children Aboard:", value = 0),
            selectInput("embarked", "Embarked:", choices = levels(titanic$Embarked))
          ),
          box(
            title = "Prediction Results",
            status = "primary",
            verbatimTextOutput("prediction"),
            width = 8
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  # EDA outputs
  output$barPlotSurvived <- renderPlot({
    ggplot(titanic, aes(x = Survived)) +
      geom_bar(fill = "lightblue") +
      theme_dark()
  })
  
  output$ageDist <- renderPlot({
    ggplot(titanic, aes(x = Age)) +
      geom_histogram(fill = "lightgreen", bins = 30) +
      theme_minimal()
  })
  
  output$fareDist <- renderPlot({
    ggplot(titanic, aes(x = Fare)) +
      geom_histogram(fill = "lightcoral", bins = 30) +
      theme_minimal()
  })
  
  output$pclassDist <- renderPlot({
    ggplot(titanic, aes(x = Pclass)) +
      geom_bar(fill = "gold") +
      theme_minimal()
  })
  
  # Train model
  model <- reactiveValues()
  observeEvent(input$train, {
    set.seed(123)
    trainIndex <- createDataPartition(titanic$Survived, p = 0.8, list = FALSE)
    trainData <- titanic[trainIndex, ]
    testData <- titanic[-trainIndex, ]
    
    if (input$model == "Logistic Regression") {
      model$fit <- train(
        Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
        data = trainData,
        method = "glm",
        family = "binomial"
      )
    } else if (input$model == "Naive Bayes") {
      model$fit <- train(
        Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
        data = trainData,
        method = "nb"
      )
    } else if (input$model == "Decision Tree") {
      model$fit <- train(
        Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
        data = trainData,
        method = "rpart"
      )
    } else if (input$model == "Random Forest") {
      model$fit <- train(
        Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
        data = trainData,
        method = "rf"
      )
    } else if (input$model == "XGBoost") {
      model$fit <- train(
        Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
        data = trainData,
        method = "xgbTree"
      )
    } else if (input$model == "SVM") {
      model$fit <- train(
        Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
        data = trainData,
        method = "svmRadial"
      )
    }
    
    predictions <- predict(model$fit, newdata = testData)
    cm <- confusionMatrix(predictions, testData$Survived)
    
    metrics <- data.frame(
      Model = input$model,
      Accuracy = cm$overall["Accuracy"],
      Precision = cm$byClass["Pos Pred Value"],
      Recall = cm$byClass["Sensitivity"],
      `F-1 Score`= 2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) / (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"]),
      AUC = if ("prob" %in% colnames(predict(model$fit, newdata = testData, type = "prob"))) {
        pred <- prediction(predict(model$fit, newdata = testData, type = "prob")[, 2], testData$Survived)
        performance(pred, "auc")@y.values[[1]]
      } else {
        NA
      }
    )
    
    output$metricsTable <- renderDataTable({
      metrics
    })
  })
  
  # Prediction logic
  observeEvent(input$predict, {
    inputData <- data.frame(
      Pclass = factor(input$pclass, levels = levels(titanic$Pclass)),
      Sex = factor(input$sex, levels = levels(titanic$Sex)),
      Age = input$age,
      SibSp = input$sibsp,
      Parch = input$parch,
      Fare = input$fare,
      Embarked = factor(input$embarked, levels = levels(titanic$Embarked))
    )
    
    if (is.null(model$fit)) {
      output$prediction <- renderPrint({
        "Please train a model first."
      })
    } else {
      prediction <- predict(model$fit, newdata = inputData)
      output$prediction <- renderPrint({
        ifelse(prediction == "1", "Survived", "Did Not Survive")
      })
    }
  })
}

# Run app
shinyApp(ui, server)
