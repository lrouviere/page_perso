library(shiny)
library(colourpicker)

# Define UI for application that draws a histogram
shinyUI(
  # navbarPage
  navbarPage("Second app",
             
             # premier onglet Data
             tabPanel("Data", 
                      navlistPanel(
                        widths = c(2, 10), 
                        tabPanel("Table", dataTableOutput("table")),
                        tabPanel("Summary",verbatimTextOutput("summary"))
                      )
             ), 
             
             # second onglet Visualisation
             tabPanel("Visualisation", 
                      
                      fluidRow(
                        # premier colonne
                        column(width = 3, 
                               # wellPanel pour griser
                               wellPanel(
                                 sliderInput("bins",
                                             "Number of bins:",
                                             min = 1,
                                             max = 50,
                                             value = 30),
                                 
                                 # input pour la couleur
                                 colourInput(inputId = "color", label = "Couleur :", value = "purple"),
                                 
                                 # titre du graphique
                                 textInput(inputId = "titre", label = "Titre :", value = "Histogram"),
                                 
                                 # selection de la colonne
                                 radioButtons(inputId = "var", label = "Variable : ", choices = colnames(faithful))
                               )
                        ), 
                        # deuxieme colonne
                        column(width = 9, 
                               tabsetPanel(
                                 tabPanel("Histogram", 
                                          plotOutput("distPlot"),
                                          # classes (div centr??e)
                                          div(textOutput("n_bins"), align = "center")
                                 ),
                                 tabPanel("Boxplot", plotOutput("boxplot"))
                               )
                        )
                      )
             )
  )
)