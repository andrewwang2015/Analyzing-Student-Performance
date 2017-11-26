library(ggplot2)
library(waffle)
library(plyr)
library(gridExtra)
library(extrafont)
library(dplyr)

setwd('C:/Users/Andrew/Desktop/Caltech/BEM150/StudentPerformance')
math = read.csv("Datasets/math_transformed.csv")
por = read.csv("Datasets/port_transformed.csv")

total = rbind(math, por)

total$Dalc <- as.factor(total$Dalc)
total$Dalc <- mapvalues(total$Dalc, 1:5, c("Very Low (1)", "Low (2)", "Medium (3)", 
                                                     "High (4)", "Very High (5)"))
total$Walc <- as.factor(total$Walc)
total$Walc <- mapvalues(total$Walc, 1:5, c("Very Low (1)", "Low (2)", "Medium (3)", 
                                           "High (4)", "Very High (5)"))

# Dealing with Weekday consumption
AlcConsumption <- as.data.frame(table(total$Dalc))
colnames(AlcConsumption) <- c("Consumption", "Frequency")

values <- as.numeric(AlcConsumption$Frequency)
names(values) <- AlcConsumption$Consumption
values <- round(values/10.05)

colors <- c("#00d134","#45d100","#d1d100","#d17a00","#d11b00")

weekday <- waffle(values, rows = 5, title = "Student Weekday Alcohol Intake", 
            glyph_size = 8, xlab = "1 tile is about 10 students", colors = colors, legend_pos = "top")

# Dealing with Weekend consumption
AlcConsumption <- as.data.frame(table(total$Walc))
colnames(AlcConsumption) <- c("Consumption", "Frequency")

values <- as.numeric(AlcConsumption$Frequency)
names(values) <- AlcConsumption$Consumption
values <- round(values/10)

colors <- c("#00d134","#45d100","#d1d100","#d17a00","#d11b00")

weekend <- waffle(values, rows = 5, title = "Student Weekend Alcohol Intake", 
                      glyph_size = 8, xlab = "1 tile is about 10 students", colors = colors, legend_pos = "top")

png(filename="alcohol.png")
grid.arrange(weekday, weekend, nrow=2)
dev.off()


