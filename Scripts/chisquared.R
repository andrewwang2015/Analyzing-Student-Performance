library(ggplot2)
library(waffle)
library(plyr)
library(gridExtra)
library(extrafont)
library(dplyr)
library(MASS)

setwd("/Users/michaelli/Desktop/Student")

math = read.csv("math_transformed.csv")
por = read.csv("port_transformed.csv")

total = rbind(math, por)

total$alc <- total$Dalc * 5/7 + total$Walc *2/7

tblD = table(total$G1, total$Dalc)
tblW = table(total$G1, total$Walc)
tbl = table(total$G1, total$alc)

chisq.test(tblD)
chisq.test(tblW)
chisq.test(tbl)
