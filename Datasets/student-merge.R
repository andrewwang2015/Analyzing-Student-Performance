d1=read.table("student-mat.csv",sep=";",header=TRUE)
write.csv(d1, 'math.csv')

d2=read.table("student-por.csv",sep=";",header=TRUE)
write.csv(d2, 'port.csv')
d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students
write.csv(d3, 'allstudents.csv')
