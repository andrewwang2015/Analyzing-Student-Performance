# Analyzing-Student-Performance

Using this student performance dataset from http://archive.ics.uci.edu/ml/datasets/Student+Performance, I attempt to predict student performance based on demographic, social, and school related features. 

- Motivation: Predict academic performance so schools can give personalized learning experiences to all students. Learning on this dataset will allow educators and schools flag students at risk for poor academic performance early and thus shuttle resources to adjust these concerns.

- Secondary objective: Look at alcohol's effect on school performance. Drinking age of Portugal is 18, while the drinking age in America is 21. Is there a legitimate reason why high school students should not be allowed to drink?

## Workflow:
- Preprocessing: labeled binary variables, and one hot vector encoded categorical variables. 
- Handling missing data: Applied a KNN (k = 10) approach to impute missing values. About 10% of students in our math dataset had missing values while ~5% of students in our portuguese dataset had missing values. 
- Early detection of students at risk / Predicting G1 (quarter 1 grades): Apply linear regression, SVM, trees, and KNN models to predict G1. See which factors are most significant.
- Predicting progress over time: Apply similar models to predict the change from G1 (quarter 1 grades) to G3 (quarter 3 grades). Which factors most motivate students to improve and likewise, which factors encourag slacking?
- Analyzing alcohol: Produce visualizations of alchohol vs. student grades in both classes

## Results:
See Student_Performance.pdf for pitch/ presentation that summarizes our project.


