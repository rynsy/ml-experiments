Name: Ryan Lindsey

Description:
  This project is meant to use two methods of creating a regression line for 
  arbitrary data to predict the value of unseen data. The program is meant 
  to create a regression line using 0th - 4th order polynomials, by way of 
  two different methods (gradient descent, and normal equations).

Problems:
  I had trouble getting my gradient descent method to work by measuring change
  in the cost function between iterations. Instead, I hardcoded the method to
  iterate 100000 times, and had better luck. I didn't have as much luck with
  the normal equation method, which fails on uninvertible matricies. 
  
  I also had trouble generating plots for my equations. I was able to generate 
  plots for the first degree polynomial, but it was inaccurate. If you run my
  program as it is now it will generate the models, and calculate the mean
  squared error for each of the models, and then crash when trying to generate
  the graphs. 

  Since I was unable to generate these graphs for the testing/training data
  before the due date, I didn't attempt to apply this to the housing data since
  I didn't think it would produce anything useful/interesting. 

Files:
  data_wrangler.py - used for importing data
  linear.py - all of my regression functions
  main.py - main driver of the program
  plot.py - contains plotting functions, including one
    for creating a graph that updates the model for each
    iteration. (Need to make some code changes for it to work.)

TODO:
    Fix plots for higher order polynomials. 
    Fix the normal function to work with polynomials of varying degree. 
