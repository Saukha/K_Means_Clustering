# Author: John Sokol
# Machine Learning in R
# 6 February 2018

# K Nearest Clusters unsupervised learning algorithm

# Algorithm steps: 
# 1. Randomly assign each point to one of the k clusters 
# 2. Calculate the centers (or centroids) of each of the clusters
# 3. Each point in the dataset is assigned to the cluster of the nearest centroid 
# 4. Recalculate coordinates of centroid again with new dataset cluster assignments 
# 5. Repeat until dataset cluster assignments do not change (error = 0)

# setwd("./Desktop")

# import dataset using read.csv function
km_data <- read.csv("kmeans_data.csv", sep = "\t")

# function that calculates euclidean distance (traditional distance formula)
euclid_dist <- function(x, y) {
  
    # initialize distance to 0           
    distance = 0        
    
    # loops by number of columns in x (should be 2)
    for (i in (1:length(x))) {  
        
        # subtract each value from their respective columns;
        # square the difference, then add this value to distance
        distance = distance + (x[[i]] - y[[i]])^2 
    }
    # square root of the distance
    distance = sqrt(distance) 
    return (distance)
}

# function that calculates k nearest clusters of a dataset
# k = number of clusters
# returns total within sum of squares and color coded plot based on cluster assignment
km_function <- function(km_data, k) {
  
    # initializing random cluster datapoints based on points from dataset
    # concatentate the x and y coordinates into data frame 
    cluster_loc <- km_data[sample.int(nrow(km_data),k),]
    
    # initialize data frame to store old cluster coordinates
    cluster_loc_old <- as.data.frame(matrix(0, ncol = 2, nrow = k))
    
    # empty data frames to store datapoint cluster ID (0, 1, 2, ...) and datapoint of particular cluster assignment
    cluster_id <- vector("numeric", nrow(km_data))
    clu_assign = as.data.frame(matrix(0, ncol = 2, nrow = nrow(km_data)))
    
    # error: the distance between previous cluster and new calculated cluster locations 
    # algorithm reaches convergence when clusters no longer change (error = 0)
    error <- sum(euclid_dist(cluster_loc, cluster_loc_old))
    
    # while loop will continue until error is not greater than 0 
    while (error > 0) {
      
        # empty vector to store distances between datapoints and clusters 
        euc_dist = vector("numeric", nrow(cluster_loc))
        
        # for loop assigns each dataset value to its closest cluster 
        for (i in 1:nrow(km_data)) {
            
            # distances between each datapoint and the k clusters
            euc_dist <- euclid_dist(km_data[i,], cluster_loc[1:k,])  
            
            # selects the index of the minimum distance; index serves as cluster ID
            cluster_id[i] <- which.min(euc_dist)                     
        }
        
        # assigns old cluster locations before modifying based on data point coordinate means
        cluster_loc_old <- cluster_loc
        
        # for loop iterates through cluster assignments 
        for (i in 1:k) {
            
            #  determines points data frame size based on number of cluster selections 
            cluster_length <- length(which(cluster_id == 1))
            
            # resets points data frame with 0's after every iteration
            clu_assign = as.data.frame(matrix(0, ncol = 2, nrow = nrow(km_data))) 
            
            # for loop iterates through each row in the dataset 
            for (j in 1:nrow(km_data)) {
                
                # checks if the cluster assignment matches the for loop iterator
                if (cluster_id[j] == i) {     
                  
                    # assigns the datapoint of the particular cluster assignment to data frame 'cluster_assign'
                    clu_assign[j,] <- km_data[j,]  
                }
            }
            
            # replaces 0's in dataframe with NA
            clu_assign[clu_assign == 0] <- NA  
            
            # omits NA values from dataframe
            clu_assign <- na.omit(clu_assign)      
            
            # calculates the mean of all x coordinates for the kth cluster 
            cluster_loc[i,1] <- mean(clu_assign[,1])  
            
            # calculcates the mean of all y coordinates for the kth cluster
            cluster_loc[i,2] <- mean(clu_assign[,2])  
            
            # replaces 0's in dataframe with NA
            cluster_loc[cluster_loc == 0] <- NA  
            
            # omits NA values from dataframe
            # this new location data stored in cluster_loc serves as the new location for the k cluster 
            cluster_loc <- na.omit(cluster_loc)      
            
        #  recalcuates the error from new cluster location and old cluster location 
        #  after sufficient while loop iterations, error = 0
        error <- sum(euclid_dist(cluster_loc, cluster_loc_old))
        }
    }
    
    # dataframe to store within sum of squares for each cluster 
    wss = as.data.frame(matrix(0, ncol = 1, nrow = nrow(cluster_loc)))
    
    # loop based on number of clusters 
    for (i in 1:k) {
      
        # temporary dataframe to store distances of individual datapoints to final cluster coordinates
        # within sum of squares (wss): Total distance of data points from their respective cluster centroids 
        wss_intv = as.data.frame(matrix(0, ncol = 1, nrow = nrow(km_data)))
        
        # loop through all data in the dataset
        for (j in 1:nrow(km_data)) {
          
            # verifies datapoint correspondings to cluster in question
            if (cluster_id[j] == i)  
                
                # calculcates euclidean distance between datapoint and cluster
                wss_intv[j,] <-  euclid_dist(km_data[j,], cluster_loc[k,]) 
        }
    
    # replaces 0's in dataframe with NA
    wss_intv[wss_intv == 0] <- NA    
    
    # omits NA in dataframe
    wss_intv <- na.omit(wss_intv)       
    
    # sums all distances to calculate within sum of squares
    wss[k,] <- sum(wss_intv)           
    }
     
    # sums all within sum of squares to calculate total within sum of squares
    totalss <- sum(wss) 
    
    return(totalss)
    plot(km_data$Col1,km_data$Col2, col = cluster_id)
    points(cluster_loc, pch = 19)
}

# function that displays elbow plot; uses built in kmeans() instead of self created K nearest clusters function 
elbow_func_built_in <- function(dataset, clusters) {
  # initialize total within sum of squares 
  wss = 0
  
  # loop iterates according to argument given number of clusters 
  for (i in 1:clusters) {
    
    # call the kmeans function; stores total within sum of squares in initialized vector
    kmeans_output <- kmeans(dataset, centers = i, nstart = 25)
    wss[i] <- kmeans_output$tot.withinss
  }
  
  # plots number of clusters vs. wss; ideal number of clusters is where 
  # a sharp "elbow" occurs in the plot
  plot(1:clusters, wss, type = 'b', pch = 19, xlab = "Number of Clusters", 
  ylab = "Total Within Sum of Squares")
}

# function that displays elbow plot of number of clusters vs. total within sum of squares
# uses self created k means clusters function 
# prone to totalss variability due to randomized inital cluster locations;
# plan to implement nstart km_function parameter in future work to prevent this
elbow_func <- function(km_data, clusters) {
  
  within_sum_squares = vector("numeric", clusters)
  
    for (i in 1:clusters) {
        kmeans_output <- km_function(km_data, clusters)
        within_sum_squares[i] <- kmeans_output
    }
    plot(1:clusters, within_sum_squares, type = 'b', pch = 19, xlab = "Number of Clusters", 
    ylab = "Total Within Sum of Squares")
}
