# First, lets get a better idea of what we are dealing with
print(boston.keys())
print(boston['feature_names'])

for i in range(X.shape[1]):
    plt.plot(X[:,i],y,'o',color="C"+str(i % 10))
    plt.show()
    
# It seems like regression algorithm deals with features overlaping response values
# This make sense beacuse KNN simply picks the the number of neighors you tell it without optimizing the number of 
# neighors or weighting them. This is sometimes called "lazy learning"
# Some thing we could do to address this:
# -Implement weighting into KNN
# -Optimize the number of neighbors ourselves (let try and few now)
# -Git rid of some features (feature selection)
# -User a different method (model selection)
