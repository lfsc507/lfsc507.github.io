x_index = 3
y_index = 0

for x_index in range(0,4):
    for y_index in range(x_index+1,4):
        for label in range(len(iris.target_names)):
            plt.scatter(iris.data[iris.target==label, x_index], 
                        iris.data[iris.target==label, y_index],
                        label=iris.target_names[label])

        plt.xlabel(iris.feature_names[x_index])
        plt.ylabel(iris.feature_names[y_index])
        plt.legend(loc='upper left')
        plt.show()
