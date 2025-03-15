import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D




# Import data file we have to header set none otherwise we lost first data
data = pd.read_csv('./data/iris.data', header = None)   

# Change columns names
data.columns = ['Sepal length', 'Sepal width', 'Petal length','Petal width','Species']


print("Task 1 --> Number of missing values : ")
nullValue = 0
nullValue = data.isnull().sum()
print(nullValue['Sepal width'])

# Null value cunts for each Iris
nullCountIrisSetosa = data.groupby('Species').get_group('Iris-setosa').isnull().sum()
nullCountIrisVersicolor = data.groupby('Species').get_group('Iris-versicolor').isnull().sum()
nullCountIrisVirginica = data.groupby('Species').get_group('Iris-virginica').isnull().sum()

# Make sure we don't have null values in 'Sepal width'
if (nullValue['Sepal width'] != 0) :
    data = data.dropna()



# Minimum value for each Iris
print('----------------------------------------------------------------')
print("Task 2 --> Minimum value : ")
min = data.groupby('Species')['Sepal width'].min()
print(min)


# Q1 is the 25th percentile of the 'Sepal width' column for each species
print('----------------------------------------------------------------')
print("Task 3 -->  First quartile (Q1) values : ")
q1 = data.groupby('Species')['Sepal width'].quantile(0.25)
print(q1)


# Calculate median 
print('----------------------------------------------------------------')
print("Task 4 -->  Median value : ")
median = data.groupby('Species')['Sepal width'].median()
print(median)


# Q3 is the 75th percentile of the 'Sepal width' column for each species and to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 5 -->  Third quartile (Q3) : ")
q3 = round(data.groupby('Species')['Sepal width'].quantile(0.75),3)
print(q3)


# We use 0.95 to get 95th percentile value and to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 6 -->  95th percentile : ")
p95 = round(data.groupby('Species')['Sepal width'].quantile(0.95),3)
print(p95)


# Calculate maximum sepal width for each species
print('----------------------------------------------------------------')
print("Task 7 -->  Maximum value : ")
max = data.groupby('Species')['Sepal width'].max()
print(max)


# Calculate mean sepal width for each species and round it to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 8 -->  Mean value : ")
mean = round(data.groupby('Species')['Sepal width'].mean(),3)
print(mean)


# Calculate Data rage and to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 9 -->  Data range is  : ")
range = round (max-min,3)
print(range)


# Calculate Interquartile range (IQR) and to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 10 -->  Interquartile range (IQR)  : ")
IQR = round(q3 - q1,3)
print(IQR)


# Calculate Sample standard deviation and to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 11 -->  Sample standard deviation  : ")
# Default value for ddof is 1 and we don't need to write it
sampleStandardDeviation = round(data.groupby('Species')['Sepal width'].std(),3)
print(sampleStandardDeviation)


# Calculate Population standard deviation and to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 12 -->  Population standard deviation  : ")
populationStandardDeviation = round(data.groupby('Species')['Sepal width'].std(ddof=0),3)
print(populationStandardDeviation)

# Calculate Median absolute deviation(MAD) for each species and round them to 3 to three decimal places
print('----------------------------------------------------------------')
print("Task 13 -->  Median absolute deviation (MAD)  : ")
madForIrisSetosa = round(data['Sepal width'][data['Species']=='Iris-setosa'].apply(lambda x: abs(x - median['Iris-setosa'])).median(),3)
madForIrisVersicolor = round(data['Sepal width'][data['Species']=='Iris-versicolor'].apply(lambda x: abs(x - median['Iris-versicolor'])).median(),3)
madForIrisVirginica = round(data['Sepal width'][data['Species']=='Iris-virginica'].apply(lambda x: abs(x - median['Iris-virginica'])).median(),3)
print('Iris-setosa',madForIrisSetosa)
print('Iris-versicolor',madForIrisVersicolor)
print('Iris-virginica',madForIrisVirginica)


# Export to CSV file
statisticsFlie = pd.DataFrame([
    {'label' :'Iris-setosa','missing':nullCountIrisSetosa['Sepal width'],'min': min['Iris-setosa'],'q1': q1['Iris-setosa'],'med': median['Iris-setosa'],'q3': q3['Iris-setosa'],'p95': p95['Iris-setosa'],'max': max['Iris-setosa'],'mean': mean['Iris-setosa'],'range': range['Iris-setosa'],'iqr': IQR['Iris-setosa'],'std': sampleStandardDeviation['Iris-setosa'],'std_pop': populationStandardDeviation['Iris-setosa'],'mad': madForIrisSetosa},
    {'label' :'Iris-versicolor','missing':nullCountIrisVersicolor['Sepal width'],'min': min['Iris-versicolor'],'q1': q1['Iris-versicolor'],'med': median['Iris-versicolor'],'q3': q3['Iris-versicolor'],'p95': p95['Iris-versicolor'],'max': max['Iris-versicolor'],'mean': mean['Iris-versicolor'],'range': range['Iris-versicolor'],'iqr': IQR['Iris-versicolor'],'std': sampleStandardDeviation['Iris-versicolor'],'std_pop': populationStandardDeviation['Iris-versicolor'],'mad':madForIrisVersicolor},
    {'label' :'Iris-virginica','missing':nullCountIrisVirginica['Sepal width'],'min': min['Iris-virginica'],'q1': q1['Iris-virginica'],'med': median['Iris-virginica'],'q3': q3['Iris-virginica'],'p95': p95['Iris-virginica'],'max': max['Iris-virginica'],'mean': mean['Iris-virginica'],'range': range['Iris-virginica'],'iqr': IQR['Iris-virginica'],'std': sampleStandardDeviation['Iris-virginica'],'std_pop': populationStandardDeviation['Iris-virginica'],'mad':madForIrisVirginica},
    
])


statisticsFlie.to_csv('./dist/statistics.csv', index=False)

print("file statistics.csv created successfully!  \n")




################################ assignment number 2 #####################################

print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
print('||||||||||||||||||||||||assignment number 2|||||||||||||||||||||||||')
print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
# Correlation matrix for 'Sepal length','Sepal width','Petal length','Petal width' 
correlationMatrix = round(data[['Sepal length','Sepal width','Petal length','Petal width']].corr(),3)
print("correation Martix is : ")
print(correlationMatrix)



# Create a copy of the matrix and set the original diameter value to NaN
correlationMatrixNoDiag = correlationMatrix.mask(np.eye(correlationMatrix.shape[0], dtype=bool))

# Find minimum and maximum correlation values ​​(ignore diagonal values)
minCorr = correlationMatrixNoDiag.min().min()
maxCorr = correlationMatrixNoDiag.max().max()

print("\nminimum correlation is : ", minCorr)
print("maximum correlation is : ", maxCorr)



# Export to CSV file
correlationMatrix.to_csv('./dist/correlations.csv', index=False, header=False)


###############################  assignment number 3  ###############################

print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
print('||||||||||||||||||||||||assignment number 3|||||||||||||||||||||||||')
print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

# Count the number of samples of each species
labelCounts = data['Species'].value_counts()
# Create a bar plot
plt.figure(figsize=(10, 7))
labelCounts.plot(kind='bar', color=['blue', 'red', 'green'])
plt.xlabel('Species')
plt.ylabel('number of samples')
plt.title('Species distribution in the dataset')
plt.show()




# Histogram for Petal length
plt.figure(figsize=(10, 7))
data['Petal length'].hist(bins=20, color='green', edgecolor='black')
plt.xlabel('Petal Length')
plt.ylabel('numbers')
plt.title('Histogram for Petal Length')
plt.show()


# Histogram for Sepal width
plt.figure(figsize=(10, 7))
data['Sepal width'].hist(bins=20, color='yellow', edgecolor='black')
plt.xlabel('Sepal Width')
plt.ylabel('numbers')
plt.title('Histogram for Sepal width')
plt.show()






# First plot initialization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


# Determining the ranges of the axes of petal length and sepal length
x = data['Petal length']
y = data['Sepal length']

# Create a histogram
hist, xedges, yedges = np.histogram2d(x, y, bins=20)


# Set the center of each dimension on the z-axis to display the number of samples
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0


# The height of each column to display the number of samples
dx = dy = 0.5
dz = hist.ravel()


# Drawing 3D Histogram plot
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='lightblue', edgecolor='k')
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Length')
ax.set_zlabel('number of samples')
plt.title('3D histogram for petal length and sepal length')
plt.show()






# Box plot for petal length according to species
plt.figure(figsize=(10, 7))
data.boxplot(column='Petal length', by='Species', grid=False, patch_artist=True)
plt.xlabel('Species')
plt.ylabel('Petal Length')
plt.title('Box plot of petal length for each species')  
plt.show()

# Box plot for sepal width according to species
plt.figure(figsize=(10, 7))
data.boxplot(column='Sepal width', by='Species', grid=False, patch_artist=True)
plt.xlabel('Species')
plt.ylabel('Sepal Width')
plt.title('Box plot of Sepal Width for each species')
plt.show()

## THIS IS MY FIRST CODE WITCHE COULD NOT DO TASK 
# # A plot of the probability distribution of petal length for each species
# plt.figure(figsize=(10, 6))
# sns.kdeplot(data=data, x='Petal length', hue='Species', fill=True)
# plt.title('Probability estimate (PDF) for petal length in each species')
# plt.xlabel('Petal Length')
# plt.ylabel('Probability density')
# plt.show()





# Two dimensional box plot for petal length and sepal width
x = data['Petal length'].values
y = data['Sepal width'].values

def boxplot_2d(x, y, ax, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    # Draw box
    box = Rectangle(
        (xlimits[0], ylimits[0]),
        (xlimits[2] - xlimits[0]),
        (ylimits[2] - ylimits[0]),
        ec='k',
        facecolor='blue',
        zorder=0
    )
    ax.add_patch(box)

    # X middleline
    vline = Line2D(
        [xlimits[1], xlimits[1]], [ylimits[0], ylimits[2]],
        color='k', zorder=1
    )
    ax.add_line(vline)

    # Y meddileline
    hline = Line2D(
        [xlimits[0], xlimits[2]], [ylimits[1], ylimits[1]],
        color='k', zorder=1
    )
    ax.add_line(hline)

    # Central point
    ax.plot([xlimits[1]], [ylimits[1]], color='k', marker='o')

    # Calculate IRQ for axes
    iqr_x = xlimits[2] - xlimits[0]
    iqr_y = ylimits[2] - ylimits[0]

    # Draw x and y vectors
    left = np.min(x[x > xlimits[0] - whis * iqr_x])
    right = np.max(x[x < xlimits[2] + whis * iqr_x])
    bottom = np.min(y[y > ylimits[0] - whis * iqr_y])
    top = np.max(y[y < ylimits[2] + whis * iqr_y])

    # Left and right 
    ax.add_line(Line2D([left, xlimits[0]], [ylimits[1], ylimits[1]], color='k', zorder=1))
    ax.add_line(Line2D([right, xlimits[2]], [ylimits[1], ylimits[1]], color='k', zorder=1))
    ax.add_line(Line2D([left, left], [ylimits[0], ylimits[2]], color='k', zorder=1))
    ax.add_line(Line2D([right, right], [ylimits[0], ylimits[2]], color='k', zorder=1))

    # Up and down 
    ax.add_line(Line2D([xlimits[1], xlimits[1]], [bottom, ylimits[0]], color='k', zorder=1))
    ax.add_line(Line2D([xlimits[1], xlimits[1]], [top, ylimits[2]], color='k', zorder=1))
    ax.add_line(Line2D([xlimits[0], xlimits[2]], [bottom, bottom], color='k', zorder=1))
    ax.add_line(Line2D([xlimits[0], xlimits[2]], [top, top], color='k', zorder=1))

    # Draw outliers
    mask = (x < left) | (x > right) | (y < bottom) | (y > top)
    ax.scatter(x[mask], y[mask], facecolors='none', edgecolors='k')

# Create a diagram and draw a two-dimensional box diagram
fig, ax = plt.subplots()
boxplot_2d(x, y, ax=ax, whis=1.5)

# Plot the box diagram
ax.set_title("2D Box Plot")
ax.set_xlabel("Petal Length")
ax.set_ylabel("Sepal Width")
plt.show()






# Select features
features = ['Petal length' , 'Sepal width']

plt.figure(figsize=(10, 7))

for feature in features:
    # Sorting data using selected features
    sortedData = np.sort(data[feature])
    quantiles = np.arange(1, len(sortedData) + 1) / len(sortedData)

    # Create Quantile Plot
    plt.plot(sortedData, quantiles, label=feature)

# Plot settings
plt.xlabel("attribute values")
plt.ylabel("percentile")
plt.title("Quantile Plot for Petal length and Sepal width")
plt.legend()
plt.grid(True)
plt.show()






# # Scatter plot for pairs of features

# Define the desired pair of features to draw a scatter plot
featurePairs = [
    ('Sepal length', 'Sepal width'),
    ('Sepal length', 'Petal length'),
    ('Sepal length', 'Petal width'),
    ('Sepal width', 'Petal length'),
    ('Sepal width', 'Petal width'),
    ('Petal length', 'Petal width')
]

plt.figure(figsize=(15, 10))

for i, (xFeature, yFeature) in enumerate(featurePairs, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x = xFeature, y = yFeature, hue='Species', data=data)
    plt.xlabel(xFeature)
    plt.ylabel(yFeature)
    plt.title(f"{xFeature} vs {yFeature}")

plt.tight_layout()
plt.show()





fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot data by species with different colors
species = data['Species'].unique()
colors = ['blue', 'green', 'red']

for sp, color in zip(species, colors):
    subset = data[data['Species'] == sp]
    ax.scatter(subset['Sepal width'], subset['Sepal length'], subset['Petal length'],
               color=color, label=sp, s=50, edgecolors='k')

# Axes labeling
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title("3D scatter plot for species separation")
plt.legend()
plt.show()






plt.figure(figsize=(8, 6))

# Drawing plot for each species
sns.kdeplot(data[data['Species'] == 'Iris-setosa']['Petal length'], fill=True, label='Iris-setosa')
sns.kdeplot(data[data['Species'] == 'Iris-versicolor']['Petal length'], fill=True, label='Petal length')
sns.kdeplot(data[data['Species'] == 'Iris-virginica']['Petal length'], fill=True, label='Iris-virginica')

# Plot settings
plt.title('PDF of Petal Length by Species')
plt.xlabel('Petal Length')
plt.ylabel('Density')
plt.legend(title='Species')
plt.show()


