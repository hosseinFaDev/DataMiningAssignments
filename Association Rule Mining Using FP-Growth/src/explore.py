import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('data/adult.csv')
#select column by data_type
numericalAttributes = df.select_dtypes(include=['int']).columns

print("List of all numeric attributes :\n")
print(list(numericalAttributes))



print("\n-----------------------------------------------------------------------------------------------\n")
#number of uniqe value for each column
uniqueValuesCount = df.nunique()
print("Table of the number of unique values ​​for each attribute:\n")
print(uniqueValuesCount)




print("\n-----------------------------------------------------------------------------------------------\n")
uniqueWorkclassValues = df['workclass'].unique()
print("list of uniqe workclass:\n")
print(uniqueWorkclassValues)




print("\n-----------------------------------------------------------------------------------------------\n")
#print number of missing value for each column witch are  > 0
missingValues = (df == '?').sum()
columnsWithMissingValue = missingValues[missingValues > 0]
print("the missing valuse Columns name with their number:\n")
print(columnsWithMissingValue)



print("\n-----------------------------------------------------------------------------------------------\n")
americans = (df['native-country'] == 'United-States').sum()
total = len(df)
americansPercentage = (americans / total) * 100
print(f"percentage of individuals who are natively from the United States is {americansPercentage:.2f}%")





print("\n---------------------------------------------Drowing a plot------------------------------------\n")
#list of pepole how are not native born in america
listCountrysExceptUsa = df[df['native-country'] != 'United-States']
#select only native-country
countryCounts = listCountrysExceptUsa['native-country'].value_counts()
sortedCountry = countryCounts.sort_values(ascending=False)

#drowing the plot and set settings for plot
plt.figure(figsize=(12, 6))
sortedCountry.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Native Country (Excluding United States)')
plt.xlabel('Native Country')
plt.ylabel('Number of Individuals')
plt.show()



print("\n-----------------------------------------------------------------------------------------------\n")


