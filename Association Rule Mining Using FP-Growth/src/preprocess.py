import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/adult.csv')
#select only columns with at least a column with ?
missingColumns = df.columns[(df == '?').any()]

# calculate mode & fill missing columns
for column in missingColumns:
    modeValue = df[column].mode()[0]  
    df[column] = df[column].replace('?', modeValue) 

outputPath = 'dist/adult_preprocessed.csv'
df.to_csv(outputPath, index=False)

print(f"\n preprocessed data saved successfully in {outputPath}")




print("\n-----------------------------------------------------------------------------------------------\n")
## change country below 40 to others 
# number of each country
countryCounts = df['native-country'].value_counts()
infrequentCountries = countryCounts[countryCounts < 40].index
df['native-country'] = df['native-country'].replace(infrequentCountries, 'Others')
print(f"change country below 40 to others are :\n{df['native-country'].value_counts()}" )



print("\n-----------------------------------------------------------------------------------------------\n")
## "Dataset binarization with One-Hot Encoding 
categoricalAttributes = ['workclass', 'education', 'marital-status', 
                         'occupation', 'relationship', 'race', 'sex', 
                         'native-country', 'income']

df = pd.get_dummies(df, 
                    columns=categoricalAttributes, 
                    prefix=categoricalAttributes, 
                    prefix_sep=' = ')

print(f"Dataset binarization successfully.\n")


# Save the preprocessed dataset
outputPath = 'dist/adult_preprocessed.csv'
df.to_csv(outputPath, index=False)

print(f"Preprocessed dataset has been successfully saved to: {outputPath}\n")





print("\n-----------------------------------------------------------------------------------------------\n")

# split a column to multiple part    numBins for qcut and eual depth
def createBinaryAttributes(df, column, bins=None, labelsPrefix=None, useQcut=False, numBins=None):
    # ues retbins to know where does each interval begin and end
    if useQcut:
        df[f"{column}_bins"], bins = pd.qcut(df[column], q=numBins, retbins=True, duplicates='drop')
    else:
        df[f"{column}_bins"] = pd.cut(df[column], bins=bins, include_lowest=True)
    
  # generate binary attributes for each bins
    for interval in df[f"{column}_bins"].cat.categories:
        #lower assing to left
        lower = interval.left
        upper = interval.right
        
        if lower == float('-inf'):
            label = f"{labelsPrefix} <= {upper:.2f}"
        elif upper == float('inf'):
            label = f"{labelsPrefix} > {lower:.2f}"
        else:
            label = f"{lower:.2f} < {labelsPrefix} <= {upper:.2f}"
        
        # create binary column     astype(int)=>  ture/false to boolean
        df[label] = (df[f"{column}_bins"] == interval).astype(int)
    
    # remove 2 rows
    df.drop(columns=[column, f"{column}_bins"], inplace=True)




createBinaryAttributes(
    df=df, 
    column='age', 
    labelsPrefix='age', 
    useQcut=True, 
    numBins=12
)

createBinaryAttributes(
    df=df, 
    column='education-num', 
    bins=8,  
    labelsPrefix='education-num'
)


capitalGainBins = [0, 2000, 5700, 11500, 21500, 64000, np.inf]
createBinaryAttributes(
    df=df, 
    column='capital-gain', 
    bins=capitalGainBins, 
    labelsPrefix='capital-gain'
)


capitalLossBins = [0, 900, 2000, 3100, np.inf]
createBinaryAttributes(
    df=df, 
    column='capital-loss', 
    bins=capitalLossBins, 
    labelsPrefix='capital-loss'
)


createBinaryAttributes(
    df=df, 
    column='hours-per-week', 
    bins=5, 
    labelsPrefix='hours-per-week'
)


print(f"Dataset now has {df.shape[1]} features.")


print("\n-----------------------------------------------------------------------------------------------\n")








def drawStripPlot(column, bins, title, ax):

    df = pd.read_csv('dist/adult_preprocessed.csv')

    colors = sns.color_palette("hsv", len(bins))
    
  
    # drow plot
    sns.stripplot(
        x=df[column], 
        jitter=True, #جلوگیری از روی هم افتادن نقاط 
        alpha=0.6, 
        size=3,  
        ax=ax 
    )
    
    # add color line to plot
    for split, color in zip(bins, colors):
        ax.axvline(split, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    
    # add color colors to Legend
    for split, color in zip(bins, colors):
        ax.plot([], [], color=color, linestyle='--', linewidth=1.5, label=f"Split: {split:.2f}")
    
    
    ax.legend(title="Split Points", loc='upper right', bbox_to_anchor=(1.2, 1))
    
    
    ax.set_title(title)
    ax.set_xlabel(column.capitalize())


# defining ranges for each column
ageBins = [21, 24, 28, 31, 34, 37, 40, 44, 48, 52, 59]
educationNumBins = [2.88, 4.75, 6.62, 8.50, 10.38, 12.25, 14.12]
capitalGainBins = [2000, 5700, 11500, 21500, 64000]
capitalLossBins = [900, 2000, 3100]
hoursPerWeekBins = [20.60, 40.20, 59.80, 79.40]

# List of columns and ranges
columns_bins = [
    ('age', ageBins, 'Strip Plot for Attribute "age"\n Strategy: Equal Frequency, Number of Intervals: 12'),
    ('education-num', educationNumBins, 'Strip Plot for Attribute "Education-Num"\n Strategy: Equal Width, Number of Intervals: 8'),
    ('capital-gain', capitalGainBins, 'Strip Plot for Attribute "Capital-Gain"\n Strategy: Equal Width, Number of Intervals: 6'),
    ('capital-loss', capitalLossBins, 'Strip Plot for Attribute "Capital-Loss"\n Strategy: Equal Width, Number of Intervals: 4'),
    ('hours-per-week', hoursPerWeekBins, 'Strip Plot for Attribute "Strip Plot for Hours-per-Week"\n Strategy: Equal Width, Number of Intervals: 5')
]


fig, axes = plt.subplots(nrows=len(columns_bins), ncols=1, figsize=(12, 16), constrained_layout=True)

# drowing plots
for ax, (column, bins, title) in zip(axes, columns_bins):
    drawStripPlot(column=column, bins=bins, title=title, ax=ax)

plt.show()



# drop column fnlwgt
df.drop(columns=['fnlwgt'], inplace=True)
outputPath = 'dist/adult_preprocessed.csv'
df = df.replace({0: False, 1: True})
df.to_csv(outputPath, index=False)

print("\nfnlwgt deleted succesflly and now we lose one of or column now we have only 119 column\n")




## **************** this is my firt code witch we have 5 seperated plot it can be uncomment for better view****************##
# def drawStripPlot(column, bins, title="Strip Plot", figsize=(12, 6)):   
#     df = pd.read_csv('dist/adult_preprocessed.csv')
#     colors = sns.color_palette("hsv", len(bins))   
#     plt.figure(figsize=figsize)
        
#     sns.stripplot(
#         x=df[column], 
#         jitter=True,  
#         alpha=0.6,  
#         size=3,  
#     )
    
#     for split, color in zip(bins, colors):
#         plt.axvline(split, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
     
#     for split, color in zip(bins, colors):
#         plt.plot([], [], color=color, linestyle='--', linewidth=1.5, label=f"Split: {split:.2f}")
    
#     plt.legend(title="Split Points", loc='upper right')
    
#     plt.title(title, fontsize=16)
#     plt.xlabel(column.capitalize(), fontsize=14)  
#     plt.show()

# ageBins = [21, 24, 28, 31, 34, 37, 40, 44, 48, 52, 59]
# educationNumBins = [2.88, 4.75, 6.62, 8.50, 10.38, 12.25, 14.12]
# capitalGainBins = [2000, 5700, 11500, 21500, 64000]
# capitalLossBins = [900, 2000, 3100]
# hoursPerWeekBins = [20.60, 40.20, 59.80, 79.40]

# drawStripPlot(
#     column='age',
#     bins=ageBins,
#     title='Strip Plot for Attribute "age"\n Strategy: Equal Frequency, Number of Intervals: 12'
# )

# drawStripPlot(
#     column='education-num',
#     bins=educationNumBins,
#     title='Strip Plot for Attribute "Education-Num"\n Strategy: Equal Width, Number of Intervals: 8'
# )

# drawStripPlot(
#     column='capital-gain',
#     bins=capitalGainBins,
#     title='Strip Plot for Attribute "Capital-Gain"\n Strategy: Equal Width, Number of Intervals: 6'
# )

# drawStripPlot(
#     column='capital-loss',
#     bins=capitalLossBins,
#     title='Strip Plot for Attribute "Capital-Loss"\n Strategy: Equal Width, Number of Intervals: 4'
# )

# drawStripPlot(
#     column='hours-per-week',
#     bins=hoursPerWeekBins,
#     title='Strip Plot for Attribute "Strip Plot for Hours-per-Week"\n Strategy: Equal Width, Number of Intervals: 5'

# )