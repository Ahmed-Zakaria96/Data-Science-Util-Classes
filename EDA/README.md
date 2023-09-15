# Exploratory Data Analysis class

Create instance of the class
``` eda = EDA() ```

## Instance variables
``` def __init__(self, train_data, target, test_data=None, val_data=None, skip=None, null_threshold=.6, dup_threshold=.8, corr_threshold=.7, alpha=.05) ```

## List of Functions
```
* grabNumeric()
* grabCategorical()
* visualize()
* grabNulls()
* handleNulls()
* handleDuplicates()
* checkOutliers()
* boxplotOutliers()
* calcSkew()
* handleSkew()
* drawQQ()
* check Distibution()
* featureScale()
```

`grabNumeric(self, target=True, skip=True)`
> Grabs numeric columns in you dataframe
- target=True -> will include target column if it is numeric
-skip=True -> works with class variable "self.skip which is a list of column names" that you may want to skip based on the current task

`grabCategorical(self, target=True)`
basically the same as grabNumeric

`visualize(self, plot=None, whis=1.5)`
> runs grabNumeric() and grabCategorical and plot a visual based on column data types
- plot=None -> will plot distribution plot, boxplot, countplot and correlation matrix all based on data type of the columns
- whis is used in box plot which is set as a multiplier of the interquartile range (IQR)

`grabNulls(self, threshold=None, rthreshold=.1)`
> checks for columns and rows that have Null values and returns columns or rows with nulls percentage more the threshold
- threshold=None -> null threshold for columns
- rthreshold=.1 -> null threshold for rows

`handleNulls(self, threshold=None, rthreshold=.1)`
> calls grabNulls and drop columns or rows returned by it based on passed thresholds

`handleCorrFeature(self, threshold=None)`
> removes highly correlated features based on threshold

`handleDuplicates(self, threshold=None)`
> removes duplicated values based on threshold

`checkOutliers(self, threshold=1.5, skip=None)`
> returns a dict with column name and a list of outlies index based on the threshold in that column







