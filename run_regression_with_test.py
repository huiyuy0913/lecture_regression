import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")
# print(dataset)

target = dataset.iloc[:,0].values #means column 1, target means y variable
# print(target)

data = dataset.iloc[:,3:9].values
# print(data)

kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

print(kfold_object)

i = 0
for train_index, test_index in kfold_object.split(data):
	i=i+1
	print("Round:", str(i))
	print("Training index:")
	print(train_index)
	print("Testing index")
	print(test_index)
	
	data_train = data[train_index]
	target_train = target[train_index]
	data_test = data[test_index]
	target_test = target[test_index]
	machine = linear_model.LinearRegression()
	machine.fit(data_train, target_train)

	prediction = machine.predict(data_test)
	r2 = metrics.r2_score(target_test, prediction)
	print("R square score: ",r2)
	# print(prediction)
	# print(target_test)
	print("\n\n") # means two new lines


# regress target_test on our prediction and get r2 of this regression to measure how good the prediction is, this is not the r2 of the original regression (y and x)
new_dataset = pandas.read_csv("new_dataset.csv")
new_data = new_dataset.values
prediction = machine.predict(new_data)
print(prediction)

print("done")


