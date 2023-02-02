# from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics


def run_kfold(data, target, machine, n, use_r2=True, use_accuracy=False):
	print("run kfold")
	kfold_object = KFold(n_splits=n)
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
		# machine = linear_model.LinearRegression()
		machine.fit(data_train, target_train)

		prediction = machine.predict(data_test)
		if (use_r2 == True):	
			r2 = metrics.r2_score(target_test, prediction)
			print("R square score: ",r2)
		if (use_accuracy == True):
			accuracy = metrics.accuracy_score(target_test, prediction)
			print("Accuracy score: ",accuracy)
		# print(prediction)
		# print(target_test)
		print("\n\n") # means two new lines

if __name__ == '__main__':
    run_kfold()
