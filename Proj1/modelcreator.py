import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from yellowbrick.classifier import ClassificationReport


def create_model(df):
    # select columns other than 'Persons'
    cols = [col for col in df.columns if col not in ['Persons']]
    # dropping the 'Date' and 'Persons' columns
    data = df[cols]
    # assign the Persons column as target
    target = df['Persons']

    # split data set into train and val+test set
    data_train, data_val_test, target_train, target_val_test = train_test_split(data, target,
                                                                                test_size=0.3, random_state=10)
    # split val+test set into validation and test set
    data_val, data_test, target_val, target_test = train_test_split(data_val_test, target_val_test,
                                                                    test_size=0.5, random_state=10)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 2), random_state=1)

    # Train -> fit the parameters of the model
    clf.fit(data_train, target_train)
    visualizer = ClassificationReport(clf, classes=[0, 1, 2, 3])
    visualizer.fit(data_train, target_train)

    # Validation -> # Evaluate the model on the validation data
    visualizer.score(data_val, target_val)
    g = visualizer.show()

    return clf
