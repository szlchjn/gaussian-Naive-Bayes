import numpy as np
from random import random
from scipy.stats import norm
import csv

def load_data(file, split, training_data=[], test_data=[]):
    with open(file, 'r') as csv_file:
        data_set = list(csv.reader(csv_file))
        for i in range(len(data_set)):
            data_set[i] = [float(j) for j in data_set[i]]
            if random() < split:
                training_data.append(data_set[i])
            else:
                test_data.append(data_set[i])
        print('Loaded {0} data set\n{1} entries, with {2} parameters\n'.format(
                                            file, len(data_set), len(data_set[0])))
        
def class_divide(training_data):
    classes = (row[-1] for row in training_data)
    separated = {class_:[] for class_ in classes}
    for row in training_data:
        separated[row[-1]].append(row)
    return separated

def attr_sum(training_data):
    summary = [(np.mean(attr), np.std(attr, ddof=1)) for attr in zip(*training_data)]
    return summary[:-1]

def class_sum(separated_data):
    summary = {class_:attr_sum(param) for class_, param in separated_data.items()}
    return summary

def gaussian_density(class_sum, item):
    probability = {}
    for class_, summary in class_sum.items():
        probability[class_] = 1
        for i in range(len(summary)):
            mean, stddev = summary[i]
            probability[class_] *= norm.pdf(item[i], mean, stddev)
    return probability

def predict(class_sum, item):
    probabilities = gaussian_density(class_sum, item)
    return max(probabilities, key=probabilities.get)

def is_correct(prediction, instance):
    return prediction == instance[-1]

def accuracy(test_data, predictions):
    return (predictions / len(test_data)) * 100.0


def main():
    training_data = []
    test_data = []
    correct = 0
    
    load_data('pima indians.data', 0.67, training_data, test_data)
    
    separated_data = class_divide(training_data)
    summaries = class_sum(separated_data)

    for item in test_data:
        prediction = predict(summaries, item)

        if is_correct(prediction, item):
            print('Prediction:', int(prediction), '-> Label:', int(item[-1]))
            correct += 1
        else:
            print('Prediction:', int(prediction), '-> Label:', int(item[-1]), 'FALSE')

    acc = accuracy(test_data, correct)
    print('\nAccuracy: {} %'.format(round(acc, 5)))


if __name__ == '__main__':
    main()