# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import itertools
import random
import numpy as np
from scipy.stats import bernoulli
from keras.datasets import mnist
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from tmu.models.classification.vanilla_classifier import TMClassifier

'''
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

# Convert the data to pandas DataFrames for easier sampling
df_X = pd.DataFrame(X_train)
df_Y = pd.DataFrame(Y_train, columns=['label'])

# Combine the X and Y DataFrames
df = pd.concat([df_X, df_Y], axis=1)

# Initialize an empty DataFrame for the final balanced dataset
balanced_data = pd.DataFrame()

# Loop through each class
for i in range(10):
    # Get a subset of the data for the current class
    class_subset = df[df['label'] == i]

    # Sample 1000 instances from the current class
    class_sample = class_subset.sample(n=1000, random_state=42)

    # Append the sample to the final dataset
    balanced_data = pd.concat([balanced_data, class_sample], axis=0)

# Shuffle the final dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the balanced dataset back into data and target
X_train_1k = balanced_data.drop('label', axis=1).values
Y_train_1k = balanced_data['label'].values

Y_train_1k[Y_train_1k != 1] = 0
Y_test[Y_test != 1] = 0

print(X_train_1k.shape)
print(Y_train_1k.shape)
'''


def success(prob):
    if (bernoulli.rvs(prob) == 1):
        return 1
    else:
        return 0


def action(state, states):
    if state <= states:
        return 0
    else:
        return 1


def TMC(n, epochs, examples, labels, clauses, states, probabilitie):
    ta_state = np.random.choice([states, states + 1], size=(clauses, n, 2)).astype(dtype=np.int32)

    for i in range(epochs):
        #print('')
        #print("###################################Epoch", i, "###################################")
        #print('')
        for e in range(len(examples)):
            # print("------------------------",e,"------------------")
            # FeedBack on Positives
            if labels[e] == 1:
                for f in range(n):
                    for j in range(clauses):
                        p = probabilitie[j]
                        # Include with some probability
                        if action(ta_state[j, f, examples[e][f]], states) == 1:
                            if success(p):
                                if ta_state[j, f, examples[e][f]] < 2 * states:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1
                            else:
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1

                        # Exclude Negations
                        if action(ta_state[j, f, (not examples[e][f]).__int__()], states) == 1:
                            ta_state[j, f, (not examples[e][f]).__int__()] = ta_state[j, f, (
                                not examples[e][f]).__int__()] - 1

                        # Include with some probability
                        if action(ta_state[j, f, examples[e][f]], states) == 0:
                            if success(1 - p):
                                if ta_state[j, f, examples[e][f]] > 1:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1
                            else:
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                        # Exclude Negations
                        if action(ta_state[j, f, (not examples[e][f]).__int__()], states) == 0:
                            if ta_state[j, f, (not examples[e][f]).__int__()] > 1:
                                ta_state[j, f, (not examples[e][f]).__int__()] = ta_state[j, f, (
                                    not examples[e][f]).__int__()] - 1

            if labels[e] == 0:
                for f in range(n):
                    for j in range(clauses):
                        p = probabilitie[j]
                        if action(ta_state[j, f, examples[e][f]], states) == 1:
                            if success(p):
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1
                            else:
                                if ta_state[j, f, examples[e][f]] < 2 * states:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                        if action(ta_state[j, f, (not examples[e][f]).__int__()], states) == 1:
                            if success(p):
                                if ta_state[j, f, (not examples[e][f]).__int__()] < 2 * states + 1:
                                    ta_state[j, f, (not examples[e][f]).__int__()] = ta_state[j, f, (
                                        not examples[e][f]).__int__()] + 1
                            else:
                                ta_state[j, f, (not examples[e][f]).__int__()] = ta_state[j, f, (
                                    not examples[e][f]).__int__()] - 1

                        if action(ta_state[j, f, examples[e][f]], states) == 0:
                            if success(1 - p):
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                            else:
                                if ta_state[j, f, examples[e][f]] > 1:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1

                        if action(ta_state[j, f, (not examples[e][f]).__int__()], states) == 0:
                            if success(1 - p):
                                if ta_state[j, f, (not examples[e][f]).__int__()] > 1:
                                    ta_state[j, f, (not examples[e][f]).__int__()] = ta_state[j, f, (
                                        not examples[e][f]).__int__()] - 1
                            else:
                                ta_state[j, f, (not examples[e][f]).__int__()] = ta_state[j, f, (
                                    not examples[e][f]).__int__()] + 1

    return ta_state


def Accuracy(examples, formulas, n, labels):
    accur = 0

    for e in range(len(examples)):
        allLabels = []
        # print("------------", e,"---------------")
        predicted = 0
        for c in formulas:
            label = 1
            for f in range(n):
                if c.__contains__((f + 1)) and examples[e][f] == 0:
                    label = 0
                    continue
                if c.__contains__(-1 * (f + 1)) and examples[e][f] == 1:
                    label = 0
                    continue
            allLabels.append(label)
        if allLabels.__contains__(1):
            predicted = 1
        #if sum(allLabels) > len(formulas) / 2:
        #    predicted = 1
        if predicted == labels[e]:
            accur = accur + 1
    return accur / len(examples)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Use a breakpoint in the code line below to debug your script.
    startAll = time.process_time()
    n = 10
    examples = list(map(list, itertools.product([0, 1], repeat=n)))
    #target = np.random.choice(3, n, replace=True)
    target = [0, 1, 2, 0, 0, 1, 2, 2, 0, 1]
    labels = [1] * len(examples)
    #tm = TMClassifier(number_of_clauses=2, T=1, s=2)
    for X in examples:
        change = False
        for i in range(len(X)):
            if X[i] == 1 and target[i] == 0:
                change = True
            if X[i] == 0 and target[i] == 1:
                change = True
        if change:
            labels[examples.index(X)] = 0
    examples = np.array(examples)
    labels = np.array(labels)
    states = 10000
    epochs = 100
    clauses = 2
    runs = 10
    # pl < pu
    pl = 0.6
    pu = 0.8
    Accuracies = []
    for i in range(runs):
        print("-----Run", i, "--------")
        #np.random.shuffle(examples)

        X_training = examples[:int(examples.shape[0]*0.8)]
        y_training = labels[:int(labels.shape[0]*0.8)]

        X_test = examples[int(examples.shape[0]*0.8):]  # Input features
        y_test = labels[int(labels.shape[0]*0.8):]
        #print(np.unique(y_test, return_counts=True))

        probabilitie = [random.uniform(pl, pu) for i in range(clauses)]
        # probabilitie = [0.9]
        #print(probabilitie)

        start = time.process_time()
        n = len(X_training[0])

        result = TMC(n, epochs, X_training, y_training, clauses, states, probabilitie)
        # print(result)
        
        formulas = []
        for i in range(clauses):
            c = []
            for j in range(n):
                if (result[i, j, 1] > states):
                    c.append(j + 1)
                if (result[i, j, 0] > states):
                    c.append(-j - 1)
            formulas.append(c)
        print("Learned: ", formulas)
        tm.fit(X_training, y_training)
        print("Time = ", time.process_time() - start)
        a = Accuracy(X_test, formulas, n, y_test)
        #a = 100 * (tm.predict(X_test) == y_test).mean()
        Accuracies.append(a)
        print("Accuracy: ", a)
    print(Accuracies)
    print("MIN: ", min(Accuracies))
    print("MAX: ", max(Accuracies))
    print("AVG: ", sum(Accuracies)/len(Accuracies))
    print("STD: ", np.std(Accuracies))

