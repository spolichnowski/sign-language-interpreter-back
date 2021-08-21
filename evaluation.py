import numpy as np
import matplotlib.pyplot as plt  
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay


def evaluate(dataset_path, model_path):
    """
    Evaluation returns all needed scores and takes paths of model 
    and dataset as an argument.
    """
    # Creates dataset from directory
    test = keras.preprocessing.image_dataset_from_directory(
        model_path,
        labels='inferred',
        subset="validation",
        validation_split=0.2,
        label_mode='categorical',
        image_size=(300, 300),
        batch_size=128
    )

    # Loads model
    model = load_model(model_path)

    # Creates np arrays of predictions and actual labels
    predictions = np.array([])
    labels =  np.array([])
    for x, y in test:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    # Methods imported from sklearn for accuracy, f1 score , recall and precision
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')

    # Creates confusion matrix, displays it nad save in default directory
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=["A", "B", "C", "D", "F", "G", "L", "M", "N", "P", "Q", "R", "V", "W", "X", "Y", "Z"])
    disp.plot() 
    plt.savefig('Confusion-Matrix.png')
    plt.show()

    # Prints all the scores
    print('Accuracy: {}, F1: {}, Recall: {}, Precision: {}'.format(accuracy, f1, recall, precision))
