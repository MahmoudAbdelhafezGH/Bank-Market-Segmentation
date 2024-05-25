import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def extract_metrics(y_true, y_pred):
    
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    print(cm.T)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [1, 0])

    cm_display.plot()
    
    Accuracy = metrics.accuracy_score(y_true, y_pred) 
    
    Precision = metrics.precision_score(y_true, y_pred)

    Sensitivity_recall = metrics.recall_score(y_true, y_pred)

    Specificity = metrics.recall_score(y_true, y_pred, pos_label=0)

    F1_score = metrics.f1_score(y_true, y_pred)

    print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})
    
    plt.show()

def evaluate(model, test_loader):

    y_true = np.empty((0, ))

    y_pred = np.empty((0, ))

    for inputs, labels in test_loader:

            output = model(inputs)

            output = (output > 0.5).float()

            output = output.detach().cpu().numpy()

            y_pred = np.append(y_pred, output)

            labels = labels.detach().cpu().numpy()

            y_true =  np.append(y_true, labels) # Save Truth
    
    extract_metrics(y_true, y_pred)
