from keras.models import Sequential 
from keras.layers import Dense
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from train import y_test, y_train, X_train, X_test
import numpy as np


print("Shape of y_train_encoded",y_train.shape)
print("Shape of y_test_encoded",y_test.shape)

model=Sequential()
model.add(Dense(100,input_dim=4,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)

prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",accuracy )

model.save("model.h5")

predictions = model.predict(X_test)

y_pred_labels = np.argmax(predictions, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

print("Confusion Matrix:\n", conf_matrix)

print("\nClassification Report:\n", classification_report(y_true_labels, y_pred_labels))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):  
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {:.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

for i in range(3):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {} (area = {:.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()