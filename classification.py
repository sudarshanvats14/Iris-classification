from keras.models import load_model
from keras.utils import to_categorical
import numpy as np


model = load_model("model.h5")  

sample_input = np.array([[5.1, 3.5, 1.4, 0.2],[6.1, 2.8, 4.7, 1.2 ],
    [6.0, 3.0, 4.8, 1.8],
    [7.3, 2.9, 6.3, 1.8],])  

sample_prediction = model.predict(sample_input)

predicted_label = np.argmax(sample_prediction, axis=1)

# label_names = {
#     0: "Setosa",
#     1: "Versicolor",
#     2: "Virginica",
# }

# Print the results with label names
for i in range(len(sample_input)):
    print("Sample Input:", sample_input[i])
    print("Predicted Label:", [predicted_label[i]])
    print()
