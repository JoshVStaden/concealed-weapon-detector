from keras.models import load_model
import cv2

img = cv2.imread("pistol.jpg")
img = cv2.resize(img, (224, 224))
# img = cv2.resize(img, (224, 224))
# img = cv2.resize(img, (224, 224))
# img = cv2.resize(img, (224, 224))
model = load_model("pistol_classification")
y = model.predict(img)
print(y)

