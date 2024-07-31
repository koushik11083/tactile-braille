import torch
from torchvision.transforms import transforms
from torch.nn import Linear
from PIL import Image,ImageOps
import copy
import numpy as np

import serial
import time


ser = serial.Serial('COM3', 9600, timeout=1)
ser.close()
ser.open()


class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for input images
tfm = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
num_classes = 26  # Assuming there are 26 classes
model = CNN(num_classes)
model.load_state_dict(torch.load('model_simple_cnn.pth'))
model.eval()
model.to(device)

# Class labels
class_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
lab= 'ABKLMNOPQRSTCUVWXYZDEFGHIJ'

img = Image.open('datasets/roma.png')
im2 = ImageOps.grayscale(img)

# image to numpy array
arr = np.array(im2)
arr.setflags(write=1)

# dimen
n, m = arr.shape

# Binary img
for i in range(n):
    for j in range(m):
        if arr[i][j] > 190:
            arr[i][j] = 255
        else:
            arr[i][j] = 0

list_images = []
li = []
b = False
w = 0

# Iteration
for j in range(m):
    x = []
    cnt = 0
    for i in range(n):
        if arr[i][j] == 0:
            cnt += 1
        x.append(arr[i][j])
    li.append(x)
    if cnt > 30:
        b = True
        w = 0
    if cnt < 30:
        w += 1
    if w == 5 and b:
        w = 0
        list_images.append(copy.deepcopy(li))
        li = []
        b = False

print(len(list_images))

# segmentation
processed_images = []
for x in list_images:
    nparr = np.array(x).T  # Transpose to get the correct orientation
    # Convert list of lists to a NumPy array
    nparr = np.array(nparr, dtype=np.uint8)

    # non-zero rows and columns
    non_zero_columns = np.where(nparr.min(axis=0) < 255)[0]
    non_zero_rows = np.where(nparr.min(axis=1) < 255)[0]

    if non_zero_columns.size and non_zero_rows.size:
        # croop
        crop_box = (non_zero_rows[0], non_zero_rows[-1], non_zero_columns[0], non_zero_columns[-1])
        nparr_cropped = nparr[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]

        #  new size
        new_size = max(nparr_cropped.shape)
        square_image = np.ones((new_size, new_size), dtype=np.uint8) * 255  # white background

        # padding calc
        x_offset = (new_size - nparr_cropped.shape[1]) // 2
        y_offset = (new_size - nparr_cropped.shape[0]) // 2

        # square image
        square_image[y_offset:y_offset+nparr_cropped.shape[0], x_offset:x_offset+nparr_cropped.shape[1]] = nparr_cropped

        # padding
        padding = 10
        padded_image = np.ones((new_size + 2 * padding, new_size + 2 * padding), dtype=np.uint8) * 255
        padded_image[padding:padding + new_size, padding:padding + new_size] = square_image

        processed_images.append(padded_image)


for i, letter_image in enumerate(processed_images):
    img = Image.fromarray(letter_image).convert('RGB')  # Convert image to RGB
    img = img.resize((28, 28))
    img.show()
    img_tensor = tfm(img).unsqueeze(0).to(device)  # Add batch dimension
    pred_prob = model(img_tensor)
    pred = torch.max(pred_prob, 1).indices.item()
    print(f"Model prediction {pred}, {lab[pred]}")
    print("========================================================")
    ser.write(str.encode(lab[pred]))

ser.close()



