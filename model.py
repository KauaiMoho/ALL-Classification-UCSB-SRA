import torch
import math
import cv2
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import efficientnet_b0
from torchvision.models import resnet18
from torchvision.models import vgg16
from torchvision.models import inception_v3
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import tensorflow as tf
import xgboost as xgb
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import PIL
import os

transform = transforms.Compose([
    transforms.CenterCrop(448),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(10),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ToTensor()
    
])

path = 'training_data/'
other_path = 'Other/'
train_dataset = datasets.ImageFolder(path, transform=transform)
csv_path = 'C-NMC_test_prelim_phase_data_labels.csv'
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {1:0, 0:1}

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index, "new_names"]
        label = self.class2index[self.df.loc[index, "labels"]]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def test_val_dataset(dataset, val_split=0.5):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['test'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

raw_dataset = CustomDataset(csv_path, other_path, transform=transform)
raw_dataset_split = test_val_dataset(raw_dataset)
test_dataset = raw_dataset_split['test']
val_dataset = raw_dataset_split['val']

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_samples = len(train_loader.dataset)
class_0_samples = sum(targets == 0 for _, targets in train_loader.dataset)
class_1_samples = sum(targets == 1 for _, targets in train_loader.dataset)

class_0_weight = total_samples / (2.0 * class_0_samples)
class_1_weight = total_samples / (2.0 * class_1_samples)

class_weight = torch.tensor([class_0_weight, class_1_weight]).to(device)

model_raw = efficientnet_b0(weights='IMAGENET1K_V1')
model_raw.aux_logits = False
softmax_layer = nn.Sequential(nn.Linear(1000, 2),
                              nn.Softmax(dim=1))
model = nn.Sequential(model_raw, softmax_layer)
# model.load_state_dict(torch.load('feature_extractor_81'))

criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adamax(model.parameters(), lr=0.001)
num_epochs = 5

model.to(device)

accuracies = []
losses = []
total = 0
correct = 0

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

for epoch in range(num_epochs):
     model.train()
     for images, labels in train_loader:
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # val_preds = []
      # val_truth = []
      # val_total = 0
      # val_correct = 0
      # with torch.no_grad():
      #   for images, labels in val_loader:
      #       images = images.to(device)
      #       outputs = model(images)
      #       _, predicted = torch.max(outputs.data, 1)
      #       predicted = predicted.to('cpu')
      #       val_total += labels.size(0)
      #       val_correct += (predicted == labels).sum().item()
      # print('Validation Accuracy:', val_correct/val_total)

     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Train Accuracy {100 * correct/total}")
     losses.append(loss.item())
     accuracies.append(100* correct/total)

fig = plt.figure('Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Feature Extractor Loss')
plt.plot(losses)
fig = plt.figure('Training Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title('Feature Extractor Accuracy')
plt.plot(accuracies)

model.eval()
target_layers = []
for l in range(10,20):
   target_layers.append(list(model.modules())[l])
# path = 'Images/test/'
# test_dataset = datasets.ImageFolder(path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

preds = []
truth = []

X_test_features = []
Y_test_labels = []

def grad_image(path):
  image = Image.open(path)
  rgb_tensor = transform(image).unsqueeze(0)
  rgb_img = rgb_tensor.detach().numpy()
  cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
  grayscale_cam = cam(input_tensor=rgb_tensor, targets=None)
  grayscale_cam = grayscale_cam[0, :]
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
  cv2.imwrite(f'Heatmaps/CAM_' + path.split('/')[2].replace('.bmp','') + '_.png', visualization)

# grad_image('Alt_images/all/UID_1_1_1_all.bmp')
# grad_image('Alt_images/all/UID_1_2_1_all.bmp')
# grad_image('Alt_images/hem/UID_H1_1_1_hem.bmp')
# grad_image('Alt_images/hem/UID_H1_2_1_hem.bmp')

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.to('cpu')
        for p in range(len(predicted.numpy())):
          preds.append(predicted.numpy()[p])
          truth.append(labels.numpy()[p])

report = classification_report(truth, preds, output_dict=True)
print(classification_report(truth, preds))
save_path = 'Models/feature_extractors/feature_extractor_' + str(math.floor(100*report['accuracy']))
torch.save(model.state_dict(), save_path)

model = model[:-1]

with torch.no_grad():
    for images, labels in test_loader:
      for label in labels:
          Y_test_labels.append(label.numpy())
      images = images.to(device)
      outputs = model(images)
      outputs = outputs.to('cpu')
      for output in outputs:
        X_test_features.append(output.numpy())
       

X_train_features = []
Y_train_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images).to('cpu')
        for output in outputs:
          X_train_features.append(output.numpy())
        for label in labels:
          Y_train_labels.append(label.numpy())

# path = 'Images/val/'
# val_dataset = datasets.ImageFolder(path, transform=transform)

X_val_features = []
Y_val_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images).to('cpu')
        for output in outputs:
          X_val_features.append(output.numpy())
        for label in labels:
          Y_val_labels.append(label.numpy())


X_val_features = np.array(X_val_features)
Y_val_labels = np.array(Y_val_labels).reshape(-1)
X_train_features = np.array(X_train_features)
Y_train_labels = np.array(Y_train_labels).reshape(-1)
X_test_features = np.array(X_test_features)
Y_test_labels = np.array(Y_test_labels).reshape(-1)

eval_set = [ (X_train_features, Y_train_labels), (X_val_features, Y_val_labels), (X_test_features, Y_test_labels)]
eval_metric = ["auc","error"]

xgbm = xgb.XGBClassifier(eta=0.1, max_depth=6,min_child_weight=3, eval_metric=eval_metric)
xgbm.fit(X_train_features, Y_train_labels, eval_set=eval_set, verbose=True)
yPreds = xgbm.predict(X_test_features)
report = classification_report(Y_test_labels, yPreds, output_dict=True)
print(classification_report(Y_test_labels, yPreds))
xgbm.save_model('Models/classifiers/classifier_'  + str(math.floor(100*report['accuracy'])) + '.txt')
results = xgbm.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Validation')
ax.plot(x_axis, results['validation_2']['error'], label='Test')
ax.legend()
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('XGBoost Classifier Error')
cm = confusion_matrix(Y_test_labels, yPreds, labels=xgbm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=xgbm.classes_)
disp.plot()
plt.show()