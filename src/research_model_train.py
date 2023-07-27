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
from sklearn.model_selection import GridSearchCV

transform = transforms.Compose([
    transforms.CenterCrop(448),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(0.2),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

train_path = 'training_data/'
test_val_path = 'Other/'
train_dataset = datasets.ImageFolder(train_path, transform=transform)
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

raw_dataset = CustomDataset(csv_path, test_val_path, transform=transform)
raw_dataset_split = test_val_dataset(raw_dataset)
test_dataset = raw_dataset_split['test']
val_dataset = raw_dataset_split['val']

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights='IMAGENET1K_V1')
model.aux_logits = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=0.001)
num_epochs = 5

model.to(device)

accuracies = []
losses = []
val_accs = []
total = 0
correct = 0

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=40, shuffle=False)

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
  val_preds = []
  val_truth = []
  val_total = 0
  val_correct = 0
  with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.to('cpu')
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Train Accuracy: {100 * correct/total}, Val Accuracy: {100* val_correct/val_total}")
  losses.append(loss.item())
  accuracies.append(100* correct/total)
  val_accs.append(100* val_correct/val_total)

fig = plt.figure('Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Feature Extractor Loss')
plt.plot(losses)
epochs = len(accuracies)
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, accuracies, label='Train')
ax.plot(x_axis, val_accs, label='Validation')
ax.legend()

model.eval()

model.load_state_dict(torch.load('feature_extractor_84'))

target_layers = model.features[7]
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


preds = []
truth = []

X_test_features = []
Y_test_labels = []

def grad_image(path):
  image = Image.open(path)
  rgb_tensor = transform(image).unsqueeze(0)
  rgb_img = rgb_tensor.detach().numpy().squeeze()
  rgb_img = np.rollaxis(rgb_img, 0, 3)  
  print(rgb_img.shape)
  cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
  targets = [ClassifierOutputTarget(181)]#281
  grayscale_cam = cam(input_tensor=rgb_tensor, targets=targets, aug_smooth=True)
  grayscale_cam = grayscale_cam[0, :]
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
  cv2.imwrite(f'Heatmaps/CAM_' + path.split('/')[2].replace('.bmp','') + '_.png', visualization)

# grad_image('Alt_images/all/UID_1_4_1_all.bmp')
# grad_image('Alt_images/all/UID_1_4_2_all.bmp')
# grad_image('Alt_images/hem/UID_H1_5_1_hem.bmp')
# grad_image('Alt_images/hem/UID_H1_6_1_hem.bmp')
# grad_image('Alt_images/hem/UID_H1_7_1_hem.bmp')
# grad_image('Alt_images/hem/UID_H1_8_1_hem.bmp')
# grad_image('Alt_images/hem/UID_H1_9_1_hem.bmp')
# grad_image('Alt_images/hem/UID_H1_10_1_hem.bmp')
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

xgbm = xgb.XGBClassifier(eta = 0.35, tree_method='gpu_hist',eval_metric=eval_metric)
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