# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Прочитать оригинальное изображение
# img = cv2.imread('./smoke1.png')
#
# # Получить высоту и ширину изображения
# height = img.shape[0]
# width = img.shape[1]
#
# # Создать изображение
# grayimg = np.zeros((height, width, 3), np.uint8)
#
# # Максимальная обработка изображения в градациях серого
# for i in range(height):
#     for j in range(width):
#         # Получить изображение R G B максимум
#         gray = max(img[i, j][0], img[i, j][1], img[i, j][2])
#         # Назначение пикселей в градациях серого серого цвета = максимум (R, G, B)
#         grayimg[i, j] = np.uint8(gray)
#
# # Показать изображение
# cv2.imshow("src", img)
# cv2.imshow("gray", grayimg)
#
# # Ждать показа
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def Origin_histogram(img):
#     # Создайте таблицу соответствия между значением серого каждого уровня серого исходного изображения и количеством пикселей
#     histogram = {}
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             k = img[i][j]
#             if k in histogram:
#                 histogram[k] += 1
#             else:
#                 histogram[k] = 1
#
#     sorted_histogram = {}  # Создать отсортированную таблицу сопоставления
#     sorted_list = sorted(histogram)  # Сортировка от низкого до высокого в соответствии со значением серого
#
#     for j in range(len(sorted_list)):
#         sorted_histogram[sorted_list[j]] = histogram[sorted_list[j]]
#
#     return sorted_histogram
#
#
# def equalization_histogram(histogram, img):
#     pr = {}  # Создать таблицу отображения распределения вероятностей
#
#     for i in histogram.keys():
#         pr[i] = histogram[i] / (img.shape[0] * img.shape[1])
#
#     tmp = 0
#     for m in pr.keys():
#         tmp += pr[m]
#         pr[m] = max(histogram) * tmp
#
#     new_img = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
#
#     for k in range(img.shape[0]):
#         for l in range(img.shape[1]):
#             new_img[k][l] = pr[img[k][l]]
#
#     return new_img
#
#
# def GrayHist(img):
#     # Рассчитать серую гистограмму
#     height, width = img.shape[:2]
#     grayHist = np.zeros([256], np.uint64)
#     for i in range(height):
#         for j in range(width):
#             grayHist[img[i][j]] += 1
#     return grayHist
#
#
# if __name__ == '__main__':
#     # Прочитать оригинальное изображение
#     img = cv2.imread('./smoke1.jpg', cv2.IMREAD_GRAYSCALE)
#     # Рассчитать гистограмму градаций серого исходного изображения
#     origin_histogram = Origin_histogram(img)
#     # Выравнивание гистограммы
#     new_img = equalization_histogram(origin_histogram, img)
#
#     origin_grayHist = GrayHist(img)
#     equaliza_grayHist = GrayHist(new_img)
#     x = np.arange(256)
#     # Нарисовать гистограмму в оттенках серого
#     plt.figure(num=1)
#     plt.subplot(2, 2, 1)
#     plt.plot(x, origin_grayHist, 'r', linewidth=2, c='black')
#     plt.title("Origin")
#     plt.ylabel("number of pixels")
#     plt.subplot(2, 2, 2)
#     plt.plot(x, equaliza_grayHist, 'r', linewidth=2, c='black')
#     plt.title("Equalization")
#     plt.ylabel("number of pixels")
#     plt.subplot(2, 2, 3)
#     plt.imshow(img, cmap=plt.cm.gray)
#     plt.title('Origin')
#     plt.subplot(2, 2, 4)
#     plt.imshow(new_img, cmap=plt.cm.gray)
#     plt.title('Equalization')
#     plt.show()
# import numpy as np
# import cv2
# import time
# import datetime
#
# colour=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))# Определить цвет прямоугольника
#
# cap = cv2.VideoCapture("./00001.MTS") # Параметр 0, чтобы открыть камеру, имя файла, чтобы открыть видео
#
# fgbg = cv2.createBackgroundSubtractorMOG2()# Гибридный алгоритм гауссовского фонового моделирования
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')# Установить сохранить формат изображения
# out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.MTS',fourcc, 10.0, (1280,800))# Разрешение должно соответствовать оригинальному видео
#
#
# while True:
#     #time.sleep(1)
#     ret, frame = cap.read()  # Читать картинку
#     fgmask = fgbg.apply(frame)
#
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))  # Морфологический шум
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # Открыть операцию шумоподавления
#
#     contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Найти перспективы
#
#     count=0
#     for cont in contours:
#         Area = cv2.contourArea(cont)  # Рассчитать площадь контура
#         if Area < 300:  # Форма с площадью фильтра менее 10
#             continue
#
#         count += 1  # Количество плюс один
#
#         print("{}-prospect:{}".format(count,Area),end="  ") # Распечатать область каждого переднего плана
#
#         rect = cv2.boundingRect(cont) # Извлечь прямоугольные координаты
#
#         print("x:{} y:{}".format(rect[0],rect[1]))#Печать координат
#
#         cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),colour[count%6],1)# Нарисуйте прямоугольник на исходном изображении
#         cv2.rectangle(fgmask,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0xff, 0xff, 0xff), 1)  # Рисуем прямоугольник на черном и белом переднем плане
#
#         y = 10 if rect[1] < 10 else rect[1]  # Предотвратить нумерацию за пределы картинки
#         cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # Напишите число на переднем плане
#
#
#
#     cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1) # Показать всего
#     cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
#     print("----------------------------")
#
#     cv2.imshow('frame', frame)# Отметить исходное изображение
#     cv2.imshow('frame2', fgmask)  # Отображение переднего плана и фона в черно-белом
#     out.write(frame)
#     k = cv2.waitKey(30)&0xff  # Нажмите Esc для выхода
#     if k == 27:
#         break
#
#
# out.release()# Выпустить файл
# cap.release()
# cv2.destoryAllWindows()# Закрыть все окна


import os
from torchvision import datasets
import torchvision
import torch
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
from PIL import ImageFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ion()

batch_size = 64
num_workers = 0

transform = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = './dataset/Train/'
valid_set = './dataset/Test/'


train_data = datasets.ImageFolder(train_set, transform=transform)
valid_data = datasets.ImageFolder(valid_set, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

loaders = {
    'train': train_loader,
    'valid': valid_loader
}

class_names = ['Fire', 'Neutral', 'Smoke']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(loaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

model = models.resnet50(pretrained=True)

use_cuda = torch.cuda.is_available()

if use_cuda:
    model = model.cuda()

print(model)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Sequential(torch.nn.Linear(2048,128),
                               torch.nn.ReLU(),
                               torch.nn.Linear(128,3),
                               torch.nn.Softmax()
                               )

for param in model.fc.parameters():
    param.requires_grad = True

if use_cuda:
    model_transfer = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_transfer.fc.parameters(), lr=0.0005)

n_epochs = 10

train_accuracy_list = []
train_loss_list = []
valid_accuracy_list = []
valid_loss_list = []

def train(n_epochs, loader, model, optimizer, criterion, use_cuda, save_path):

    valid_loss_min = np.Inf

    for epoch in range(1, (n_epochs+1)):

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):

            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_acc = train_acc + torch.sum(preds == target.data)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):

            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)

            valid_acc = valid_acc + torch.sum(preds == target.data)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        train_loss = train_loss/len(loaders['train'].dataset)
        valid_loss = valid_loss/len(loaders['valid'].dataset)
        train_acc = train_acc/len(loaders['train'].dataset)
        valid_acc = valid_acc/len(loaders['valid'].dataset)

        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        valid_accuracy_list.append(valid_acc)
        valid_loss_list.append(valid_loss)

        print('Epoch: {} \tTraining Acc: {:6f} \tTraining Loss: {:6f} \tValidation Acc: {:6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_acc,
            train_loss,
            valid_acc,
            valid_loss
        ))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    return model

model = train(n_epochs, loaders, model, optimizer, criterion, use_cuda, './trained-models/model_transfer.pt')

plt.style.use("ggplot")
plt.figure()
plt.plot(train_loss_list, label="train_loss")
plt.title("Train-Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

plt.style.use("ggplot")
plt.figure()

plt.plot(train_accuracy_list, label="train_acc")
plt.plot(valid_accuracy_list, label="valid_acc")

plt.title("Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

torch.save(model, './trained-models/model_final.pth')

from PIL import Image

class_names = class_names= ['Fire', 'Neutral', 'Smoke']

def predict(image):
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    image = image.cuda()

    pred = model(image)
    idx = torch.argmax(pred)
    prob = pred[0][idx].item()*100

    return class_names[idx], prob

def app(path):
    img = Image.open(path)
    plt.imshow(img)
    plt.show()

    prediction, prob = predict(img)
    print(prediction, prob)


for img_file in os.listdir('./test-imgs/'):
    img_path = os.path.join('./test-imgs/', img_file)
    app(img_path)


