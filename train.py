import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from progressbar import ProgressBar
from data_module import DIV2K_x2, RandomHorizontalFlip, RandomVerticalFlip, Normalize, ToTensor, Compose
from EDSR import edsr
import imshow
import pdb
import torchvision
from torch.utils.tensorboard import SummaryWriter
tensorboard_name = 'first_test'
writer = SummaryWriter('run/'+tensorboard_name)

def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

train_dir = '../data/images'
val_dir = '../data/images'

train_transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor()])
                            #Normalize([0.449, 0.438, 0.404],
                                      #[1.0, 1.0, 1.0])])

valid_transforms = Compose([RandomHorizontalFlip(p=0.5),
                            RandomVerticalFlip(p=0.5),
                            ToTensor()])
                            #Normalize([0.440, 0.435, 0.403],
                                      #[1.0, 1.0, 1.0])])

trainset = DIV2K_x2(root_dir=train_dir, im_size=300, scale=4, transform=train_transforms)
validset = DIV2K_x2(root_dir=val_dir, im_size=300, scale=4, transform=valid_transforms)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
validloader = DataLoader(validset, batch_size=4, shuffle=True)

# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision
# def imshow(img1, img2):
#     img1 = torchvision.utils.make_grid(img1)
#     img2 = torchvision.utils.make_grid(img2)
#     plt.figure()
#     plt.imshow(np.transpose(img1.numpy(), (1,2,0)))
#     plt.figure()
#     plt.imshow(np.transpose(img2.numpy(), (1,2,0)))
#     plt.show()

input_images, output_images = iter(trainloader).next()
torchvision.utils.save_image(torchvision.utils.make_grid(input_images), 'input_images.png')
torchvision.utils.save_image(torchvision.utils.make_grid(output_images), 'output_images.png')


# exit('done')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = edsr().to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

epochs = 2
print_every = 25
batch_num = 0

for epoch_num in range(epochs):
    for img, label in trainloader:
        print(batch_num)
        optimizer.zero_grad()
        pred = model(img)

        # print(pred.shape, label.shape)
        batch_num += 1
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        writer.add_scalar('training loss', loss, batch_num)

        if batch_num % 5 == 0:
            with torch.no_grad():
                model.eval()
                val_ims, val_lbs = iter(validloader).next()
                test_pred = model(val_ims)
                grid_input = torchvision.utils.make_grid(val_ims)
                writer.add_image('validation input', grid_input, batch_num)
                grid_output = torchvision.utils.make_grid(test_pred)
                writer.add_image('validation output', grid_output, batch_num)
                vloss = criterion(test_pred, val_lbs)
                val_loss = vloss.item()
                writer.add_scalar('validation loss', vloss, batch_num)
                model.train()


import pickle
bk = {}
for k in dir():
    obj = globals()[k]
    if is_picklable(obj):
        try:
            bk.update({k: obj})
        except TypeError:
            pass
with open('./save/'+tensorboard_name+'_'+str(batch_num)+'.pkl', 'wb') as f:
    pickle.dump(bk, f)

writer.close()


