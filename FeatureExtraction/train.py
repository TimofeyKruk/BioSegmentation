import os
import torch
import time
import json
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.sampler import SubsetRandomSampler
from FeatureExtraction.dataset import CAM17Dataset
from FeatureExtraction.model import FEModel

# To find where gradients explode
torch.autograd.set_detect_anomaly(True)

# TODO: MAKE IT CONFIG!
dataset_path = "/media/krukts/HDD/BioDiploma/Timofey/dataset_PINK_300k_Stage2_WithErrorNormCls"
model_name = "efficientnet-b0"
save_name = "SecondTry"

tb_name = "log/tb/" + save_name

validation_split = 0.2
batch_size = 40
lr_start = 1e-4
epochs = 15
log_every = 25
save_model = True
img_size = (224, 224)

if __name__ == '__main__':
    print("Script started to work!")

    # Dataset creation
    dataset = CAM17Dataset(dataset_path=dataset_path, img_size=img_size)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)

    train_indices, validation_indices = indices[split:], indices[:split]

    # Creating samplers
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_dataloader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  sampler=train_sampler,
                                  drop_last=True)
    validation_dataloader = DataLoader(dataset,
                                       batch_size=batch_size,
                                       num_workers=2,
                                       sampler=validation_sampler,
                                       drop_last=True)

    print("Train dataloader len: ", len(train_dataloader))
    print("Validation dataloader len: ", len(validation_dataloader))
    print("Datasets were loaded!________\n")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Model initialization
    model = FEModel(model_name)
    model.to(device)

    # SummaryWriter initialization
    tensorboard = SummaryWriter(tb_name)
    tensorboard.add_graph(model, torch.rand(8, 3, 224, 224).to(device))

    # LOSS & OPTIMIZER
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    # --------------------------------------------
    # TODO: try bigger momentum (0.9 for instance)
    # TODO: Maybe try another optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

    # TODO: As Ksenia recommended, maybe I should tri cyclic schedulers
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 35, 50, 70], gamma=0.7)
    # Cyclic scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=lr_start, step_size_up=2000)
    # # TODO: Registering hooks to crop the gradients
    # clip_value = 1.0
    # for p in model.parameters():
    #     p.register_hook(lambda
    #     grad: torch.clamp(grad, -clip_value, clip_value))

    # TRAINING
    i = 0
    for epoch in range(epochs):
        model.train()
        print("Epoch: ", epoch, ", i: ", i)

        j = 0
        total_loss = 0
        for batch, labels in train_dataloader:
            i += 1
            j += 1

            # Processing and sending data to cuda
            batch = batch.to(device)
            labels = labels.to(device)

            # Zero previous gradients
            optimizer.zero_grad()

            outputs = model(batch)

            # TODO: Compare predicted next frame with the real one
            current_loss = criterion(outputs, labels)
            # print(current_loss.item())

            # The gradient descent
            current_loss.backward()

            # Cropping the gradients
            # TODO: maybe return back to 0.5
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += current_loss.item()

            if j % log_every == 0:
                print("Train loss: ", total_loss / log_every)
                print("Last used LR: ", scheduler.get_last_lr())

                tensorboard.add_scalar("Train loss", total_loss / log_every, i // log_every)
                tensorboard.add_scalar("LR", scheduler.get_last_lr()[0], i // log_every)
                total_loss = 0
                j = 0
                # Logging weights and grad
                for name, weight in model.named_parameters():
                    tensorboard.add_histogram(name, weight, i)
                    if weight.grad is not None:
                        tensorboard.add_histogram(f'{name}.grad', weight.grad, i // log_every)
                        # print("Grad max {}.grad --> {}".format(name, weight.grad.max().item()))
                    else:
                        print("___ !!None!! grad!:  {}.grad".format(name))

            scheduler.step()

        print("____ Last used LR: ", scheduler.get_last_lr())

        # ____________ Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total_accuracy = 0.0
            for batch, labels in validation_dataloader:
                # Processing and sending data to cuda
                batch = batch.to(device)
                labels = labels.to(device)

                outputs = model(batch)
                pred = outputs[:, 1]
                pred = pred > 0.5
                total_accuracy += torch.sum(pred == labels).item() / batch_size

                current_loss = criterion(outputs, labels)
                val_loss += current_loss.item()

            tensorboard.add_scalar("Validation loss", val_loss / len(validation_dataloader), epoch)
            tensorboard.add_scalar("Accuracy", total_accuracy / len(validation_dataloader), epoch)
            print("Validation loss: ", val_loss / len(validation_dataloader))
            print("Epoch accuracy: ", total_accuracy / len(validation_dataloader))

        # Saving model weights
        if save_model is True and (epoch + 1) % 4 == 0:
            torch.save(model.state_dict(), os.path.join("log", save_name + "after{}".format(epoch + 1)))
            print("Model was saved as file: ", os.path.join("log", save_name + "after{}".format(epoch + 1)))

    # Saving model weights
    if save_model is True:
        torch.save(model.state_dict(), os.path.join("log", save_name))
        print("Model was saved as file: ", os.path.join("log", save_name))

    tensorboard.flush()
    tensorboard.close()
    print("Script train.py worked!")
