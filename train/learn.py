import time
import sys
import os

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

# from tools.utils import *


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)

    return g


def print_gradients(model):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p.grad, p.requires_grad, p.shape)
    pass


def test(model, test_loader, device, threshold):
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """

    model.eval()
    recons_loss_tot = 0
    test_size = len(test_loader)
    correct = 0.0
    total_nodes = 0.0
    correct_true = 0.0
    predicted_true = 0.0
    target_true = 0.0
    iterator = iter(test_loader)
    for batch_idx in range(test_size):
        try:
            graph, inds, _ = next(iterator)
        except StopIteration:
            iterator = iter(test_loader)
            batch_idx -= 1
            continue
        # Get data on the devices
        graph = send_graph_to_device(graph, device)
        batch_size = graph.number_of_nodes()
        # Do the computations for the forward pass
        with torch.no_grad():
            out = model(graph).squeeze()

            #TODO: get labels from graph
            labels = graph.ndata['interface'].to(torch.float32)

            loss = F.binary_cross_entropy(out, labels)

            recons_loss_tot += loss

            preds = (out > threshold).float()
            correct += (preds == labels).float().sum()

            target_true += labels.sum()
            predicted_true += preds.sum()
            correct_true += ((labels==preds)*(preds==1)).float().sum()

            total_nodes += batch_size

    try:
        acc = correct/total_nodes
    except(ZeroDivisionError):
        print('zero division error')
        for graph in test_loader:
            print(graph.number_of_nodes())

    precision = correct_true / predicted_true
    recall = correct_true / target_true
    f1 = 2*precision*recall / (precision + recall)

    return recons_loss_tot / test_size, acc, precision, recall, f1


def train_model(model, optimizer, train_loader, test_loader,
                # save_path,
                threshold = 0.5, verbose=True,
                writer=None, num_epochs=25, wall_time=None, embed_only=-1):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param train_loader: data loader for training
    :param test_loader: data loader for validation
    :param save_path: where to save the model
    :param writer: a pytorch writer object
    :param num_epochs: int number of epochs
    :param wall_time: The number of hours you want the model to run
    :param embed_only: number of epochs before starting attributor training.
    :return:
    """
    device = model.current_device
    epochs_from_best = 0
    attributions = 0
    early_stop_threshold = 60
    start_time = time.time()
    best_loss = sys.maxsize
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0.0
        total_nodes = 0.0
        num_batches = len(train_loader)
        # print('num_batches:', num_batches)
        # if num_batches < 10:
            # print("ERROR not enough batches")
            # raise ValueError
        # for graph in train_loader:
            # print(graph)

        iterator = iter(train_loader)
        for batch_idx in range(len(train_loader)):

            try:
                graph, inds, _ = next(iterator)
            except:
                iterator = iter(train_loader)
                batch_idx -= 1
                continue
            # Get data on the devices
            graph = send_graph_to_device(graph, device)
            labels = graph.ndata['interface'].to(torch.float32)
            batch_size = graph.number_of_nodes()
            # print(b_size)
            # Do the computations for the forward pass
            out = model(graph).squeeze()

            # print('out:\n', out, out.shape)
            # print('label:\n', labels, labels.shape)
            loss = F.binary_cross_entropy(out, labels)

            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            loss = loss.item()
            preds = (out > threshold).float()
            correct += (preds == labels).float().sum()
            total_nodes += batch_size
            # print(f'batch: {batch_idx} \t batch_size: {batch_size},\t total_nodes:', total_nodes)
            running_loss += loss
            time_elapsed = time.time() - start_time

            if batch_idx % 20 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Time: {:.6f}'.format(
                    epoch + 1,
                    (batch_idx + 1),
                    num_batches,
                    100. * (batch_idx + 1) / num_batches,
                    loss,
                    time_elapsed))

                # tensorboard logging
                step = epoch * num_batches + batch_idx
                writer.add_scalar("Training loss", loss, step)

        # # Log training metrics
        try:
            train_acc = correct/total_nodes
        except(ZeroDivisionError):
            print('ERROR: zero division error')
            print('total_nodes:', total_nodes)
            print('num_batches:', num_batches)
            for graph in train_loader:
                print(graph[0].number_of_nodes())

        train_loss = running_loss / num_batches
        if verbose: print('Train Epoch: {} [100%]\t Loss: {:.6f} \t Accuracy: {:.4f} \t Time: {:.2f}'.format(
                epoch + 1,
                train_loss,
                train_acc,
                time_elapsed))
        writer.add_scalar("Training epoch loss", train_loss, epoch)
        writer.add_scalar("Training accuracy", train_acc, epoch)

        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight.grad, epoch)

        # Test phase
        test_loss, test_acc, precision, recall, f1\
                = test(model, test_loader, device, threshold)
        writer.add_scalar("Test Accuracy", test_acc, epoch)
        writer.add_scalar("Precision", precision, epoch)
        writer.add_scalar("Recall", recall, epoch)
        writer.add_scalar("F1", f1, epoch)
        writer.add_scalar("Test Accuracy", test_acc, epoch)
        writer.add_scalar("Test loss during training", test_loss, epoch)
        print(f"Test loss: {test_loss:.4f} \t test accuracy: {test_acc:.4f}")
        #
        # Checkpointing
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_from_best = 0

            # model.cpu()
            # print(">> saving checkpoint")
            # torch.save({
                # 'epoch': epoch,
                # 'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict()
            # }, save_path)
            # model.to(device)

        # Early stopping
        else:
            epochs_from_best += 1
            if epochs_from_best > early_stop_threshold:
                print('This model was early stopped')
                break

        # Sanity Check
        if wall_time is not None:
            # Break out of the loop if we might go beyond the wall time
            time_elapsed = time.time() - start_time
            if time_elapsed * (1 + 1 / (epoch + 1)) > .95 * wall_time * 3600:
                break
    return best_loss


def make_predictions(data_loader, model, optimizer, model_weights_path):
    """
    :param data_loader: an iterator on input data
    :param model: An empty model
    :param optimizer: An empty optimizer
    :param model_weights_path: the path of the model to load
    :return: list of predictions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    predictions = []

    for batch_idx, inputs in enumerate(data_loader):
        inputs = inputs.to(device)
        predictions.append(model(inputs))
    return predictions


if __name__ == "__main__":
    pass
