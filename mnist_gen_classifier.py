import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from os import mkdir
from os.path import isdir, join
from typing import List, Tuple

from tqdm import trange
from st_models import *

params = {'n_classes': 2,  # first n_classes digits are used
          'sigma_weights': 1.,
          'sigma_init': 0.01,
          'shape': [28, 28],
          'train_data_path': './data/train.npy',
          'val_data_path': './data/val.npy',
          'test_data_path': './data/test.npy',
          'images_path': './images',
          'dtype': torch.float32,
          'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
          'batch_size': 128,
          'gen_learning_rate': 1e-1,
          'tree_loss_weight': 1e-2,
          'gen_n_iters': 1,  # for each class
          'n_samples': 5,  # number of trees
          'burn_in': 30000,
          'gibbs_n_iters': 10000,
          'disc_learning_rate': 1e-3,
          'disc_n_iters': 10,
          'n_trees': 10,  # for the ensemble
          'model_type': 'ensemble'}


def generate_images(samples: List[np.ndarray], tag: str) -> None:
    """Save the mean and the sample images for each class."""
    n_classes = len(samples)
    for c in range(n_classes):
        plt.imsave(join(params['images_path'], f'{c}_{params["model_type"]}_{tag}_sample.png'), samples[c][-1])
        plt.imsave(join(params['images_path'], f'{c}_{params["model_type"]}_{tag}_mean.png'), samples[c].mean(axis=0))


def generate_trees(trees: List[List[Tuple[torch.Tensor, torch.Tensor]]]) -> None:
    """Generate the tree images for each class."""

    def _generate_image(tree: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Convert the tree to an image."""
        horizontal_edges, vertical_edges = tree
        image_shape = horizontal_edges.shape[0], horizontal_edges.shape[1] + 1
        image = np.zeros((image_shape[0] * 9, image_shape[1] * 9), dtype=np.int64)
        for i in range(horizontal_edges.shape[0]):
            for j in range(horizontal_edges.shape[1]):
                if horizontal_edges[i, j]:
                    image[4 + i * 9, 4 + j * 9: 13 + j * 9] = 1

        for i in range(vertical_edges.shape[0]):
            for j in range(vertical_edges.shape[1]):
                if vertical_edges[i, j]:
                    image[4 + i * 9: 13 + i * 9, 4 + j * 9] = 1

        return image

    n_classes = len(trees)
    for c in range(n_classes):
        plt.imsave(join(params['images_path'], f'{c}_first_tree.png'), _generate_image(trees[c][0]))
        plt.imsave(join(params['images_path'], f'{c}_last_tree.png'), _generate_image(trees[c][-1]))


def evaluate_classifier(classifier: Classifier, data: np.ndarray, labels: np.ndarray,
                        batch_size: int) -> Tuple[float, float]:
    """Evaluate the classifier over the given dataset."""
    classifier.eval()
    cross_entropy = 0.
    correct = 0
    n_batches = (len(data) - 1) // batch_size
    classifier.eval()
    with torch.no_grad():
        for i in range(n_batches):
            batch = torch.tensor(data[i * batch_size: (i + 1) * batch_size], dtype=params['dtype'],
                                 device=params['device'])
            labs = torch.tensor(labels[i * batch_size: (i + 1) * batch_size], dtype=torch.int64,
                                device=params['device'])
            log_probs = classifier(batch)
            correct += (torch.argmax(log_probs, dim=1) == labs).sum().item()
            cross_entropy += -log_probs[torch.arange(labs.shape[0]), labs].sum()
    return correct / len(data), cross_entropy / len(data)


def main():
    train_data = np.load(params['train_data_path'])
    val_data = np.load(params['val_data_path'])
    test_data = np.load(params['test_data_path'])
    n_classes = params['n_classes']
    batch_size = params['batch_size']
    if params['model_type'] == 'learned':
        model_class = LearnedSpanningTreeModel
    elif params['model_type'] == 'ensemble':
        model_class = EnsembleSpanningTreeModel
    else:
        raise ValueError('Unknown model type')
    generative_models = []
    trees = []
    samples = []
    for c in range(n_classes):
        print(f'Class: {c}')
        curr_train = train_data[c].reshape((-1, 28, 28))
        curr_val = val_data[c].reshape((-1, 28, 28))
        model = model_class(params).to(params['device'])
        n_train_batches = (len(curr_train) - 1) // batch_size + 1
        n_val_batches = (len(curr_val) - 1) // batch_size + 1
        optimizer = optim.Adam(model.parameters(), lr=params['gen_learning_rate'])
        class_trees = []
        for i in range(params['gen_n_iters']):
            model.train()
            order = np.random.permutation(len(curr_train))
            print(f'Iteration {i + 1}:')
            bar = trange(n_train_batches, file=sys.stdout)
            bar.set_description('Current log-prob: NaN')
            for j in bar:
                batch = torch.tensor(curr_train[order[j * batch_size: (j + 1) * batch_size]],
                                     dtype=params['dtype'], device=params['device'])
                optimizer.zero_grad()
                if params['model_type'] == 'learned':
                    conditionals, tree = model(batch, params['n_samples'])
                    loss_tree = -(tree * conditionals.detach()).mean() * params['tree_loss_weight']
                    loss_tree.backward()
                    if len(class_trees) < 2:
                        class_trees.append(model.tree)
                    else:
                        class_trees[-1] = model.tree
                else:
                    conditionals = model(batch)
                bar.set_description(f'Current log-prob: {conditionals.mean().item()}')
                loss_params = -conditionals.mean()
                loss_params.backward()
                optimizer.step()
            total_val_lp = 0.
            model.eval()
            for j in range(n_val_batches):
                batch = torch.tensor(curr_val[j * batch_size: (j + 1) * batch_size],
                                     dtype=params['dtype'], device=params['device'])
                total_val_lp += model.log_prob(batch).mean().item()
            print(f'Validation log-probability: {total_val_lp / n_val_batches}')
            print('---------------------------------------------------')
        print('---------------------------------------------------')
        trees.append(class_trees)
        samples.append(model.sample(params['burn_in'], params['gibbs_n_iters']))
        generative_models.append(model)

    if not isdir(params['images_path']):
        mkdir(params['images_path'])
    generate_images(samples, 'gen_train')
    if params['model_type'] == 'learned':
        generate_trees(trees)

    classifier = Classifier(generative_models)
    train_labels = np.array(list(itertools.chain(*[[i] * len(train_data[i]) for i in range(n_classes)])))
    val_labels = np.array(list(itertools.chain(*[[i] * len(val_data[i]) for i in range(n_classes)])))
    test_labels = np.array(list(itertools.chain(*[[i] * len(test_data[i]) for i in range(n_classes)])))
    train_data = np.concatenate(train_data[:n_classes], axis=0).reshape((-1, 28, 28))
    val_data = np.concatenate(val_data[:n_classes], axis=0).reshape((-1, 28, 28))
    test_data = np.concatenate(test_data[:n_classes], axis=0).reshape((-1, 28, 28))
    print('Testing classification without discriminative training...')
    accuracy, cross_entropy = evaluate_classifier(classifier, test_data, test_labels, batch_size)
    print(f'Test accuracy: {accuracy}')
    print(f'Test cross-entropy: {cross_entropy}')
    print('---------------------------------------------------')
    print('Training for classification...')
    optimizer = optim.Adam(classifier.parameters(), lr=params['disc_learning_rate'])
    n_train_batches = (len(train_data) - 1) // batch_size + 1
    for i in range(params['disc_n_iters']):
        classifier.train()
        order = np.random.permutation(len(train_data))
        print(f'Iteration {i + 1}')
        bar = trange(n_train_batches, desc='Curr loss: NaN', file=sys.stdout)
        for j in bar:
            batch = torch.tensor(train_data[order[j * batch_size: (j + 1) * batch_size]], dtype=params['dtype'],
                                 device=params['device'])
            labels = torch.tensor(train_labels[order[j * batch_size: (j + 1) * batch_size]], dtype=torch.int64,
                                  device=params['device'])
            log_probs = classifier(batch)
            loss = -log_probs[torch.arange(batch.shape[0]), labels].mean()
            bar.set_description(f'Curr loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Validating...')
        accuracy, cross_entropy = evaluate_classifier(classifier, val_data, val_labels, batch_size)
        print(f'Validation accuracy: {accuracy}')
        print(f'Validation cross-entropy: {cross_entropy}')
        print('---------------------------------------------------')
    print('Testing...')
    accuracy, cross_entropy = evaluate_classifier(classifier, test_data, test_labels, batch_size)
    print(f'Test accuracy: {accuracy}')
    print(f'Test cross-entropy: {cross_entropy}')
    print('---------------------------------------------------')
    print('Generating new images...')
    samples = []
    for c in range(n_classes):
        print(f'Generating class {c}')
        samples.append(generative_models[c].sample(params['burn_in'], params['gibbs_n_iters']))
    generate_images(samples, 'disc_train')
    print('---------------------------------------------------')
    print('---------------------------------------------------')


if __name__ == '__main__':
    main()
