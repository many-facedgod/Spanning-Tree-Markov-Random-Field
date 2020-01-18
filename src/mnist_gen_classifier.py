import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from os import mkdir
from os.path import isdir, join
from typing import List, Tuple

from tqdm import trange
from mst_ising import MSTIsingModel, Classifier

params = {'sigma_weights': 1.,
          'sigma_init': 0.01,
          'shape': [28, 28],
          'train_data_path': '../data/train.npy',
          'val_data_path': '../data/val.npy',
          'test_data_path': '../data/test.npy',
          'images_path': '../images',
          'dtype': torch.float32,
          'batch_size': 128,
          'learning_rate': 1e-1,
          'tree_loss_weight': 1e-2,
          'n_iters': 10,
          'n_samples': 10,
          'burn_in': 30000,
          'n_gibbs_iters': 10000,
          'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')}


def generate_images(samples: List[np.ndarray]) -> None:
    """Save the mean and the sample images for each class."""
    n_classes = len(samples)
    for c in range(n_classes):
        plt.imsave(join(params['images_path'], f'{c}_sample.png'), samples[c][-1])
        plt.imsave(join(params['images_path'], f'{c}_mean.png'), samples[c].mean(axis=0))


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


def main():
    train_data = np.load(params['train_data_path'])
    val_data = np.load(params['val_data_path'])
    test_data = np.load(params['test_data_path'])
    n_classes = 10
    generative_models = []
    trees = []
    samples = []
    for c in range(n_classes):
        print(f'Class: {c}')
        curr_train = train_data[c].reshape((-1, 28, 28))
        curr_val = val_data[c].reshape((-1, 28, 28))
        model = MSTIsingModel(params).to(params['device'])
        batch_size = params['batch_size']
        n_train_batches = (len(curr_train) - 1) // batch_size + 1
        n_val_batches = (len(curr_val) - 1) // batch_size + 1
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        class_trees = []
        for i in range(params['n_iters']):
            model.train()
            order = np.random.permutation(len(curr_train))
            print(f'Iteration {i + 1}:')
            bar = trange(n_train_batches, file=sys.stdout)
            bar.set_description('Current log-prob: NaN')
            for j in bar:
                batch = torch.tensor(curr_train[order[j * batch_size: (j + 1) * batch_size]],
                                     dtype=params['dtype'], device=params['device'])
                conditionals, tree = model(batch, params['n_samples'])
                bar.set_description(f'Current log-prob: {conditionals.mean().item()}')
                loss_params = -conditionals.mean()
                loss_tree = -(tree * conditionals.detach()).mean() * params['tree_loss_weight']
                optimizer.zero_grad()
                loss_params.backward()
                loss_tree.backward()
                optimizer.step()
                class_trees.append(model.tree)
            total_val_lp = 0.
            model.eval()
            class_trees = [class_trees[0], class_trees[-1]]
            for j in range(n_val_batches):
                batch = torch.tensor(curr_val[j * batch_size: (j + 1) * batch_size],
                                     dtype=params['dtype'], device=params['device'])
                total_val_lp += model.log_prob(batch).item()
            print(f'Validation log-probability: {total_val_lp / n_val_batches}')
            print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        trees.append(class_trees)
        samples.append(model.sample(params['burn_in'], params['n_gibbs_iters']))
        generative_models.append(model)
    classifier = Classifier(generative_models)
    print('Testing classification...')
    log_prob_sum = 0.
    correct = 0
    for c in range(n_classes):
        data = test_data[c].reshape((-1, 28, 28))
        for point in data:
            batch = torch.tensor(point[np.newaxis, :, :], dtype=params['dtype'], device=params['device'])
            prediction, log_probs = classifier.predict(batch)
            correct += prediction == c
            log_prob_sum += log_probs[c]
    total_test_points = sum([len(test_data[i]) for i in range(n_classes)])
    print(f'Accuracy: {correct / total_test_points}')
    print(f'Cross-entropy: {-log_prob_sum / total_test_points}')
    print('---------------------------------------------------')
    if not isdir(params['images_path']):
        mkdir(params['images_path'])
    generate_images(samples)
    generate_trees(trees)


if __name__ == '__main__':
    main()