import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtree

import numpy as np
from typing import Sequence
import itertools


# Define the DLN model
class DeepLinearNetwork(hk.Module):
    def __init__(self, layer_widths: Sequence[int], name: str = None, with_bias=False):
        super().__init__(name=name)
        self.layer_widths = layer_widths
        self.with_bias = with_bias

    def __call__(self, x):
        for width in self.layer_widths:
            x = hk.Linear(width, with_bias=self.with_bias)(x)
        return x


def dln_forward_fn(x, layer_widths):
    # Function to initialize and apply the DLN model
    net = DeepLinearNetwork(layer_widths)
    return net(x)


def create_dln_model(layer_widths):
    """Create a Haiku-transformed version of the model"""
    model = hk.without_apply_rng(hk.transform(lambda x: dln_forward_fn(x, layer_widths)))
    return model


def generate_training_data(true_param, model, input_dim, num_samples):
    # Generate random inputs
    inputs = np.random.uniform(-10, 10, size=(num_samples, input_dim))
    true_outputs = model.apply(true_param, inputs)
    return inputs, true_outputs

def mse_loss(param, model, inputs, targets):
    predictions = model.apply(param, inputs)
    return jnp.mean((predictions - targets) ** 2)


def create_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    else:
        indices = np.arange(len(inputs))

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]


def true_dln_learning_coefficient(true_rank, layer_widths, input_dim, verbose=False):
    """
    Theoretical learning coefficient of DLN model. 
    Reference: Aoyagi, Miki. 2023. 
    “Consideration on the Learning Efficiency Of Multiple-Layered Neural Networks 
    with Linear Units.” https://doi.org/10.2139/ssrn.4404877.
    """
    M_list = np.array([input_dim] + list(layer_widths)) - true_rank
    indices = brute_force_search_subset(M_list, early_return=verbose)
    M_subset = M_list[indices]
    if verbose:
        print(f"M_list: {M_list}, indices: {indices}, M_subset: {M_subset}")
    M_subset_sum = np.sum(M_subset)
    ell = len(M_subset) - 1
    M = np.ceil(M_subset_sum / ell)
    a = M_subset_sum - (M - 1) * ell
    output_dim = layer_widths[-1]

    term1 = (-true_rank**2 + true_rank * (output_dim + input_dim)) / 2
    term2 = a * (ell - a) / (4 * ell)
    term3 = -ell * (ell - 1) / 4 * (M_subset_sum / ell)**2
    term4 = 1 / 2 * np.sum([M_subset[i] * M_subset[j] for i in range(ell + 1) for j in range(i + 1, ell + 1)])
    learning_coefficient = term1 + term2 + term3 + term4
    return learning_coefficient

def _condition(indices, intlist, verbose=False):
    intlist = np.array(intlist)
    ell = len(indices) - 1
    subset = intlist[indices]
    complement = intlist[[i for i in range(len(intlist)) if i not in indices]]
    has_complement = len(complement) > 0
    # print(indices, subset, complement)
    if has_complement and not (np.max(subset) < np.min(complement)):
        if verbose: print(f"max(subset) = {np.max(subset)}, min(complement) = {np.min(complement)}")
        return False
    if not (np.sum(subset) >= ell * np.max(subset)):
        if verbose: print(f"sum(subset) = {sum(subset)}, ell * max(subset) = {ell * np.max(subset)}")
        return False
    if has_complement and not (np.sum(subset) < ell * np.min(complement)):
        if verbose: print(f"sum(subset) = {sum(subset)}, ell * min(complement) = {ell * np.min(complement)}")
        return False
    return True


def generate_indices_subsets(length):
    indices = list(range(length))
    for size in range(1, length + 1):
        for subset in itertools.combinations(indices, size):
            subset = np.array(subset)
            yield subset


def brute_force_search_subset(intlist, early_return=False):
    candidates = []
    for indices in generate_indices_subsets(len(intlist)):
        if _condition(indices, intlist):
            if early_return:
                return indices
            candidates.append(indices)
    if len(candidates) == 0:
        raise RuntimeError("No candidates")
    if len(candidates) > 1:
        print("More than one candidate")
    return candidates[0]