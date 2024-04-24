import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import haiku as hk
import tensorflow_datasets as tfds

import numpy as np
import optax
import pandas as pd 
import matplotlib.pyplot as plt

from sgld_utils import (
    SGLDConfig, 
    run_sgld
)
from utils import to_json_friendly_tree, running_mean
from typing import NamedTuple

from sacred import Experiment
# Create a new experiment
ex = Experiment('llc_curve')


class TrainingConfig(NamedTuple):
    optim: str
    learning_rate: float
    batch_size: int
    num_steps: int
    momentum: float = None

# Haiku module for ResNet18 with is_training flag
def net_fn(x, is_training=True):
    net = hk.nets.ResNet18(num_classes=10)
    return net(x, is_training=is_training)

# Transformed function with state
def make_resnet18():
    return hk.transform_with_state(net_fn)


# Load and preprocess CIFAR-10 dataset
def load_cifar10():
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_images, train_labels = train_ds['image'], train_ds['label']
    test_images, test_labels = test_ds['image'], test_ds['label']
    # Normalize images
    train_images = train_images.astype(jnp.float32) / 255.0
    test_images = test_images.astype(jnp.float32) / 255.0
    return train_images, train_labels, test_images, test_labels


# Initialize the model and optimizer
def initialize_model(rng, learning_rate=0.001):
    model = make_resnet18()
    dummy_input = jnp.ones([1, 32, 32, 3], jnp.float32)
    params, state = model.init(rng, dummy_input, True)
    optimizer = optax.adam(learning_rate)
    return model, params, state, optimizer


def batch_generator(x, y, batch_size, rngkey):
    num_examples = len(x)
    while True:  # This creates an infinite loop, each time reshuffling and starting over
        perm = np.random.permutation(num_examples)
        for i in range(0, num_examples, batch_size):
            batch_indices = perm[i:i + batch_size]
            yield x[batch_indices], y[batch_indices]
        rngkey, _ = jax.random.split(rngkey)  # Reshuffle RNG for next epoch's shuffling

def evaluate_accuracy(model, params, state, x, y, rngkey):
    """Evaluation metric (classification accuracy)."""
    logits, _ = model.apply(params, state, rngkey, x, False)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)


@ex.config
def cfg():
    expt_name = None
    sgld_config = {
        'epsilon': 5e-6,
        'gamma': 1.0,
        'num_steps': 1000,
        "num_chains": 1, # TODO: not implemented chains
        "batch_size": 1024
    }
    loss_trace_minibatch = True # if True SGLD loss_trace uses minibatch, else use full dataset. 
    model_data_config = { # TODO: currently only RESNET18 + CIFAR10 is implemented
        "model_name": "resnet18",
        "data_name": "cifar10"
    }
    training_config = {
        "optim": "sgd", 
        "learning_rate": 1e-3, 
        "momentum": None, 
        "batch_size": 128, 
        "num_steps": 20000
    }
    force_realisable = False # if True use LLC realisable i.e. y = model(param_init, x) 
    seed = 42
    logging_period = 200
    verbose = False
    do_plot = False


@ex.automain
def run_experiment(
    _run, 
    expt_name,
    sgld_config, 
    loss_trace_minibatch,
    model_data_config,
    training_config,
    force_realisable,
    seed,
    logging_period,
    verbose,
    do_plot,
):
    # seeding
    np.random.seed(seed)
    rngkey = jax.random.PRNGKey(seed)

    ####################
    # Initialisations
    ####################
    sgld_config = SGLDConfig(**sgld_config)
    training_config = TrainingConfig(**training_config)

    x_train, y_train, x_test, y_test = load_cifar10()
    
    rngkey, subkey = jax.random.split(rngkey)
    model, trained_param, model_state, optimizer = initialize_model(rngkey)
    rngkey, subkey = jax.random.split(rngkey)
    opt_state = optimizer.init(trained_param)
        
    rngkey, subkey = jax.random.split(rngkey)
    train_dataset_iter = batch_generator(
        x_train, 
        y_train, 
        training_config.batch_size, 
        rngkey
    )
    rngkey, subkey = jax.random.split(rngkey)
    num_training_data = x_train.shape[0]

    if training_config.optim.lower() == "sgd":
        optimizer = optax.sgd(
            learning_rate=training_config.learning_rate, 
            momentum=training_config.momentum
        )
    elif training_config.optim.lower() == "adam":
        optimizer = optax.adam(
            learning_rate=training_config.learning_rate, 
            momentum=training_config.momentum
        )

    
    def compute_loss(params, state, rngkey, x, y, is_training):
        labels_one_hot = jax.nn.one_hot(y, 10)
        logits, new_state = model.apply(params, state, rngkey, x, is_training)
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)), new_state


    @jax.jit
    def update_step(params, state, rngkey, x, y, opt_state):
        (loss_val, new_state), grad = jax.value_and_grad(compute_loss, has_aux=True)(params, state, rngkey, x, y, True)
        updates, new_opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss_val, new_params, new_state, new_opt_state

    
    ##############################################
    # Train the model and do SGLD at some interval
    ##############################################
    itemp = 1 / np.log(num_training_data)
    max_steps = training_config.num_steps
    
    _run.info = []
    t = 0
    while t < max_steps:
        for x_batch, y_batch in train_dataset_iter:
            rngkey, subkey = jax.random.split(rngkey)
            train_loss, trained_param, model_state, opt_state = update_step(
                trained_param, 
                model_state, 
                rngkey, 
                x_batch, 
                y_batch, 
                opt_state
            )
            
            if t % logging_period == 0: 
                if force_realisable:
                    y = model.apply(trained_param, x_train)
                else:
                    y = y_train
                
                rngkey, subkey = jax.random.split(rngkey)
                loss_fn = jax.jit(
                    lambda parameter, x, y: compute_loss(
                        parameter, model_state, rngkey, x, y, False
                    )[0]
                )
                
                loss_trace, distances, acceptance_probs = run_sgld(
                    subkey, 
                    loss_fn, 
                    sgld_config, 
                    trained_param, 
                    x_train, 
                    y,
                    itemp=itemp, 
                    trace_batch_loss=loss_trace_minibatch, 
                    compute_distance=False, 
                    verbose=verbose
                )
                
                init_loss = loss_fn(trained_param, x_train, y)
                lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp

                rngkey, subkey = jax.random.split(rngkey)
                test_accuracy = evaluate_accuracy(model, trained_param, model_state, x_test, y_test, rngkey)
                rngkey, subkey = jax.random.split(rngkey)
                train_accuracy = evaluate_accuracy(model, trained_param, model_state, x_train, y_train, rngkey)

                rec = {
                    "t": t + 1, 
                    "train_loss": float(train_loss),
                    "lambdahat": float(lambdahat),
                    "loss_trace": loss_trace, 
                    "init_loss": float(init_loss),
                    "sgld_distances": distances, 
                    "sgld_acceptance_probs": acceptance_probs, 
                    "test_accuracy": float(test_accuracy),
                    "train_accuracy": float(train_accuracy),
                }
                if verbose:
                    print(rec["t"], rec["train_loss"], rec["lambdahat"])

                _run.info.append(to_json_friendly_tree(rec))
            
            t += 1
            if t >= max_steps:
                break
    if do_plot:
        df = pd.DataFrame(_run.info)
        fig, ax = plt.subplots()
        ax.plot(df["t"], df["train_loss"])
        ax.set_yscale("log")
        ax.set_ylabel("Training Loss")
        ax.set_xlabel("Num SGD Steps")

        ax = ax.twinx()
        title = (
            f"$T={training_config.num_steps}$, " 
            f"$k={logging_period}$, "
            f"RNG seed$={seed}$"
        )
        ax.plot(df["t"], np.clip(df["lambdahat"], a_min=0, a_max=np.inf), "kx", alpha=0.3)
        yvals = running_mean(df["lambdahat"])
        ax.plot(df["t"], yvals, "g-")
        ax.set_ylabel("Estimated LLC, $\hat{\lambda}(w^*)$")
        ax.set_title(title, fontsize="large")
        filepath = f"llc_curve_plot_{expt_name}_{seed}.pdf"
        fig.savefig(filepath, bbox_inches="tight")
        print(f"Image saved at: {filepath}")
    return 
