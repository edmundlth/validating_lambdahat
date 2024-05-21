import jax
import jax.numpy as jnp
import jax.tree_util as jtree

import numpy as np
import optax
import pandas as pd 
import matplotlib.pyplot as plt

from dln import (
    create_dln_model, 
    create_minibatches, 
    true_dln_learning_coefficient, 
    mse_loss, 
)
from sgld_utils import (
    SGLDConfig, 
    run_sgld
)
from utils import to_json_friendly_tree, running_mean
import os
from sacred import Experiment
# Create a new experiment
ex = Experiment('dln_saddle_dynamics')


@ex.config
def cfg():
    expt_name = None
    in_out_dim = 5 # input output dimension, control teacher matrix shape
    sgld_config = {
        'epsilon': 1e-6,
        'gamma': 1.0,
        'num_steps': 100,
        "num_chains": 1, # TODO: not implemented chains
        "batch_size": 128
    }
    loss_trace_minibatch = True # if True loss_trace uses minibatch, else use full dataset. 
    width = 10
    initialisation_exponent = 1.5
    num_hidden_layers = 4
    num_training_data = 10000
    itemp = 1 / np.log(num_training_data)
    training_config = {
        "optim": "sgd", 
        "learning_rate": 1e-4, 
        "momentum": None, 
        "batch_size": 128, 
        "num_steps": 20000
    }
    seed = 42
    logging_period = 50
    verbose = False
    do_plot = False


@ex.automain
def run_experiment(
    _run, 
    expt_name,
    in_out_dim,
    sgld_config, 
    loss_trace_minibatch,
    width,
    initialisation_exponent,
    num_hidden_layers, 
    num_training_data, 
    itemp, 
    training_config,
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
    # Teacher matrix
    initialisation_sigma = np.sqrt(width ** (-initialisation_exponent))
    teacher_matrix = 10.0 * np.diag(np.arange(in_out_dim) + 1)
    input_dim = teacher_matrix.shape[0]
    output_dim = teacher_matrix.shape[1]
    layer_widths = [width] * num_hidden_layers + [output_dim]

    # Training data from teacher matrix
    rngkey, key = jax.random.split(rngkey)
    x_train = jax.random.normal(key, shape=(num_training_data, input_dim))
    y_train = x_train @ teacher_matrix 

    # DLN model
    model = create_dln_model(layer_widths, sigma=initialisation_sigma)
    rngkey, subkey = jax.random.split(rngkey)
    init_param = model.init(rngkey, jnp.zeros((1, input_dim)))
    jtree.tree_map(lambda x: print(x.shape), init_param)
    loss_fn = jax.jit(lambda param, inputs, targets: mse_loss(param, model, inputs, targets))

    ##############################################
    # Train the model and do SGLD at some interval
    ##############################################
    
    sgld_config = SGLDConfig(**sgld_config)

    optimizer = optax.sgd(learning_rate=training_config["learning_rate"])
    max_steps = training_config["num_steps"]
    t = 0
    rngkey, subkey = jax.random.split(rngkey)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))
    trained_param = model.init(rngkey, jnp.zeros((1, input_dim)))
    opt_state = optimizer.init(trained_param)
    _run.info = []
    while t < max_steps:
        for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=training_config["batch_size"]):
            train_loss, grads = grad_fn(trained_param, x_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            trained_param = optax.apply_updates(trained_param, updates)
            
            if t % logging_period == 0: 
                rngkey, subkey = jax.random.split(rngkey)
                y_realisable = model.apply(trained_param, x_train)
                y = y_realisable
                # y = y_train
                
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
                    verbose=False
                )
                
                init_loss = loss_fn(trained_param, x_train, y)
                lambdahat = float(np.mean(loss_trace) - init_loss) * num_training_data * itemp

                true_matrix = jnp.linalg.multi_dot(
                    [trained_param[f'deep_linear_network/linear{loc}']['w'] for loc in [''] + [f'_{i}' for i in range(1, len(layer_widths))]]
                )
                true_rank = jnp.linalg.matrix_rank(true_matrix)
                true_lambda, true_multiplicity = true_dln_learning_coefficient(
                    true_rank, 
                    layer_widths, 
                    input_dim, 
                )
                
                rec = {
                    "t": t + 1, 
                    "train_loss": float(train_loss),
                    "lambdahat": float(lambdahat),
                    "true_lambda": true_lambda, 
                    "true_multiplicity": true_multiplicity, 
                    "loss_trace": loss_trace, 
                    "true_rank": true_rank,
                    "init_loss": float(init_loss),
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
            f"$M={num_hidden_layers}$, "
            f"$H={width}$, "
            f"$T={training_config['num_steps']}$, " 
            f"$k={logging_period}$, "
            f"RNG seed$={seed}$"
        )
        ax.plot(df["t"], np.clip(df["lambdahat"], a_min=0, a_max=np.inf), "kx", alpha=0.3)
        yvals = running_mean(df["lambdahat"])
        ax.plot(df["t"], yvals, "g-")
        ax.set_ylabel("Estimated LLC, $\hat{\lambda}(w^*)$")
        ax.set_title(title, fontsize="large")
        filepath = f"dln_saddle_to_saddle_plot_{expt_name}_{seed}.pdf"
        fig.savefig(filepath, bbox_inches="tight")
        print(f"Image saved at: {filepath}")
    return 
            

