from sacred import Experiment
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import optax
from dln import (
    create_dln_model, 
    create_minibatches, 
    true_dln_learning_coefficient, 
    mse_loss, 
    generate_training_data,
)
from utils import (
    to_json_friendly_tree, 
    rand_reduce_matrix_rank, 
    create_random_matrix_with_rank, 
    get_singular_values,
)
from sgld_utils import SGLDConfig, optim_sgld, create_local_logposterior, generate_rngkey_tree


# Create a new experiment
ex = Experiment('dln_lambdahat')


@ex.config
def cfg():
    expt_name = None
    sgld_config = {
        'epsilon': 1e-5,
        'gamma': 10.0,
        'num_steps': 5000,
        "num_chains": 1, # TODO: not implemented chains
        "batch_size": 100
    }
    layer_widths = [10, 10]
    input_dim = 10
    true_param_config = {
        "method": "random", # random / zero / rand_rank / rand_rank_sv
        "prop_rank_reduce": 0.2, # rand_rank, proportion of matrices selected to have random rank. 
        "mean": 0.0, # rand_rank_sv, mean and std of gaussian to generate SV (singular values) 
        "std": 5.0, 
    }
    param_init = None
    num_training_data = 10000
    itemp = 1 / np.log(num_training_data)
    seed = 0
    verbose=False


def initialise_expt(rngkey, layer_widths, input_dim, num_training_data, true_param_config):
    """
    Handles complex initialisation procedure for various objects required for the experiments.
    """
    model = create_dln_model(layer_widths)
    dummy_input = jnp.zeros((1, input_dim))
    rngkey, subkey = jax.random.split(rngkey)
    true_param = model.init(rngkey, dummy_input)
    if true_param_config["method"] == "random":
        pass
    elif true_param_config["method"] == "zero": 
        true_param = jtree.tree_map(lambda x: x * 0.0, true_param) # zero true parameter
    elif true_param_config["method"] == "rand_rank":
        # randomly reduce rank of random matrices at a given rate
        rate = true_param_config["prop_rank_reduce"]
        jtree.tree_map(
            lambda x: rand_reduce_matrix_rank(x) if np.random.rand() < rate else x, 
            true_param
        ) 
    elif true_param_config["method"] == "rand_rank_sv":
        # randomly reduce rank of random matrices at a given rate with singular values drawn from gaussian
        rate = true_param_config["prop_rank_reduce"]
        def _helper(rngkey, x):
            rngkey, subkey = jax.random.split(rngkey)
            if jax.random.uniform(rngkey, shape=()) < rate:
                rngkey, subkey = jax.random.split(rngkey)
                rank = int(jax.random.randint(subkey, shape=(), minval=0, maxval=min(x.shape)))
            else:
                rank = min(x.shape) # i.e. full rank
            rngkey, subkey = jax.random.split(rngkey)
            return create_random_matrix_with_rank(
                rngkey, 
                x.shape, 
                rank=rank, 
                mean=true_param_config["mean"], 
                std=true_param_config["std"]
            )
        rngkey, subkey = jax.random.split(rngkey)
        jtree.tree_map(_helper, generate_rngkey_tree(subkey, true_param), true_param)
    else:
        raise RuntimeError(f"Unsupported true parameter config: {true_param_config}")
    # create training data
    x_train, y_train = generate_training_data(true_param, model, input_dim, num_training_data)
    
    return model, true_param, x_train, y_train,


@ex.automain
def run_experiment(
    _run, 
    sgld_config, 
    layer_widths,
    input_dim,
    true_param_config,
    num_training_data, 
    itemp, 
    seed,
    verbose,
):
    # seeding
    np.random.seed(seed)
    rngkey = jax.random.PRNGKey(seed)

    ####################
    # Initialisations
    ####################
    sgld_config = SGLDConfig(**sgld_config)
    rngkey, subkey = jax.random.split(rngkey)
    model, true_param, x_train, y_train = initialise_expt(
        subkey, 
        layer_widths, 
        input_dim, 
        num_training_data, 
        true_param_config
    )
    loss_fn = jax.jit(lambda param, inputs, targets: mse_loss(param, model, inputs, targets))

    # for SGLD
    rngkey, subkey = jax.random.split(rngkey)
    sgldoptim = optim_sgld(sgld_config.epsilon, subkey)
    param_init = true_param
    local_logprob = create_local_logposterior(
        avgnegloglikelihood_fn=loss_fn,
        num_training_data=num_training_data,
        w_init=param_init,
        gamma=sgld_config.gamma,
        itemp=itemp,
    )
    sgld_grad_fn = jax.jit(jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0))

    ####################
    # SGLD loop
    ####################
    loss_trace = []
    nlls = []
    opt_state = sgldoptim.init(param_init)
    param = param_init
    t = 0
    while t < sgld_config.num_steps:
        for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=sgld_config.batch_size):
            nll, grads = sgld_grad_fn(param, x_batch, y_batch)
            nlls.append(float(nll))
            updates, opt_state = sgldoptim.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            loss_trace.append(loss_fn(param, x_train, y_train))
            t += 1
    
    ######################
    # Recording experiment
    ######################
    
    # compute lambdahat from loss trace
    init_loss = loss_fn(param_init, x_train, y_train)
    lambdahat = (np.mean(loss_trace) - init_loss) * num_training_data * itemp


    _run.info.update(to_json_friendly_tree({
        "lambdahat": lambdahat,
        "loss_trace": loss_trace,
        "init_loss": init_loss,
    }))

    # Computing true lambda
    true_matrix = jnp.linalg.multi_dot(
        [true_param[f'deep_linear_network/linear{loc}']['w'] for loc in [''] + [f'_{i}' for i in range(1, len(layer_widths))]]
    )

    true_rank = jnp.linalg.matrix_rank(true_matrix)
    true_lambda = true_dln_learning_coefficient(true_rank, layer_widths, input_dim, verbose=verbose)

    _run.info.update(to_json_friendly_tree(
        {
            "true_lambda": true_lambda, 
            "true_rank": true_rank, 
            "true_param_singular_values": jtree.tree_map(get_singular_values, true_param),
            "truth_check": np.allclose(model.apply(true_param, x_train), x_train @ true_matrix, atol=1e-4)
        }
    ))

    return
