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
    param_lp_dist,
    pack_params, 
    unpack_params
)
from sgld_utils import (
    SGLDConfig, optim_sgld, create_local_logposterior, generate_rngkey_tree, 
    run_sgld
)


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
    input_dist = "uniform" # uniform / unit_ball
    true_param_config = {
        "method": "random", # random / zero / rand_rank / rand_rank_sv
        "prop_rank_reduce": 0.2, # rand_rank, proportion of matrices selected to have random rank. 
        "mean": 0.0, # rand_rank_sv, mean and std of gaussian to generate SV (singular values) 
        "std": 5.0, 
    }
    
    param_init = None
    num_training_data = 10000
    itemp = 1 / np.log(num_training_data)

    training_config = {
        "optim": "sgd", 
        "learning_rate": 0.01, 
        "momentum": 0.9, 
        "batch_size": 200, 
        "num_steps": 5000
    }
    do_training = False
    do_functional_rank = False
    seed = 0
    verbose=False


def initialise_expt(rngkey, layer_widths, input_dim, input_dist, num_training_data, true_param_config):
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
    x_train, y_train = generate_training_data(true_param, model, input_dim, num_training_data, input_dist=input_dist)
    
    return model, true_param, x_train, y_train


@ex.automain
def run_experiment(
    _run, 
    sgld_config, 
    layer_widths,
    input_dim,
    input_dist,
    true_param_config,
    num_training_data, 
    itemp, 
    training_config,
    do_training, 
    do_functional_rank,
    seed,
    verbose,
):
    # seeding
    np.random.seed(seed)
    rngkey = jax.random.PRNGKey(seed)

    ####################
    # Initialisations
    ####################
    # for model
    sgld_config = SGLDConfig(**sgld_config)
    rngkey, subkey = jax.random.split(rngkey)
    model, true_param, x_train, y_train = initialise_expt(
        subkey, 
        layer_widths, 
        input_dim, 
        input_dist,
        num_training_data, 
        true_param_config
    )
    loss_fn = jax.jit(lambda param, inputs, targets: mse_loss(param, model, inputs, targets))
    
    ########################################
    # Train the model if specified
    ########################################
    if do_training:
        optimizer = optax.sgd(
            learning_rate=training_config["learning_rate"], 
            momentum=training_config["momentum"]
        )
        max_steps = training_config["num_steps"]
        t = 0
        rngkey, subkey = jax.random.split(rngkey)
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
        trained_param = model.init(rngkey, jnp.zeros((1, input_dim)))
        opt_state = optimizer.init(trained_param)
        while t < max_steps:
            for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=training_config["batch_size"]):
                grads = grad_fn(trained_param, x_batch, y_batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                trained_param = optax.apply_updates(trained_param, updates)
                t += 1
                if t >= max_steps:
                    break

    ####################
    # SGLD lambdahat
    ####################
    param_init = true_param
    rngkey, subkey = jax.random.split(rngkey)
    loss_trace, distances = run_sgld(subkey, loss_fn, sgld_config, param_init, x_train, y_train, itemp=itemp)

    # compute lambdahat from loss trace
    init_loss = loss_fn(param_init, x_train, y_train)
    lambdahat = (np.mean(loss_trace) - init_loss) * num_training_data * itemp
    
    # record
    _run.info.update(to_json_friendly_tree({
        "lambdahat": lambdahat,
        "loss_trace": loss_trace,
        "init_loss": init_loss,
        "sgld_distances": distances,
    }))

    
    # do SGLD sampling centred on trained parameter if there is one. 
    if do_training:
        param_init = trained_param
        rngkey, subkey = jax.random.split(rngkey)
        loss_trace, distances = run_sgld(subkey, loss_fn, sgld_config, param_init, x_train, y_train, itemp=itemp)

        # compute lambdahat from loss trace
        init_loss = loss_fn(param_init, x_train, y_train)
        lambdahat = (np.mean(loss_trace) - init_loss) * num_training_data * itemp
        
        # record
        _run.info["trained_param_info"] = to_json_friendly_tree({
            "lambdahat": lambdahat,
            "loss_trace": loss_trace,
            "init_loss": init_loss,
            "sgld_distances": distances,
        })


    # Computing true lambda
    true_matrix = jnp.linalg.multi_dot(
        [true_param[f'deep_linear_network/linear{loc}']['w'] for loc in [''] + [f'_{i}' for i in range(1, len(layer_widths))]]
    )
    true_rank = jnp.linalg.matrix_rank(true_matrix)
    true_lambda, true_multiplicity = true_dln_learning_coefficient(true_rank, layer_widths, input_dim, verbose=verbose)
    model_dim = sum([np.prod(x.shape) for x in jtree.tree_leaves(true_param)])

    _run.info.update(to_json_friendly_tree(
        {
            "true_lambda": true_lambda, 
            "true_multiplicity": true_multiplicity, 
            "true_rank": true_rank, 
            "true_matrix_shape": list(true_matrix.shape), 
            "true_param_singular_values": jtree.tree_map(get_singular_values, true_param),
            "truth_check": np.allclose(model.apply(true_param, x_train), x_train @ true_matrix, atol=1e-4),
            "model_dim": model_dim,
        }
    ))

    ####################
    # Jacobian rank
    ####################
    if do_functional_rank:
        n_small = 100
        threshold = 0.001
        x_small, y_small = generate_training_data(true_param, model, input_dim, n_small)
        true_param_packed, pack_info = pack_params(true_param)
        fwd_fn = jax.jit(lambda p_packed: jnp.ravel(model.apply(unpack_params(p_packed, pack_info), x_small)))
        jacobian_matrix = np.asarray(jax.jacfwd(fwd_fn)(true_param_packed))
        svdvals = np.linalg.svd(jacobian_matrix, compute_uv=False)

        _run.info["functional_rank_info"] = to_json_friendly_tree({
            "matrix_rank": jnp.linalg.matrix_rank(jacobian_matrix), 
            "singular_values": svdvals,
            "functional_rank": int(np.sum(svdvals > threshold)), 
            "threshold": threshold, 
            "num_data_used": n_small
        })

    return
