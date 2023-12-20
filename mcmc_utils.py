import jax
import jax.tree_util as jtree
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
from collections import namedtuple

MCMCConfig = namedtuple(
    "MCMCConfig", ["num_posterior_samples", "num_warmup", "num_chains", "thinning"]
)

def run_mcmc(model, X, Y, rngkey, mcmc_config, step_size=None, init_params=None, itemp=1.0, progress_bar=True):
    if step_size is None:
        kernel = NUTS(model)
    else:
        kernel = NUTS(model, step_size=step_size)
    mcmc = MCMC(
        kernel, 
        num_warmup=mcmc_config.num_warmup, 
        num_samples=mcmc_config.num_posterior_samples, 
        thinning=mcmc_config.thinning, 
        num_chains=mcmc_config.num_chains, 
        progress_bar=progress_bar
    )
    print("Running MCMC")
    mcmc.run(rngkey, X, Y, itemp=itemp, init_params=init_params)
    return mcmc



def build_model(forward_fn, prior_center_tree, prior_std=1.0, sigma_obs=1.0):
    leaves, treedef = jtree.tree_flatten(prior_center_tree)
    def model(X, Y=None, itemp=1.0):
        # parameter from gaussian prior
        prior_samples = [
            numpyro.sample(i, dist.Normal(x, scale=prior_std)) for i, x in enumerate(leaves)
        ]
        param = jtree.tree_unflatten(treedef, prior_samples)
        # mean
        mu = forward_fn(param, X)
        numpyro.sample('obs', dist.Normal(mu, sigma_obs / jnp.sqrt(itemp)), obs=Y)
        return 
    return model


def mala_acceptance_probability(current_point, proposed_point, loss_and_grad_fn, step_size):
    """
    Calculate the acceptance probability for a MALA transition.

    Args:
    current_point: The current point in parameter space.
    proposed_point: The proposed point in parameter space.
    loss_and_grad_fn (function): Function to compute loss and loss gradient at a point.
    step_size (float): Step size parameter for MALA.

    Returns:
    float: Acceptance probability for the proposed transition.
    """
    # Compute the gradient of the loss at the current point
    current_loss, current_grad = loss_and_grad_fn(current_point)
    proposed_loss, proposed_grad = loss_and_grad_fn(proposed_point)

    # Compute the log of the proposal probabilities (using the Gaussian proposal distribution)
    log_q_proposed_to_current = -jnp.sum((current_point - proposed_point - (step_size * 0.5 * -proposed_grad)) ** 2) / (2 * step_size)
    log_q_current_to_proposed = -jnp.sum((proposed_point - current_point - (step_size * 0.5 * -current_grad)) ** 2) / (2 * step_size)

    # Compute the acceptance probability
    acceptance_log_prob = log_q_proposed_to_current - log_q_current_to_proposed + current_loss - proposed_loss
    return jnp.minimum(1.0, jnp.exp(acceptance_log_prob))
