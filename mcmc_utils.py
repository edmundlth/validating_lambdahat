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

