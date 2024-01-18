import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import optax 
from typing import NamedTuple
from dln import create_minibatches
from utils import param_lp_dist

class SGLDConfig(NamedTuple):
  epsilon: float
  gamma: float
  num_steps: int
  num_chains: int = 1 
  batch_size: int = None


def run_sgld(rngkey, loss_fn, sgld_config, param_init, x_train, y_train, itemp=None, trace_batch_loss=True, compute_distance=False):
    num_training_data = len(x_train)
    if itemp is None:
        itemp = 1 / jnp.log(num_training_data)
    local_logprob = create_local_logposterior(
        avgnegloglikelihood_fn=loss_fn,
        num_training_data=num_training_data,
        w_init=param_init,
        gamma=sgld_config.gamma,
        itemp=itemp,
    )
    sgld_grad_fn = jax.jit(jax.grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0))
    
    sgldoptim = optim_sgld(sgld_config.epsilon, rngkey)
    loss_trace = []
    distances = []
    opt_state = sgldoptim.init(param_init)
    param = param_init
    t = 0
    while t < sgld_config.num_steps:
        for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=sgld_config.batch_size):
            grads = sgld_grad_fn(param, x_batch, y_batch)
            updates, opt_state = sgldoptim.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            if compute_distance: 
                distances.append(param_lp_dist(param_init, param, ord=2))
            if trace_batch_loss:
                loss_trace.append(loss_fn(param, x_batch, y_batch))
            else:
                loss_trace.append(loss_fn(param, x_train, y_train))
            t += 1
    return loss_trace, distances


def generate_rngkey_tree(key_or_seed, tree_or_treedef):
    rngseq = hk.PRNGSequence(key_or_seed)
    return jtree.tree_map(lambda _: next(rngseq), tree_or_treedef)

def optim_sgld(epsilon, rngkey_or_seed):
    @jax.jit
    def sgld_delta(g, rngkey):
        eta = jax.random.normal(rngkey, shape=g.shape) * jnp.sqrt(epsilon)
        return -epsilon * g / 2 + eta

    def init_fn(_):
        return rngkey_or_seed

    @jax.jit
    def update_fn(grads, state):
        rngkey, new_rngkey = jax.random.split(state)
        rngkey_tree = generate_rngkey_tree(rngkey, grads)
        updates = jax.tree_map(sgld_delta, grads, rngkey_tree)
        return updates, new_rngkey
    return optax.GradientTransformation(init_fn, update_fn)


def create_local_logposterior(avgnegloglikelihood_fn, num_training_data, w_init, gamma, itemp):
    def helper(x, y):
        return jnp.sum((x - y)**2)

    def _logprior_fn(w):
        sqnorm = jax.tree_util.tree_map(helper, w, w_init)
        return jax.tree_util.tree_reduce(lambda a,b: a + b, sqnorm)

    def logprob(w, x, y):
        loglike = -num_training_data * avgnegloglikelihood_fn(w, x, y)
        logprior = -gamma / 2 * _logprior_fn(w)
        return itemp * loglike + logprior
    return logprob
