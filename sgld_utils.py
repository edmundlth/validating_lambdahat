import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import optax 
from typing import NamedTuple
from dln import create_minibatches
from utils import param_lp_dist, pack_params, unpack_params, param_l2_dist

class SGLDConfig(NamedTuple):
  epsilon: float
  gamma: float
  num_steps: int
  num_chains: int = 1 
  batch_size: int = None

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

def run_sgld(rngkey, loss_fn, sgld_config, param_init, x_train, y_train, itemp=None, trace_batch_loss=True, compute_distance=False, compute_mala_acceptance=True, verbose=False, logging_period=200):
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
    sgld_grad_fn = jax.jit(jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0))
    
    sgldoptim = optim_sgld(sgld_config.epsilon, rngkey)
    if compute_mala_acceptance: # For memory efficiency, no need to store if not computing
        old_param = param.copy()

    loss_trace = []
    distances = []
    accept_probs = []
    opt_state = sgldoptim.init(param_init)
    param = param_init
    t = 0
    while t < sgld_config.num_steps:
        for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=sgld_config.batch_size):

            if compute_distance: 
                distances.append(param_l2_dist(param_init, param))
            
            if trace_batch_loss:
                loss_val = loss_fn(param, x_batch, y_batch)
            else:
                loss_val = loss_fn(param, x_train, y_train)
            loss_trace.append(loss_val)
            
            if compute_mala_acceptance and t % 20 == 0: # Compute acceptance probability every 20 steps
                old_param_packed, pack_info = pack_params(old_param)
                param_packed, _ = pack_params(param)
                def grad_fn_packed(w):
                    nll, grad = sgld_grad_fn(unpack_params(w, pack_info), x_batch, y_batch)
                    grad_packed, _ = pack_params(grad)
                    return nll, grad_packed
                prob = mala_acceptance_probability(
                    old_param_packed, 
                    param_packed, 
                    grad_fn_packed, 
                    sgld_config.epsilon
                )
                accept_probs.append([t, prob])
            
            if t % logging_period == 0 and verbose:
                print(f"Step {t}, loss: {loss_trace[-1]}")
            
            if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                print(f"Step {t}, loss is NaN. Exiting.")
                return loss_trace, distances, accept_probs
            
            if compute_mala_acceptance:
                old_param = param.copy()

            _, grads = sgld_grad_fn(param, x_batch, y_batch)
            updates, opt_state = sgldoptim.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            t += 1
            if t >= sgld_config.num_steps:
                break
    return loss_trace, distances, accept_probs


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
