import jax
import jax.tree_util as jtree
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import linregress



def to_float_or_list(x):
    if isinstance(x, (float, int)):
        return float(x)
    elif isinstance(x, (list, tuple)):
        return [float(el) for el in x]
    elif hasattr(x, "tolist"):  # For JAX or numpy arrays
        return x.tolist()
    elif isinstance(x, str):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")

def to_json_friendly_tree(tree):
    return jtree.tree_map(to_float_or_list, tree)


def reduce_matrix_rank(matrix, reduction):
    """
    Reduce the rank of the matrix by 'reduction' amount.

    :param matrix: Input matrix.
    :param reduction: The amount by which the rank should be reduced.
    :return: A matrix similar to the input but with reduced rank.
    """
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    # Reduce the number of non-zero singular values by 'reduction'
    new_rank = max(len(S) - reduction, 0)
    S[new_rank:] = 0
    # Reconstruct the matrix with the reduced number of singular values
    reduced_matrix = np.dot(U * S, Vh)
    # print(np.linalg.matrix_rank(reduced_matrix), np.linalg.matrix_rank(matrix), reduction)
    return reduced_matrix

def rand_reduce_matrix_rank(rngkey, matrix):
    r = jnp.linalg.matrix_rank(matrix)
    reduction = int(jax.random.randint(rngkey, shape=(), minval=0, maxval=max(1, r)))
    return reduce_matrix_rank(matrix, reduction)


def create_random_matrix_with_rank(rng_key, shape, rank=None, mean=0.0, std=5.0):
    """
    Create a random matrix of a specified rank and shape.

    :param rng_key: JAX random key.
    :param shape: Shape of the desired matrix (rows, cols).
    :param rank: Desired rank of the matrix.
    :param mean: Mean of the Gaussian distribution for singular values.
    :param std: Standard deviation of the Gaussian distribution for singular values.
    :return: A random matrix with the specified rank.
    """
    rows, cols = shape
    if rank is None:
        rank = min(rows, cols)
    if rank > min(rows, cols):
        raise ValueError("Rank cannot be greater than the smallest dimension of the matrix")

    # Generate singular values from a Gaussian distribution
    rng_key, sub_key = jax.random.split(rng_key)
    singular_values = jax.random.normal(sub_key, (rank,)) * std - mean

    # Create random orthogonal matrices U and V
    rng_key, sub_key = jax.random.split(rng_key)
    U, _ = jnp.linalg.qr(jax.random.normal(sub_key, (rows, rank)))
    rng_key, sub_key = jax.random.split(rng_key)
    V, _ = jnp.linalg.qr(jax.random.normal(sub_key, (cols, rank)))

    # Construct the matrix with the desired rank
    S = jnp.diag(singular_values)
    matrix = U @ S @ V.T
    return matrix



def get_singular_values(matrix):
    """
    Return the singular values of a matrix.

    :param matrix: Input matrix.
    :return: A list of singular values.
    """
    # Perform Singular Value Decomposition
    U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)
    
    # S contains the singular values
    return S

def param_lp_dist(param1, param2, ord=2):
    return sum([
        jnp.linalg.norm(x - y, ord=ord) 
        for x, y in zip(jtree.tree_leaves(param1), jtree.tree_leaves(param2))
    ])


def pack_params(params):
    params_flat, treedef = jax.tree_util.tree_flatten(params)
    shapes = [p.shape for p in params_flat]
    indices = np.cumsum([p.size for p in params_flat])
    params_packed = jnp.concatenate([jnp.ravel(p) for p in params_flat])
    pack_info = (treedef, shapes, indices)
    return params_packed, pack_info

def unpack_params(params_packed, pack_info):
    treedef, shapes, indices = pack_info
    params_split = jnp.split(params_packed, indices)
    params_flat = [jnp.reshape(p, shape) for p, shape in zip(params_split, shapes)]
    params = jax.tree_util.tree_unflatten(treedef, params_flat)
    return params


def make_hessian_vector_prod_fn(func, x_test, jit=True):
    def hvp(v):
        return jax.grad(lambda x: jnp.dot(jax.grad(func)(x), v))(x_test)
    if jit:
        return jax.jit(hvp)
    else:
        return hvp    

def hessian_trace_estimate(rng_key, hvp, dimension, num_samples=100):
    # Use Rademacher distribution and Hutchinson's method
    z_samples = (jax.random.randint(rng_key, (num_samples, dimension), 0, 2) * 2) - 1
    trace_estimates = jax.vmap(lambda z: jnp.dot(z, hvp(z)))(z_samples)
    # Average over all samples
    return jnp.mean(trace_estimates)


def stable_weighted_average(Ls, n, delta_beta, factor=None):
    """
    Compute: 1/m \sum_{j = 1}^m n Ls[j] exp(- delta_beta * n * Ls[j])
    """
    m = len(Ls)
    Ls = np.array(Ls)
    log_terms = - delta_beta * n * Ls
    if factor is not None:
        log_terms += np.log(factor)
    average = np.exp(logsumexp(log_terms) - np.log(m))
    return average

def extrapolated_wbic(losses, n, itemp1, itemp2):
    delta_beta = itemp2 - itemp1
    numerator = stable_weighted_average(losses, n, delta_beta, factor=n * losses)
    normalisation = stable_weighted_average(losses, n, delta_beta)
    return numerator / normalisation
    
def extrapolated_multiitemp_lambdahat(losses, n, itemp_og, num_extrapolation=5, return_full=False):
    itemps = linspaced_itemps_by_n(n, num_extrapolation)
    wbics = [
        extrapolated_wbic(losses, n, itemp_og, itemp_new) for itemp_new in itemps
    ]
    result = linregress(1 / itemps, wbics)
    if return_full: 
        return result
    else:
        return result.slope


def linspaced_itemps_by_n(n, num_itemps):
    """
    Returns a 1D numpy array of length `num_itemps` that contains `num_itemps`
    evenly spaced inverse temperatures between the values calculated from the
    formula 1/log(n) * (1 - 1/sqrt(2log(n))) and 1/log(n) * (1 + 1/sqrt(2ln(n))).
    The formula is used in the context of simulating the behavior of a physical
    system at different temperatures using the Metropolis-Hastings algorithm.
    """
    return np.linspace(
        1 / np.log(n) * (1 - 1 / np.sqrt(2 * np.log(n))),
        1 / np.log(n) * (1 + 1 / np.sqrt(2 * np.log(n))),
        num_itemps,
    )