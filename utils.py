import jax
import jax.tree_util as jtree
import jax.numpy as jnp

def to_float_or_list(x):
    if isinstance(x, (float, int)):
        return float(x)
    elif isinstance(x, (list, tuple)):
        return [float(el) for el in x]
    elif hasattr(x, "tolist"):  # For JAX or numpy arrays
        return x.tolist()
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
    U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    # Reduce the number of non-zero singular values by 'reduction'
    new_rank = max(len(S) - reduction, 0)
    S[new_rank:] = 0

    # Reconstruct the matrix with the reduced number of singular values
    reduced_matrix = jnp.dot(U * S, Vh)
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