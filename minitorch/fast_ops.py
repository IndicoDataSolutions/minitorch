import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange


# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.1.
        for i in range(len(out)):
            out_index = [-1 for _ in out_shape]
            in_index = [-1 for _ in in_shape]
            # Fills in out_index
            to_index(i, out_shape, out_index)
            # Fills in in_index
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out[index_to_position(out_index, out_strides)] = fn(
                in_storage[index_to_position(in_index, in_strides)]
            )

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        for i in range(len(out)):
            out_index = [-1 for _ in out_shape]
            a_index = [-1 for _ in a_shape]
            b_index = [-1 for _ in b_shape]
            # Fills in out_index
            to_index(i, out_shape, out_index)

            # Fills in a_index and b_index
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out[index_to_position(out_index, out_strides)] = fn(
                a_storage[index_to_position(a_index, a_strides)],
                b_storage[index_to_position(b_index, b_strides)],
            )

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function.

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`

    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        for i in range(a_storage.shape[0]):
            a_index = [-1 for _ in out_shape]
            to_index(i, a_shape, a_index)
            out_index = [0 if i == reduce_dim else idx for i, idx in enumerate(a_index)]
            out[index_to_position(out_index, out_strides)] = fn(
                a_storage[index_to_position(a_index, a_strides)],
                out[index_to_position(out_index, out_strides)],
            )


    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


@njit(parallel=True, fastmath=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Optimizations:

        * Outer loop in parallel
        * No index buffers or function calls
        * Inner loop should have no global writes, 1 multiply.


    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # index_to_position(out_index, out_strides)
    
    layout = [1] 
    offset = 1
    for s in out_shape[::-1]:
        layout.append(s * offset)
        offset = s * offset
    stride_product = layout[:-1][::-1]

    for i in range(out.shape[0]):
        out[i] = 0 # Possibly unnecessary
        
        out_index = [-1 for _ in out_shape]

        ordinal = i
        for k in range(len(stride_product)):
            s = stride_product[k]
            out_index[k] = ordinal // s
            ordinal = ordinal % s

        a_offset = 0
        b_offset = 0        
        
        out_index_slice = out_index[-3::-1]
        a_shape_slice = a_shape[-3::-1]
        a_stride_slice = a_strides[-3::-1]
        b_shape_slice = b_shape[-3::-1]
        b_stride_slice = b_strides[-3::-1]

        for j in range(len(out_index_slice)):

            big_index_j = out_index_slice[j]
            if j < len(a_shape_slice):
                a_shape_j = a_shape_slice[j]
                a_stride_j = a_stride_slice[j]
                if a_shape_j != 1:
                    a_offset += a_stride_j * big_index_j
        
            if j < len(b_shape_slice):
                b_shape_j = b_shape_slice[j]
                b_stride_j = b_stride_slice[j]
                if b_shape_j != 1:
                    b_offset += b_stride_j * big_index_j
    

        for a_idx in range(a_shape[-1]):
            a_offset_new = a_offset + a_strides[-2] * out_index[-2] + a_strides[-1] * a_idx
            b_offset_new = b_offset + b_strides[-2] * a_idx + b_strides[-1] * out_index[-1]
            out[i] += (
                a_storage[a_offset_new]
                * b_storage[b_offset_new]
            )


#
#   [a, b]
#   [c, d]
#   [e, f]
    
#   [g, h]
#   [i, j]

#   [k, l]
#   [m, n]
#   [o, p]

# = [a * g + b * i, a * h + b * j]
#   [c * g + d * i, c * h + d * j]
#   [e * g + g * i, e * h + f * j]


shape =        [1, 5]
big_shape = [6, 5, 5]

index = [5, 4, 1]

out_index = [0, 1]




def matrix_multiply(a, b):
    """
    Batched tensor matrix multiply ::

        for n:
          for i:
            for j:
              for k:
                out[n, i, j] += a[n, i, k] * b[n, k, j]

    Where n indicates an optional broadcasted batched dimension.

    Should work for tensor shapes of 3 dims ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b

    Returns:
        :class:`Tensor` : new tensor data
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
