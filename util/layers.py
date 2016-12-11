import prettytensor as pt
from prettytensor import layers
import tensorflow as tf
import collections

@pt.Register(assign_defaults=('activation_fn', 'l2loss',
                                        'parameter_modifier', 'phase'))
class transposed_fully_connected(pt.VarStoreMethod):

  def __call__(self,
               input_layer,
               in_size,
               activation_fn=None,
               l2loss=None,
               weights=None,
               bias=tf.zeros_initializer,
               transpose_weights=False,
               phase=pt.Phase.train,
               parameter_modifier=None,
               name="transposed_fully_connected"):
    in_size = in_size
    size = input_layer.shape[0]
    books = input_layer.bookkeeper
    if weights is None:
      weights = layers.he_init(in_size, size, activation_fn)

    dtype = input_layer.tensor.dtype
    weight_shape = [size, in_size] if transpose_weights else [in_size, size]

    params = self.variable('transposed_fully_connected_weights', weight_shape, weights, dt=dtype)
    y = tf.matmul(params, input_layer, transpose_b=transpose_weights)
    layers.add_l2loss(books, params, l2loss)
    if bias is not None:
      y += self.variable('transposed_fully_connected_bias', [input_layer.shape[1]], bias, dt=dtype)

    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(books,
                                  y,
                                  activation_fn[0],
                                  activation_args=activation_fn[1:])
    books.add_histogram_summary(y, '%s/activations' % y.op.name)
    return input_layer.with_tensor(y, parameters=self.vars)