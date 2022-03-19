import itertools
from typing import Any, Optional
from cv2 import reduce

import tensorflow as tf
from tensorflow.keras import backend as K

_EPSILON = tf.keras.backend.epsilon()


def sparse_categorical_focal_loss(y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1
                                  ) -> tf.Tensor:
    # Process focusing parameter
    gamma = tf.convert_to_tensor(gamma, dtype=y_pred.dtype)
    gamma_rank = gamma.shape.rank
    scalar_gamma = gamma_rank == 0

    # Process class weight
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight,
                                            dtype=tf.dtypes.float32)

    # Process prediction tensor
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred_rank = y_pred.shape.rank
    if y_pred_rank is not None:
        axis %= y_pred_rank
        if axis != y_pred_rank - 1:
            # Put channel axis last for sparse_softmax_cross_entropy_with_logits
            perm = list(itertools.chain(range(axis),
                                        range(axis + 1, y_pred_rank), [axis]))
            y_pred = tf.transpose(y_pred, perm=perm)
    elif axis != -1:
        raise ValueError(
            f'Cannot compute sparse categorical focal loss with axis={axis} on '
            'a prediction tensor with statically unknown rank.')
    y_pred_shape = tf.shape(y_pred)

    # Process ground truth tensor
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank

    if y_true_rank is None:
        raise NotImplementedError('Sparse categorical focal loss not supported '
                                  'for target/label tensors of unknown rank')

    reshape_needed = (y_true_rank is not None and y_pred_rank is not None and
                      y_pred_rank != y_true_rank + 1)
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if from_logits:
        logits = y_pred
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=logits,
    )

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)
    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
    
    focal_modulation = tf.pow(1 - probs, gamma)
    loss = focal_modulation * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0,
                                 batch_dims=y_true_rank)
        loss *= class_weight

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma, class_weight: Optional[Any] = None,
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits

    def get_config(self):
        """Returns the config of the layer.
        A layer config is a Python dictionary containing the configuration of a
        layer. The same layer can be re-instantiated later (without its trained
        weights) from this configuration.
        Returns
        -------
        dict
            This layer's config.
        """
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight,
                      from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):
        """Compute the per-example focal loss.
        This method simply calls
        :meth:`~focal_loss.sparse_categorical_focal_loss` with the appropriate
        arguments.
        Parameters
        ----------
        y_true : tensor-like, shape (N,)
            Integer class labels.
        y_pred : tensor-like, shape (N, K)
            Either probabilities or logits, depending on the `from_logits`
            parameter.
        Returns
        -------
        :class:`tf.Tensor`
            The per-example focal loss. Reduction to a scalar is handled by
            this layer's
            :meth:`~focal_loss.SparseCateogiricalFocalLoss.__call__` method.
        """
        return sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                             class_weight=self.class_weight,
                                             gamma=self.gamma,
                                             from_logits=self.from_logits)



def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        y_true = tf.one_hot(y_true, depth=y_pred.shape[1], dtype=y_pred.dtype)
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        loss = tf.reduce_mean(loss)
        return loss
    
    return focal_loss


class CeLoss(tf.keras.losses.Loss):
    def __init__(self, ohem=False, n_classes=15587, class_weight: Optional[Any] = None,
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ohem = ohem
        self.class_weight = class_weight
        self.n_classes = n_classes
        self.from_logits = from_logits
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def get_config(self):
        config = super().get_config()
        config.update(ohem=self.ohem, class_weight=self.class_weight,
                      from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):
        loss = self.ce(y_true, y_pred)
        if self.ohem:
            ohem_percent = 0.7
            _, indices = tf.math.top_k(loss, k=int(self.n_classes * ohem_percent))
            loss = tf.gather(loss, indices, batch_dims=-1)
        return tf.reduce_mean(loss)