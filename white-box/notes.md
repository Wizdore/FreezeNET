a neat way to watermark the network would be to flatten all weights, fix a ratio of the array according to the chosen distribution, and reassign the weights to the model. this could be done in a callback after each batch. however using get weights, update, and set weights does not work for all weights. what's the difference between assignable weights and the rest ?

8 hours of debugging later : if idx has duplicate elements, arr[idx] = vals will obviously have a strange behaviour

another idea would be to initalize the weights to the signaure+random and zero out the gradients that would change the fixed weights during backpropagation. problem with this approach is that it requires to know which weights in which layers should be fixed, rather than just picking a distribution.

there seems to be a turning point where changing a single weights completely crashes the model accuracy. maybe restrict the signature to inner layers ? -> this is due to some weights being non trainable, changing them makes the model unusable : len(model.get_weights()) != len(model.trainable_weights)

maybe add kernel regularizer to model ?

# models

* 10epochs_nosig : baseline simplenet