from tensorflow import custom_gradient, identity, boolean_mask, not_equal, reduce_all
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, concatenate, Layer
from keras.metrics import binary_crossentropy, categorical_crossentropy, categorical_accuracy, binary_accuracy


@custom_gradient
def grad_reverse(x):
  y = identity(x)
  def custom_grad(dy):
    return -dy
  return y, custom_grad

class GradReverse(Layer):
  def __init__(self):
    super().__init__()

  def call(self, x):
    return grad_reverse(x)

def masked_bce(y_true, y_pred):
  y_pred = boolean_mask(y_pred, not_equal(y_true, -1))
  y_true = boolean_mask(y_true, not_equal(y_true, -1))
  return binary_crossentropy(y_true, y_pred)

def masked_cce(y_true, y_pred):
  mask = reduce_all(not_equal(y_true, -1),1)
  y_pred = boolean_mask(y_pred, mask)
  y_true = boolean_mask(y_true, mask)
  return categorical_crossentropy(y_true, y_pred)
    
def masked_categorical_accuracy(y_true, y_pred):
  mask = reduce_all(not_equal(y_true, -1),1)
  y_true = boolean_mask(y_true, mask)
  y_pred = boolean_mask(y_pred, mask)
  return categorical_accuracy(y_true, y_pred)
  
def masked_binary_accuracy(y_true, y_pred):
  mask = not_equal(y_true, -1)
  y_pred = boolean_mask(y_pred, mask)
  y_true = boolean_mask(y_true, mask)
  return binary_accuracy(y_true, y_pred)

def construct_model(input_shape, domain_adaptation=False, da_weight=1):
    model_in = Input(input_shape)
    h = Conv2D(128, 3, activation="relu", padding="same", name="conv1_1")(
        model_in
    )
    h = Conv2D(64, 3, activation="relu", padding="same", name="conv1_2")(h)
    h = MaxPooling2D(pool_size=3, name="pool1", padding="same")(h)
    h = Dropout(0.15, name="drop1")(h)
    h = Flatten(name="flaten1")(h)

    dh = Conv2D(
        128,
        2,
        activation="relu",
        dilation_rate=[1, 3],
        padding="same",
        name="dconv1_1",
    )(model_in)
    dh = Conv2D(
        64,
        2,
        activation="relu",
        dilation_rate=[1, 3],
        padding="same",
        name="dconv1_2",
    )(dh)
    dh = MaxPooling2D(pool_size=2, name="dpool1")(dh)
    dh = Dropout(0.15, name="ddrop1")(dh)
    dh = Flatten(name="dflaten1")(dh)

    dh1 = Conv2D(
        128,
        2,
        activation="relu",
        dilation_rate=[1, 4],
        padding="same",
        name="dconv4_1",
    )(model_in)
    dh1 = Conv2D(
        64,
        2,
        activation="relu",
        dilation_rate=[1, 4],
        padding="same",
        name="dconv4_2",
    )(dh1)
    dh1 = MaxPooling2D(pool_size=2, name="d1pool1")(dh1)
    dh1 = Dropout(0.15, name="d1drop1")(dh1)
    dh1 = Flatten(name="d1flaten1")(dh1)

    h_concated = concatenate([h, dh, dh1])
    h = Dense(512, name="512dense", activation="relu")(h_concated)
    h = Dropout(0.2, name="drop7")(h)
    h = Dense(128, name="last_dense", activation="relu")(h)
    h = Dropout(0.1, name="drop8")(h)
    output = Dense(5, name="predictor", activation="softmax")(h)
    if domain_adaptation:
        da = GradReverse()(h_concated)
        da = Dense(512, name="DA512dense", activation="relu")(da)
        #da = Dropout(0.2, name="DADrop1")(da)
        da = Dense(128, name="DA128dense", activation="relu")(da)
        #da = Dropout(0.1, name="DADrop2")(da)
        domain_output = Dense(1, name="discriminator", activation="sigmoid")(da)
        model = Model(inputs=[model_in], outputs=[output, domain_output])
        model.compile(optimizer='adam',
                        loss={'predictor': masked_cce, 'discriminator': masked_bce},
                        loss_weights = [1, da_weight], # equal weighing of two tasks
                        metrics={'predictor': masked_categorical_accuracy, 'discriminator': masked_binary_accuracy})
    else:
        model = Model(inputs=[model_in], outputs=[output])
        model.compile(loss=masked_cce, optimizer="adam", metrics=[masked_categorical_accuracy])
    return model
