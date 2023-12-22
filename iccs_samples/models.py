from keras import Sequential
from keras.src.layers import BatchNormalization, MaxPool2D, Dropout, AveragePooling2D, LSTM, Conv2D, Flatten, Dense
from keras.src.layers import Activation
import keras
import tensorflow as tf


class Involution(keras.layers.Layer):
    """ Implementation of the involution layer (source: https://keras.io/examples/vision/involution/) """

    def __init__(
            self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)

        # Initialize the parameters.
        self.output_reshape = None
        self.input_patches_reshape = None
        self.kernel_gen = None
        self.stride_layer = None
        self.kernel_reshape = None
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output, kernel


def convolution_block(model, rows, columns, n_filters):
    model.add(Conv2D(filters=n_filters, kernel_size=(1, 3), input_shape=(rows, columns, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))


def convolution_only_model(rows, columns, output_dim, n_filters):
    cnn = Sequential()
    convolution_block(cnn, rows, columns, n_filters)
    convolution_block(cnn, rows, columns, n_filters)
    convolution_block(cnn, rows, columns, n_filters)
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(output_dim, activation='sigmoid'))  # Output layer (adjust for your specific task)

    print("compiling the combined model...")
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return cnn


def involution_only_model(rows, columns, output_dim):
    inputs = keras.Input(shape=(rows, columns, 1))

    x, _ = Involution(channel=1, group_number=1, kernel_size=3, stride=1, reduction_ratio=1, name="inv_1")(inputs)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x, _ = Involution(channel=1, group_number=1, kernel_size=3, stride=1, reduction_ratio=1, name="inv_2")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(output_dim)(x)

    inv_model = keras.Model(inputs=[inputs], outputs=[outputs], name="inv_model")

    # Compile the mode with the necessary loss function and optimizer.
    print("compiling the involution model...")
    inv_model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return inv_model


def combined_model(rows, columns, output_dim, n_filters):
    inputs = keras.Input(shape=(rows, columns, 1))

    x = Conv2D(filters=n_filters, kernel_size=(1, 3))(inputs)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x, _ = Involution(channel=1, group_number=1, kernel_size=3, stride=1, reduction_ratio=1, name="inv_1")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x, _ = Involution(
        channel=1, group_number=1, kernel_size=3, stride=1, reduction_ratio=1, name="inv_2"
    )(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(output_dim)(x)

    combined_model = keras.Model(inputs=[inputs], outputs=[outputs], name="combined_model")

    # Compile the mode with the necessary loss function and optimizer.
    print("compiling the combined model...")
    combined_model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return combined_model
