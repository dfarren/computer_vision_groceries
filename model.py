import tensorflow as tf
import os
import pdb
import random
import matplotlib.pyplot as plt

VGG_MEAN = [123.68, 116.78, 103.94]


def list_images(directory, testing_code=False):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    files_and_labels = []

    for label in labels:
        if label.startswith('.'):   # this is a hidden file, not a directory
            continue
        for i, f in enumerate(os.listdir(os.path.join(directory, label))):
            files_and_labels.append((os.path.join(directory, label, f), label))
            if testing_code and i>10:
                break

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels


def get_data(validation_percentage=0.3, testing_code=False):
    # Get the list of filenames and corresponding list of labels for training et validation
    filenames, labels = list_images(data_dir, testing_code)

    # if looking for bugs, the data doesn't matter as long as it is small
    if testing_code:
        train_filenames, train_labels = filenames, labels
        val_filenames, val_labels = filenames, labels
    else:
        nbr_examples = len(filenames)
        nbr_examples_val = int(nbr_examples * validation_percentage)

        train_filenames, train_labels = [], []
        val_filenames, val_labels = [], []

        index_examples_val = random.sample(range(nbr_examples), nbr_examples_val)
        for i, f in enumerate(filenames):
            l = labels[i]
            if i in index_examples_val:
                val_filenames.append(f)
                val_labels.append(l)
            else:
                train_filenames.append(f)
                train_labels.append(l)

    num_classes = len(set(train_labels))

    return train_filenames, train_labels, val_filenames, val_labels, num_classes


def check_accuracy(sess, correct_prediction, keep_prob, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {keep_prob: 1})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def calculate_mean_image(sess, train_init_op, sum_image, nbr_samples, keep_prob):
    sess.run(train_init_op)
    sum_total, num_samples = 0, 0
    while True:
        try:
            sum_img, nbr_smpls = sess.run([sum_image, nbr_samples], {keep_prob: 1})
            sum_total += sum_img
            num_samples += nbr_smpls
        except tf.errors.OutOfRangeError:
            break

    mean_image = float(sum_total) / num_samples
    return mean_image


def plot_images(sess, train_init_op, images, labels, keep_prob, labels_pred = None, img_size=224, num_channels=3):
    sess.run(train_init_op)
    # run just one epoch
    img, lbl = sess.run([images, labels], {keep_prob:1})

    #index = random.sample(range(len(img)), k = 9)
    #img = img[index]
    #lbl = lbl[index]

    if len(img) == 0:
        print("no images to show")
        return
    else:
        random_indices = random.sample(range(len(img)), min(len(img), 9))

    img, lbl = zip(*[(img[i], lbl[i]) for i in random_indices])

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    #pdb.set_trace()
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(img[i].reshape(img_size, img_size, num_channels))

        xlabel = "True: {0}".format(lbl[i])

        # Show true and predicted classes.
        #if labels_pred is None:
        #    xlabel = "True: {0}".format(lbl[i])
        #else:
        #    xlabel = "True: {0}, Pred: {1}".format(lbl[i], labels_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def create_model(data_dir, num_workers, batch_size, learning_rate, validation_percentage=0.3, testing_code=False):

    train_filenames, train_labels, val_filenames, val_labels, num_classes = get_data(validation_percentage, testing_code)

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # Preprocessing (for both training and validation):
        # (1) Decode the image from jpg format
        # (2) Resize the image so its smaller side is 256 pixels long
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_png(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            return resized_image, label

        # Preprocessing (for training)
        # (3) Take a random 224x224 crop to the scaled image
        # (4) Horizontally flip the image with probability 1/2
        # (5) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
        def training_preprocess(image, label):
            crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
            #flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

            #means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            #centered_image = flip_image - means                                     # (5)

            #return centered_image, label
            return crop_image, label

        # Preprocessing (for validation)
        # (3) Take a central 224x224 crop to the scaled image
        # (4) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
        def val_preprocess(image, label):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means                                     # (4)

            return centered_image, label


        '''
        A convolutional layer produces an output tensor with 4 dimensions. We will add
        fully-connected layers after the convolution layers, so we need to reduce the 4-dim
        tensor to 2-dim which can be used as input to the fully-connected layer.
        '''
        def flatten_layer(layer):
            # Get the shape of the input layer.
            layer_shape = layer.get_shape()

            # The shape of the input layer is assumed to be:
            # layer_shape == [num_images, img_height, img_width, num_channels]

            # The number of features is: img_height * img_width * num_channels
            # We can use a function from TensorFlow to calculate this.
            num_features = layer_shape[1:4].num_elements()

            # Reshape the layer to [num_images, num_features].
            # Note that we just set the size of the second dimension
            # to num_features and the size of the first dimension to -1
            # which means the size in that dimension is calculated
            # so the total size of the tensor is unchanged from the reshaping.
            layer_flat = tf.reshape(layer, [-1, num_features])

            # The shape of the flattened layer is now:
            # [num_images, img_height * img_width * num_channels]

            # Return both the flattened layer and the number of features.
            return layer_flat, num_features

        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data

        # Training dataset
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_parse_function, num_threads=num_workers, output_buffer_size=batch_size)
        train_dataset = train_dataset.map(training_preprocess, num_threads=num_workers, output_buffer_size=batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(batch_size)

        # Validation dataset
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function, num_threads=num_workers, output_buffer_size=batch_size)
        val_dataset = val_dataset.map(val_preprocess, num_threads=num_workers, output_buffer_size=batch_size)
        batched_val_dataset = val_dataset.batch(batch_size)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in test mode
        #is_training = tf.placeholder(tf.bool)

        mean_image = tf.placeholder(tf.float32, shape=[1, 1, 1])

        sum_image = tf.reduce_sum(images, axis=0)
        nbr_samples = images[0]


        ##############################




        conv_size1 = 5
        conv_dep1 = 19
        aff_size1 = 25

        # AlexNet
        regularizers = 0
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        ###########################################
        #             CONV1 layer                 #
        ###########################################
        _, H, W, nbr_channels = images.shape
        Wconv1 = tf.get_variable("Wconv1", shape=[conv_size1, conv_size1, nbr_channels, conv_dep1])
        bconv1 = tf.get_variable("bconv1", shape=[conv_dep1])
        # Wconv1_summary = tf.summary.image("Wconv1_summary", Wconv1)
        # Spatial Batch Normalization Layer (,F)
        betabatch1 = tf.get_variable("betabatch1", shape=[conv_dep1])
        gammabatch1 = tf.get_variable("gammabatch1", shape=[conv_dep1])

        # GRAPH SETUP
        # Convolutional layer
        conv1 = tf.nn.conv2d(images, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + bconv1
        # ReLU Activation
        relu1 = tf.nn.relu(conv1)
        relu1 = tf.nn.dropout(relu1, keep_prob)
        # Max pooling
        pool1 = tf.nn.max_pool(relu1, strides=[1, 2, 2, 1], padding='VALID', ksize=[1, 2, 2, 1])
        # Spatial Batch Normalization Layer
        meanbatch1, variancebatch1 = tf.nn.moments(pool1, axes=[0, 1, 2], keep_dims=True)
        batch1 = tf.nn.batch_norm_with_global_normalization(t=pool1,
                                                            m=meanbatch1,
                                                            v=variancebatch1,
                                                            beta=betabatch1,
                                                            gamma=gammabatch1,
                                                            variance_epsilon=1e-5,
                                                            scale_after_normalization=True)

        # Add regulerization
        regularizers += tf.nn.l2_loss(Wconv1)

        ############################################
        #           AFFINE layer                   #
        ############################################
        pool5_flat, num_features = flatten_layer(batch1)
        #_, H_, W_, D = batch1.get_shape()
        #pool5_flat = tf.reshape(batch1, [-1, H_ * W_ * D])
        W6 = tf.get_variable("W6", shape=[num_features, aff_size1])
        b6 = tf.get_variable("b6", shape=[aff_size1])

        # GRAPH SETUP
        aff6 = tf.matmul(pool5_flat, W6) + b6
        # Relu Activation
        y_out = tf.nn.relu(aff6)
        # relu6 = tf.nn.dropout(relu6, dropout)
        # Add regulerization
        regularizers += tf.nn.l2_loss(W6)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(y_out, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #accuracy = tf.metrics.accuracy(labels, prediction)
        #pdb.set_trace()
        total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, num_classes), logits=y_out)
        mean_loss = tf.reduce_mean(total_loss + regularizers * reg)

        # define our optimizer
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)  # select optimizer and set learning rate
        #optimizer = tf.train.AdamOptimizer(1e-6)

        # batch normalization in tensorflow requires this extra dependency
        # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(extra_update_ops):
        #    train_step = optimizer.minimize(mean_loss)
        train_op = optimizer.minimize(mean_loss)

        init_op = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        sess.run(init_op)

        plot_images(sess, train_init_op, images, labels, keep_prob)

        mean_image = calculate_mean_image(sess, train_init_op, sum_image, nbr_samples, keep_prob)

        for epoch in range(num_epochs):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            while True:
                try:
                    _, loss, acc = sess.run([train_op, mean_loss, accuracy], {keep_prob: 0.9, mean_image: mean_image})
                    print('loss: %f' % loss)
                    print('accuracy: %f' % acc)
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, keep_prob, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, keep_prob, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)




if __name__ == '__main__':
    data_dir = "../datasets/freiburg_groceries_dataset/images"
    num_workers = 4
    batch_size = 100

    # Hyperparameters
    reg = 0  # regularization
    learning_rate = 1e-6  # learning rate
    decay_rate = 0.8  # decay rate
    decay_steps = 10  # decay cut

    num_epochs = 10

    #correct_prediction, train_op, images, labels, num_classes, train_init_op, val_init_op, graph, is_training, init_op = create_model(data_dir, num_workers, batch_size, learning_rate, decay_steps, decay_rate, validation_percentage=0.3, testing_code=True)
    #run_model(graph, num_epochs, train_init_op, val_init_op, train_op, is_training, correct_prediction, init_op)

    create_model(data_dir, num_workers, batch_size, learning_rate, validation_percentage=0.3, testing_code=True)

    #pdb.set_trace()