"""
This is a straightforward Python implementation of a generative adversarial network.
The code is drawn directly from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

A version of this model with explanatory notes is also available on GitHub
at https://github.com/jonbruner/generative-adversarial-networks.

This script requires TensorFlow and its dependencies in order to run. Please see
the readme for guidance on installing TensorFlow.

This script won't print summary statistics in the terminal during training;
track progress and see sample images in TensorBoard.
"""

import tensorflow as tf
import numpy as np
import datetime

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads:
            List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# Define the discriminator network
def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4


# Define the generator network
def generator(z, batch_size, z_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim / 2], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim / 2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim / 2, z_dim / 4], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim / 4, 1], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4


def main(server, log_dir, context):
    """ Accept parameters from wrapper script """

    z_dimensions = context.get("z_dimensions") or 100
    batch_size = context.get("batch_size") or 50
    pre_train_steps = context.get("pre_train_steps") or 300
    g_learning_rate = context.get("g_learning_rate") or 0.0001
    d_fake_learning_rate = context.get("g_learning_rate") or 0.0003
    d_real_learning_rate = context.get("g_learning_rate") or 0.0003
    beta1 = context.get("beta1") or 0.9
    beta2 = context.get("beta2") or 0.999
    run_name = context.get("run_name") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if len(server.server_def.cluster.job) > 1:
        num_workers = len(server.server_def.cluster.job[1].tasks)
    else:
        num_workers = len(server.server_def.cluster.job[0].tasks)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    g_opt = tf.train.AdamOptimizer(g_learning_rate, beta1=beta1, beta2=beta2)
    d_opt_fake = tf.train.AdamOptimizer(d_fake_learning_rate, beta1=beta1, beta2=beta2)
    d_opt_real = tf.train.AdamOptimizer(d_real_learning_rate, beta1=beta1, beta2=beta2)

    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
    x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x_placeholder')

    gs = []
    d_fakes = []
    d_reals = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(num_workers):
            with tf.device("/job:worker/task:%d" % i):
                with tf.name_scope("worker_%d" % i) as scope:

                    # Define generator
                    Gz = generator(z_placeholder, batch_size, z_dimensions)
                    Dx = discriminator(x_placeholder)

                    # Define discriminator
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
                    Dg = discriminator(Gz, reuse_variables=True)
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
                    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

                    # Reuse variables for next worker
                    tf.get_variable_scope().reuse_variables()

                    # Define variable lists
                    tvars = tf.trainable_variables()
                    d_vars = [var for var in tvars if 'd_' in var.name]
                    g_vars = [var for var in tvars if 'g_' in var.name]

                    g_grad = g_opt.compute_gradients(g_loss, var_list=g_vars)
                    gs.append(g_grad)

                    d_fake_grad = d_opt_fake.compute_gradients(d_loss_fake, var_list=d_vars)
                    d_fakes.append(d_fake_grad)

                    d_real_grad = d_opt_real.compute_gradients(d_loss_real, var_list=d_vars)
                    d_reals.append(d_real_grad)

                    if not i:
                        # Send summary statistics to TensorBoard
                        tf.summary.scalar('Generator_loss', g_loss)
                        tf.summary.scalar('Discriminator_loss_real', d_loss_real)
                        tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

                        images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
                        tf.summary.image('Generated_images', images_for_tensorboard, 5)

    merged = tf.summary.merge_all()

    g_grads = average_gradients(gs)
    d_fake_grads = average_gradients(d_fakes)
    d_real_grads = average_gradients(d_reals)

    g_train_op = g_opt.apply_gradients(g_grads, global_step)
    d_fake_train_op = d_opt_fake.apply_gradients(d_fake_grads, global_step)
    d_real_train_op = d_opt_fake.apply_gradients(d_real_grads, global_step)

    # var_avgs = tf.train.ExponentialMovingAverage(0.999, global_step)
    # var_avgs_op = var_avgs.apply(tf.trainable_variables())

    # g_train_op = tf.group(g_train, var_avgs_op)
    # d_fake_train_op = tf.group(d_fake_train, var_avgs_op)
    # d_real_train_op = tf.group(d_real_train, var_avgs_op)

    is_chief = server.server_def.task_index == 0
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief) as sess:

        log_dir = log_dir + "/" + run_name + "/"
        writer = tf.summary.FileWriter(log_dir, sess.graph) if is_chief else None

        local_step = 0
        while tf.train.global_step(sess, global_step) < 1000000:
            gstep = tf.train.global_step(sess, global_step)

            # Train discriminator
            real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            sess.run([d_real_train_op, d_fake_train_op],
                     feed_dict={x_placeholder: real_image_batch, z_placeholder: z_batch})

            if gstep > pre_train_steps:
                # Train generator
                z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
                sess.run([g_train_op],
                         feed_dict={z_placeholder: z_batch})

            if is_chief and (local_step % 100 == 0):
                # Update TensorBoard with summary statistics
                print("Saving summary at global step {} (local step {})".format(gstep, local_step))
                z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
                summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
                writer.add_summary(summary, gstep)

            local_step += 1
