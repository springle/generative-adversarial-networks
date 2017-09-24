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
    g_learning_rate = context.get("g_learning_rate") or 0.0001
    d_fake_learning_rate = context.get("g_learning_rate") or 0.0003
    d_real_learning_rate = context.get("g_learning_rate") or 0.0003
    total_steps = context.get("total_steps") or 100000
    pre_train_steps = context.get("pre_train_steps") or 1000
    num_workers = len(server.server_def.cluster.job[1].tasks)

    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
    # z_placeholder is for feeding input noise to the generator

    x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x_placeholder')
    # x_placeholder is for feeding input images to the discriminator

    Gz = generator(z_placeholder, batch_size, z_dimensions)
    # Gz holds the generated images

    Dx = discriminator(x_placeholder)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
    # Dx will hold discriminator prediction probabilities
    # for the real MNIST images

    Dg = discriminator(Gz, reuse_variables=True)
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
    # Dg will hold discriminator prediction probabilities for generated images

    # Define variable lists
    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    g_global_step = tf.Variable(0, trainable=False, name='g_global_step')
    d_fake_global_step = tf.Variable(0, trainable=False, name='d_fake_global_step')
    d_real_global_step = tf.Variable(0, trainable=False, name='d_real_global_step')

    # Train the generator
    g_opt = tf.train.AdamOptimizer(g_learning_rate)
    g_opt = tf.train.SyncReplicasOptimizer(g_opt, replicas_to_aggregate=num_workers-1)
    g_trainer = g_opt.minimize(g_loss, var_list=g_vars, global_step=g_global_step)

    # Train the fake discriminator
    d_opt_fake = tf.train.AdamOptimizer(d_fake_learning_rate)
    d_opt_fake = tf.train.SyncReplicasOptimizer(d_opt_fake, replicas_to_aggregate=num_workers-1)
    d_trainer_fake = d_opt_fake.minimize(d_loss_fake, var_list=d_vars, global_step=d_fake_global_step)

    # Train the real discriminator
    d_opt_real = tf.train.AdamOptimizer(d_real_learning_rate)
    d_opt_real = tf.train.SyncReplicasOptimizer(d_opt_real, replicas_to_aggregate=num_workers-1)
    d_trainer_real = d_opt_real.minimize(d_loss_real, var_list=d_vars, global_step=d_real_global_step)

    # From this point forward, reuse variables
    tf.get_variable_scope().reuse_variables()

    # Send summary statistics to TensorBoard
    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_loss_real', d_loss_real)
    tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

    images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
    tf.summary.image('Generated_images', images_for_tensorboard, 5)
    merged = tf.summary.merge_all()

    is_chief = server.server_def.task_index == 0
    hooks = [g_opt.make_session_run_hook(is_chief),
             d_opt_fake.make_session_run_hook(is_chief),
             d_opt_real.make_session_run_hook(is_chief)]
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           hooks=hooks) as sess:

        if is_chief:
            print("I am the chief!")
            print("Creating writer")
            writer = tf.summary.FileWriter(log_dir, sess.graph)

        local_step = 0
        while sess.run(g_global_step) < 1000000:
            d_fake_step, d_real_step, g_step = sess.run([d_fake_global_step, d_real_global_step, g_global_step])
            if (d_fake_step < pre_train_steps) and (d_real_step < pre_train_steps):
                print("[step] pre-training... local_step={}, d_fake_global_step={}, d_real_global_step={}, "
                      "g_global_step={}".format(local_step, d_fake_step, d_real_step, g_step))
                z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
                real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
                _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                      {x_placeholder: real_image_batch, z_placeholder: z_batch})
                continue

            print("[step] training together ... local step {}".format(local_step))
            real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

            # Train discriminator on both real and fake images
            _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                  {x_placeholder: real_image_batch, z_placeholder: z_batch})

            # Train generator
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

            if is_chief and (local_step % 10 == 0):
                # Update TensorBoard with summary statistics
                print("Saving summary {}".format(str(local_step / 10)))
                z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
                summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
                writer.add_summary(summary, local_step / 10)

            if local_step % 50 == 0:
                print("real training... local_step={}, d_fake_global_step={}, d_real_global_step={}, "
                      "g_global_step={}".format(local_step, d_fake_step, d_real_step, g_step))

            local_step += 1
