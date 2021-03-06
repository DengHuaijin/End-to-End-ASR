from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from six.moves import range

from asr_e2e.optimizers.optimizers import optimize_loss
from asr_e2e.optimizers.lr_policies import fixed_lr

class OptimizerTests(tf.test.TestCase):

    def SetUp(self):
        pass

    def tearDown(self):
        pass

    def test_updates(self):
        dtype = tf.float32

        with tf.Graph().as_default() as g:
            
            n_samples = 10
            n_hid = 10
            var_dtype = tf.float32 if dtype == tf.float32 else tf.float16

            np.random.seed(0)

            X = np.random.rand(n_samples, n_hid)
            y = np.random.rand(n_samples, 1)
            w = np.dot(np.linalg.inv(X), y)
            # w = np.linalg.solve(X.T.dot(X), X.T.dot(y))
            
            x_ph = tf.placeholder(var_dtype, [n_samples, n_hid])
            y_ph = tf.placeholder(var_dtype, [n_samples, 1])

            y_pred = tf.layers.dense(x_ph, 1, use_bias = False)
            loss = tf.losses.mean_squared_error(y_ph, y_pred)
            loss += tf.losses.get_regularization_loss()
            skip_update_ph = tf.placeholder(tf.bool)
            # iter_size = 8
            train_op, grads_and_vars = optimize_loss(loss, "SGD", {},
                                     lambda gs: fixed_lr(gs, 0.3), dtype = dtype,
                                     skip_update_ph = skip_update_ph)
            
            for var in tf.global_variables():
                print(var.name)
            
            var = tf.trainable_variables()[0]
            grad, _ = grads_and_vars[0]
            with self.test_session(g, use_gpu = True) as sess:
                
                sess.run(tf.global_variables_initializer())
                print("Before update: {}".format(sess.run(var)))
                # print("y label: {}".format(y))
                for i in range(5):
                    _, v_value, new_grad, loss_value, y_value = sess.run([train_op, var, grad, loss, y_pred],  {x_ph: X, y_ph: y})
                    
                    true_grad = 2 * (X.T.dot(X).dot(v_value) - X.T.dot(y)) / X.shape[0]
                    npt.assert_allclose(new_grad, true_grad, atol = 1e-7)
                    """
                    v = sess.run([var])
                    # npt.assert_allclose(g, np.zeros(g.shape))

                    true_g = 2 * (X.T.dot(X).dot(v) - X.T.dot(y)) / X.shape[0]
                    
                    sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: True})
                    v_new = sess.run([var])
                    # npt.assert_allclose(g_new, true_g, atol = 1e-7)
                    npt.assert_allclose(v_new, v)

                    sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: True})
                    v_new = sess.run([var])
                    # npt.assert_allclose(g_new, true_g * 2, atol = 1e-7)
                    npt.assert_allclose(v_new, v)

                    sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: True})
                    v_new = sess.run([var])
                    # npt.assert_allclose(g_new, true_g * 3, atol = 1e-7)
                    npt.assert_allclose(v_new, v)

                    sess.run(train_op, {x_ph: X, y_ph: y, skip_update_ph: False})
                    v_new = sess.run([var])
                    # npt.assert_allclose(g_new, np.zeros(g.shape))
                    npt.assert_allclose(v_new, v - 0.1 * true_g * 4, atol = 1e-7)
                    """
                    print("epoch {} var: {}  Loss: {}".format(i, sess.run([var]), loss_value))
                    # print("y_pred: {}".format(y_value))
                print(w)
                print("y label: {}".format(y))
                print("y_pred: {}".format(y_value))
                print(np.mean((y_value - y)**2))
tf.test.main()

