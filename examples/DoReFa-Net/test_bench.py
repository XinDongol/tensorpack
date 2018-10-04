import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import dorefa
import seaborn as sns
import numpy as np
#plt.style.use('seaborn')

sess=tf.Session()

plt.figure()
fa = dorefa.get_hwgq(5)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='5-bit', alpha=0.5)
fa = dorefa.get_hwgq(2)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='2-bit', alpha=0.5)
fa = dorefa.get_hwgq(3)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='3-bit', alpha=0.5)
fa = dorefa.get_hwgq(4)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='4-bit', alpha=0.5)
plt.title('Forward')
plt.legend()
plt.savefig('test_forward_hwgq.pdf')

plt.figure()
fa = dorefa.get_hwgq(5)
x = tf.range(-5,5,0.01)
y = fa(x)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='5-bit', alpha=0.5)

fa = dorefa.get_hwgq(2)
x = tf.range(-5,5,0.01)
y = fa(x)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit', alpha=0.5)

fa = dorefa.get_hwgq(3)
x = tf.range(-5,5,0.01)
y = fa(x)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='3-bit', alpha=0.5)

fa = dorefa.get_hwgq(4)
x = tf.range(-5,5,0.01)
y = fa(x)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='4-bit', alpha=0.5)
plt.legend()
plt.savefig('test_backward_hwgq.pdf')


plt.figure()

fa = dorefa.get_warmbin(2)
plt.plot(sess.run(tf.range(-1,2,0.01)),sess.run(fa(tf.range(-1,2,0.01), 50)),label='2-bit', alpha=0.5)
fa = dorefa.get_warmbin(3)
plt.plot(sess.run(tf.range(-1,2,0.01)),sess.run(fa(tf.range(-1,2,0.01), 50)),label='3-bit', alpha=0.5)

plt.plot(sess.run(tf.range(-1,2,0.01)),sess.run(tf.nn.relu(tf.range(-1,2,0.01))),label='Relu', alpha=0.9)

plt.title('Forward')
plt.legend()
plt.savefig('test_forward_warmbin.pdf')


plt.figure()

fa = dorefa.get_warmbin(2)
x = tf.range(-1,2,0.01)
y = fa(x, 50)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit', alpha=0.5)

fa = dorefa.get_warmbin(3)
x = tf.range(-1,2,0.01)
y = fa(x, 50)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='3-bit', alpha=0.5)

x = tf.range(-1,2,0.01)
y = tf.nn.relu(x)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='Relu', alpha=0.9)

plt.legend()
plt.savefig('test_backward_warmbin.pdf')



plt.figure()

fa = dorefa.get_warmbin(2)
x = tf.range(-0.1,1,0.001)
for i in [1,5,10,15,20,30,50,100,120,200,400,1000,2000,10000]:
	y = fa(x, i)
	plt.plot(sess.run(x),sess.run(y),label='2-bit-'+str(i), alpha=0.5, linewidth=0.5)
plt.legend()
plt.savefig('test_backward_warmbin_gradual.pdf')
