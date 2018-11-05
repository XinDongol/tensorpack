import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import dorefa
import seaborn as sns
import numpy as np
import time
from tqdm import tqdm
import functools
#plt.style.use('seaborn')
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
'''
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
plt.savefig('./test_figs/test_forward_hwgq.pdf')

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
plt.savefig('./test_figs/test_backward_hwgq.pdf')


plt.figure()

fa = dorefa.get_warmbin(2)
plt.plot(sess.run(tf.range(-1,2,0.01)),sess.run(fa(tf.range(-1,2,0.01), 50)),label='2-bit', alpha=0.5)
fa = dorefa.get_warmbin(3)
plt.plot(sess.run(tf.range(-1,2,0.01)),sess.run(fa(tf.range(-1,2,0.01), 50)),label='3-bit', alpha=0.5)

plt.plot(sess.run(tf.range(-1,2,0.01)),sess.run(tf.nn.relu(tf.range(-1,2,0.01))),label='Relu', alpha=0.9)

plt.title('Forward')
plt.legend()
plt.savefig('./test_figs/test_forward_warmbin.pdf')


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
plt.savefig('./test_figs/test_backward_warmbin.pdf')



plt.figure()

fa = dorefa.get_warmbin(2)
x = tf.range(-0.1,1,0.001)
for i in [1,5,10,15,20,30,50,100,120,200,400,1000,2000,10000]:
	y = fa(x, i)
	plt.plot(sess.run(x),sess.run(y),label='2-bit-'+str(i), alpha=0.5, linewidth=0.5)
plt.legend()
plt.savefig('./test_figs/test_forward_warmbin_gradual.pdf')



plt.figure()
num_iters = 5000
relax_schduler = dorefa.Schdule_Relax(1,num_iters,1,1000)
fa = dorefa.get_warmbin(2)
x = tf.range(-0.1,1,0.001)
x_np = sess.run(x)
start = time.time()
relax_placeholder = tf.placeholder(tf.float32,(),'relax')
y = fa(x, relax_placeholder)

for i in tqdm(range(1,num_iters+1)):
	y_np = sess.run(y, feed_dict={relax_placeholder:relax_schduler.step()})
	plt.plot(x_np, y_np, alpha=0.1, linewidth=0.1, c='r')
print('Time for per iters: ', (time.time()-start)/num_iters)
plt.savefig('./test_figs/test_forward_warmbin_gradual_with_schduler-1.pdf')


plt.figure()
def cabs(x):
    return tf.minimum(1.0, tf.abs(x), name='cabs')
def fa(x):
	return fa_fake(cabs(x))
_,fa_fake,_ = dorefa.get_dorefa(32,5,32)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='5-bit', alpha=0.5)

_,fa_fake,_ = dorefa.get_dorefa(32,2,32)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='2-bit', alpha=0.5)

_,fa_fake,_ = dorefa.get_dorefa(32,3,32)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='3-bit', alpha=0.5)

_,fa_fake,_ = dorefa.get_dorefa(32,4,32)
plt.plot(sess.run(tf.range(-5,5,0.01)),sess.run(fa(tf.range(-5,5,0.01))),label='4-bit', alpha=0.5)

plt.xlim(0,1.2)
plt.ylim(0,1.2)
plt.title('Forward')
plt.legend()
plt.savefig('./test_figs/test_forward_derefa.pdf')


plt.figure()
def quantize(x, k):
    n = float(2 ** k - 1)

    @tf.custom_gradient
    def _quantize(x):
        return tf.round(x * n) / n, lambda dy: dy

    return _quantize(x)
x = tf.range(0,1,0.01)
fa = quantize
plt.plot(sess.run(x),sess.run(fa(x,2)),label='2-bit', alpha=0.5)
plt.plot(sess.run(x),sess.run(fa(x,3)),label='3-bit', alpha=0.5)
plt.plot(sess.run(x),sess.run(fa(x,4)),label='4-bit', alpha=0.5)
plt.title('Forward')
plt.legend()
plt.savefig('./test_figs/test_forward_derefa_quantize.pdf')


plt.figure()


x = tf.range(0,1.1,0.01)
_,soft_quantize,_ = dorefa.get_warmbin(32,3,15)
fa = quantize
plt.plot(sess.run(x),sess.run(fa(x,3)),label='2-bit-dorefa_quantize', alpha=0.5)
plt.plot(sess.run(x),sess.run(soft_quantize(x,15)),label='2-bit-soft_quantize-15', alpha=0.5)
plt.plot(sess.run(x),sess.run(soft_quantize(x,5)),label='2-bit-soft_quantize-5', alpha=0.5)
plt.plot(sess.run(x),sess.run(soft_quantize(x,55)),label='2-bit-soft_quantize-55', alpha=0.5)
plt.plot(sess.run(x),sess.run(x),label='y=x', alpha=0.5)
plt.title('Forward')
plt.legend()
plt.savefig('./test_figs/test_forward_soft_quantize.pdf')
'''



plt.figure(figsize=(5,15))

plt.subplot(312)

_,fa,_ = dorefa.get_warmbin_match(2, 2, 32)
x = tf.range(-1,2,0.01)
y = fa(x, 5)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-5', alpha=0.5)


y = fa(x, 10)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-10', alpha=0.5)


y = fa(x, 1)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-1', alpha=0.5)


y = fa(x, 20)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-20', alpha=0.5)

y = tf.nn.relu(x)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='Relu', alpha=0.9)
plt.legend()

plt.subplot(313)

_,fa,_ = dorefa.get_warmbin(2, 2, 32)
x = tf.range(-1,2,0.01)
y = fa(x, 5)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-5', alpha=0.5)

y = fa(x, 10)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-10', alpha=0.5)

y = fa(x, 1)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-1', alpha=0.5)

y = fa(x, 20)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-20', alpha=0.5)

y = fa(x, 50)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='2-bit-50', alpha=0.5)

y = tf.nn.relu(x)
plt.plot(sess.run(x),np.transpose(sess.run(tf.gradients(y, x))),label='Relu', alpha=0.9)
plt.legend()

plt.subplot(311)

_,fa,_ = dorefa.get_warmbin(32, 2, 32)
x = tf.range(0,1,0.01)

plt.plot(sess.run(x),sess.run(fa(x, 5)),label='2-bit-5', alpha=0.5)

plt.plot(sess.run(x),sess.run(fa(x, 1)),label='2-bit-1', alpha=0.5)

plt.plot(sess.run(x),sess.run(fa(x, 10)),label='2-bit-10', alpha=0.5)

plt.plot(sess.run(x),sess.run(fa(x, 20)),label='2-bit-20', alpha=0.5)

y = tf.nn.relu(x)
plt.plot(sess.run(x),sess.run(y),label='Relu', alpha=0.9)
plt.legend()


plt.savefig('./test_figs/test_backward_warmbin_match.pdf')

