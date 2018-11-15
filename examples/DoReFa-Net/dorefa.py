# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu
from scipy.io import loadmat
import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized
from numpy import tanh
from tensorpack import *
from math import sqrt

@graph_memoized
def get_dorefa(bitW, bitA, bitG):
    """
    Return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    def quantize(x, k):
        n = float(2 ** k - 1)

        @tf.custom_gradient
        def _quantize(x):
            return tf.round(x * n) / n, lambda dy: dy

        return _quantize(x)

    def fw(x, relax):
        # relax is just for API consistance
        if bitW == 32:
            return x

        if bitW == 1:   # BWN
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

            @tf.custom_gradient
            def _sign(x):
                return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E, lambda dy: dy

            return _sign(x)

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

    def fa(x, relax):
        # relax is just for API consistance
        if bitA == 32:
            return x

        return quantize(x, bitA)

    def fg(x):
        if bitG == 32:
            return x

        @tf.custom_gradient
        def _identity(input):
            def grad_fg(x):
                rank = x.get_shape().ndims
                assert rank is not None
                maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
                x = x / maxx
                n = float(2**bitG - 1)
                x = x * 0.5 + 0.5 + tf.random_uniform(
                    tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
                x = tf.clip_by_value(x, 0.0, 1.0)
                x = quantize(x, bitG) - 0.5
                return x * maxx * 2

            return input, grad_fg

        return _identity(x)
    return fw, fa, fg


@graph_memoized
def get_hwgq_bwn(bitW, bitA, bitG):

    def quantize_act(x, k):
        # in order of 
        assert k in [2], 'Does not support %d bits' % k
        code_book={
        '2':[0.5380, 0., 0.5380*(2**2-1)],
        '3':[0.3218, 0., 0.3218*(2**3-1)],
        '4':[0.1813, 0., 0.1813*(2**4-1)],
        '5':[0.1029, 0., 0.1029*(2**5-1)]
        }
        delta, minv, maxv = code_book[str(k)]
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            return tf.to_float(x>0.)*(tf.clip_by_value((tf.floor(x/delta + 0.5)+tf.to_float(x<0.5*delta))*delta, minv, maxv)), lambda dy: dy*tf.to_float(x>minv)*tf.to_float(x<maxv)

        return _quantize(x)   
    
    
    def fw(x, relax):
 
        if bitW == 1: #BWN
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

            @tf.custom_gradient
            def _sign(x):
                return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E, lambda dy: dy

            return _sign(x)
            
        else:
            raise NameError('You should use bitW=1 for HWGQ !')


    def fa(x, relax):
        if bitA != 2:
            raise NameError('You should use bitA=2 for HWGQ !')
            
        return quantize_act(x, bitA)
    def fg(x):
        if bitG == 32:
            return x
        else:
            raise NameError('Don not support gradients !')
    return fw, fa, fg


@graph_memoized
def get_hwgq_bwn_warm(bitW, bitA, bitG):
    
    def scale_tanh(x, x_scale, y_scale):
        # scale tanh alone x-axis and y-axis
        return (y_scale*tf.tanh(x_scale*x))

    def move_scaled_tanh(x, x_scale, y_scale, x_range, x_move, y_move):
        # move the scaled tanh along x-axis and y-axis
        return tf.clip_by_value(scale_tanh(x+x_move, x_scale, y_scale ),-0.5*x_range,0.5*x_range)+y_move

    def tanh_appro(x, x_scale, y_scale, k, delta):
        y_scale_first = 0.5*2*delta/tf.tanh(x_scale*0.5*2*delta)
        y=  tf.clip_by_value(move_scaled_tanh(x, x_scale, y_scale_first, 2*delta, 0, 0),0,delta)
        for i in range(1, 2**k-1):
            y += move_scaled_tanh(x, x_scale, y_scale, delta, (-i-0.5)*delta, 0.5*delta)
        return y 


    def quantize_act(x, k, x_scale):
        # in order of 
        assert k in [2,3,4,5], 'Does not support %d bits' % k
        code_book={
        '2':[0.5380, 0., 0.5380*(2**2-1)],
        '3':[0.3218, 0., 0.3218*(2**3-1)],
        '4':[0.1813, 0., 0.1813*(2**4-1)],
        '5':[0.1029, 0., 0.1029*(2**5-1)]
        }
        delta, minv, maxv = code_book[str(k)]
        y_scale = 0.5*delta/tf.tanh(x_scale*0.5*delta)
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            return tf.clip_by_value(tanh_appro(x, x_scale, y_scale, k, delta),0,maxv)\
            , lambda dy: dy*tf.to_float(x>minv)*tf.to_float(x<maxv)

        return _quantize(x)

    
    
    def fw(x, relax):
 
        if bitW == 1: #BWN
            def sign_like(x, x_scale):
                delta = 2.
                y_scale = (0.5*delta)/tf.tanh(x_scale*0.5*delta)
                return move_scaled_tanh(x, x_scale, y_scale, delta, 0, 0)
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

            @tf.custom_gradient
            def _sign(x):
                return sign_like(x / E, relax) * E, lambda dy: dy

            return _sign(x)
            
        else:
            raise NameError('You should use bitW=1 for HWGQ !')


    def fa(x, relax):
        if bitA != 2:
            raise NameError('You should use bitA=2 for HWGQ !')
            
        return quantize_act(x, bitA, relax)
    def fg(x):
        if bitG == 32:
            return x
        else:
            raise NameError('Don not support gradients !')
    return fw, fa, fg




@graph_memoized
def get_hwgq_dorefa(bitW, bitA, bitG):
    
    def quantize(x, k):
        n = float(2 ** k - 1)

        @tf.custom_gradient
        def _quantize(x):
            return tf.round(x * n) / n, lambda dy: dy

        return _quantize(x)
    
    
    def quantize_act(x, k):
        # in order of 
        assert k in [2,3,4,5], 'Does not support %d bits' % k
        code_book={
        '2':[0.5380, 0., 0.5380*(2**2-1)],
        '3':[0.3218, 0., 0.3218*(2**3-1)],
        '4':[0.1813, 0., 0.1813*(2**4-1)],
        '5':[0.1029, 0., 0.1029*(2**5-1)]
        }
        delta, minv, maxv = code_book[str(k)]
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            return tf.to_float(x>0.)*(tf.clip_by_value((tf.floor(x/delta + 0.5)+tf.to_float(x<0.5*delta))*delta, minv, maxv)), lambda dy: dy*tf.to_float(x>minv)*tf.to_float(x<maxv)

        return _quantize(x)
    
    def fw(x, relax):
 
        if bitW == 1: #BWN
            x = tf.tanh(x)
            x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
            return 2 * quantize(x, bitW) - 1
        else:
            raise NameError('You should use bitW=1 for HWGQ !')


    def fa(x, relax):
        if bitA != 2:
            raise NameError('You should use bitA=2 for HWGQ !')
        return quantize_act(x, bitA)
    
    def fg(x):
        if bitG == 32:
            return x
        else:
            raise NameError('Don not support gradients !')
    return fw, fa, fg



@graph_memoized
def get_warmbin(bitW, bitA, bitG):
    '''
    This is for hwgq like
    def scale_tanh(x, x_scale, y_scale):
        # scale tanh alone x-axis and y-axis
        return (y_scale*tf.tanh(x_scale*x))

    def move_scaled_tanh(x, x_scale, y_scale, x_range, x_move, y_move):
        # move the scaled tanh along x-axis and y-axis
        return (scale_tanh(x+x_move, x_scale, y_scale )+y_move)* \
        tf.to_float((x+x_move)>=-0.5*x_range) *\
        tf.to_float((x+x_move)<0.5*x_range)

    def tanh_appro(x, x_scale, y_scale, k, delta):
        y=0
        for i in range(2**k):
            y += move_scaled_tanh(x, x_scale, y_scale, delta, (-i+0.5)*delta, (i-0.5)*delta)
        return y 


    def quantize(x, k, x_scale):
        # in order of 
        assert k in [2,3,4,5], 'Does not support %d bits' % k
        code_book={
        '2':[0.2662, 0., 0.2662*(2**2-1)],
        '3':[0.1139, 0., 0.1139*(2**3-1)],
        '4':[0.1813, 0., 0.1813*(2**4-1)],
        '5':[0.1029, 0., 0.1029*(2**5-1)]
        }
        delta, minv, maxv = code_book[str(k)]
        y_scale = 0.5*delta/tf.tanh(x_scale*0.5*delta)
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            return tf.to_float(x>0.)*tanh_appro(x, x_scale, y_scale, k, delta)+tf.to_float(x>maxv)*maxv\
            , lambda dy: dy*tf.to_float(x>minv)*tf.to_float(x<maxv)

        return _quantize(x)
    '''
    def scale_tanh(x, x_scale, y_scale):
        # scale tanh alone x-axis and y-axis
        return (y_scale*tf.tanh(x_scale*x))

    def move_scaled_tanh(x, x_scale, y_scale, x_range, x_move, y_move):
        # move the scaled tanh along x-axis and y-axis
        return tf.clip_by_value(scale_tanh(x+x_move, x_scale, y_scale ),-0.5*x_range,0.5*x_range)+y_move
        #* \
        #tf.to_float((x+x_move)>=-0.5*x_range) *\
        #tf.to_float((x+x_move)<0.5*x_range)

    def tanh_appro(x, x_scale, y_scale, k, delta):
        y=0
        
        for i in range(1,2**k):
            y += move_scaled_tanh(x, x_scale, y_scale, delta, (-i+0.5)*delta, (0.5)*delta)
        #i=1
        #y = move_scaled_tanh(x, x_scale, y_scale, delta, (-i+0.5)*delta, (0.5)*delta)
        return y 


    def quantize(x, k, x_scale):

        delta = float(1./(2**k-1.))
        y_scale = (0.5*delta)/tf.tanh(x_scale*0.5*delta)
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            return tanh_appro(x, x_scale, y_scale, k, delta), lambda dy: dy

        return _quantize(x)

    def fw(x, relax):
        if bitW == 32:
            return x
        
        if bitW == 1: #BWN
            def sign_like(x, x_scale):
                delta = 2.
                y_scale = (0.5*delta)/tf.tanh(x_scale*0.5*delta)
                return move_scaled_tanh(x, x_scale, y_scale, delta, 0, 0)
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

            @tf.custom_gradient
            def _sign(x):
                return sign_like(x / E, relax) * E, lambda dy: dy

            return _sign(x)
            

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW, relax) - 1

    def fa(x, relax):
        # relax is just for API consistance
        if bitA == 32:
            return x
        #print(bitA)
        return quantize(x, bitA, relax)

    def fg(x):
        if bitG == 32:
            return x
        else:
            raise NameError('Don not support gradients !')
    return fw, fa, fg


@graph_memoized
def get_warmbin_match(bitW, bitA, bitG):
    '''
    use the same function for forward and backward
    '''
    def scale_tanh(x, x_scale, y_scale):
        # scale tanh alone x-axis and y-axis
        return (y_scale*tf.tanh(x_scale*x))

    def move_scaled_tanh(x, x_scale, y_scale, x_range, x_move, y_move):
        # move the scaled tanh along x-axis and y-axis
        return tf.clip_by_value(scale_tanh(x+x_move, x_scale, y_scale ),-0.5*x_range,0.5*x_range)+y_move
        #* \
        #tf.to_float((x+x_move)>=-0.5*x_range) *\
        #tf.to_float((x+x_move)<0.5*x_range)

    def tanh_appro(x, x_scale, y_scale, k, delta):
        y=0
        
        for i in range(1,2**k):
            y += move_scaled_tanh(x, x_scale, y_scale, delta, (-i+0.5)*delta, (0.5)*delta)
        #i=1
        #y = move_scaled_tanh(x, x_scale, y_scale, delta, (-i+0.5)*delta, (0.5)*delta)
        return y 


    def quantize(x, k, x_scale, x_scale_back=10.):
    
        x_scale_back = tf.minimum(x_scale, tf.constant(10.0))
        delta = float(1./(2**k-1.))
        y_scale = (0.5*delta)/tf.tanh(x_scale*0.5*delta)
        y_scale_back = (0.5*delta)/tf.tanh(x_scale_back*0.5*delta)
        #print(delta,minv,maxv)
        def _quantize(x):
            return tanh_appro(x, x_scale_back, y_scale_back, k, delta) + \
                    tf.stop_gradient(tanh_appro(x, x_scale, y_scale, k, delta) - tanh_appro(x, x_scale_back, y_scale_back, k, delta))

        return _quantize(x)

    def fw(x, relax, relax_back):
        if bitW == 32:
            return x

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW, relax, relax_back) - 1

    def fa(x, relax, relax_back):
        # relax is just for API consistance
        if bitA == 32:
            return x

        return quantize(x, bitA, relax, relax_back)

    def fg(x):
        if bitG == 32:
            return x
        else:
            raise NameError('Don not support gradients !')
    return fw, fa, fg


@graph_memoized
def get_warmbin_clip(bitW, bitA, bitG):
    '''
    use the same function for forward and backward
    '''
    def scale_tanh(x, x_scale, y_scale):
        # scale tanh alone x-axis and y-axis
        return (y_scale*tf.tanh(x_scale*x))

    def move_scaled_tanh(x, x_scale, y_scale, x_range, x_move, y_move):
        # move the scaled tanh along x-axis and y-axis
        return tf.clip_by_value(scale_tanh(x+x_move, x_scale, y_scale ),-0.5*x_range,0.5*x_range)+y_move
        #* \
        #tf.to_float((x+x_move)>=-0.5*x_range) *\
        #tf.to_float((x+x_move)<0.5*x_range)

    def tanh_appro(x, x_scale, y_scale, k, delta):
        y=0
        
        for i in range(1,2**k):
            y += move_scaled_tanh(x, x_scale, y_scale, delta, (-i+0.5)*delta, (0.5)*delta)
        #i=1
        #y = move_scaled_tanh(x, x_scale, y_scale, delta, (-i+0.5)*delta, (0.5)*delta)
        return y 

    def quantize(x, k, x_scale, x_scale_back):

        delta = float(1./(2**k-1.))
        y_scale = (0.5*delta)/tf.tanh(x_scale*0.5*delta)
                
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            grad_y = 0
            y_scale_back = (0.5*delta)/tf.tanh(x_scale_back*0.5*delta)
            for i in range(1,2**k):
                grad_y += (y_scale_back*x_scale_back*(1-tf.tanh(x_scale_back*(x+(-i+0.5)*delta))**2)) \
                                * tf.to_float((x+(-i+0.5)*delta)>=-0.5*delta)\
                                * tf.to_float((x+(-i+0.5)*delta)<0.5*delta)
                    #return dy * (tf.clip_by_value(grad_y, -100, 1))
            def deriv(dy):
                return dy * tf.clip_by_value(grad_y,-100,1.5)
            return tanh_appro(x, x_scale, y_scale, k, delta), deriv

        return _quantize(x)

    def fw(x, relax, relax_back):
        if bitW == 32:
            return x

        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW, relax, relax_back) - 1

    def fa(x, relax, relax_back):
        # relax is just for API consistance
        if bitA == 32:
            return x

        return quantize(x, bitA, relax, relax_back)

    def fg(x):
        if bitG == 32:
            return x
        else:
            raise NameError('Don not support gradients !')
    return fw, fa, fg



class Schdule_Relax():
    def __init__(self, start_iter, end_iter, start_value, end_value, mode):
        if mode == 'expo':
            self.p = pow((end_value / start_value), 1./ (end_iter - start_iter))
        elif mode == 'linear':
            self.p = (end_value - start_value) / (end_iter - start_iter)
        elif mode == 'sqrt':
            self.a = (end_value - start_value) / (sqrt(end_iter)-1)
            self.b = start_value - self.a
        elif mode == 'from_file':
            self.mat = loadmat(start_value)['c'][0]
        self.start_value = start_value
        self.start_iter = start_iter 
        self.end_iter = end_iter
        self.now_iter = start_iter -1 
        self.now_value = start_value
        self.mode = mode
    def get_relax(self, now_iter):
        if self.mode == 'expo':
            return self.start_value * pow(self.p, (now_iter-self.start_iter))  
        if self.mode == 'from_file':
            return self.mat[int(self.now_iter*self.mat.shape[0]/(self.end_iter - self.start_iter + 1))]
    def step(self):
        self.now_iter += 1
        if self.now_iter <= self.end_iter:
            if self.mode == 'expo':
                self.now_value *= self.p
            elif self.mode == 'linear':
                self.now_value += self.p
            elif self.mode == 'sqrt':
                self.now_value = self.a * sqrt(self.now_iter) + self.b
            elif self.mode == 'from_file':
                self.now_value = self.get_relax(self.now_iter)
        return self.now_value 
    def ident(self, x):
        # for test
        return x
class RelaxSetter(Callback):
    def __init__(self, start_iter, end_iter, start_value, end_value, mode='linear'):
        assert mode in ['linear','expo', 'sqrt', 'from_file']
        self.relax_schduler = Schdule_Relax(start_iter, end_iter, start_value, end_value, mode)
    def _setup_graph(self):
        self._relax = [k for k in tf.global_variables() if k.name == 'relax_para:0'][0]
    def _trigger_step(self):
        self._relax.load(self.relax_schduler.step())
        
class RelaxSetterBack(Callback):
    def __init__(self, start_iter, end_iter, start_value, end_value, mode='linear'):
        assert mode in ['linear','expo', 'sqrt', 'from_file']
        self.relax_schduler = Schdule_Relax(start_iter, end_iter, start_value, end_value, mode)
    def _setup_graph(self):
        self._relax = [k for k in tf.global_variables() if k.name == 'relax_para_back:0'][0]
    def _trigger_step(self):
        self._relax.load(self.relax_schduler.step())
        
def ternarize(x, thresh=0.05):
    """
    Implemented Trained Ternary Quantization:
    https://arxiv.org/abs/1612.01064

    Code modified from the authors' at:
    https://github.com/czhu95/ternarynet/blob/master/examples/Ternary-Net/ternary.py
    """
    shape = x.get_shape()

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thresh)

    w_p = tf.get_variable('Wp', initializer=1.0, dtype=tf.float32)
    w_n = tf.get_variable('Wn', initializer=1.0, dtype=tf.float32)

    tf.summary.scalar(w_p.op.name + '-summary', w_p)
    tf.summary.scalar(w_n.op.name + '-summary', w_n)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    @tf.custom_gradient
    def _sign_mask(x):
        return tf.sign(x) * mask_z, lambda dy: dy

    w = _sign_mask(x)

    w = w * mask_np

    tf.summary.histogram(w.name, w)
    return w

if __name__ == '__main__':
    '''
    setter = Schdule_Relax(start_iter=1, end_iter=200, start_value=1, end_value=100, mode='sqrt')
    for i in range(202):
        print(setter.step())
    '''
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    sess = tf.Session(config=config)
    
    #-----------------------------------------------#
    def scale_tanh(x, x_scale, y_scale):
        # scale tanh alone x-axis and y-axis
        return (y_scale*tf.tanh(x_scale*x))

    def move_scaled_tanh(x, x_scale, y_scale, x_range, x_move, y_move):
        # move the scaled tanh along x-axis and y-axis
        return tf.clip_by_value(scale_tanh(x+x_move, x_scale, y_scale ),-0.5*x_range,0.5*x_range)+y_move

    def tanh_appro(x, x_scale, y_scale, k, delta):
        y_scale_first = 0.5*2*delta/tf.tanh(x_scale*0.5*2*delta)
        y=  tf.clip_by_value(move_scaled_tanh(x, x_scale, y_scale_first, 2*delta, 0, 0),0,delta)
        for i in range(1, 2**k-1):
            y += move_scaled_tanh(x, x_scale, y_scale, delta, (-i-0.5)*delta, 0.5*delta)
        return y 


    def quantize(x, k, x_scale):
        # in order of 
        assert k in [2,3,4,5], 'Does not support %d bits' % k
        code_book={
        '2':[0.5380, 0., 0.5380*(2**2-1)],
        '3':[0.3218, 0., 0.3218*(2**3-1)],
        '4':[0.1813, 0., 0.1813*(2**4-1)],
        '5':[0.1029, 0., 0.1029*(2**5-1)]
        }
        delta, minv, maxv = code_book[str(k)]
        y_scale = 0.5*delta/tf.tanh(x_scale*0.5*delta)
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            return tf.clip_by_value(tanh_appro(x, x_scale, y_scale, k, delta),0,maxv)\
            , lambda dy: dy*tf.to_float(x>minv)*tf.to_float(x<maxv)

        return _quantize(x)

    def quantize_act(x, k, placeholder):
        # in order of 
        assert k in [2], 'Does not support %d bits' % k
        code_book={
        '2':[0.5380, 0., 0.5380*(2**2-1)],
        '3':[0.3218, 0., 0.3218*(2**3-1)],
        '4':[0.1813, 0., 0.1813*(2**4-1)],
        '5':[0.1029, 0., 0.1029*(2**5-1)]
        }
        delta, minv, maxv = code_book[str(k)]
        #print(delta,minv,maxv)
        @tf.custom_gradient
        def _quantize(x):
            return tf.to_float(x>0.)*(tf.clip_by_value((tf.floor(x/delta + 0.5)+tf.to_float(x<0.5*delta))*delta, minv, maxv)), lambda dy: dy*tf.to_float(x>minv)*tf.to_float(x<maxv)

        return _quantize(x)   
    
    
    plt.figure(figsize=(8,15))

    plt.subplot(211)

    x = tf.range(-0.1,2.5,0.01)
    fa = quantize
    bit = 2
    
    plt.plot(sess.run(x),sess.run(fa(x,bit,500)),label='2-bit-500', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,50)),label='2-bit-50', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,5)),label='2-bit-5', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,2)),label='2-bit-2', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,10)),label='2-bit-10', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,15)),label='2-bit-15', alpha=0.5)
    plt.plot(sess.run(x),sess.run(x),label='x', alpha=0.5)
    #plt.plot(sess.run(x),sess.run(fa(x,3,500)),label='3-bit', alpha=0.5)
    #plt.plot(sess.run(x),sess.run(fa(x,4,500)),label='4-bit', alpha=0.5)
    #plt.title('Forward')
    plt.legend()
    
    plt.subplot(212)

    x = tf.range(-0.1,2.5,0.01)
    fa = quantize_act
    bit = 2
    
    plt.plot(sess.run(x),sess.run(fa(x,bit,500)),label='2-bit-500', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,50)),label='2-bit-50', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,5)),label='2-bit-5', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,2)),label='2-bit-2', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,10)),label='2-bit-10', alpha=0.5)
    plt.plot(sess.run(x),sess.run(fa(x,bit,15)),label='2-bit-15', alpha=0.5)
    plt.plot(sess.run(x),sess.run(x),label='x', alpha=0.5)
    #plt.plot(sess.run(x),sess.run(fa(x,3,500)),label='3-bit', alpha=0.5)
    #plt.plot(sess.run(x),sess.run(fa(x,4,500)),label='4-bit', alpha=0.5)
    #plt.title('Forward')
    plt.legend()
    
    
    
    plt.savefig('./test_figs/test_hwgq_new.pdf')