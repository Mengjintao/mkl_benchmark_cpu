import tvm
import numpy
import timeit
from tvm import autotvm
from tvm import te
#import tvm.testing
import logging
import sys
import os
# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = int(sys.argv[1])
K = int(sys.argv[2])
N = int(sys.argv[3])
print('M=%d, K=%d, N=%d' % (M, K, N))

#M=5
#N=16384
#K=8192
M=5
N=4096
K=3524

# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = 'llvm -mcpu=skylake-avx512'
ctx = tvm.context(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)

np_repeat = 1000
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)
print("Numpy running time: %f" % (np_runing_time / np_repeat))

answer = numpy.dot(a.asnumpy(), b.asnumpy())

@autotvm.template('matmul')
def matmul():
    # Algorithm
    k = te.reduce_axis((0, K), 'k')
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", M, num_outputs=3)
    cfg.define_split("tile_y", N, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=2)
    ##### define space end #####

    # We have to re-write the algorithm slightly.
    bn = cfg["tile_y"].size[-1]
    packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
    C = te.compute((M, N),
                    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, y % bn], axis=k),
                    name = 'C')
    s = te.create_schedule(C.op)
    x, y = s[C].op.axis
    k, = s[C].op.reduce_axis

    # schedule according to config
    # Allocate write cache
    CC = s.cache_write(C, 'global')
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    s[C].reorder(xt, yt, xo, yo, xi, yi)
    xyt = s[C].fuse(xt, yt)
    # parallel
    s[C].parallel(xyt)
    xyo = s[C].fuse(xo, yo)
    s[C].unroll(xi)
    s[C].vectorize(yi)

    # Write cache is computed at xyo
    s[CC].compute_at(s[C], xyt)

    # New inner axes
    xc, yc = s[CC].op.axis

    k, = s[CC].op.reduce_axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, xc)
    code = tvm.lower(s, [A, B, C], simple_mode=True)
    cfg.define_reorder("reorder", [ko, xc, ki, yc], "all")
    cfg["reorder"].apply(s, CC, [ko, xc, ki, yc])
    cfg.define_annotate('ann', [ko, xc, ki, yc], policy='try_unroll_vec')
    cfg['ann'].apply(s, CC, [ko, xc, ki, yc])


    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)

    return s, [A, B, C]


task = autotvm.task.create('matmul', args=[], target=target)


measure_option = autotvm.measure_option(
    #builder='local',
     builder=autotvm.LocalBuilder(n_parallel=56), 
     runner=autotvm.LocalRunner(number=3))

# begin tuning, log records to file `matmul.log`
#tuner = autotvm.tuner.XGBTuner(task, argsDict=None)
#tuner = autotvm.tuner.XGBTuner(task)
#tuner = autotvm.tuner.RandomTuner(task)
tuner = autotvm.tuner.GridSearchTuner(task)
n_trial = 4000
early_stopping = None
if os.path.exists('matmul_skx.log.tmp'):
    os.remove('matmul_skx.log.tmp')
tuner.tune(n_trial=n_trial,
           early_stopping=early_stopping,        
           measure_option=measure_option,
           callbacks=[autotvm.callback.progress_bar(n_trial),
                       autotvm.callback.log_to_file('matmul_skx.log.tmp')])
# pick best records to a cache file
autotvm.record.pick_best('matmul_skx.log.tmp', 'matmul_skx.log')

with autotvm.apply_history_best('matmul_skx.log'):
    with tvm.target.create('llvm -mcpu=skylake-avx512'):
        s, arg_buf = matmul()
        func = tvm.build(s, arg_buf)
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)


func(a, b, c)
#tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
# print(func.get_source("asm"))

evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
# print('TVM: %f' % evaluator(a, b, c).mean)
print('time: %f ms, GFLOPS: %f' % (evaluator(a, b, c).mean * 1000, 2 * M * N * K / evaluator(a, b, c).mean / 1e9))
print(tvm.lower(s, arg_buf, simple_mode=True))
