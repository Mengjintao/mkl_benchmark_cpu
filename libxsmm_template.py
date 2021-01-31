import tvm
from tvm import te
import numpy as np
import sys
from tvm import testing
from tvm import autotvm
import os

import sys

M=int(sys.argv[1])
K=int(sys.argv[2])
N=int(sys.argv[3])

print(M)
print(K)
print(N)

#print(type(M))
#M, K, N = 5, 8192, 16384

dtype = "float32"

target = "llvm -mcpu=skylake-avx512"
ctx = tvm.context(target, 0)

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)

answer = np.dot(a.asnumpy(), b.asnumpy())

@autotvm.template('matmul')
def matmul():
  k = te.reduce_axis((0, K), 'k')
  A = te.placeholder((M, K), name='A')

  cfg = autotvm.get_config()
  cfg.define_split('tile_x', M, num_outputs=3)
  cfg.define_split('tile_y', N, num_outputs=3)
  cfg.define_split('tile_k', K, num_outputs=3)

  bn = cfg['tile_y'].size[-1]
  packedB = te.placeholder((N // bn, K, bn), name='packedB')
 # packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
  C = te.compute((M, N),
                  lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, y % bn], axis=k),
                  name="C")
 # C = te.compute((M, N),
 #                lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
 #                name="C")
  s = te.create_schedule(C.op)
  x, y = s[C].op.axis
  k, = s[C].op.reduce_axis

  xt, xo, xi = cfg["tile_x"].apply(s, C, x)
  yt, yo, yi = cfg["tile_y"].apply(s, C, y)
  kt, ko, ki = cfg["tile_k"].apply(s, C, k)
  s[C].reorder(kt, xo, yt, xt, yo, ko, xi, yi, ki)

  cfg.define_reorder("reorder", [kt, xo, yt, xt, yo, ko], "all")
  new_order = cfg["reorder"].apply(s, C, [kt, xo, yt, xt, yo, ko])

  sibling_axes = []
  first_non_k_axis_met = False
  for axis in new_order:
    if not first_non_k_axis_met:
      if axis not in [kt, ko]:
        sibling_axes.append(axis)
        first_non_k_axis_met = True
    else:
      if axis in [kt, ko]:
        break
      sibling_axes.append(axis)
      if len(sibling_axes) >= 2:
        break

  assert len(sibling_axes) <= 2
  parallel_axis = None
  if len(sibling_axes) == 1:
    parallel_axis = sibling_axes[0]
  else:
    parallel_axis = s[C].fuse(*sibling_axes)

  assert parallel_axis is not None
  s[C].parallel(parallel_axis)
    
  

  #xoytxt = s[C].fuse(xo, yt, xt)
  
  #s[C].parallel(xoytxt)

 # x, y, z = s[packedB].op.axis
 # s[packedB].vectorize(z)
 # s[packedB].parallel(x)
 # s[C].parallel(yt);

  def intrin_libxsmm(m, k, n):
    a = te.placeholder((m, k), name='a')
    b = te.placeholder((k, n), name='b')
    k = te.reduce_axis((0, k), name='k')
    c = te.compute((m, n), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name='c')
    a_buffer = tvm.tir.decl_buffer(a.shape, a.dtype, name='a_buffer', offset_factor=1, strides=[te.var('s1'), 1])#[te.var('s1'), te.var('s11')])
    b_buffer = tvm.tir.decl_buffer(b.shape, b.dtype, name='b_buffer', offset_factor=1, strides=[te.var('s2'), 1])
    c_buffer = tvm.tir.decl_buffer(c.shape, c.dtype, name='c_buffer', offset_factor=1, strides=[te.var('s3'), 1])

    def intrin_func(ins, outs):
      def _body():
        ib = tvm.tir.ir_builder.create()
        ib.emit(
          tvm.tir.call_packed(
            "tvm.contrib.libxsmm.matmul", ins[0], ins[1], outs[0], False, False
          )
        )
        return ib.get()
      def _update():
        ib = tvm.tir.ir_builder.create()
        ib.emit(
          tvm.tir.call_packed(
            "tvm.contrib.libxsmm.matmul", ins[0], ins[1], outs[0], False, False, 1, 1
          )
        )
        return ib.get()


      return _body(), None, _update()
    
    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: a_buffer, b: b_buffer, c: c_buffer})

  micro_kernel = intrin_libxsmm(cfg["tile_x"].size[-1], cfg["tile_k"].size[-1], cfg["tile_y"].size[-1])
  #s[C].tensorize(k, micro_kernel)
  s[C].tensorize(xi, micro_kernel)

  return s, [A, packedB, C]

task = autotvm.task.create('matmul', args=[], target=target)

@autotvm.template('fake_matmul')
def matmul():
  k = te.reduce_axis((0, K), 'k')
  A = te.placeholder((M, K), name='A')

  cfg = autotvm.get_config()
  cfg.define_split('tile_x', M, num_outputs=3)
  cfg.define_split('tile_y', N, num_outputs=3)
  cfg.define_split('tile_k', K, num_outputs=3)

  bn = cfg['tile_y'].size[-1]
  packedB = te.placeholder((N // bn, K, bn), name='packedB')
 # packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
  C = te.compute((M, N),
                  lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, y % bn], axis=k),
                  name="C")
 # C = te.compute((M, N),
 #                lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
 #                name="C")
  s = te.create_schedule(C.op)
  x, y = s[C].op.axis
  k, = s[C].op.reduce_axis

  xt, xo, xi = cfg["tile_x"].apply(s, C, x)
  yt, yo, yi = cfg["tile_y"].apply(s, C, y)
  kt, ko, ki = cfg["tile_k"].apply(s, C, k)
  s[C].reorder(kt, xo, yt, xt, yo, ko, xi, yi, ki)

  cfg.define_reorder("reorder", [kt, xo, yt, xt, yo, ko], "all")
  new_order = cfg["reorder"].apply(s, C, [kt, xo, yt, xt, yo, ko])

  sibling_axes = []
  first_non_k_axis_met = False
  for axis in new_order:
    if not first_non_k_axis_met:
      if axis not in [kt, ko]:
        sibling_axes.append(axis)
        first_non_k_axis_met = True
        break
    else:
      if axis in [kt, ko]:
        break
      sibling_axes.append(axis)

  assert len(sibling_axes) == 1
  parallel_axis = None
  if len(sibling_axes) == 1:
    parallel_axis = sibling_axes[0]
  else:
    parallel_axis = s[C].fuse(*sibling_axes)

  assert parallel_axis is not None
  s[C].parallel(parallel_axis)
    
  

  #xoytxt = s[C].fuse(xo, yt, xt)
  
  #s[C].parallel(xoytxt)

 # x, y, z = s[packedB].op.axis
 # s[packedB].vectorize(z)
 # s[packedB].parallel(x)
 # s[C].parallel(yt);

  def intrin_libxsmm(m, k, n):
    a = te.placeholder((m, k), name='a')
    b = te.placeholder((k, n), name='b')
    k = te.reduce_axis((0, k), name='k')
    c = te.compute((m, n), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name='c')
    a_buffer = tvm.tir.decl_buffer(a.shape, a.dtype, name='a_buffer', offset_factor=1, strides=[te.var('s1'), 1])#[te.var('s1'), te.var('s11')])
    b_buffer = tvm.tir.decl_buffer(b.shape, b.dtype, name='b_buffer', offset_factor=1, strides=[te.var('s2'), 1])
    c_buffer = tvm.tir.decl_buffer(c.shape, c.dtype, name='c_buffer', offset_factor=1, strides=[te.var('s3'), 1])

    def intrin_func(ins, outs):
      def _body():
        ib = tvm.tir.ir_builder.create()
        ib.emit(
          tvm.tir.call_packed(
            "tvm.contrib.libxsmm.matmul", ins[0], ins[1], outs[0], False, False
          )
        )
        return ib.get()
      def _update():
        ib = tvm.tir.ir_builder.create()
        ib.emit(
          tvm.tir.call_packed(
            "tvm.contrib.libxsmm.matmul", ins[0], ins[1], outs[0], False, False, 1, 1
          )
        )
        return ib.get()


      return _body(), None, _update()
    
    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: a_buffer, b: b_buffer, c: c_buffer})

  micro_kernel = intrin_libxsmm(cfg["tile_x"].size[-1], cfg["tile_k"].size[-1], cfg["tile_y"].size[-1])
  #s[C].tensorize(k, micro_kernel)
  s[C].tensorize(xi, micro_kernel)

  return s, [A, packedB, C]

fake_task = autotvm.task.create('fake_matmul', args=[], target=target)


measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(n_parallel=24), runner=autotvm.LocalRunner(number=5, timeout=20))

#tunner = autotvm.tuner.XGBTuner(task, fake_task)
tunner = autotvm.tuner.RandomTuner(task)
n_trial = 500
early_stopping = 800

if os.path.exists('matmul.log.tmp'):
  os.remove('matmul.log.tmp')
tunner.tune(n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[autotvm.callback.progress_bar(n_trial), 
                       autotvm.callback.log_to_file('matmul.log.tmp')])
