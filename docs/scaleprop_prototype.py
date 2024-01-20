#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import signal
from types import MethodWrapperType, GetSetDescriptorType
from typing import *
from warnings import warn
from functools import partial
from dataclasses import dataclass
from icecream import ic

import torch
from torch import Tensor, tensor, nn
from torch.utils._pytree import tree_map

# Make warn=print for notebooks, as otherwise outputs are not interleaved correctly
from termcolor import colored
def warn(msg): print(colored('WARNING', 'yellow'), f': {msg}')
warn('Nothing to worry about')

def numeric_info(dtype):
  return torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)

def dtype_max(dtype):
  return numeric_info(dtype).max

def _round(x : Tensor, dtype : torch.dtype) -> Tensor:
  """
  Convert to dtype, rounding if the destination is integer
  """
  if dtype.is_floating_point:
    return x.to(dtype)
  else:
    return torch.round(x).to(dtype)

def function_str(func : Callable) -> str:
  # https://stackoverflow.com/questions/251464/how-to-get-a-function-name-as-a-string
  # for future expansion e.g. properties
  if hasattr(func, '__module__'):
    return func.__module__ + '.' + func.__qualname__
  else:
    return func.__qualname__
  
def type_str(x : type) -> str:
   return x.__name__

def _uptype(dtype) -> torch.dtype:
    """
    For DTYPE, what is the type in which most arithmetic (e.g. max, abs) is defined?
    """
    GPU = False # Check more accurately, and choose bf16 as appropriate
    f16_t = torch.float16 if GPU else torch.float32
    map = {
        torch.int8: torch.int16,
        torch.int16: torch.int16,
        torch.float8_e4m3fn: f16_t,
        torch.float8_e5m2: f16_t,
        torch.float16: f16_t,
        torch.float32: torch.float32,
    }
    return map[dtype]

def _to_uptype(t : Tensor) -> Tensor:
    """
    Convert t to its _uptype
    """
    return torch.as_tensor(t, dtype=_uptype(t.dtype))


def _maxval(t : Tensor):
    """
    Max absolute value of tensor, returned in its `_uptype`
    """
    return _to_uptype(t).abs().max()

torch.set_printoptions(precision=3, threshold=32)

# From https://github.com/albanD/subclass_zoo/blob/main/utils.py
import contextlib
@contextlib.contextmanager
def _no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard



# In[2]:


def tensor_oneline_str(t):
  shape_str = "x".join(map(str, t.shape))
  quantiles = torch.tensor([0, 0.05,.25, .5, .74, 1.0])
  vals = torch.quantile(torch.flatten(t).to(torch.float32), quantiles, interpolation = 'nearest')
  if t.dtype.is_floating_point:
    # scale down vals
    max = torch.floor(torch.log10(vals.abs().max()))
    if -2 <= max <= 3:
      max = 0
    max_scale = 10 ** -max
    max_scale_str = f"10^{int(max)} x " if max != 0 else ""
    vals_str = max_scale_str + "Quants{" + "|".join(f'{v:.3f}' for v in vals*max_scale) + "}"
  else:
    # Assume integer, print as integers
    vals_str = "Quants{" + "|".join(f'{int(v)}' for v in vals) + "}"

  classname = type(t).__name__

  return f'{classname}({shape_str}) {vals_str}'

for scale in [-10, -3, -2, -1, 0, 1, 2, 3, 10]:
  kurt = 3
  print(tensor_oneline_str(torch.randn(100,300)**kurt*(10**scale)))

print(tensor_oneline_str((torch.randn(100,300) * 10000).to(torch.int16)))


# ## 1. Core functionality, no syntactic sugar
# 
# Define a `ScaleTensorData` object, with the minimal information, and with operations
# defined cleanly, but without PyTorch Tensor integration

# In[3]:


@dataclass
class ScaledTensorData:
    data: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.scale, Tensor):
            self.scale = tensor(self.scale)
        assert self.scale.dtype == torch.float32
        assert self.scale.shape == () 
        # Possible future expansion to e.g. row-scaled, column-scaled, etc, but
        # for now, insist st_scale is a single-element tensor

    def _contents_str(self) -> str:
        return f"{self.scale} * {tensor_oneline_str(self.data)}"

    def __repr__(self) -> str:
        return f"ScaledTensorData({self._contents_str()})"

    def to_tensor(self, dtype: torch.dtype = None) -> Tensor:
        dtype = dtype or self.scale.dtype
        return self.data.to(dtype) * self.scale.to(dtype)

    @property
    def shape(self) -> torch.Size:
        return self.data.shape
    
    @property
    def dtype_max(self):
        return dtype_max(self.data.dtype)

def st_quantise(x: Tensor, dtype: torch.dtype) -> ScaledTensorData:
    """
    Rescale so that max(|data|) == maxFinite(data.dtype)
    """
    maxval = _maxval(x)
    if maxval == 0:
        # Tensor is all zeros - set scale to 1
        scale = Tensor(1.0, dtype=torch.float32)
    else:
        # Scale so that largest element is the largest finite value of dtype
        scale = maxval / dtype_max(dtype)

    return ScaledTensorData(_round(x / scale, dtype), scale)


def st_requantise(st: ScaledTensorData) -> ScaledTensorData:
    """
    Rescale so that max(|data|) == maxFinite(data.dtype)

    Equivalent to quantise(st.to_tensor(torch.float32)) but avoids the conversion

    Returned tensor may share its data with input tensor
    """
    maxdataval = _maxval(st.data)
    if maxdataval == 0:
        # All zero, reset scale to 1
        return ScaledTensorData(st.data, st.scale)
    else:
         rescale = maxdataval / st.dtype_max
    return ScaledTensorData((_to_uptype(st.data) * (1 / rescale)).to(st.data.dtype), st.scale * rescale)


def wasted_bits(st_data, maxval = None) -> float:
    """
    By how much is tensor `st_data` not using the full dynamic range of its dtype?

    E.g.
       t = torch.tensor([1,2,-16], dtype=torch.int8)

    Is using only 5 (4 + sign) of the available 8 bits.
    Therefore 
       wasted_bits(t) == 3 == 8-3

    Optional argument maxval, if the maximum value in the tensor has already 
    been computed, perhaps with a higher-accuracy method (e.g. pre-rounding)
    """
    if maxval is None:
        maxval = _maxval(st_data)

    maxval = maxval.to(st_data.dtype)
    dtype_bits = numeric_info(st_data.dtype).bits
    if maxval == 0:
        # All values zero -> all bits are wasted
        return dtype_bits
    
    # Otherwise, how many bits is maxval using.
    if st_data.dtype.is_floating_point:
      # Convert maxval to integer of the same bitwidth
      ints = {
          8: torch.int8,
          16: torch.int16,
          32: torch.int32
      }
      maxval = maxval.view(ints[dtype_bits])

    # Assuming a signed type, max usable bits are dtype_bits-1
    return dtype_bits-1 - torch.log2(maxval)

def test_wasted_bits():
    t = torch.tensor([1,2,-16], dtype=torch.int8)
    assert wasted_bits(t) == 3
test_wasted_bits()

### Quick testing
f16 = tensor([1, 2, 3], dtype=torch.float16)
f8_t = torch.float8_e4m3fn
print(f16)
print(f16.to(f8_t))

st_data = st_quantise(f16, f8_t)

# TODO: wasting a lot more bits at the low end here -- range of 144->448.
print(st_data)
print(f'{wasted_bits(st_data.data)=} <-- of a max of {numeric_info(f8_t).bits} bits, should be wasting nearly zero')
print(st_requantise(st_data))
print(st_data.to_tensor(f16.dtype), "# <- rounding errors at the high end of the f8 range")


# ## 2. Operations, without syntactic sugar

# In[4]:


# Ops
# - Use worst-case scaling rules (no overflow!)
# - Placeholder impl (fast impl requires custom kernels)
# - No autograd support

def st_add(a: ScaledTensorData, b: ScaledTensorData) -> ScaledTensorData:
    out_dtype = a.data.dtype
    scale = a.scale + b.scale
    data = (_to_uptype(a.data) * (a.scale / scale) + _to_uptype(b.data) * (b.scale / scale)).to(out_dtype)
    return ScaledTensorData(data, scale)


def st_matmul(a: ScaledTensorData, b: ScaledTensorData, debug = True) -> ScaledTensorData:
    assert a.data.dtype == b.data.dtype
    in_dtype = a.data.dtype
    out_dtype = a.data.dtype

    a_maxval = a.scale * a.dtype_max
    b_maxval = b.scale * b.dtype_max

    # Predicted maxval for NxK @ KxM
    K = a.shape[-1]
    out_maxval_estimate = a_maxval * b_maxval * K

    out_scale = out_maxval_estimate / dtype_max(out_dtype) 

    # Derivation of matmul scale factors:
    # (ad * as) @ (bd * bs) = (ad @ bd) * (as * bs)
    #                       = (ad @ bd) * (as * bs / os * os)
    #                       = (ad @ bd * as * bs / os) * os
    #                       = (ad @ bd * rat) * os
    #                         where rat = as * bs / os
    #                       = (ad * sqrt(rat)) @ (bd * sqrt(rat)) * os

    rat = a.scale * b.scale / out_scale

    if numeric_info(in_dtype).bits < numeric_info(_uptype(in_dtype)).bits:
        # Assume low-precision muls will accumulate to uptype, so won't overflow
        # to simulate this on cpu, uptype before the matmul;
        # on appropriate hardware (e.g. graphcore, h100), call the special matmul
        adbd = _to_uptype(a.data) @ _to_uptype(b.data)
        if debug:
            out_maxval = _maxval(adbd) * (a.scale * b.scale)
        out_data = adbd * rat
        out_data = out_data.to(out_dtype)
    else:
        # Inputs are in 16+ bits, and we know the products will certainly 
        # overflow, as they are scaled to dtype_max, so downscale before multiplying 
        sqrt_rat = torch.sqrt(rat)
        a_down = _to_uptype(a.data) * sqrt_rat
        b_down = _to_uptype(b.data) * sqrt_rat
        out_data = a_down @ b_down
        if debug:
            out_maxval = _maxval(out_data) * out_scale
        out_data = out_data.to(out_dtype)

    # debug check how bad out_maxval_estimate was
    if debug:
        assert out_maxval_estimate > out_maxval # Should always be an upper bound
        wasted = wasted_bits(out_data)
        if wasted > numeric_info(out_dtype).bits/2:
            warn(f'st_matmul: Very bad maxval estimate {out_maxval_estimate} vs {out_maxval}, {out_maxval_estimate/out_maxval:.1f}x too large - will lose at least {wasted} bits of precision')

        if _maxval(out_data) == 0:
            raise ValueError("All-data zero - rerun with debug and view st_matmul: WARNING above")

    return ScaledTensorData(out_data, out_scale)


def st_relu(a: ScaledTensorData) -> ScaledTensorData:
    data = nn.functional.relu(a.data).to(a.data.dtype)
    return ScaledTensorData(data, a.scale)

# Check operators behave sensibly
st1 = ScaledTensorData(tensor(32, dtype=torch.int8), 0.5)
st2 = ScaledTensorData(tensor(64, dtype=torch.int8), 0.25)

f32 = lambda x: x.to_tensor(torch.float32) if isinstance(x, ScaledTensorData) else x.to(dtype=torch.float32)


ic(st1)
ic(st2)
ic(st_add(st1, st2))
ic(f32(st_add(st1, st2)) )
ic(f32(st1) + f32(st2))

hidden = 32
t3 = torch.randn(2, hidden)
t4 = torch.randn(hidden, 3)
st3 = st_quantise(t3, torch.int8)
st4 = st_quantise(t4, torch.int8)
ic(st3)
ic(st4)
ic(st_matmul(st3, st4))
print(f'{f32(st_matmul(st3, st4)) = } <-- quantized')
print(f'{f32(st3) @ f32(st4) = } <-- intermediate')
print(f'{t3 @ t4 = } <-- exact')

# st5 = st_quantise(tensor([-2, 3, -0.06, 4]), torch.int8)
# ic(st5, f32(st5))
# rt5 = st_relu(st5)
# ic(rt5, f32(rt5))


# In[5]:


### [Aside: Possibly surprising rounding]

# print('Starting point: ', tensor([-2, 0.09, 4]))

# st5 = st_quantise(tensor([-2, 0.09, 4]), torch.int8)
# print(f'{st5=} {f32(st5)=} <-- 0.0900 input rounds to 0.0630')

# print('So, put in 0.0630 to begin with')
# st5 = st_quantise(tensor([-2, 0.0630, 4]), torch.int8) 
# print(f'{st5=} {f32(st5)=} <-- great, 0.0630 rounds to 0.0630')

# print('But now, put in 0.06')
# st5 = st_quantise(tensor([-2, 0.06, 4]), torch.int8) 
# print(f'{st5=} {f32(st5)=} <-- Eh, 0.06 rounds down to 0.0315?')


# ## 3. Sugar hit: override Tensor operations in subclass
# 
# We define a `ScaledTensor` object that behaves like a torch Tensor.

# In[6]:


# See https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-like-type
# TODO: check
#   - _make_wrapper_subclass THPVariable_make_wrapper_subclass https://github.com/pytorch/pytorch/pull/65340
#   - _make_subclass
#   - See [Note] at https://github.com/albanD/subclass_zoo/blob/276d2f005484d80ebbcd9e274d79685adb6a1da2/negative_tensor.py#L24
#     - Doesn't apply in this case as we are composing, not deriving?
# Looking at https://github.com/albanD/subclass_zoo
#   - trivial_tensor doesn't do autograd
#   - inner_autograd_tensor explicitly defers to its `elem`, which is incorrect
# In PyTorch core
#   - MaskedTensor

class ScaledTensor(Tensor):
    @staticmethod
    def __new__(cls, st: ScaledTensorData, *, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(cls, st.data.shape, dtype=st.scale.dtype, requires_grad=requires_grad)

    def __init__(self, st_data: ScaledTensorData):
        self.st = st_data

    def tolist(self):
        return self.st.to_tensor().tolist()

    def __repr__(self, tensor_contents=None) -> str:
        assert not tensor_contents
        # See https://github.com/pytorch/pytorch/issues/73665
        with _no_dispatch():
            return super().__repr__(tensor_contents=str(self.st._contents_str()))

    @property
    def shape(self) -> torch.Size:
        return self.st.data.shape

    def size(self, *args, **kwargs) -> torch.Size:
        return self.st.data.size(*args, **kwargs)

    def sizes(self):
        return self.st.data.sizes()

    def numel(self) -> int:
        return self.st.data.numel()

    # def detach(self) -> Tensor:
    #     return ScaledTensor(ScaledTensorData(self.st.data.detach(), self.st.scale))

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Have we registered a handler?
        # 1. Turn "func" into something we can lookup in the HANDLED_FUNCTIONS dict.
        func_to_lookup = func
        if isinstance(func, MethodWrapperType): # TODO: remove this logic - sems not to be needed for dispatch
            # For methods, e.g. Tensor.T, we want to use that name, so unpack it
            assert func.__qualname__ == 'getset_descriptor.__get__' # This is the only case tested so far
            func_to_lookup = func.__self__
        else:
            # Probably other special cases here
            pass

        if func_to_lookup in cls.DELEGATE_TO_SUPER:
            # 2. Delegate to super - just pass the args upwards
            if func_to_lookup in cls.HANDLED_FUNCTIONS:  # Keep it unsurprising
                warn(f"{func} in both DELEGATE_TO_SUPER and HANDLED_FUNCTIONS")

            with _no_dispatch():
                ret = func(*args, *kwargs)
            # sanity check - if it returned a Tensor, maybe not such a good idea to silently delegate upwards
            if isinstance(ret, Tensor):
                warn(f"Delegated {func} to super, but it returned a Tensor")
            return ret

        # 3. Is this one we handle?
        extra_msg = ''
        if func_to_lookup in cls.HANDLED_FUNCTIONS:
            # Call the handler
            handler = cls.HANDLED_FUNCTIONS[func_to_lookup]
            ret = handler(func, *args, **kwargs)

            # handler may return "NotImplemented" to tell us to run the fallback
            if ret != NotImplemented:
                print('RET', ret)
                return ret

            extra_msg = f' -- [Handler {function_str(handler)} returned NotImplemented]'
            # Otherwise drop through to 4. Fallback

        # 4. Fallback: Convert to float32 and call func
        func_str = f'{function_str(func_to_lookup)}@({",".join(type_str(type(x)) for x in args)})'
        warn(f"ScaledTensor.__torch_dispatch__: Upcasting to float32 for {func_str}" + extra_msg)

        def to_tensor_if_scaled(t: Any) -> Tensor:
            if isinstance(t, ScaledTensor):  # TODO: ScaledTensor -> cls
                return t.st.to_tensor(torch.float32)
            else:
                return t

        new_args = tree_map(to_tensor_if_scaled, args)
        new_kwargs = tree_map(to_tensor_if_scaled, kwargs)

        return func(*new_args, **new_kwargs)

    # __torch_function__ = __torch_dispatch__
    __torch_function__ = torch._C._disabled_torch_function_impl


ScaledTensor.DELEGATE_TO_SUPER = {
    torch.ops.aten.is_same_size.default,
    Tensor.grad_fn,
    Tensor.requires_grad_,
    Tensor.__repr__,
    Tensor.shape,
}

ScaledTensor.HANDLED_FUNCTIONS = {}


def tensor_subclass_override(cls, funcs):
    """
    Decorator to add an implementation of an operation to a Tensor subclass

    @tensor_subclass_override(MySubclass, torch.ops.aten.view.default)
    def _(func, *args, *kwargs):
      print(f'Calling wrapped {func} with {len(args)} args)
      with _no_dispatch():
        return func(*args, *kwargs)

    Calling the implementation "_" allows this decorator to overwrite the name with
    a more sensible one "@torch_function_override(MySubclass, torch.ops.aten.view.default)"
    but of course you can just call it "MySubclass_impl_view" or "foo42" if you prefer.
    """
    funcs = funcs if isinstance(funcs, tuple) else (funcs,)
    funcs_str = ",".join(map(function_str, funcs))

    def doit(impl):
        # Override impl name if it was just "_"
        if hasattr(impl, "__name__") and impl.__name__ == "_":
            impl.__name__ = f"@torch_function_override({cls.__name__}, {funcs_str})"
            if impl.__qualname__ != "_":
                print(f"torch_function_override: NOTE: {impl.__qualname__} not overridden")

            if impl.__qualname__ == "_":
                impl.__qualname__ = f"torch_function_override({cls.__name__}, {funcs_str})"

        # Record handler in the dictionary for each func
        for func in funcs:
            cls.HANDLED_FUNCTIONS[func] = impl

    return doit


@tensor_subclass_override(ScaledTensor, (torch.ops.aten.view.default, torch.ops.aten.permute.default))
def passthru_to_data(func, t_self, *args, **kwargs):
    new_data = func(t_self.st.data, *args, **kwargs)
    return ScaledTensor(ScaledTensorData(new_data, t_self.st.scale))

@tensor_subclass_override(ScaledTensor, torch.ops.aten.detach.default)
def _(func_, a:Tensor) -> Tensor:
    return ScaledTensor(a.st)


# Forward the ScaledTensorData ops above on the ScaledTensor type
def quantise(x: Tensor, dtype: torch.dtype) -> ScaledTensor:
    return ScaledTensor(st_quantise(x, dtype))


def requantise(st) -> ScaledTensor:
    return ScaledTensor(st_requantise(st.st))


# Check basic to/from Subclass
f16 = tensor([[1, 2, 3]], dtype=torch.float16)
f8_t = torch.int8
print(f16)

st = quantise(f16, f8_t)

print(f"{st.shape=}")
assert st.shape == f16.shape
s = str(st)
print(f"{st=}")

print(f"{requantise(st)=}")
print("Rounding errors at the high end of the f8 range:", st.st.to_tensor(f16.dtype))

print("Reshaped:", st.T)

print(colored('Expect a warning...', 'yellow'))
print('Addition, but note warning above about "Upcasting to float32", so prints as a normal tensor:', st + 2)

st.requires_grad_(True)


# ### Now overrides work, but just punt up to f32 for all ops
# 
# We will do some adds/multiplies etc, and note that the torch function
# implementation issues a sequence of "WARNING: Upcasting to float32"

# In[7]:


print(colored('Expect four warnings...', 'yellow'))

print(st + 2)
print(2 * st)
print(st + st)

st3 = quantise(torch.full((2, 3), 100.0), torch.int8)
st4 = quantise(torch.full((3, 4), 200.0), torch.int8)

print(f'{f32(st3 @ st4)       = } <-- should be ~21000 quantized, but was done in f32 so exact')
print(f'{f32(st3) @ f32(st4)  = } <-- 60000 exact')


# ## Autograd

# In[8]:
import os, signal

class ScaledTensor_add(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        print('st fwd')
        if (isinstance(a, ScaledTensor) and isinstance(b, ScaledTensor)):
          return ScaledTensor(st_add(a.st, b.st))
        return a + b

    @staticmethod
    def backward(ctx, dout):
        print('st bwd')
        return dout, dout

@tensor_subclass_override(ScaledTensor, torch.ops.aten.add.Tensor)
def _(func_, a:Tensor, b: Tensor) -> Tensor:
    if not (isinstance(a, ScaledTensor) and isinstance(b, ScaledTensor)):
        return NotImplemented

    ret = ScaledTensor_add.apply(a, b)
    print('ret', ret)
    #os.kill(os.getpid(), signal.SIGTRAP)
    return ret

x = tensor([1.1, 2.2, 3.3])

qx =quantise(x, torch.int8)
qx.requires_grad_(True)

print(f'{qx=}')


print('---------------------------------------')
qy = qx + qx
print('---------------------------------------')
print(f'{qy.grad_fn=}')
print('---------------------------------------')
# qy = ScaledTensor_add.apply(qx, qx)
print(f'{qy=}')


# dx = x * .001 # no actual need to do 0.001
# qdx = quantise(dx, torch.int8)
# qy.backward(qdx)
# print(f'{qx.grad=}')
