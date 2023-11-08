from ..core import ScaledArray


def scaled_mul(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)


__all__ = ["scaled_mul"]
