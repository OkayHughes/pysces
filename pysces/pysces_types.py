from typing import Dict, NewType
from jaxtyping import Array, Float, Int64

ProcIdx = NewType("ProcIdx", int)

ElemIdx = NewType("ElemIdx", int)
ElemIdxLocal = NewType("ElemIdxLocal", ElemIdx)
ElemIdxGlobal = NewType("ElemIdxGlobal", ElemIdx)
GllIdx = NewType("GllIdx", int)
GllIdxI = NewType("GllIdxI", GllIdx)
GllIdxJ = NewType("GllIdxJ", GllIdx)

VertRedundancyLocal = Dict[ElemIdxLocal, Dict[tuple[GllIdxI, GllIdxJ], list[tuple[ElemIdxLocal, GllIdxI, GllIdxJ]]]]
VertRedundancyRemote = Dict[ProcIdx, list[tuple[ElemIdxLocal, GllIdxI, GllIdxJ]]]

AssemblyTriple = tuple[Float[Array, "point_idx"], Int64[Array, "point_idx"], Int64[Array, "point_idx"]]
Decomp = list[tuple[int]]
