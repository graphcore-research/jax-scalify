# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List

from jax.stages import Compiled, Lowered


@dataclass
class HloOperationInfo:
    """HLO module operation (raw) info.

    Parse from raw `as_text` compiled HloModule.

    Args:
        cmd: Raw HLO operation (function + inputs/outputs).
        metadata: JAX metadata (line, ...)
        backend_config: Optional backend config dictionary.
    """

    cmd: str
    indent: int = 0
    metadata: str | None = None
    backend_config: Dict[Any, Any] | None = None

    def as_text(self, metadata: bool = False, backend_cfg: bool = False, indent: int = 2) -> str:
        """Convert to raw text, with formatting issues."""
        indent_txt = " " * (indent * self.indent)
        line = indent_txt + self.cmd
        if backend_cfg and self.backend_config:
            # A bit hacky text formating of backend config!
            backend_cfg_raw = json.dumps(self.backend_config, indent=indent)
            backend_cfg_raw = "backend_cfg: " + backend_cfg_raw
            backend_cfg_raw = textwrap.indent(backend_cfg_raw, indent_txt + " " * indent)
            line += "\n" + backend_cfg_raw
        return line


def parse_hlo_operation_raw_line(raw_line: str) -> HloOperationInfo:
    """Very crude and ugly parsing of an Hlo operation raw line!

    Returns:
        Parsed Hlo operation line.
    """
    metadata: str | None = None
    backend_cfg = None

    # Parse "metadata={...}" block.
    metadata_prefix = ", metadata={"
    lidx = raw_line.find(metadata_prefix)
    if lidx >= 0:
        ridx = raw_line[lidx:].find("}") + lidx
        metadata = raw_line[lidx : ridx + 1]
        raw_line = raw_line.replace(metadata, "")
        metadata = metadata[2:]

    # Parse "backend_config={...}" block.
    backend_cfg_prefix = ", backend_config="
    lidx = raw_line.find(backend_cfg_prefix)
    if lidx >= 0:
        backend_cfg_str = raw_line[lidx + len(backend_cfg_prefix) :]
        # TODO: deal with exception raised.
        backend_cfg = json.loads(backend_cfg_str)
        raw_line = raw_line[:lidx]

    # Clean the raw line.
    raw_line = raw_line.rstrip()
    size = len(raw_line)
    raw_line = raw_line.lstrip()
    indent = (size - len(raw_line)) // 2
    return HloOperationInfo(raw_line, indent, metadata, backend_cfg)


def parse_hlo_module(module: Lowered | Compiled) -> List[HloOperationInfo]:
    """Parse an Hlo module, to be human-readable!

    Note: `m.hlo_modules()[0].computations()[0].render_html()`
        is also generating a nice HTML output!

    Args:
        module: HLO module or JAX stages compiled instance.
    Returns:
        List of HLO operation info.
    """
    assert isinstance(module, (Lowered, Compiled))
    if isinstance(module, Lowered):
        module = module.compile()
    module_raw_txt = module.as_text()
    module_lines = module_raw_txt.split("\n")
    ops = [parse_hlo_operation_raw_line(line) for line in module_lines]
    return ops


def print_hlo_module(
    module: Lowered | Compiled, metadata: bool = False, backend_cfg: bool = False, indent: int = 2
) -> None:
    """Human-readable Hlo module printing.

    Args:
        module: AOT Lowered or Compiled JAX module.
        metadata: Print op metadata as well.
        backend_cfg: Print op backend config as well.
    """
    cmds = parse_hlo_module(module)
    for c in cmds:
        print(c.as_text(metadata=metadata, backend_cfg=backend_cfg, indent=indent))
