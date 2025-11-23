"""
Utility to scale a MuJoCo XML model (positions, sizes, masses, inertias, and mesh scaling).

Usage:
  python tools/scale_mujoco_xml.py --input assets/robots/g1/g1.xml --output assets/robots/g1/g1_scaled.xml --scale 1.2

Default scaling rules:
  - linear scale: `scale` (applied to positions and sizes)
  - mass scale: scale**3 (can be overridden with --mass_scale)
  - inertia (diaginertia) scale: scale**5 (can be overridden with --inertia_scale)

The script will also add `scale="s s s"` to every <mesh> asset element when requested.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import argparse


def _scale_floats(s: str, factor: float) -> str:
    parts = s.strip().split()
    out = []
    for p in parts:
        try:
            v = float(p)
            scaled = v * factor
            out.append(f"{scaled:.6f}".rstrip("0").rstrip("."))
        except Exception:
            out.append(p)
    return " ".join(out)


def _scale_qpos(s: str, pos_factor: float) -> str:
    parts = s.strip().split()
    out = []
    for i, p in enumerate(parts):
        try:
            v = float(p)
            if i < 3:
                out.append(f"{(v * pos_factor):.6f}".rstrip("0").rstrip("."))
            else:
                out.append(p)
        except Exception:
            out.append(p)
    return " ".join(out)


def scale_xml(
    input_xml: str,
    output_xml: str,
    scale: float = 1.0,
    mass_scale: float = None,
    inertia_scale: float = None,
    ctrl_scale: float = None,
    add_mesh_scale: bool = True,
):
    """Scale a MuJoCo XML file and write the result.

    Args:
        input_xml: path to original xml
        output_xml: path to write scaled xml
        scale: linear scaling factor (positions, sizes, mesh scale)
        mass_scale: override mass scaling (default scale**3)
        inertia_scale: override inertia scaling for `diaginertia` (default scale**5)
        ctrl_scale: control/force scaling factor for ctrlrange and actuatorfrcrrange (default 1.0)
        add_mesh_scale: add `scale` attribute to mesh elements
    """
    in_path = Path(input_xml)
    out_path = Path(output_xml)

    if not in_path.exists():
        raise FileNotFoundError(f"Input XML not found: {in_path}")

    if mass_scale is None:
        mass_scale = scale**3
    if inertia_scale is None:
        inertia_scale = scale**5
    if ctrl_scale is None:
        ctrl_scale = 1.0

    text = in_path.read_text()
    root = ET.fromstring(text)

    # Add scale attribute to mesh elements if requested
    if add_mesh_scale:
        for mesh in root.findall(".//mesh"):
            # if mesh already has a scale attribute, override it
            mesh.set("scale", f"{scale} {scale} {scale}")

    for elem in root.iter():
        # Copy keys to avoid runtime mutation while iterating
        for attr in list(elem.attrib.keys()):
            val = elem.attrib[attr]
            if attr == "pos" or attr.endswith("pos"):
                elem.attrib[attr] = _scale_floats(val, scale)
            elif attr == "size":
                elem.attrib[attr] = _scale_floats(val, scale)
            elif attr == "mass":
                try:
                    m = float(val)
                    elem.attrib[attr] = f"{(m * mass_scale):.6f}".rstrip("0").rstrip(
                        "."
                    )
                except Exception:
                    pass
            elif attr == "diaginertia":
                elem.attrib[attr] = _scale_floats(val, inertia_scale)
            elif attr == "qpos":
                elem.attrib[attr] = _scale_qpos(val, scale)
            elif attr in ("ctrlrange", "actuatorfrcrange"):
                elem.attrib[attr] = _scale_floats(val, ctrl_scale)
            # leave other attributes as-is (quat, ranges, etc.)

    try:
        ET.indent(root, space="  ")
    except Exception:
        pass

    out_path.write_text(ET.tostring(root, encoding="unicode"))
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input MuJoCo XML")
    parser.add_argument(
        "--output", "-o", required=True, help="Output scaled MuJoCo XML"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Linear scale factor (positions/sizes)"
    )
    parser.add_argument(
        "--mass_scale",
        type=float,
        default=None,
        help="Mass scale factor (default scale**3)",
    )
    parser.add_argument(
        "--inertia_scale",
        type=float,
        default=None,
        help="Inertia scale factor (default scale**5)",
    )
    parser.add_argument(
        "--ctrl_scale",
        type=float,
        default=None,
        help="Control/force scale factor (default 1.0)",
    )
    parser.add_argument(
        "--no_mesh_scale", action="store_true", help="Do not add mesh scale attribute"
    )
    args = parser.parse_args()

    out = scale_xml(
        args.input,
        args.output,
        scale=args.scale,
        mass_scale=args.mass_scale,
        inertia_scale=args.inertia_scale,
        ctrl_scale=args.ctrl_scale,
        add_mesh_scale=(not args.no_mesh_scale),
    )
    print("Wrote", out)
