import pandas as pd
from pathlib import Path

tree_id = "SERC_usb1_job28_LITU"
in_dir = Path(r"E:\TLS\2025_summer\2025_SERC\Tree_scan\ForestGEO_SERC\L1_segmented") / tree_id / "L2" / "TLSeparation_test"

wood = pd.read_csv(in_dir / f"{tree_id}_wood.xyz", sep=r"\s+", header=None, names=["x","y","z"], engine="python")
leaf = pd.read_csv(in_dir / f"{tree_id}_leaf.xyz", sep=r"\s+", header=None, names=["x","y","z"], engine="python")

wood["class"] = 0
leaf["class"] = 1

out = pd.concat([wood, leaf], ignore_index=True)
out.to_csv(in_dir / f"{tree_id}_wood_leaf_labeled.csv", index=False)
print("Wrote:", in_dir / f"{tree_id}_wood_leaf_labeled.csv")


csv_path = Path(r"E:\TLS\2025_summer\2025_SERC\Tree_scan\ForestGEO_SERC\L1_segmented\SERC_usb1_job28_LITU\L2\TLSeparation_test\SERC_usb1_job28_LITU_wood_leaf_labeled.csv")

df = pd.read_csv(csv_path)


# keep only valid rows
df = df[["x", "y", "z", "class"]].copy()
for c in ["x", "y", "z", "class"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

wood = df[df["class"] == 0][["x", "y", "z"]].copy()
leaf = df[df["class"] == 1][["x", "y", "z"]].copy()

print("Wood points:", len(wood))
print("Leaf points:", len(leaf))

if len(wood) > 0:
    print("Wood XYZ min:", wood.min().to_dict())
    print("Wood XYZ max:", wood.max().to_dict())

if len(leaf) > 0:
    print("Leaf XYZ min:", leaf.min().to_dict())
    print("Leaf XYZ max:", leaf.max().to_dict())


def write_ascii_ply_xyz(path, arr, rgb=None):
    """
    Write Nx3 XYZ or Nx6 XYZRGB ASCII PLY
    rgb should be a tuple like (255,0,0) or None
    """
    n = arr.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if rgb is None:
            for x, y, z in arr:
                f.write(f"{x} {y} {z}\n")
        else:
            r, g, b = rgb
            for x, y, z in arr:
                f.write(f"{x} {y} {z} {r} {g} {b}\n")


out_dir = csv_path.parent

wood_ply = out_dir / "SERC_usb1_job28_LITU_wood_only.ply"
leaf_ply = out_dir / "SERC_usb1_job28_LITU_leaf_only.ply"
combined_ply = out_dir / "SERC_usb1_job28_LITU_wood_leaf_colored.ply"

write_ascii_ply_xyz(wood_ply, wood[["x", "y", "z"]].to_numpy(), rgb=(139, 69, 19))   # brown
write_ascii_ply_xyz(leaf_ply, leaf[["x", "y", "z"]].to_numpy(), rgb=(34, 139, 34))   # green

combined = pd.concat([
    wood.assign(red=139, green=69, blue=19),
    leaf.assign(red=34, green=139, blue=34)
], ignore_index=True)

with open(combined_ply, "w", encoding="utf-8") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(combined)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for row in combined.itertuples(index=False):
        f.write(f"{row.x} {row.y} {row.z} {int(row.red)} {int(row.green)} {int(row.blue)}\n")

print("Wrote:")
print(wood_ply)
print(leaf_ply)
print(combined_ply)