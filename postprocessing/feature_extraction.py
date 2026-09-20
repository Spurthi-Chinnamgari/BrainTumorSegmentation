import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area
import math


# ============================================================
# FILE
# ============================================================

INPUT_MASK = "postprocessed/cleaned_prediction_mask.nii.gz"


# ============================================================
# LOAD MASK
# ============================================================

nii = nib.load(INPUT_MASK)

mask = nii.get_fdata().astype(np.int16)

spacing = np.asarray(nii.header.get_zooms()[:3], dtype=float)
affine = nii.affine

print("\n" + "=" * 70)
print("                 TUMOR FEATURE REPORT")
print("=" * 70)


# ============================================================
# REGIONS
# ============================================================

regions = {
    "WT": {
        "name": "Whole Tumor",
        "labels": [1, 2, 3]
    },

    "TC": {
        "name": "Tumor Core",
        "labels": [1, 3]
    },

    "ET": {
        "name": "Enhancing Tumor",
        "labels": [3]
    }
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(region_mask):

    voxel_count = np.count_nonzero(region_mask)

    if voxel_count == 0:
        return None

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    voxel_volume = np.prod(spacing)

    volume_mm3 = voxel_count * voxel_volume
    volume_cm3 = volume_mm3 / 1000.0


    # --------------------------------------------------------
    # TUMOR SIZE
    # --------------------------------------------------------

    coords = np.argwhere(region_mask)

    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)

    dimensions_voxels = maximum - minimum + 1

    dimensions_mm = dimensions_voxels * spacing

    length = dimensions_mm[0]
    width = dimensions_mm[1]
    height = dimensions_mm[2]


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    centroid_voxel = ndimage.center_of_mass(region_mask)

    centroid_mm = nib.affines.apply_affine(
        affine,
        centroid_voxel
    )


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    location = (
        f"({centroid_mm[0]:.2f}, "
        f"{centroid_mm[1]:.2f}, "
        f"{centroid_mm[2]:.2f}) mm"
    )


    # --------------------------------------------------------
    # SHAPE - ELONGATION
    # --------------------------------------------------------

    physical_coords = coords * spacing

    centered = (
        physical_coords -
        np.mean(physical_coords, axis=0)
    )

    if len(physical_coords) > 2:

        covariance = np.cov(
            centered,
            rowvar=False
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        eigenvalues = np.maximum(
            eigenvalues,
            1e-12
        )

        eigenvalues = np.sort(
            eigenvalues
        )

        elongation = math.sqrt(
            eigenvalues[-1] /
            eigenvalues[0]
        )

    else:

        elongation = 1.0


    # --------------------------------------------------------
    # SURFACE AREA
    # --------------------------------------------------------

    try:

        vertices, faces, _, _ = marching_cubes(
            region_mask.astype(np.uint8),
            level=0.5,
            spacing=spacing
        )

        surface_area = mesh_surface_area(
            vertices,
            faces
        )

    except Exception:

        surface_area = 0.0


    # --------------------------------------------------------
    # SPHERICITY
    # --------------------------------------------------------

    if surface_area > 0:

        sphericity = (
            (math.pi ** (1 / 3)) *
            ((6 * volume_mm3) ** (2 / 3)) /
            surface_area
        )

        sphericity = min(
            max(sphericity, 0.0),
            1.0
        )

    else:

        sphericity = 0.0


    # --------------------------------------------------------
    # COMPACTNESS
    # --------------------------------------------------------

    bounding_box_volume = (
        length *
        width *
        height
    )

    if bounding_box_volume > 0:

        compactness = (
            volume_mm3 /
            bounding_box_volume
        )

    else:

        compactness = 0.0


    return {
        "volume_cm3": volume_cm3,
        "length": length,
        "width": width,
        "height": height,
        "centroid": centroid_mm,
        "location": location,
        "elongation": elongation,
        "sphericity": sphericity,
        "compactness": compactness
    }


# ============================================================
# EXTRACT ALL THREE REGIONS
# ============================================================

results = {}

for region_code, information in regions.items():

    region_mask = np.isin(
        mask,
        information["labels"]
    )

    results[region_code] = extract_features(
        region_mask
    )


# ============================================================
# 1. TUMOR PRESENCE
# ============================================================

tumor_exists = np.any(mask > 0)

print("\n")
print("1. TUMOR PRESENCE")
print("-" * 70)

print(
    "Tumor Detected :",
    "YES" if tumor_exists else "NO"
)


# ============================================================
# 2. TUMOR VOLUME
# ============================================================

print("\n")
print("2. TUMOR VOLUME")
print("-" * 70)

for code in ["WT", "TC", "ET"]:

    result = results[code]

    name = regions[code]["name"]

    if result is not None:

        print(
            f"{code} ({name})"
            f" : {result['volume_cm3']:.2f} cm³"
        )

    else:

        print(
            f"{code} ({name})"
            f" : 0.00 cm³"
        )


# ============================================================
# 3–6. INDIVIDUAL REGION DETAILS
# ============================================================

for code in ["WT", "TC", "ET"]:

    result = results[code]

    name = regions[code]["name"]


    print("\n")
    print("=" * 70)
    print(f"{code} — {name.upper()}")
    print("=" * 70)


    if result is None:

        print("\nTumor region not detected.")
        continue


    # --------------------------------------------------------
    # TUMOR SIZE
    # --------------------------------------------------------

    print("\n3. TUMOR SIZE")
    print("-" * 70)

    print(
        "Length × Width × Height : "
        f"{result['length']:.2f} × "
        f"{result['width']:.2f} × "
        f"{result['height']:.2f} mm"
    )


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    print("\n4. CENTROID")
    print("-" * 70)

    c = result["centroid"]

    print(
        f"(X, Y, Z) : "
        f"({c[0]:.2f}, "
        f"{c[1]:.2f}, "
        f"{c[2]:.2f}) mm"
    )


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    print("\n5. LOCATION")
    print("-" * 70)

    print(
        "Tumor Center :",
        result["location"]
    )


    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    print("\n6. SHAPE")
    print("-" * 70)

    print(
        f"Elongation  : "
        f"{result['elongation']:.2f}"
    )

    print(
        f"Sphericity  : "
        f"{result['sphericity']:.2f}"
    )

    print(
        f"Compactness : "
        f"{result['compactness']:.2f}"
    )


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    print("\n7. VOLUME")
    print("-" * 70)

    print(
        f"{result['volume_cm3']:.2f} cm³"
    )


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

print("\n")
print("=" * 70)
print("                 FINAL TUMOR SUMMARY")
print("=" * 70)

print(
    f"{'Region':<8}"
    f"{'Volume(cm³)':<15}"
    f"{'Size(mm)':<30}"
)

print("-" * 70)

for code in ["WT", "TC", "ET"]:

    result = results[code]

    if result is None:

        print(
            f"{code:<8}"
            f"{0:<15.2f}"
            f"{'Not detected':<30}"
        )

    else:

        size = (
            f"{result['length']:.1f} × "
            f"{result['width']:.1f} × "
            f"{result['height']:.1f}"
        )

        print(
            f"{code:<8}"
            f"{result['volume_cm3']:<15.2f}"
            f"{size:<30}"
        )


print("\n")
print("=" * 70)
print("             FEATURE EXTRACTION COMPLETED")
print("=" * 70)