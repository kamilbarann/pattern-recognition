"""Wspólne funkcje: ładowanie danych, segmentacja liścia, cechy Hougha."""
from pathlib import Path
import random

import numpy as np
from scipy import ndimage as ndi
from skimage import io, color, feature, filters, measure, morphology
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

# DATA_DIR = Path(__file__).parent / "Ziemniaki" / "Task 1 - Simple"
# CLASSES = {"zdrowe": "Potato___healthy", "chore": "Potato___Early_blight"}
DATA_DIR = Path(__file__).parent / "Ziemniaki" / "Task 2 - Hard"
CLASSES = {"obłożnie chore": "Potato___Late_blight", "chore": "Potato___Early_blight"}


# ---------------------------------------------------------------- dane

def list_paths():
    """Zwraca listę (etykieta, ścieżka) dla wszystkich obrazów Task 1."""
    return [(label, p) for label, folder in CLASSES.items()
            for p in sorted((DATA_DIR / folder).glob("*.JPG"))]


def sample_paths(n_per_class, seed=42):
    random.seed(seed)
    out = []
    for label, folder in CLASSES.items():
        paths = sorted((DATA_DIR / folder).glob("*.JPG"))
        out += [(label, p) for p in random.sample(paths, n_per_class)]
    return out


def load(path):
    return io.imread(path)


# ---------------------------------------------------------------- segmentacja

def excess_green(img):
    """ExG = 2G - R - B (na znormalizowanych kanałach) - jasne dla zieleni."""
    f = img.astype(float) / 255
    return 2 * f[..., 1] - f[..., 0] - f[..., 2]


def smooth_contour(mask, radius):
    """Wygładzenie konturu: dylatacja, potem erozja tym samym dyskiem (= zamknięcie morfologiczne).
    Większy promień = gładszy kontur, ale zalewa też głębsze wcięcia brzegu."""
    if radius <= 0:
        return mask
    fp = morphology.disk(radius)
    # margines, żeby erozja przy krawędzi obrazu nie zjadała liścia dotykającego ramki
    padded = np.pad(mask, radius, mode="edge")
    padded = morphology.erosion(morphology.dilation(padded, fp), fp)
    return padded[radius:-radius, radius:-radius]


def segment_leaf(img, smooth_radius=5):
    """Binaryzacja liścia.

    smooth_radius - promień dysku do wygładzania konturu (0 = bez wygładzania)

    Zwraca słownik:
      exg     - mapa ExG
      green   - ExG > próg Otsu (zielona tkanka; plamy choroby wypadają jako dziury)
      leaf    - maska całego liścia (największa składowa, domknięta, z wypełnionymi dziurami)
      lesions - plamy = liść, ale nie zielone
    """
    exg = excess_green(img)
    green = exg > filters.threshold_otsu(exg)

    mask = morphology.opening(green, morphology.disk(2))
    labels = measure.label(mask)
    if labels.max() > 0:
        largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
        mask = labels == largest
    mask = smooth_contour(mask, smooth_radius)
    leaf = ndi.binary_fill_holes(mask)

    lesions = leaf & ~green
    lesions = morphology.remove_small_objects(lesions, max_size=15)
    return {"exg": exg, "green": green, "leaf": leaf, "lesions": lesions}


def crop_leaf(img, leaf, pad=5, background=0):
    """Wycina liść: tło wyzerowane, kadr przycięty do bounding boxa maski."""
    out = img.copy()
    out[~leaf] = background
    rows, cols = np.nonzero(leaf)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, img.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, img.shape[1])
    return out[r0:r1, c0:c1], leaf[r0:r1, c0:c1]



# ---------------------------------------------------------------- Hough

THETAS = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)


def inner_edges(img, leaf, sigma=2.0, erode=6):
    """Canny tylko wewnątrz liścia (obrys zerodowany, żeby kontur liścia nie dominował)."""
    gray = color.rgb2gray(img)
    edges = feature.canny(gray, sigma=sigma)
    inner = morphology.erosion(leaf, morphology.disk(erode))
    return edges & inner


def hough(edges, num_peaks=10, min_distance=9, min_angle=10):
    h, theta, d = hough_line(edges, theta=THETAS)
    peaks = hough_line_peaks(h, theta, d, num_peaks=num_peaks,
                             min_distance=min_distance, min_angle=min_angle)
    return h, theta, d, peaks


def axial_diff(a, b):
    """Różnica kątów dla prostych (okres π), w zakresie [0, π/2]."""
    diff = np.abs(a - b) % np.pi
    return np.minimum(diff, np.pi - diff)


def midrib(edges):
    """Nerw główny = najsilniejsza prosta Hougha wewnątrz liścia. Zwraca (θ, ρ).
    (Oś elipsy dopasowanej do maski myli się na okrągłych liściach.)"""
    _, _, _, (_, angles, dists) = hough(edges, num_peaks=1)
    return (angles[0], dists[0]) if len(angles) else (0.0, 0.0)


def angle_clusters(angles, gap=np.deg2rad(20)):
    """Liczba grup kątów (okręgowo, okres π): grupy rozdziela przerwa > gap."""
    if len(angles) == 0:
        return 0
    a = np.sort(np.mod(angles, np.pi))
    gaps = np.diff(np.concatenate([a, [a[0] + np.pi]]))
    return max(int(np.sum(gaps > gap)), 1)


def circular_spread(angles):
    """1 - |średni wektor| dla kątów podwojonych (proste mają okres π).
    0 = wszystkie linie równoległe, 1 = kierunki rozłożone równomiernie."""
    if len(angles) == 0:
        return np.nan
    return 1 - np.abs(np.mean(np.exp(2j * np.asarray(angles))))


def edge_features(edges, leaf):
    """Cechy 'brzydoty' krawędzi."""
    area = leaf.sum()
    n_px = edges.sum()
    labels = measure.label(edges, connectivity=2)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "edge_density": n_px / area,                         # ile krawędzi na piksel liścia
        "edge_fragments": len(sizes) / max(n_px, 1) * 100,   # fragmenty na 100 px krawędzi
        "edge_mean_len": sizes.mean() if len(sizes) else 0,  # średnia długość fragmentu
    }


def vein_segments(edges, axis_theta, line_length=20, line_gap=3, oblique=np.deg2rad(20), seed=0):
    """Probabilistyczny Hough: odcinki + podział na 'wzdłuż nerwu głównego' i 'skośne' (nerwy boczne)."""
    segs = probabilistic_hough_line(edges, threshold=10, line_length=line_length,
                                    line_gap=line_gap, theta=THETAS, rng=seed)
    if not segs:
        return segs, np.array([], dtype=bool)
    p0, p1 = np.array(segs).transpose(1, 0, 2)  # (N, 2) w (x, y)
    dx, dy = (p1 - p0).T
    # normalna odcinka w konwencji hough_line
    seg_theta = np.arctan2(dy, dx) + np.pi / 2
    is_oblique = axial_diff(seg_theta, axis_theta) > oblique
    return segs, is_oblique


def hough_features(img, leaf=None, num_peaks=10):
    """Wszystkie cechy dla jednego obrazu."""
    if leaf is None:
        leaf = segment_leaf(img)["leaf"]
    edges = inner_edges(img, leaf)
    h, theta, d, (acc, angles, dists) = hough(edges, num_peaks=num_peaks)
    axis_theta = angles[0] if len(angles) else 0.0  # najsilniejsza prosta = nerw główny
    rel = axial_diff(angles[1:], axis_theta)
    segs, is_obl = vein_segments(edges, axis_theta)

    feats = edge_features(edges, leaf)
    feats.update({
        "peak_clusters": angle_clusters(angles),
        "peak_spread": circular_spread(angles),
        "lines_oblique_frac": np.mean(rel > np.deg2rad(20)) if len(rel) else np.nan,
        "lines_mean_dev_deg": np.rad2deg(rel.mean()) if len(rel) else np.nan,
        "segments_n": len(segs),
        "segments_oblique_frac": is_obl.mean() if len(segs) else np.nan,
        "segments_oblique_n": int(is_obl.sum()),
    })

    # dla porównania: piki akumulatora z krawędzi całego obrazu (z konturem liścia)
    full_edges = feature.canny(color.rgb2gray(img), sigma=2.0)
    _, _, _, (_, full_angles, _) = hough(full_edges, num_peaks=num_peaks)
    feats["peak_clusters_full"] = angle_clusters(full_angles)
    feats["peak_spread_full"] = circular_spread(full_angles)
    return feats
