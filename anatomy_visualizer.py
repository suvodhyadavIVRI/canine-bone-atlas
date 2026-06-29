"""
anatomy_visualizer.py  —  Publication-Quality Veterinary Anatomy Annotation Renderer

This module provides a complete pipeline for rendering COCO-format anatomical
segmentation annotations onto bone images with:

  • Duplicate detection removal  (IoU-based, per-image)
  • Smart non-overlapping label placement  (spiral search + grid collision)
  • Anti-aliased leader lines connecting labels to centroids
  • Semi-transparent segmentation masks with crisp outlines
  • Rounded-rectangle label badges with high-contrast text
  • Bone isolation on a dark background
  • Rich info panel with colour-coded legend

Author : Visualization module for IVRI Bone Annotation Project
Stack  : Python 3, OpenCV, NumPy, Pillow
"""

import os
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

try:
    import pycocotools.mask as maskUtils
except ImportError:
    maskUtils = None


# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION  (all tuneable from outside via function parameters)
# ─────────────────────────────────────────────────────────────────────

# Default canvas sizes
DEFAULT_CARD_WIDTH  = 1100
DEFAULT_TOP_HEIGHT  = 740
DEFAULT_DESC_HEIGHT = 340

# Rendering
MASK_ALPHA       = 0.45       # transparency for filled segmentation masks (higher = more vivid colour on bone)
OUTLINE_THICK    = 2          # segmentation contour thickness
WHOLE_THICK      = 3          # thickness for the whole-bone outline
LEADER_MIN_DIST  = 30         # minimum displacement to draw a leader line
LABEL_PAD        = 6          # padding inside label badges
LABEL_CORNER_R   = 8          # corner radius for label badge
EDGE_MARGIN      = 6          # minimum distance from canvas edge
BADGE_DARKEN     = 0.55       # how much to darken badge background (0=black, 1=full colour)
FONT_STROKE_W    = 2          # text outline width for contrast

# Duplicate filtering
DEFAULT_IOU_THRESH = 0.85     # IoU above this → duplicate

# ─────────────────────────────────────────────────────────────────────
# CURATED COLOUR PALETTE  (warm, high-contrast, anatomy-friendly)
# ─────────────────────────────────────────────────────────────────────
REGION_COLORS = {
    "DOG-Scapula":                  (255, 255,  80),
    "Acromion process":             (255,  80,  80),
    "Area serrata":                 ( 80, 255, 130),
    "Distal angle medial surface":  (255, 160,  40),
    "Glenoid cavity":               ( 80, 200, 255),
    "Infraspinous fossa":           (210,  80, 255),
    "Lateralsurface":               (255, 230, 100),
    "Medial surface":               ( 80, 255, 220),
    "Scapular spine":               (255,  60, 180),
    "Subscapular fossa":            ( 60, 140, 255),
    "Supraspinous fossa":           (255, 130,  50),
    "supraglenoid tubercle":        (150, 255,  60),
}

REGION_DESC = {
    # Scapula
    "DOG-Scapula":                  "Whole scapula outline",
    "Acromion Process":             "The lateral extension of the scapular spine. In dogs, it is well-developed and serves as the origin for the acromiodeltoideus muscle.",
    "Area Serrata":                 "The roughened cranial area on the medial (subscapular) surface where the serratus ventralis muscle attaches.",
    "Distal Angle Medial Surface": "The distal/ventral aspect of the medial surface of the scapula near the neck.",
    "Glenoid Cavity":               "A shallow articular depression that articulates with the head of the humerus to form the shoulder joint.",
    "Infraspinous Fossa":           "The large area caudal to the scapular spine, occupied by the infraspinatus muscle.",
    "Lateral Surface":              "The outer surface of the scapula, divided into two fossae by the prominent scapular spine.",
    "Medial Surface":               "The inner (deep) costal surface facing the ribs, containing the subscapular fossa.",
    "Scapular Spine":               "A prominent shelf-like plate of bone dividing the lateral surface of the scapula.",
    "Subscapular Fossa":            "The large, shallow depression on the medial surface occupied by the subscapularis muscle.",
    "Supraglenoid Tubercle":        "A projection near the cranial aspect of the glenoid cavity where the biceps brachii muscle originates.",
    "Supraspinous Fossa":           "The area cranial to the scapular spine, occupied by the supraspinatus muscle.",
    
    # Humerus
    "Deltoid Ridge": "A prominent ridge on the craniolateral aspect of the humerus where the deltoid muscle inserts.",
    "Distal Extremity of Humerus": "The lower end of the humerus, forming the condyle which articulates with the radius and ulna.",
    "Humeral Head": "The smooth, rounded articular surface on the proximal end that articulates with the scapula's glenoid cavity.",
    "Olecranon Fossa": "A deep cavity on the caudal surface of the distal extremity of the humerus which receives the anconeal process of the ulna.",
    "Proximal Extremity of Humerus": "The upper end of the humerus, bearing the head, neck, and greater/lesser tubercles.",
    "Supratrochlear Foramen": "An opening in the bony septum between the radial and olecranon fossae. Unique to dogs; no major vessels pass through it.",
    
    # Radius & Ulna
    "Anconeal Process": "A sharp, beak-like projection on the proximal end of the ulna that fits into the olecranon fossa of the humerus.",
    "Anterior Surface of Radius": "The smooth, convex front surface of the radius facing cranially.",
    "Body of Radius": "The shaft of the radius, curved cranially to support forelimb weight.",
    "Capitular Fovea": "The articular surface at the proximal end of the radius articulating with the humeral condyle.",
    "Carpal Facets": "Articular surfaces at the distal end of the radius that connect with the carpal bones of the paw.",
    "Distal End of Radius": "The expanded lower end of the radius which participates in the wrist (carpus) joint.",
    "Facet for Ulna": "Smooth contact points where the radius and ulna articulate with one another.",
    "Nutrient Foramen": "The small opening through which blood vessels enter the shaft of the bone.",
    "Olecranon Process": "The large projection at the proximal end of the ulna, forming the point of the elbow (attachment for triceps).",
    "Posterior Surface of Radius": "The flat caudal surface facing the ulna.",
    "Styloid Process of Radius": "A pointed bony projection at the distal medial end of the radius.",
    "Styloid Process of Ulna": "A pointed projection at the distal lateral end of the ulna.",
    "Trochlear Notch": "The semi-circular groove on the ulna that receives the trochlea of the humerus.",
    "Ulna of Dog": "The longer of the two forelimb bones, acting as a lever for elbow extension.",
    "Dog Radius": "The major weight-bearing bone of the canine forearm.",
    "Proximal Radius": "The upper portion of the radius articulating with the humerus and ulna.",
    "Proximal Ulna": "The upper portion of the ulna comprising the olecranon process and trochlear notch.",
    "Radial Tuberosity": "A rough projection on the proximal medial aspect where the biceps brachii inserts.",
    
    # Os Coxae
    "Cotyloid Cavity": "Also known as the Acetabulum; the deep cup-shaped socket that articulates with the head of the femur.",
    "Gluteal Fossa": "The concave lateral surface of the wing of the ilium where gluteal muscles originate.",
    "Iliacus Surface": "The smooth ventromedial surface of the ilium where the iliacus muscle originates.",
    "Iliopectineal Eminence": "A low tubercle on the pubis near the acetabulum where the pectineus muscle originates.",
    "Ilium": "The largest and most cranial of the three pelvic bones.",
    "Ischiatic Arch": "The deep notch on the caudal aspect of the pelvic floor between the two ischial tuberosities.",
    "Ischiatic Tuberosity": "The thick caudal angle of the ischium, forming the 'pin bone' (point of the hip).",
    "Ischium": "The caudal-most bone of the pelvis forming the pelvic floor.",
    "Obturator Foramen": "The large opening on the pelvic floor bounded by the pubis and ischium.",
    "Pelvic Symphysis": "The median joint uniting the two halves of the pelvis.",
    "Pubis": "The smallest of the pelvic bones, forming the cranial portion of the pelvic floor.",
    "Sacral Surface of Ilium": "The rough medial surface that articulates with the sacrum (sacroiliac joint).",
    "Ventral Tubercle of Pubis": "A small bony tubercle on the ventral midline of the pubis.",
    "Wing of Ilium": "The wide, laterally expanded cranial wing of the ilium.",
    "Pectineal Line": "A sharp ridge on the pubis where the pectineus muscle attaches.",

    # Femur
    "Head of Femur": "A smooth, hemispherical articular surface at the proximal end of the femur articulating with the pelvis's acetabulum to form the hip joint.",
    "Greater Trochanter": "A prominent lateral bony projection on the proximal femur for insertion of the gluteal muscles.",
    "Lesser Trochanter": "A projection on the caudomedial aspect of the proximal femur where the iliopsoas muscle inserts.",
    "Third Trochanter": "A small lateral projection on the proximal shaft of the femur for superficial gluteal muscle attachment.",
    "Trochlea of Femur": "The smooth cranial groove on the distal femur where the patella glides during joint extension.",
    "Medial Condyle (Femur)": "The medial articular surface at the distal extremity of the femur that articulates with the tibia.",
    "Lateral Condyle (Femur)": "The lateral articular surface at the distal extremity of the femur that articulates with the tibia.",
    "Medial Epicondyle": "A bony projection on the medial side of the distal femur proximal to the condyle.",
    "Lateral Epicondyle": "A bony projection on the lateral side of the distal femur proximal to the condyle.",
    "Supracondylar Tuberosity": "Bony ridges proximal to the femoral condyles where the gastrocnemius muscle originates.",
    "Dog Femur": "The heaviest long bone of the canine hindlimb, extending from the hip to the stifle (knee) joint.",

    # Fibula
    "Head of Fibula": "The proximal end of the fibula that articulates with the lateral condyle of the tibia.",
    "Neck of Fibula": "The constricted portion of the fibula just distal to its head.",
    "Lateral Malleolus": "The distal extremity of the fibula, forming the lateral side of the ankle joint.",
    "Sulcus Malleolaris Lateralis": "A groove on the lateral malleolus providing a pathway for digital extensor tendons.",
    "Dog Fibula": "A long, thin, lateral leg bone running parallel to the tibia, acting as a site for muscle attachment.",

    # Tibia
    "Medial Condyle": "The medial articular surface on the proximal tibia that articulates with the femur and meniscus.",
    "Lateral Condyle": "The lateral articular surface on the proximal tibia that articulates with the femur and meniscus.",
    "Intercondylar Eminence": "A central spine-like projection on the proximal tibia separating the condyles.",
    "Tibial Tuberosity": "A large cranial projection at the proximal end of the tibia for attachment of the patellar ligament.",
    "Extensor Groove": "A notch lateral to the tibial tuberosity for passage of the long digital extensor tendon.",
    "Popliteal Notch": "A caudal notch between the proximal tibial condyles for popliteal blood vessels.",
    "Articular Facet for Fibula": "A small lateral facet near the proximal end for articulation with the head of the fibula.",
    "Tibial Crest": "A prominent, palpable cranial border of the tibial shaft extending distally from the tuberosity.",
    "Popliteal Line": "A diagonal line on the caudal surface of the proximal tibia for popliteus muscle attachment.",
    "Interosseous Crest": "A lateral ridge along the tibial shaft where the interosseous membrane attaches to the fibula.",
    "Tibial Cochlea": "The distal articular surface of the tibia articulating with the talus of the ankle.",
    "Medial Malleolus": "The prominent bony projection on the medial side of the distal extremity of the tibia.",
    "Dog Tibia": "The main weight-bearing long bone of the canine leg, extending from the stifle (knee) to the hock joint.",
    
    # Skull
    "Zygomatic Arch": "The bony arch forming the lateral boundary of the orbit and temporal fossa, serving as the origin for the masseter muscle.",
    "Occipital Condyle": "The articular prominences at the base of the skull that articulate with the atlas (first cervical vertebra).",
    "External Sagittal Crest": "A median ridge on the dorsal aspect of the canine skull, particularly prominent in dolichocephalic breeds, serving for temporalis muscle attachment.",
    "Nasal Bone": "The paired bones forming the dorsal roof of the nasal cavity.",
    "Mandible": "The lower jaw bone, forming the temporomandibular joint with the temporal bone.",
    "Dog Skull": "The complex skeletal structure of the canine head enclosing the brain and supporting the facial structures."
}

# Keep the lowercase/no-space aliases for backwards compatibility
REGION_DESC_ALIASES = {
    "DOG-Scapula":                  "Whole scapula outline",
    "Acromion process":             "Lateral bony projection",
    "Area serrata":                 "Serrated medial border",
    "Distal angle medial surface":  "Distal medial corner",
    "Glenoid cavity":               "Shoulder joint socket",
    "Infraspinous fossa":           "Below scapular spine",
    "Lateralsurface":               "Outer (lateral) face",
    "Medial surface":               "Inner (medial) face",
    "Scapular spine":               "Dorsal bony ridge",
    "Subscapular fossa":            "Costal (ventral) face",
    "Supraspinous fossa":           "Above scapular spine",
    "supraglenoid tubercle":        "Biceps tendon origin",
}
for k, v in REGION_DESC_ALIASES.items():
    if k not in REGION_DESC:
        REGION_DESC[k] = v

# Fallback colour for unknown labels
_FALLBACK_COLOR = (200, 200, 200)


# ═════════════════════════════════════════════════════════════════════
# 1. FONT HELPERS
# ═════════════════════════════════════════════════════════════════════

def get_font(size: int = 18, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Load the best available TrueType font, falling back gracefully."""
    if bold:
        candidates = [
            "arialbd.ttf", "Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/ARIALBD.TTF",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "arial.ttf", "Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.Draw, text: str, font) -> tuple:
    """Return (width, height) of rendered text."""
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def wrap_text_pil(text: str, font, max_width: int, draw: ImageDraw.Draw) -> list:
    """Wrap text to fit within a given maximum width in pixels."""
    words = text.split(" ")
    lines = []
    current_line = []
    
    for word in words:
        test_line = " ".join(current_line + [word]) if current_line else word
        tw, _ = _text_size(draw, test_line, font)
        if tw <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            
    if current_line:
        lines.append(" ".join(current_line))
        
    return lines


# ═════════════════════════════════════════════════════════════════════
# 2. COLOUR HELPERS
# ═════════════════════════════════════════════════════════════════════

def color_for(label: str) -> tuple:
    """Return the curated RGB colour for an anatomical region."""
    # 1. Exact match in REGION_COLORS
    if label in REGION_COLORS:
        return REGION_COLORS[label]
    
    # 2. Case-insensitive and clean match
    norm_label = label.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
    for k, v in REGION_COLORS.items():
        norm_k = k.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
        if norm_k == norm_label:
            return v
            
    # 3. Dynamic high-contrast color generation based on label hash
    palette = [
        (255, 80, 80),   # Coral Red
        (80, 255, 130),  # Mint Green
        (80, 200, 255),  # Sky Blue
        (210, 80, 255),  # Lavender Purple
        (255, 230, 100), # Warm Yellow
        (80, 255, 220),  # Turquoise
        (255, 60, 180),  # Hot Pink
        (60, 140, 255),  # Royal Blue
        (255, 130, 50),  # Bright Orange
        (150, 255, 60),  # Lime Green
        (255, 170, 170), # Light Pink
        (170, 255, 170), # Light Green
        (170, 170, 255), # Light Blue
        (255, 220, 170), # Peach
        (220, 170, 255), # Mauve
        (170, 255, 220)  # Pale Teal
    ]
    idx = abs(hash(norm_label)) % len(palette)
    return palette[idx]


def darken(color: tuple, factor: float = BADGE_DARKEN) -> tuple:
    """Darken an RGB colour by *factor* (0 = black, 1 = unchanged)."""
    return tuple(int(c * factor) for c in color)


def _contrast_text_color(bg_rgb: tuple) -> tuple:
    """Return white or black text depending on background luminance."""
    lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    return (255, 255, 255) if lum < 160 else (20, 20, 20)


# ═════════════════════════════════════════════════════════════════════
# 3. DUPLICATE FILTERING  (IoU-based, in-memory)
# ═════════════════════════════════════════════════════════════════════

def _bbox_iou(a, b) -> float:
    """Compute IoU between two COCO-format bboxes [x, y, w, h]."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def filter_duplicates(annotations: list, iou_thresh: float = DEFAULT_IOU_THRESH) -> list:
    """
    Remove duplicate annotations within a single image.

    Two annotations are considered duplicates if they share the same
    category label AND their bounding-box IoU exceeds *iou_thresh*.
    When duplicates are found, the one with the larger area is kept.

    Parameters
    ----------
    annotations : list[dict]
        COCO-style annotation dicts.  Must contain 'bbox' and 'label'.
    iou_thresh : float
        IoU threshold above which two same-class annotations are duplicates.

    Returns
    -------
    list[dict]
        Deduplicated annotations.
    """
    # Sort by area (largest first) so we keep the most prominent one
    sorted_anns = sorted(
        annotations,
        key=lambda a: a["bbox"][2] * a["bbox"][3],
        reverse=True,
    )

    unique = []
    for ann in sorted_anns:
        is_dup = False
        for kept in unique:
            if ann.get("label") == kept.get("label"):
                if _bbox_iou(ann["bbox"], kept["bbox"]) > iou_thresh:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(ann)

    return unique


# ═════════════════════════════════════════════════════════════════════
# 4. SEGMENTATION HELPERS
# ═════════════════════════════════════════════════════════════════════

def get_annotation_mask(ann: dict, canvas_shape: tuple, scale: float, nw: int, nh: int, ox: int, oy: int) -> np.ndarray:
    """Returns a full-canvas mask (0 or 255) for the given annotation, handling RLE, Polygons, or Bbox fallbacks."""
    h_cv, w_cv = canvas_shape[:2]
    canvas_mask = np.zeros((h_cv, w_cv), dtype=np.uint8)
    
    seg = ann.get("segmentation")
    has_drawn = False

    if isinstance(seg, dict) and "counts" in seg and maskUtils is not None:
        # COCO compressed RLE
        try:
            binary_mask = maskUtils.decode(seg) # shape: (orig_h, orig_w)
            # resize mask precisely
            resized_mask = cv2.resize(binary_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            canvas_mask[oy:oy+nh, ox:ox+nw] = resized_mask * 255
            has_drawn = True
        except Exception:
            pass
    elif isinstance(seg, list) and len(seg) > 0:
        # COCO Polygon format
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = pts[:, 0] * scale + ox
            pts[:, 1] = pts[:, 1] * scale + oy
            cv2.fillPoly(canvas_mask, [pts.astype(np.int32)], 255)
            has_drawn = True

    if not has_drawn and "bbox" in ann:
        # Fallback to an ellipse inside the bounding box
        x, y, w, h = ann["bbox"]
        c_x = int((x + w / 2) * scale + ox)
        c_y = int((y + h / 2) * scale + oy)
        r_x = max(1, int((w / 2) * scale))
        r_y = max(1, int((h / 2) * scale))
        cv2.ellipse(canvas_mask, (c_x, c_y), (r_x, r_y), 0, 0, 360, 255, -1)
        has_drawn = True
        
    return canvas_mask if has_drawn else None


def seg_centroid(ann: dict, scale: float, ox: int, oy: int) -> tuple:
    """Compute centroid approximation directly from bbox (extremely reliable)."""
    x, y, w, h = ann["bbox"]
    return int((x + w / 2) * scale + ox), int((y + h / 2) * scale + oy)


# ═════════════════════════════════════════════════════════════════════
# 5. BONE ISOLATION  (remove background → black)
# ═════════════════════════════════════════════════════════════════════

def isolate_bone(image: np.ndarray, annotations: list) -> np.ndarray:
    """
    Create a version of *image* with everything outside the annotated
    segmentation regions set to black.
    """
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for ann in annotations:
        # Scale=1.0 and nw/nh=w/h because isolate operates on original dimensions
        mask = get_annotation_mask(ann, (h, w), 1.0, w, h, 0, 0)
        if mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    result = np.zeros_like(image)
    result[combined_mask > 0] = image[combined_mask > 0]
    return result


# ═════════════════════════════════════════════════════════════════════
# 6. SMART LABEL PLACEMENT  (guaranteed non-overlap, spiral search)
# ═════════════════════════════════════════════════════════════════════

class SmartLabelPlacer:
    """
    Place text labels on a canvas without overlapping each other or
    overflowing the canvas boundaries.

    Uses a spiral search outward from each label's preferred anchor
    position.  A simple grid-based occupancy map accelerates collision
    checks to near-O(1).
    """

    GRID_CELL = 16  # pixels per grid cell — smaller = more precise

    def __init__(self, canvas_w: int, canvas_h: int, font, draw: ImageDraw.Draw):
        self.cw = canvas_w
        self.ch = canvas_h
        self.font = font
        self.draw = draw

        # Occupancy grid (True = occupied)
        self.cols = math.ceil(canvas_w / self.GRID_CELL)
        self.rows = math.ceil(canvas_h / self.GRID_CELL)
        self.grid = [[False] * self.cols for _ in range(self.rows)]

        # Results
        self.placements = []  # (text, lx, ly, tw, th, color, anchor_cx, anchor_cy)

    def _measure(self, text: str) -> tuple:
        tw, th = _text_size(self.draw, text, self.font)
        return tw + LABEL_PAD * 2, th + LABEL_PAD * 2

    def _grid_overlaps(self, lx: int, ly: int, tw: int, th: int) -> bool:
        """Check if rectangle [lx, ly, lx+tw, ly+th] overlaps any occupied cell."""
        gc = self.GRID_CELL
        c1 = max(0, lx // gc)
        r1 = max(0, ly // gc)
        c2 = min(self.cols - 1, (lx + tw) // gc)
        r2 = min(self.rows - 1, (ly + th) // gc)

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.grid[r][c]:
                    return True
        return False

    def _grid_mark(self, lx: int, ly: int, tw: int, th: int):
        """Mark rectangle cells as occupied."""
        gc = self.GRID_CELL
        c1 = max(0, lx // gc)
        r1 = max(0, ly // gc)
        c2 = min(self.cols - 1, (lx + tw) // gc)
        r2 = min(self.rows - 1, (ly + th) // gc)

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.grid[r][c] = True

    def _clamp(self, lx: int, ly: int, tw: int, th: int) -> tuple:
        """Ensure the label rectangle stays inside canvas bounds."""
        lx = max(EDGE_MARGIN, min(lx, self.cw - tw - EDGE_MARGIN))
        ly = max(EDGE_MARGIN, min(ly, self.ch - th - EDGE_MARGIN))
        return lx, ly

    def place(self, text: str, anchor_cx: int, anchor_cy: int, color: tuple):
        """
        Find the best non-overlapping position for *text* near (anchor_cx, anchor_cy).
        """
        tw, th = self._measure(text)

        # Preferred: centred above the anchor
        pref_x = anchor_cx - tw // 2
        pref_y = anchor_cy - th - 10

        # Spiral search
        found = False
        max_radius = max(self.cw, self.ch)

        for radius in range(0, max_radius, 5):
            steps = max(12, radius // 3)

            if radius == 0:
                offsets = [(0, 0)]
            else:
                offsets = [
                    (int(radius * math.cos(2 * math.pi * s / steps)),
                     int(radius * math.sin(2 * math.pi * s / steps)))
                    for s in range(steps)
                ]

            for dx, dy in offsets:
                lx, ly = self._clamp(pref_x + dx, pref_y + dy, tw, th)
                if not self._grid_overlaps(lx, ly, tw, th):
                    self._grid_mark(lx, ly, tw, th)
                    self.placements.append((text, lx, ly, tw, th, color, anchor_cx, anchor_cy))
                    found = True
                    break

            if found:
                break

        # Absolute fallback — place it anyway (at top of canvas)
        if not found:
            lx, ly = self._clamp(pref_x, EDGE_MARGIN, tw, th)
            self._grid_mark(lx, ly, tw, th)
            self.placements.append((text, lx, ly, tw, th, color, anchor_cx, anchor_cy))


# ═════════════════════════════════════════════════════════════════════
# 7. DRAWING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def draw_masks_and_contours(canvas: np.ndarray, annotations: list,
                            scale: float, nw: int, nh: int, ox: int, oy: int) -> np.ndarray:
    """
    Draw crisp contour outlines (no fills) for every annotation onto *canvas*.
    Returns the modified canvas.
    """
    # Draw contour outlines
    for ann in annotations:
        if ann["label"] == "DOG-Scapula":
            continue # skip outlining the entire isolated bone as it frames the image

        c = color_for(ann["label"])
        mask = get_annotation_mask(ann, canvas.shape, scale, nw, nh, ox, oy)
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Double-line border for better visibility:
            # 1. Thicker dark "outer glow" or boundary
            shadow_c = (0, 0, 0)
            cv2.drawContours(canvas, contours, -1, shadow_c, OUTLINE_THICK + 2, cv2.LINE_AA)
            
            # 2. Main color contour inside it
            cv2.drawContours(canvas, contours, -1, c, OUTLINE_THICK, cv2.LINE_AA)

    return canvas


def draw_labels_and_leaders(canvas: np.ndarray, annotations: list,
                            scale: float, ox: int, oy: int,
                            font_size: int = 17) -> np.ndarray:
    """
    Place rounded-rectangle label badges and anti-aliased leader lines
    for each annotation.  Labels are guaranteed non-overlapping.

    Returns the modified canvas (numpy array, RGB).
    """
    h_cv, w_cv = canvas.shape[:2]
    font = get_font(font_size, bold=True)

    # Use PIL for high-quality text rendering
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    # ---------- collect candidates sorted by area (large → small) ----------
    candidates = []
    for ann in annotations:
        label = ann["label"]
        c = color_for(label)
        cx, cy = seg_centroid(ann, scale, ox, oy)
        area = ann["bbox"][2] * ann["bbox"][3]
        candidates.append((area, label, cx, cy, c))

    # Sort large-area annotations first → they get priority placement
    candidates.sort(key=lambda x: -x[0])

    # ---------- place labels ----------
    placer = SmartLabelPlacer(w_cv, h_cv, font, draw)
    for _, label, cx, cy, c in candidates:
        placer.place(label, cx, cy, c)

    # ---------- draw leader lines, dots, and badges ----------
    for (text, lx, ly, tw, th, c, anch_cx, anch_cy) in placer.placements:
        # Badge centre
        badge_cx = lx + tw // 2
        badge_cy = ly + th // 2

        # Distance from badge centre to anchor
        dist = math.hypot(badge_cx - anch_cx, badge_cy - anch_cy)

        # Leader line (only if label was displaced significantly)
        if dist > LEADER_MIN_DIST:
            # Compute the point on the badge edge closest to the anchor
            # for a cleaner connection
            angle = math.atan2(anch_cy - badge_cy, anch_cx - badge_cx)
            edge_x = int(badge_cx + (tw / 2) * math.cos(angle))
            edge_y = int(badge_cy + (th / 2) * math.sin(angle))

            # Draw anti-aliased line
            draw.line([(edge_x, edge_y), (anch_cx, anch_cy)],
                      fill=c, width=2)

            # Centroid dot
            dot_r = 5
            draw.ellipse(
                [anch_cx - dot_r, anch_cy - dot_r,
                 anch_cx + dot_r, anch_cy + dot_r],
                fill=c, outline=(0, 0, 0), width=1,
            )

        # Rounded-rectangle badge: background matching the outline colour
        bg_color = darken(c, 0.70) # slightly darkened for depth, but clearly the same colour
        _draw_rounded_rect(draw, lx, ly, lx + tw, ly + th,
                           radius=LABEL_CORNER_R, fill=bg_color,
                           outline=c, outline_width=2)

        # Text with maximum contrast
        tx = lx + LABEL_PAD
        ty = ly + LABEL_PAD
        text_color = _contrast_text_color(bg_color)
        
        # Add a subtle dark text stroke if text is white
        if sum(text_color) > 100:
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if ddx == 0 and ddy == 0:
                        continue
                    draw.text((tx + ddx, ty + ddy), text, fill=(0, 0, 0), font=font)
                    
        # Main text
        draw.text((tx, ty), text, fill=text_color, font=font)

    return np.array(pil_img)


def _draw_rounded_rect(draw: ImageDraw.Draw, x1, y1, x2, y2,
                        radius=8, fill=None, outline=None, outline_width=1):
    """Draw a rounded rectangle using PIL's rounded_rectangle (Pillow ≥ 8.2)."""
    try:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius,
                               fill=fill, outline=outline, width=outline_width)
    except AttributeError:
        # Fallback for older Pillow: plain rectangle
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=outline_width)


# ═════════════════════════════════════════════════════════════════════
# 8. LEGEND / INFO PANEL
# ═════════════════════════════════════════════════════════════════════

def draw_legend_panel(panel_w: int, panel_h: int,
                      present_labels: list,
                      annotations: list,
                      img_w: int, img_h: int,
                      file_name: str = "",
                      bone_type: str = "Scapula") -> np.ndarray:
    """
    Render a rich bottom panel with:
      - Title line
      - Statistics bar
      - Multi-column colour-coded legend with descriptions
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Subtle top border
    panel[0, :] = [55, 55, 75]

    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)

    font_title = get_font(24, bold=True)
    font_stat  = get_font(16, bold=False)
    font_hdr   = get_font(18, bold=True)
    font_item  = get_font(14, bold=False)

    y = 14

    # Title
    draw.text((24, y), f"Canine {bone_type} — Regional Anatomy",
              fill=(255, 210, 80), font=font_title)
    y += 36

    # Statistics line
    n_regions = len(set(present_labels))
    n_total   = len(annotations)
    stat_text = (f"Regions annotated: {n_regions}   │   "
                 f"Total instances: {n_total}   │   "
                 f"Image: {img_w}×{img_h} px")
    if file_name:
        stat_text += f"   │   {file_name}"
    draw.text((24, y), stat_text, fill=(100, 220, 255), font=font_stat)
    y += 30

    # Section header
    draw.text((24, y), "Anatomical Regions:", fill=(180, 220, 255), font=font_hdr)
    y += 28

    # Multi-column legend
    items = sorted(set(present_labels))
    cols     = 2
    col_w    = (panel_w - 48) // cols
    per_col  = max(1, (len(items) + cols - 1) // cols)

    col_y = [y] * cols
    max_text_w = col_w - 32

    for idx, label in enumerate(items):
        col = idx // per_col
        cx  = 24 + col * col_w
        cy  = col_y[col]
        c   = color_for(label)

        # Colour swatch
        draw.rectangle([cx, cy + 3, cx + 14, cy + 16],
                       fill=c, outline=(255, 255, 255), width=1)

        # Label + description
        desc = REGION_DESC.get(label, "")
        if not desc:
            norm_label = label.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
            for k, v in REGION_DESC.items():
                norm_k = k.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
                if norm_k == norm_label:
                    desc = v
                    break
        legend_text = f"{label}  —  {desc}" if desc else label
        
        # Wrap the legend text
        lines = wrap_text_pil(legend_text, font_item, max_text_w, draw)
        
        # Draw wrapped lines
        for line_idx, line in enumerate(lines):
            draw.text((cx + 22, cy + line_idx * 16), line,
                      fill=(220, 220, 220), font=font_item)
                      
        col_y[col] += len(lines) * 16 + 8

    return np.array(pil)


# ═════════════════════════════════════════════════════════════════════
# 9. HEADER BAR
# ═════════════════════════════════════════════════════════════════════

def draw_header_bar(canvas: np.ndarray, file_name: str,
                    bar_height: int = 40, bone_type: str = "Scapula") -> np.ndarray:
    """Draw a dark header bar with the filename at the top of the canvas."""
    h, w = canvas.shape[:2]
    # Dark bar
    canvas[:bar_height, :] = [20, 20, 32]

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = get_font(17, bold=True)

    title = f"Dog {bone_type} — Full Anatomical Segmentation   │   {file_name}"
    draw.text((16, 10), title, fill=(200, 200, 220), font=font)

    return np.array(pil)


# ═════════════════════════════════════════════════════════════════════
# 10. CARD COMPOSITION  (main entry point)
# ═════════════════════════════════════════════════════════════════════

def make_publication_card(
    image_rgb: np.ndarray,
    annotations: list,
    file_name: str = "",
    img_w: int = 0,
    img_h: int = 0,
    card_width: int = DEFAULT_CARD_WIDTH,
    top_height: int = DEFAULT_TOP_HEIGHT,
    desc_height: int = DEFAULT_DESC_HEIGHT,
    isolate: bool = True,
    dedup_iou: float = DEFAULT_IOU_THRESH,
    font_size: int = 17,
    bone_type: str = "",
) -> np.ndarray:
    """
    Create a publication-quality annotated card from a bone image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input image in RGB colour space.
    annotations : list[dict]
        COCO-style annotations, each having 'bbox', 'segmentation', 'label'.
    file_name : str
        Image filename (shown in the header bar).
    img_w, img_h : int
        Original image dimensions (for the stats line).
    card_width, top_height, desc_height : int
        Layout dimensions.
    isolate : bool
        If True, remove background (bone on black).
    dedup_iou : float
        IoU threshold for duplicate removal.  Set to 1.0 to disable.
    font_size : int
        Font size for annotation labels.

    Returns
    -------
    np.ndarray
        The composed card image in RGB, ready for saving.
    """
    # ── Step 1: deduplicate ────────────────────────────────────────
    anns = filter_duplicates(annotations, iou_thresh=dedup_iou)

    # ── Step 2: isolate bone (optional) ────────────────────────────
    if isolate:
        image_rgb = isolate_bone(image_rgb, anns)

    # ── Step 3: scale and centre on canvas ─────────────────────────
    h, w = image_rgb.shape[:2]
    scale = min(card_width / w, top_height / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((top_height, card_width, 3), dtype=np.uint8)
    ox = (card_width - nw) // 2
    oy = (top_height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized

    # ── Step 4: draw segmentation masks & outlines ─────────────────
    canvas = draw_masks_and_contours(canvas, anns, scale, nw, nh, ox, oy)

    # ── Step 5: draw labels and leader lines ───────────────────────
    canvas = draw_labels_and_leaders(canvas, anns, scale, ox, oy,
                                     font_size=font_size)

    # Determine bone_type if empty
    if not bone_type:
        present_labels = [a["label"] for a in anns]
        bone_type = "Scapula"  # Default fallback
        for label in present_labels:
            label_lower = label.lower()
            if any(h in label_lower for h in ["humerus", "humeral", "olecranon", "deltoid", "supratrochlear"]):
                bone_type = "Humerus"
                break
            elif any(o in label_lower for o in ["coxae", "ilium", "ischium", "pubis", "obturator", "symphysis", "acetabulum", "cotyloid"]):
                bone_type = "Os Coxae"
                break
            elif any(r in label_lower for r in ["radius", "ulna", "styloid", "anconeal", "trochlear", "carpal", "capitular"]):
                bone_type = "Radius-Ulna"
                break

    # ── Step 6: header bar ─────────────────────────────────────────
    canvas = draw_header_bar(canvas, file_name, bone_type=bone_type)

    # ── Step 7: legend / info panel ────────────────────────────────
    present_labels = [a["label"] for a in anns]
    legend = draw_legend_panel(
        panel_w=card_width,
        panel_h=desc_height,
        present_labels=present_labels,
        annotations=anns,
        img_w=img_w or w,
        img_h=img_h or h,
        file_name=file_name,
        bone_type=bone_type,
    )

    # ── Step 8: stack vertically ───────────────────────────────────
    return np.vstack([canvas, legend])


# ═════════════════════════════════════════════════════════════════════
# 11. UTILITY:  ensure output directory
# ═════════════════════════════════════════════════════════════════════

def ensure_dir(path: str):
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
