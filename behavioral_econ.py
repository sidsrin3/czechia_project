import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ==============================================================================
# 1. EDIT YOUR TITLE, CENTER HUB & NODE LABELS HERE
# ==============================================================================
CONFIG = {
    "title": "Behavioral Economics Feedback Loop — Why Czech Tennis Output Is Self-Reinforcing",
    "output_path": "fig_behavioral_flywheel.png",
    "hub_title": "THE CZECH TENNIS\nFLYWHEEL",
    "hub_subtitle": "Path Dependence & +36% TFP\n(Violating Statistical Independence)",
}

NODES = [
    {
        "title": "1. International Success",
        "desc": "Grand Slams, Davis & BJK Cups\n(58 Top-100 Pros, 2000–2025)",
        "color": "#D62728",
        "bg": "#FDF2F2",
        "w": 3.25,
        "h": 1.10,
    },
    {
        "title": "2. Media & Role Models",
        "desc": "High national salience;\nrelatable local champions",
        "color": "#2B5B84",
        "bg": "#F2F6FA",
        "w": 3.00,
        "h": 1.10,
    },
    {
        "title": "3. Mass Youth Participation",
        "desc": "4.04% population participation;\ndefault childhood club sport",
        "color": "#238B45",
        "bg": "#F1F9F4",
        "w": 3.15,
        "h": 1.10,
    },
    {
        "title": "4. Coaching & Club Density",
        "desc": "107.1 courts & 78.6 coaches/100k;\nformer pros mentor juniors",
        "color": "#D95F02",
        "bg": "#FFF6EF",
        "w": 3.30,
        "h": 1.10,
    },
    {
        "title": "5. Deep Elite Pipeline",
        "desc": "Dense inter-club leagues &\nhigh-level domestic sparring",
        "color": "#6A51A3",
        "bg": "#F6F3FA",
        "w": 3.15,
        "h": 1.10,
    },
]

ARROW_CALLOUTS = [
    ("Availability Heuristic\n& Salience Bias", (2.85, 2.85)),
    ("Social Norms, Herding\n& Identity Economics", (4.05, -1.25)),
    ("Network Externalities &\nKnowledge Spillovers", (0.0, -4.35)),
    ("Short-Term Feedback Loops\n(Countering Present Bias)", (-4.05, -1.25)),
    ("Positive Serial Correlation\n& Talent Agglomeration", (-2.85, 2.85)),
]

# ==============================================================================
# 2. BUILD THE CIRCULAR FLYWHEEL DIAGRAM
# ==============================================================================
fig, ax = plt.subplots(figsize=(11.0, 9.6))
ax.set_xlim(-6.6, 6.6)
ax.set_ylim(-6.0, 6.0)
ax.axis("off")

# Center Hub
center_box = FancyBboxPatch(
    (-1.55, -0.80),
    3.1,
    1.6,
    boxstyle="round,pad=0.16,rounding_size=0.35",
    facecolor="#1F3A60",
    edgecolor="#0F2340",
    linewidth=2,
    zorder=4,
)
ax.add_patch(center_box)
ax.text(
    0,
    0.22,
    CONFIG["hub_title"],
    ha="center",
    va="center",
    color="white",
    fontsize=11.5,
    fontweight="bold",
    zorder=5,
)
ax.text(
    0,
    -0.42,
    CONFIG["hub_subtitle"],
    ha="center",
    va="center",
    color="#D9E6F2",
    fontsize=8.4,
    zorder=5,
)

# 5 Outer Stage Nodes (Clockwise starting from top: 90, 18, -54, -126, 162 deg)
R = 3.95
angles_deg = [90, 18, -54, -126, 162]
coords = []

for ang, node in zip(angles_deg, NODES):
    rad = np.deg2rad(ang)
    cx, cy = R * np.cos(rad), R * np.sin(rad)
    coords.append((cx, cy))

    box = FancyBboxPatch(
        (cx - node["w"] / 2, cy - node["h"] / 2),
        node["w"],
        node["h"],
        boxstyle="round,pad=0.12,rounding_size=0.25",
        facecolor=node["bg"],
        edgecolor=node["color"],
        linewidth=2.2,
        zorder=4,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy + 0.23,
        node["title"],
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=node["color"],
        zorder=5,
    )
    ax.text(
        cx,
        cy - 0.21,
        node["desc"],
        ha="center",
        va="center",
        fontsize=8.5,
        color="#222222",
        zorder=5,
    )

# Curved Clockwise Arrows & Behavioral Callout Badges
for i in range(5):
    x1, y1 = coords[i]
    x2, y2 = coords[(i + 1) % 5]
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        connectionstyle="arc3,rad=-0.20",
        arrowstyle="-|>,head_length=9,head_width=5.5",
        color="#334E68",
        linewidth=2.2,
        shrinkA=62,
        shrinkB=62,
        zorder=3,
    )
    ax.add_patch(arrow)

    lbl_text, (lx, ly) = ARROW_CALLOUTS[i]
    ax.text(
        lx,
        ly,
        lbl_text,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#102A43",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#FFFBEA",
            edgecolor="#D9B24C",
            linewidth=1.2,
        ),
        zorder=6,
    )

plt.title(
    CONFIG["title"],
    fontsize=13,
    fontweight="bold",
    pad=12,
    color="#102A43",
)
plt.tight_layout()
plt.savefig(CONFIG["output_path"], dpi=300, bbox_inches="tight")
plt.show()
