"""
Terminal-based visualization for HVAC placement results.
Renders colored floor maps and heatmaps using ANSI escape codes.
"""

import numpy as np
from typing import List, Tuple, Optional


# ANSI color helpers
def rgb_bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"

def rgb_fg(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def heat_color(v: float):
    """Map 0..1 to a cold-to-hot color."""
    t = max(0.0, min(1.0, v))
    if t < 0.25:
        s = t / 0.25
        return (int(30 + 20*s), int(30 + 70*s), int(80 + 120*s))
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return (int(50), int(100 + 100*s), int(200 - 100*s))
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return (int(50 + 180*s), int(200), int(100 - 50*s))
    else:
        s = (t - 0.75) / 0.25
        return (int(230 + 25*s), int(200 - 120*s), int(50 - 20*s))


def render_floor_map(floor_map: np.ndarray, title: str = "Floor Map"):
    """Render a floor map with walls, floors, and doors."""
    rows, cols = floor_map.shape
    print(f"\n  {BOLD}{rgb_fg(0,229,255)}{'─'*cols*2}{'──'}{RESET}")
    print(f"  {BOLD}{rgb_fg(0,229,255)} {title}{RESET}")
    print(f"  {BOLD}{rgb_fg(0,229,255)}{'─'*cols*2}{'──'}{RESET}")

    # Column headers
    hdr = "    "
    for c in range(cols):
        hdr += f"{c%10} "
    print(f"  {DIM}{hdr}{RESET}")

    for r in range(rows):
        line = f"  {DIM}{r:>2}{RESET} "
        for c in range(cols):
            cell = floor_map[r, c]
            if cell == 0:
                line += rgb_bg(26, 26, 46) + "  " + RESET
            elif cell == 2:
                line += rgb_bg(196, 149, 106) + "░░" + RESET
            else:
                line += rgb_bg(232, 224, 212) + "  " + RESET
        print(line)

    print(f"  {DIM}Legend: {RESET}"
          f"{rgb_bg(26,26,46)}  {RESET} Wall  "
          f"{rgb_bg(232,224,212)}  {RESET} Floor  "
          f"{rgb_bg(196,149,106)}░░{RESET} Door")


def render_result(
    floor_map: np.ndarray,
    positions: List[Tuple[int, int]],
    heatmap: Optional[np.ndarray],
    coverage: float,
    title: str = "HVAC Placement Result"
):
    """Render the optimized placement with cooling heatmap overlay."""
    rows, cols = floor_map.shape
    pos_set = set((int(r), int(c)) for r, c in positions)

    print(f"\n  {BOLD}{rgb_fg(0,229,255)}{'═'*cols*2}{'══'}{RESET}")
    print(f"  {BOLD}{rgb_fg(0,229,255)} {title}{RESET}")
    print(f"  {BOLD}{rgb_fg(0,229,255)}{'═'*cols*2}{'══'}{RESET}")

    hdr = "    "
    for c in range(cols):
        hdr += f"{c%10} "
    print(f"  {DIM}{hdr}{RESET}")

    for r in range(rows):
        line = f"  {DIM}{r:>2}{RESET} "
        for c in range(cols):
            cell = floor_map[r, c]
            if (r, c) in pos_set:
                line += rgb_bg(0, 229, 255) + rgb_fg(0, 40, 50) + BOLD + "AC" + RESET
            elif cell == 0:
                line += rgb_bg(20, 20, 35) + "  " + RESET
            elif heatmap is not None and cell > 0:
                v = min(1.0, heatmap[r, c])
                cr, cg, cb = heat_color(v)
                line += rgb_bg(cr, cg, cb) + "  " + RESET
            else:
                line += rgb_bg(232, 224, 212) + "  " + RESET
        print(line)

    print(f"\n  {DIM}Legend: {RESET}"
          f"{rgb_bg(20,20,35)}  {RESET} Wall  "
          f"{rgb_bg(0,229,255)}{rgb_fg(0,40,50)}{BOLD}AC{RESET} HVAC  "
          f"{rgb_bg(*heat_color(0.1))}  {RESET}→"
          f"{rgb_bg(*heat_color(0.5))}  {RESET}→"
          f"{rgb_bg(*heat_color(0.9))}  {RESET} Cool→Warm")

    # Summary
    print(f"\n  {BOLD}{rgb_fg(0,229,255)}┌────────────────────────────────────┐{RESET}")
    print(f"  {BOLD}{rgb_fg(0,229,255)}│{RESET}  HVAC Units Required: {BOLD}{rgb_fg(0,255,160)}{len(positions)}{RESET}             {BOLD}{rgb_fg(0,229,255)}│{RESET}")
    print(f"  {BOLD}{rgb_fg(0,229,255)}│{RESET}  Floor Coverage:      {BOLD}{rgb_fg(0,255,160)}{coverage:.1f}%{RESET}          {BOLD}{rgb_fg(0,229,255)}│{RESET}")
    print(f"  {BOLD}{rgb_fg(0,229,255)}│{RESET}                                    {BOLD}{rgb_fg(0,229,255)}│{RESET}")

    for i, (r, c) in enumerate(positions):
        pos_str = f"  Unit #{i+1}:  row={r:<3} col={c:<3}"
        padding = 36 - len(pos_str)
        print(f"  {BOLD}{rgb_fg(0,229,255)}│{RESET}{pos_str}{' '*padding}{BOLD}{rgb_fg(0,229,255)}│{RESET}")

    print(f"  {BOLD}{rgb_fg(0,229,255)}└────────────────────────────────────┘{RESET}")


def render_training_progress(episode, total, best_units, best_coverage, epsilon):
    """Single-line training progress bar."""
    pct = episode / total
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {rgb_fg(0,229,255)}[{bar}]{RESET} "
          f"{episode:>4}/{total}  "
          f"Best: {rgb_fg(0,255,160)}{best_units} units @ {best_coverage:.1f}%{RESET}  "
          f"ε={epsilon:.3f}", end="", flush=True)
