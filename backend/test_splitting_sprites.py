# slice_spritesheet_test.py
# pip install pillow
# This mirrors editor_introspect → operation="slice_spritesheet" (same defaults/behavior)

import argparse, os, statistics
from PIL import Image

def parse_wh(s, default=None):
    if not s:
        return default
    s = str(s).lower().replace(" ", "")
    w, h = s.split("x")
    return int(float(w)), int(float(h))

def estimate_bg(img):
    px = img.load(); w, h = img.size
    cs = [px[0,0], px[w-1,0], px[0,h-1], px[w-1,h-1]]
    if img.mode != "RGBA":
        cs = [Image.new("RGBA",(1,1),c).getpixel((0,0)) for c in cs]
    r = sum(c[0] for c in cs)/4.0
    g = sum(c[1] for c in cs)/4.0
    b = sum(c[2] for c in cs)/4.0
    a = sum((c[3] if len(c)>3 else 255) for c in cs)/4.0
    return (r,g,b,a)

def is_bg(c, bg, tol, a_thresh):
    if len(c)==4 and c[3] <= a_thresh: return True
    return abs(c[0]-bg[0])<=tol and abs(c[1]-bg[1])<=tol and abs(c[2]-bg[2])<=tol

def median_or(x, d): return int(statistics.median(x)) if x else int(d)

def auto_infer(img, tile_w, tile_h, tol, a_thresh, margin, spacing):
    w, h = img.size; px = img.load(); bg = estimate_bg(img)
    col_ne = [0]*w; row_ne = [0]*h
    for x in range(w):
        col_ne[x] = 1 if any(not is_bg(px[x,y], bg, tol, a_thresh) for y in range(h)) else 0
    for y in range(h):
        row_ne[y] = 1 if any(not is_bg(px[x,y], bg, tol, a_thresh) for x in range(w)) else 0
    # margins from outermost empty bands
    left = 0
    while left < w and col_ne[left] == 0:
        left += 1
    right = w - 1
    while right >= 0 and col_ne[right] == 0:
        right -= 1
    top = 0
    while top < h and row_ne[top] == 0:
        top += 1
    bottom = h - 1
    while bottom >= 0 and row_ne[bottom] == 0:
        bottom -= 1
    if left<right and top<bottom:
        margin = max(margin, min(left, top))
    # spacing = median empty run between content bands
    def est_space(flags):
        gaps=[]; run=0; prev=False
        for f in flags:
            if f==0: run+=1; prev=True
            else:
                if prev and run>0: gaps.append(run)
                run=0; prev=False
        return median_or(gaps, 0)
    if spacing==0:
        spacing = max(0, min(est_space(col_ne), est_space(row_ne)))
    # grid from usable area
    usable_w = w - margin*2 + spacing
    usable_h = h - margin*2 + spacing
    cols = max(1, (usable_w + spacing)//(tile_w + spacing))
    rows = max(1, (usable_h + spacing)//(tile_h + spacing))
    return margin, spacing, cols, rows

def tight_crop_alpha(img, a_thresh):
    px = img.load(); w,h = img.size
    minx,miny,maxx,maxy = w,h,-1,-1
    for y in range(h):
        for x in range(w):
            a = px[x,y][3] if img.mode=="RGBA" else 255
            if a > a_thresh:
                if x<minx: minx=x
                if y<miny: miny=y
                if x>maxx: maxx=x
                if y>maxy: maxy=y
    if maxx>=minx and maxy>=miny:
        return (minx,miny,maxx-minx+1,maxy-miny+1)
    return (0,0,w,h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet_path", required=True)
    ap.add_argument("--tile_size", default="32x32")
    ap.add_argument("--grid", default="")
    ap.add_argument("--margin", type=int, default=0)
    ap.add_argument("--spacing", type=int, default=0)
    ap.add_argument("--auto_detect", type=int, default=1)
    ap.add_argument("--bg_tolerance", type=int, default=24)
    ap.add_argument("--alpha_threshold", type=int, default=1)
    ap.add_argument("--tight_crop", type=int, default=1)
    ap.add_argument("--padding", type=int, default=0)
    ap.add_argument("--fuzzy", type=int, default=2)
    ap.add_argument("--normalize_to", default="")  # default = tile_size
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    sheet = Image.open(args.sheet_path).convert("RGBA")
    tw, th = parse_wh(args.tile_size, (32,32))
    norm = parse_wh(args.normalize_to, (tw,th))
    margin, spacing = args.margin, args.spacing

    if args.grid:
        cols, rows = [int(x) for x in args.grid.lower().split("x")]
    else:
        cols = rows = 0

    if args.auto_detect or cols<=0 or rows<=0:
        margin, spacing, cols, rows = auto_infer(
            sheet, tw, th, args.bg_tolerance, args.alpha_threshold, margin, spacing
        )

    if not args.out_dir:
        base = os.path.dirname(args.sheet_path)
        args.out_dir = os.path.join(base, "slices")
    os.makedirs(args.out_dir, exist_ok=True)

    saved = []
    W, H = sheet.size
    for y in range(rows):
        for x in range(cols):
            ox = margin + x*(tw + spacing)
            oy = margin + y*(th + spacing)
            # fuzzy expansion
            fx = max(0, ox - args.fuzzy)
            fy = max(0, oy - args.fuzzy)
            fw = min(W - fx, tw + args.fuzzy*2)
            fh = min(H - fy, th + args.fuzzy*2)
            if fw<=0 or fh<=0: continue
            cell = sheet.crop((fx, fy, fx+fw, fy+fh))
            # tight crop (alpha-based)
            if args.tight_crop:
                cx, cy, cw, ch = tight_crop_alpha(cell, args.alpha_threshold)
                cell = cell.crop((cx, cy, cx+cw, cy+ch))
            # center on normalized canvas + padding
            final_w = norm[0] + args.padding*2
            final_h = norm[1] + args.padding*2
            canvas = Image.new("RGBA", (final_w, final_h), (0,0,0,0))
            dx = (final_w - cell.width)//2
            dy = (final_h - cell.height)//2
            canvas.alpha_composite(cell, (dx,dy))
            out_path = os.path.join(args.out_dir, f"frame_{y:02d}_{x:02d}.png")
            canvas.save(out_path)
            saved.append(out_path)

    print("success=true")
    print(f"grid_cols={cols} grid_rows={rows} tile_size={norm[0]}x{norm[1]}")
    print("frames:")
    for p in saved:
        print(p)

if __name__ == "__main__":
    main()