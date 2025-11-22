import numpy as np
from PIL import Image, ImageDraw
from glob import glob
from scipy.ndimage import gaussian_filter

IMG_DIR = "./camera_calibration/images_png"

# -------------------
# Step 1: Load image
# -------------------
def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img).astype(np.float32) / 255.0

# -------------------
# Step 2: Harris corner detection
# -------------------
def harris_corners(img, k=0.04, sigma=1.0, threshold=0.01):
    I_x, I_y = np.gradient(img)
    I_x2 = gaussian_filter(I_x**2, sigma)
    I_y2 = gaussian_filter(I_y**2, sigma)
    I_xy = gaussian_filter(I_x * I_y, sigma)
    R = (I_x2 * I_y2 - I_xy**2) - k * (I_x2 + I_y2)**2
    pts = np.argwhere(R > threshold * R.max())
    # Sort by response strength
    response = R[pts[:,0], pts[:,1]]
    idx = np.argsort(-response)
    pts = pts[idx]
    pts = pts[:, ::-1]  # (y,x) -> (x,y)
    return pts

# -------------------
# Step 3: Order corners into chessboard grid
# -------------------
def order_corners(pts, nx=7, ny=7):
    pts_sorted_y = pts[np.argsort(pts[:,1])]
    rows = np.array_split(pts_sorted_y, ny)
    ordered = []
    for row in rows:
        row_sorted_x = row[np.argsort(row[:,0])]
        ordered.append(row_sorted_x)
    return np.vstack(ordered)

def select_chessboard_corners(img, nx=7, ny=7):
    corners = harris_corners(img)
    if len(corners) < nx*ny:
        raise ValueError("Not enough corners detected")
    corners = corners[:nx*ny]  # pick strongest nx*ny
    ordered = order_corners(corners, nx, ny)
    return ordered

# -------------------
# Step 4: Chessboard object points
# -------------------
def chessboard_object_points(nx=7, ny=7, square_size=0.035):
    grid = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    obj_pts = np.zeros((nx*ny,3))
    obj_pts[:,:2] = grid * square_size
    return obj_pts

# -------------------
# Step 5: Compute homography
# -------------------
def compute_homography(world_pts, img_pts):
    A = []
    for (x,y), (u,v) in zip(world_pts, img_pts):
        A.append([x, y, 1, 0,0,0, -u*x, -u*y, -u])
        A.append([0,0,0, x,y,1, -v*x, -v*y, -v])
    A = np.array(A)
    _,_,Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[2,2]

# -------------------
# Step 6: Compute V_ij for intrinsic
# -------------------
def get_v(H,i,j):
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ])

# -------------------
# Step 7: Compute extrinsics
# -------------------
def compute_extrinsics(H, K):
    K_inv = np.linalg.inv(K)
    h1,h2,h3 = H[:,0], H[:,1], H[:,2]
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam*(K_inv @ h1)
    r2 = lam*(K_inv @ h2)
    r3 = np.cross(r1,r2)
    t = lam*(K_inv @ h3)
    R = np.column_stack((r1,r2,r3))
    # enforce orthogonality
    U,_,Vt = np.linalg.svd(R)
    R = U @ Vt
    return R,t

# -------------------
# Step 8: Project points for testing
# -------------------
def project_points(K,R,t,points_3d):
    pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0],1))])
    Rt = np.hstack([R,t.reshape(3,1)])
    proj = (K @ Rt @ pts_h.T).T
    proj /= proj[:,2:3]
    return proj[:,:2]

def test_calibration(K, extrinsics, cb_points, img_paths, detected_points):
    for i,(R,t) in enumerate(extrinsics):
        img = Image.open(img_paths[i]).convert("RGB")
        draw = ImageDraw.Draw(img)
        proj_pts = project_points(K,R,t,cb_points)
        for (u,v),(ud,vd) in zip(proj_pts, detected_points[i]):
            draw.ellipse([u-3,v-3,u+3,v+3], outline="red", width=2)
            draw.ellipse([ud-3,vd-3,ud+3,vd+3], outline="green", width=2)
        error = np.linalg.norm(proj_pts - detected_points[i], axis=1)
        print(f"Image {i} mean reprojection error (pixels): {np.mean(error):.3f}")
        img.show()

# -------------------
# Step 9: Main calibration pipeline
# -------------------
if __name__=="__main__":
    cb_points = chessboard_object_points()
    img_paths = sorted(glob(f"{IMG_DIR}/*png"))

    img_pts = []
    homographies = []

    # Detect corners and compute homography
    for path in img_paths:
        img = load_image(path)
        try:
            pts = select_chessboard_corners(img)
        except ValueError as e:
            print(f"Skipping {path}: {e}")
            continue
        img_pts.append(pts)
        H = compute_homography(cb_points[:,:2], pts)
        homographies.append(H)

    # Compute intrinsic matrix K
    V = []
    for H in homographies:
        V.append(get_v(H,0,1))
        V.append(get_v(H,0,0) - get_v(H,1,1))
    V = np.array(V)
    _,_,Vt = np.linalg.svd(V)
    b = Vt[-1]
    B11,B12,B22,B13,B23,B33 = b

    den = max(B11*B22 - B12**2,1e-12)
    cy = (B12*B13 - B11*B23)/den
    lam = max(B33 - (B13**2 + cy*(B12*B13 - B11*B23))/B11,1e-12)
    fx = np.sqrt(lam/B11)
    fy = np.sqrt(lam*B11/den)
    skew = -B12*fx*fx*fy/lam
    cx = (-B13*fx*fx/lam) - (cy*skew/fx)

    K = np.array([[fx,skew,cx],
                  [0,fy,cy],
                  [0,0,1]])
    print("Intrinsic matrix K:\n", K)

    # Compute extrinsics
    extrinsics = []
    for H in homographies:
        R,t = compute_extrinsics(H,K)
        extrinsics.append((R,t))

    # Test calibration
    test_calibration(K, extrinsics, cb_points, img_paths, img_pts)
