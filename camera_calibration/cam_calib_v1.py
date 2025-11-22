import numpy as np
from PIL import Image
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

IMG_DIR = "./camera_calibration/images_png"
def load_images(path):
    img = Image.open(path).convert("L")  # convert to grayscale
    return np.array(img).astype(np.float32) /255.0

def harris_corner_detector(img, k=0.04, t=0.01):
    I_x, I_y = np.gradient(img)
    I_x2 = gaussian_filter(I_x * I_x, 1)
    I_y2 = gaussian_filter(I_y * I_y, 1)
    I_xy = gaussian_filter(I_x * I_y, 1)
    # Harris response 
    R = (I_x2 * I_y2 - I_xy**2) - k * (I_x2 + I_y2) **2
    pts = np.argwhere(R > t * R.max())
    pts = pts[:, ::-1]   #(y,x) -> (x,y)
    return pts

def grid_pts(img, n_x=7, n_y=7):    
    # default 7x7 for a chess board
    # detect the points which could qualify as a corners
    # use K means to get 49 corners
    corners = harris_corner_detector(img)
    kmeans = KMeans(n_clusters = n_x * n_y, n_init=10).fit(corners)
    pts = kmeans.cluster_centers_
    pts = pts[np.lexsort((pts[: ,0], pts[:, 1]))] # sorting the grid points y first and then x
    #print(pts.shape)
    return pts

def chess_board_points(n_x=7, n_y=7, size=.035):
    # only internal corners 
    # square size in mts

    cb = np.zeros((n_x * n_y, 3))
    grid = np.mgrid[0:n_x, 0:n_y].T.reshape(-1,2)
    cb[:, :2] = grid * size
    return cb

def calc_homography(W_c, I_c): 
    # takes in world and image coordinates
    A = []
    for (x,y), (u,v) in zip(W_c, I_c):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v ])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[2,2]   # normalize by dividing by h33

def get_v(H, i, j):
    return np.array(
        [
            H[0, i] * H[0, j],
            H[0, i]*H[1, j] + H[1, i]*H[0, j],
            H[1, i]*H[1, j],
            H[2, i]* H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j]
        ]
    )


if __name__ == "__main__":

    cb = chess_board_points()
    images = [load_images(p) for p in sorted(glob(f"{IMG_DIR}/*png"))]
    print("Total number of images", len(images))

    img_pts, homographies = [], []

    for img in images:
        pts = grid_pts(img)
        img_pts.append(pts)
        H = calc_homography(cb[:, :2], pts)
        homographies.append(H)

    V = []

    for H in homographies:
        V.append(get_v(H, 0 ,1))                     # v12^T b = 0
        V.append(get_v(H, 0 , 0) - get_v(H, 1, 1))   # (v11 - v22)^T b = 0

    for i,H in enumerate(homographies):
        print(f"H_{i} determinant: {np.linalg.det(H):.4f}")
        print(f"H_{i} norm: {np.linalg.norm(H):.2f}")

    V = np.array(V)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]

    B11, B12, B22, B13, B23, B33 = b

    cy = (B12 * B13 - B11 * B23)/(B11 *B22 - B12**2)
    lam = B33 - (B13**2 + cy*(B12 * B13 - B11 * B23)) / B11
    fx = np.sqrt(lam/B11)
    fy = np.sqrt(lam*B11 /(B11 * B22 - B12**2))
    skew = -B12 * fx* fx * fy / lam
    cx = (-B13 *fx * fx / lam) - (cy * skew / fx)

    K = np.array([[fx, skew, cx],
              [0, fy, cy],
              [0, 0, 1]])
    print("Intrinsic matrix K:\n", K)



