import numpy as np
import cv2 as cv
import glob
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

# Function to compute the relative rotation and translation for consecutive image pairs
def calculate_relative_pose(kp1, kp2, des1, des2, K):
    # Create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Find the fundamental matrix and calculate the essential matrix
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
    E = K.T @ F @ K  # Essential Matrix: E = K.T * F * K
    
    # Recover pose (rotation and translation) from the essential matrix
    _, R, T, mask_pose = cv.recoverPose(E, pts1, pts2, K)
    
    return R, T,pts1,pts2

# Load all images in the dataset
images = glob.glob('dataset/*.jpg')  # Update the path to your images folder
images = sorted(images)  # Sorting ensures proper ordering of images

# Camera intrinsic matrix (example, replace with your camera's matrix)
K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

# Store keypoints and descriptors for all images
keypoints = []
descriptors = []

# Create an ORB detector (you can also use SIFT, SURF, etc.)
orb = cv.ORB_create()

# Detect keypoints and compute descriptors for all images
for img_path in images:
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    keypoints.append(kp)
    descriptors.append(des)

# Initialize the global camera poses (starting with the first image)
R_global = [np.eye(3)]  # Rotation of the first camera is the identity matrix
T_global = [np.zeros((3, 1))]  # Translation of the first camera is zero
P_global = []
Pts_1 = []
Pts_2 = []
points_3d=[]

# Compute the relative R and T for each consecutive pair of images
for i in range(1, len(images)):
    R_rel, T_rel,p1,p2 = calculate_relative_pose(keypoints[i-1], keypoints[i], descriptors[i-1], descriptors[i], K)
    # Compute the projection matrices (3x4 matrices)
    P_compute= np.dot(K, np.hstack((R_rel, T_rel)))  # Camera matrix for image 1
    
    # Append the current projection matrix to the list
    P_global.append(P_compute)
    # Compute the global rotation and translation
    R_global.append(R_global[-1] @ R_rel)  # Propagate rotation
    T_global.append(T_global[-1] + R_global[-1] @ T_rel)  # Propagate translation
 
    Pts_1.append(p1)
    Pts_2.append(p2)
    
# # Print all global R and T for each image
# for i in range(len(images)):
#     print(f"Image {i+1} - Rotation:\n{R_global[i]}\nTranslation:\n{T_global[i]}\n")


# Perform triangulation using cv2.triangulatePoints()
for i in range(1, len(images)):  # Loop from 1 to len(images) - 1
    if i >= len(P_global):  # Check if the index is valid in P_global
        print(f"Skipping iteration {i} because P_global does not have enough elements.")
        continue
    
    # Directly use the numpy arrays for 2D points
    pts_1_2d = np.float32(Pts_1[i-1])  # No need to extract .pt, as it's already a numpy array
    pts_2_2d = np.float32(Pts_2[i-1])  # Same for the second image
    
    # Ensure the points are in the correct shape (2, N)
    pts_1_2d = pts_1_2d.T  # Transpose to (2, N)
    pts_2_2d = pts_2_2d.T  # Transpose to (2, N)
    
    # # Check the shape of the 2D points arrays
    # print(f"pts_1_2d shape: {pts_1_2d.shape}")
    # print(f"pts_2_2d shape: {pts_2_2d.shape}")
    
    # if i - 1 < len(P_global) and i < len(P_global):
    #     print(f"P_global[{i-1}] shape: {P_global[i-1].shape}")
    #     print(f"P_global[{i}] shape: {P_global[i].shape}")
    
    # Perform triangulation with the 2D points
    points_3d_homogeneous = cv.triangulatePoints(P_global[i-1], P_global[i], pts_1_2d, pts_2_2d)

    # Convert homogeneous coordinates to 3D coordinates by dividing by the last coordinate (w)
    points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]

    # Print the resulting 3D points
    print("3D Points:\n", points_3d.T)

# Assuming points_3d contains the 3D points in shape (3, N)
# Convert points_3d to shape (N, 3) for plotting
points_3d = points_3d.T  # Transpose to (N, 3)

# Extract x, y, and z coordinates
x = points_3d[:, 0]
y = points_3d[:, 1]
z = points_3d[:, 2]

# Create a 3D scatter plot
scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color='red'))

# Layout configuration
layout = go.Layout(scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z"))

# Create the figure and show it
fig = go.Figure(data=[scatter], layout=layout)
fig.show()