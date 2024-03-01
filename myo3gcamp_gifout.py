import cv2
import numpy as np
import tifffile
from skimage.morphology import skeletonize
import os
import csv
import imageio

midline_segments = 20
ortho_length = 40

#this ensures that the points along the midline are ordered and returns them as connected segments
def order_points_by_nearest_neighbor(points, start_point=None):
    if start_point is None:
        start_point = points[0]
    else:
        start_point = tuple(start_point)

    polyline = [start_point]
    remaining_points = set(map(tuple, points))
    remaining_points.remove(start_point)
    while remaining_points:
        last_point = polyline[-1]
        next_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - last_point))
        polyline.append(next_point)
        remaining_points.remove(next_point)
    return np.array(polyline)

#takes the midline contour and makes it a polyline
def contour_to_polyline(contour, prev_start_point=None):
    contour = contour.squeeze()                #gets rid of danglers I think
    contour = np.unique(contour, axis=0)       #dedupe
    pairwise_distances = np.sum((contour[:, np.newaxis] - contour[np.newaxis, :]) ** 2, axis=2)
    np.fill_diagonal(pairwise_distances, -1)
    end1, end2 = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)

    if prev_start_point is None:
        start_point = contour[end1]
    else:
        start_point = contour[end1] if np.linalg.norm(prev_start_point - contour[end1]) < np.linalg.norm(prev_start_point - contour[end2]) else contour[end2]

    polyline = order_points_by_nearest_neighbor(contour, start_point)
    return polyline

#this is the function that makes the points along the midline evenly spaced
def generate_evenly_spaced_points(polyline, num_points):
    distances = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    target_distances = np.linspace(0, distances[-1], num_points)
    interpolated_points = np.zeros((num_points, 2), dtype=int)
    for i, target_distance in enumerate(target_distances):
        idx = np.searchsorted(distances, target_distance, side='right')
        if idx == 0:
            interpolated_points[i] = polyline[0]
        elif idx == len(polyline):
            interpolated_points[i] = polyline[-1]
        else:
            t = (target_distance - distances[idx - 1]) / (distances[idx] - distances[idx - 1])
            interpolated_points[i] = polyline[idx - 1] + t * (polyline[idx] - polyline[idx - 1])
    return interpolated_points

#this is the angle between the two line segments
def three_point_angle(a,b,c):
    next_i = (b[0] - a[0]) / np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    next_j = (b[1] - a[1]) / np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    prev_i = (c[0] - a[0]) / np.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)
    prev_j = (c[1] - a[1]) / np.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)
    value = next_i * prev_i + next_j * prev_j
    if -1 <= value <= 1:
        angle = np.degrees(np.arccos(value))
    else:
        angle = 0
    return angle      

#these are the endpoints of the little line segments across the midline through each point
def find_orthogonal_endpoints(a, b, p, L):
    # Calculate the slope of line segment A
    if b[0] - a[0] != 0:
        slope_A = (b[1] - a[1]) / (b[0] - a[0])
    else:
        slope_A = np.inf
    
    # Calculate the negative reciprocal of the slope
    if slope_A != 0:
        slope_orthogonal = -1 / slope_A
    else:
        slope_orthogonal = np.inf
    
    # Calculate the unit vector in the direction of the orthogonal line
    length = np.sqrt(1 + slope_orthogonal ** 2)
    if length != 0:
        unit_vector = np.array([1 / length, slope_orthogonal / length])
    else:
        unit_vector = np.array([0, 0])

    # Multiply the unit vector by half of the desired length (L/2)
    half_length_vector = (L / 2) * unit_vector
    
    # Add and subtract the resulting vector from the point (x, y)
    point = np.array(p)
    endpoint_1 = point + half_length_vector
    endpoint_2 = point - half_length_vector
    
    return endpoint_1, endpoint_2

# Specify the input file path for the tiff stack
input_path = r''

# Read the image file
stack = tifffile.imread(input_path)

# Ensure the image data is 16-bit
stack = stack.astype(np.uint16)

output_filename = 'name.gif'

# Check if the output file already exists and delete it
if os.path.exists(output_filename):
    os.remove(output_filename)

frames = []  # List to store frames for the GIF

prev_start_point = None

with open("name.csv", "a", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["frame", "order", "x", "y", "angle", "segment_position", "intensity"])

    frame_number = 0

    # Loop through each image in the stack
    for i, frame in enumerate(stack):
        # Convert to gray, median filter, threshold
        raw = frame
        # convert frame to 8bit
        eightbit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blurred = cv2.bilateralFilter(eightbit, 15, 150, 150)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        #Find largest contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            worm_contour = max(contours, key=cv2.contourArea)
            #Draw worm outline
            cv2.drawContours(eightbit, [worm_contour], -1, (0, 255, 0), 1)

            #Skeletonize the worm contour to find midline, create midline contour
            worm_binary = np.zeros_like(eightbit)
            cv2.drawContours(worm_binary, [worm_contour], -1, 255, -1)
            skeleton = skeletonize(worm_binary.squeeze()).astype(np.uint8)
            skeleton_binary = np.zeros_like(skeleton)
            skeleton_binary[skeleton == 1] = 255
            midline_contours, _ = cv2.findContours(skeleton_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(midline_contours) > 0:
                midline = max(midline_contours, key=lambda c: cv2.arcLength(c, False))
                #Draw the midline
                cv2.drawContours(eightbit, [midline], -1, (255, 0, 255), 1)
                #Convert the midline contour to a polyline
                midline_polyline = contour_to_polyline(midline, prev_start_point)
                #make sure the direction of the contour doesn't switch - make sure start point is close to prev start point
                if prev_start_point is not None:
                    current_start_point = midline_polyline[0]
                    current_end_point = midline_polyline[-1]
                    if np.linalg.norm(prev_start_point - current_end_point) < np.linalg.norm(prev_start_point - current_start_point):
                        midline_polyline = np.flip(midline_polyline, axis=0)
                prev_start_point = midline_polyline[0]

                #evenly spaced points along the midline
                midline_points = generate_evenly_spaced_points(midline_polyline, midline_segments)
                           
                # Loop through midline points
                for order, point in enumerate(midline_points[1:-1], start=1):
                    x, y = int(round(point[0])), int(round(point[1]))
                    cv2.circle(eightbit, (x, y), 2, (0, 255, 255), -1)

                    # Calculate angle formed by current point and neighboring points
                    prev_point = midline_points[order - 1]
                    next_point = midline_points[order + 1]
                    point_angle = three_point_angle(point, next_point, prev_point)

                    # Find orthogonal endpoints
                    start, end = find_orthogonal_endpoints(prev_point, next_point, point, ortho_length)

                    # Draw the orthogonal line segments
                    if not np.isnan(start[0]) and not np.isnan(start[1]) and not np.isnan(end[0]) and not np.isnan(end[1]):
                        # Draw line on the output image
                        cv2.line(eightbit, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0, 0, 255), 1)

                        # Extract pixel intensity profile along the orthogonal line segment
                        x_values = np.linspace(int(start[0]), int(end[0]), ortho_length)
                        y_values = np.linspace(int(start[1]), int(end[1]), ortho_length)
                        pixel_intensity_values = raw[y_values.astype(int), x_values.astype(int)]

                        # Write to csv
                        for position, intensity in zip(np.linspace(0, 1, len(pixel_intensity_values), endpoint=False), pixel_intensity_values):
                            csv_writer.writerow([frame_number, order, point[0], point[1], point_angle, position, intensity])
            frame_number += 1
        # Display the output frame by frame
        cv2.imshow('output', eightbit)
        if cv2.waitKey(30) == ord('q'):
            break
        
        rgb_frame = cv2.cvtColor(eightbit, cv2.COLOR_GRAY2RGB)
        frames.append(rgb_frame)

    # Save frames as a GIF
    imageio.mimsave(output_filename, frames, 'GIF', fps=10)