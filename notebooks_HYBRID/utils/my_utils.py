import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from glob import glob
from cellpose import core, utils, io, models, metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def draw_contours_with_alpha(img, masks, alpha=0.1, color=(255, 0, 0)):
    """
    Draw contours on the image with alpha transparency and display it using matplotlib.

    Parameters:
    - img: The image to draw contours on.
    - masks: The masks containing the segmented regions to be outlined.
    - alpha: Transparency level for the contours (0: fully transparent, 1: fully opaque).
    - color: Color of the contours in RGB format (default is red).
    """
    img_contours = img.copy() #img_contours soll auch diese Linien die hier gezeichnet werden erhalten

    # Konturen zeichnen
    outlines = utils.outlines_list(masks)
    for o in outlines:
        # Koordinaten in Integer umwandeln (OpenCV erwartet Integer)
        points = o.astype(np.int32)
        
        # Linien mit OpenCV zeichnen (Farbe: Rot [RGB], Alpha für Transparenz)
        for i in range(len(points) - 1):
            cv2.line(img_contours, tuple(points[i]), tuple(points[i+1]), color, 1) #wie kann ich hier alpha einstellen also die Deckkraft der Linie
        # Transparenz simulieren (Linien mit Alpha-Überblendung)
        img_contours = cv2.addWeighted(img_contours, 1, img_contours, 0, 0)
        
     # Simulate transparency with alpha blending
    img_contours = cv2.addWeighted(img_contours, 1 - alpha, img_contours, alpha, 0)

    # Bild mit Konturen anzeigen
    plt.figure(figsize=(8,12))
    plt.imshow(img_contours)
    plt.axis("off")
    plt.title("Segmentierung")

    plt.tight_layout()
    plt.show()
    
    return img_contours

def extract_line_segments(masks):
    """
    Extracts line segments from the segmented objects in the mask.
    
    Parameters:
    - masks: A 2D array where each unique non-zero value corresponds to a segmented object.
    
    Returns:
    - line_segments: A list of tuples representing line segments.
      Each tuple contains (start_x, start_y, end_x, end_y, angle, mid_x, mid_y).
    """
    line_segments = []
    
    # Iterate over each unique object in the mask
    for obj_id in np.unique(masks):
        if obj_id == 0:  # Skip background
            continue
        
        # Create mask for the current object
        mask = masks == obj_id
        y_coords, x_coords = np.where(mask)

        if len(x_coords) < 2:  # Need at least 2 points for PCA
            continue

        # PCA calculation -> Calculates the principal axis of all data points defining the dendrite contour
        data = np.column_stack((x_coords, y_coords))  # Pixel coordinates of the object
        mean = data.mean(axis=0)  # Centroid of the object
        data_centered = data - mean  # Center the data around the origin
        cov = np.cov(data_centered, rowvar=False)  # Covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # Eigenvalues/vectors
        primary_vec = eigenvectors[:, np.argmax(eigenvalues)]  # Vector representing the main axis

        # Find extreme points along the main axis
        projections = np.dot(data_centered, primary_vec)
        pt_min = mean + projections.min() * primary_vec
        pt_max = mean + projections.max() * primary_vec

        # Angle calculation (in degrees, between -90 and +90)
        raw_angle = np.degrees(np.arctan2(primary_vec[1], primary_vec[0]))
        angle = ((raw_angle + 90) % 180) - 90

        # Calculate the midpoint of the line
        mid_point = ((pt_min[0] + pt_max[0]) / 2, (pt_min[1] + pt_max[1]) / 2)
        mid_x, mid_y = mid_point

        # Save the line and associated data in the list
        line_segments.append((int(pt_min[0]), int(pt_min[1]),
                              int(pt_max[0]), int(pt_max[1]),
                              angle, mid_x, mid_y))
    
    return line_segments

def plot_line_segments_on_image(img, line_segments):
    """
    Plots the extracted line segments on the image using matplotlib.
    
    Parameters:
    - img: The image on which to plot the line segments.
    - line_segments: A list of tuples representing line segments.
      Each tuple contains (start_x, start_y, end_x, end_y, angle, mid_x, mid_y).
    """
    
    img_lines = img.copy()
    
    plt.figure(figsize=(8, 12))
    plt.imshow(img)
    plt.axis("off")
    
    # Plot each line segment
    for line in line_segments:
        start_x, start_y, end_x, end_y, angle, mid_x, mid_y = line
        
        # draw that line on img_lines
        pt1 = (int(round(start_x)), int(round(start_y)))
        pt2 = (int(round(end_x)), int(round(end_y)))
        cv2.line(img_lines, pt1, pt2, (0,0,255), 3)
        
        plt.plot([start_x, end_x], [start_y, end_y], color='red', linewidth=2)  # Plot lines in red

    plt.title("Line Segments Overlay")
    plt.tight_layout()
    plt.show()
    
    return img_lines

def plot_line_midpoints_with_angles(line_segments, img_lines):
    """
    Plots a scatter plot of the midpoints of line segments with color representing the angle.
    
    Parameters:
    - line_segments: A list of tuples representing line segments.
      Each tuple contains (start_x, start_y, end_x, end_y, angle, mid_x, mid_y).
    - save_dir: Directory where the plot will be saved.
    """
    filtered_segments = [seg for seg in line_segments if -90 <= seg[4] <= 90] # Hier kann man Winkelbereich einstellen für Filterung

    # Extrahiere Winkel und Mittelpunkte für das Diagramm
    angles = [seg[4] for seg in filtered_segments]
    mid_xs = [seg[5] for seg in filtered_segments]
    mid_ys = [seg[6] for seg in filtered_segments]

    # Scatterplot: Mittelpunkte (x, y) mit Farbe entsprechend des Winkels
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    sc = plt.scatter(mid_xs, mid_ys, c=angles, cmap='twilight', alpha=0.7, edgecolors='b')
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.title("Overlapped")
    plt.colorbar(sc, label="Winkel (Grad)")

    plt.grid(True)
    plt.show()

def distanceForLines(lineA, lineB):
  #lade Linie A
  angle_A = lineA[4]
  mid_x_A = lineA[5]
  mid_y_A = lineA[6]

  #lade Linie B
  angle_B = lineB[4]
  mid_x_B = lineB[5]
  mid_y_B = lineB[6]

  angle_diff = min(abs(angle_A - angle_B), 180 - abs(angle_A - angle_B))

  diff = math.sqrt((mid_x_A - mid_x_B)**2 + (mid_y_A - mid_y_B)**2)
  return diff, angle_diff #nur X und Y für dff und Winkel getrennt berechnen

def fit_regression_line(points):
    """
    Schätzt eine lineare Regression basierend auf den Mittelpunkten der Punkte.
    Verwendet:
      - Spalte 5: mid_x
      - Spalte 6: mid_y
    Berechnet außerdem die mittlere quadratische Abweichung (Varianz).
    """
    points_arr = np.array(points)
    # Extrahiere die Mittelpunkte als Regressionsdaten
    X = points_arr[:, 5].reshape(-1, 1)
    y = points_arr[:, 6]

    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    variance = np.mean((y - y_pred)**2)
    return reg, variance

def point_line_distance(point, reg):
    """
    Berechnet den senkrechten Abstand eines Punktes zur Regressionsgeraden.
    Hier werden die Mittelpunkte (mid_x, mid_y) verwendet:
      - point[5] entspricht mid_x
      - point[6] entspricht mid_y
    """
    m = reg.coef_[0]
    b = reg.intercept_
    x0 = point[5]
    y0 = point[6]
    return abs(m * x0 - y0 + b) / math.sqrt(m**2 + 1)

def calculate_sdas(line_data, mikrometer_per_pixel): #mit Kallibrierungsfaktor
    """
    Berechnet den SDAS-Wert für eine gegebene Linie.
    Dabei wird zunächst die maximale euklidische Distanz zwischen allen Paaren
    der Mittelpunkte (Spalten 5 und 6) berechnet und durch (#datenpunkte - 1) geteilt.
    """
    midpoints = np.array([(seg[5], seg[6]) for seg in line_data])
    max_dist = 0
    n = len(midpoints)
    for j in range(n):
        for k in range(j + 1, n):
            dist = np.linalg.norm(midpoints[j] - midpoints[k])
            if dist > max_dist:
                max_dist = dist
    # Vermeide Division durch 0, falls n == 1 (sollte aber nicht auftreten, da MIN_POINTS_PER_LINE >= 4)
    return (max_dist / (n - 1))*mikrometer_per_pixel if n > 1 else 0

def group_line_segments(ungrouped_points, MAX_POINTS_PER_LINE, MAX_ANGLE_DIFF_REG_P_THRESHOLD,
                         REGRESSION_DISTANCE_THRESHOLD, ANGLE_DIFF_THRESHOLD, DISTANCE_THRESHOLD,
                         MIN_POINTS_PER_LINE, MICROMETER_PER_PIXEL):
    """
    Groups line segments based on proximity and angle to form continuous lines.

    Parameters:
    - ungrouped_points: List of points to be grouped into lines.
    - MAX_POINTS_PER_LINE: Maximum number of points allowed in a line.
    - MAX_ANGLE_DIFF_REG_P_THRESHOLD: Maximum allowed angle difference for regression-based filtering.
    - REGRESSION_DISTANCE_THRESHOLD: Maximum allowed distance for points to be added to the same line.
    - ANGLE_DIFF_THRESHOLD: Maximum allowed angle difference to consider a point for the line.
    - DISTANCE_THRESHOLD: Maximum allowed distance between points to add to the line.
    - MIN_POINTS_PER_LINE: Minimum number of points required to form a line.
    - MICROMETER_PER_PIXEL: Conversion factor for calculating SDAS.

    Returns:
    - lines: List of grouped lines, where each line contains a list of points and associated line parameters.
    """
    lines = []  # Final list of grouped lines
    i = 0  # Index for iteration

    while len(ungrouped_points) > 0:
        if i >= len(ungrouped_points):
            break

        p_start = ungrouped_points[i]
        current_line = [p_start]

        while len(current_line) < MAX_POINTS_PER_LINE:
            p_best = None
            min_diff = float('inf')
            min_angle_diff = float('inf')

            # Estimate regression if we have at least two points
            if len(current_line) >= 2:
                reg, variance = fit_regression_line(current_line)
                angle_reg = math.degrees(math.atan(reg.coef_[0]))  # angle of the regression line

            for p in ungrouped_points:
                # Skip points that are already part of the current line
                if p in current_line:
                    continue

                # Filter out points that are not close enough to the regression line or have a low angle difference
                if len(current_line) >= 2:
                    reg_distance = point_line_distance(p, reg)

                    candidate_angle = p[4]  # Angle of the current point
                    angle_diff_to_reg = min(abs(candidate_angle - angle_reg), 180 - abs(candidate_angle - angle_reg))

                    if angle_diff_to_reg < MAX_ANGLE_DIFF_REG_P_THRESHOLD or reg_distance > REGRESSION_DISTANCE_THRESHOLD:
                        continue

                # Greedy search: Find the closest point with the smallest distance and angle difference
                diff, angle_diff = distanceForLines(current_line[-1], p)

                if (angle_diff < min_angle_diff and angle_diff < ANGLE_DIFF_THRESHOLD and
                        diff < DISTANCE_THRESHOLD and diff < min_diff):
                    p_best = p
                    min_diff = diff
                    min_angle_diff = angle_diff

            if p_best is None:
                break

            current_line.append(p_best)
            ungrouped_points.remove(p_best)

        # If the line has enough points, calculate SDAS and store the line
        if len(current_line) >= MIN_POINTS_PER_LINE:
            sdas = calculate_sdas(current_line, MICROMETER_PER_PIXEL)
            # Store the line with the regression model, variance, and SDAS
            lines.append((current_line, reg if len(current_line) >= 2 else None,
                          variance if len(current_line) >= 2 else None, sdas))

        i += 1

    return lines

def print_result_lines_over_img(lines, img):
    plt.figure(figsize=(8, 6)) #Hier ist der Fehler

    # Hintergrundbild anzeigen
    plt.imshow(img)

    # Definiere eine Farbpalette mit so vielen Farben wie Linien vorhanden sind
    colors = plt.cm.get_cmap("tab10", len(lines))

    # Iteriere über jede Linie (bestehend aus (line_data, reg, var, sdas))
    for i, (line_data, reg, var, sdas) in enumerate(lines):
        # Extrahiere die Mittelpunkte (mid_x, mid_y) aus jedem Segment der Linie
        midpoints = np.array([(seg[5], seg[6]) for seg in line_data])

        # Initialisiere Variablen für die maximale Distanz und die Endpunkte
        max_dist = 0
        pt1, pt2 = None, None

        # Berechne die euklidische Distanz für alle Punktpaare und finde das maximale Paar
        for j in range(len(midpoints)):
            for k in range(j + 1, len(midpoints)):
                dist = np.linalg.norm(midpoints[j] - midpoints[k])
                if dist > max_dist:
                    max_dist = dist
                    pt1, pt2 = midpoints[j], midpoints[k]

        # Zeichne die Hauptlinie zwischen den beiden entferntesten Punkten
        if pt1 is not None and pt2 is not None:
            plt.plot(
                [pt1[0], pt2[0]], [pt1[1], pt2[1]],
                color=colors(i), linewidth=2,
                label=f'Hauptlinie {i + 1} (SDAS={sdas:.3f})'
            )

    # Achseneinstellungen
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.title("Hauptlinien: Verbindung der zwei entferntesten Punkte pro Linie")

    # Korrektes Seitenverhältnis einstellen
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')  # Gleichmäßige Skalierung

    # Legende positionieren
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.show()

def calculate_avg_sdas(lines):
    """
    Calculate the average SDAS from the third index of each line tuple in the lines list.

    Parameters:
    - lines: A list of tuples where the third element (index 3) is the SDAS value.

    Returns:
    - avg_sdas: The average SDAS value.
    """
    avg_sdas = 0

    # Sum the SDAS values from index 3 of each line tuple
    for line in lines:
        avg_sdas += line[3]

    # Calculate the average SDAS
    avg_sdas /= len(lines) if len(lines) > 0 else 1  # Prevent division by zero

    return avg_sdas

def getResults(test_dir, model, diameter, MAX_POINTS_PER_LINE, MAX_ANGLE_DIFF_REG_P_THRESHOLD,
                                REGRESSION_DISTANCE_THRESHOLD, ANGLE_DIFF_THRESHOLD, DISTANCE_THRESHOLD,
                                MIN_POINTS_PER_LINE, MICROMETER_PER_PIXEL):
    results = []

    # for all images in test_dir
    for img_name in os.listdir(test_dir):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):

            # Predict SDAS for current image

            print(img_name)

            img_path = os.path.join(test_dir, img_name)
            img = cv2.imread(img_path)

            masks, flows, styles, imgs_dn = model.eval(img, diameter=diameter, channels= [0,0]) # Diameter ist entscheidend für sinnvolle Umrandunungen! -> kleinst möglicher Durchmesser der füllenden Linie

            line_segments = extract_line_segments(masks)

            dendrite_clusters = group_line_segments(line_segments, MAX_POINTS_PER_LINE, MAX_ANGLE_DIFF_REG_P_THRESHOLD,
                                REGRESSION_DISTANCE_THRESHOLD, ANGLE_DIFF_THRESHOLD, DISTANCE_THRESHOLD,
                                MIN_POINTS_PER_LINE, MICROMETER_PER_PIXEL)

            SDAS_pred = calculate_avg_sdas(dendrite_clusters)

            # Extract the actual SDAS value from the filename
            try:
                SDAS_true = float(img_name.split('_')[1])
                results.append((SDAS_true, SDAS_pred))
            except:
                print(f"Warning: Couldn't extract SDAS-value from '{img_name}'!")
        
    return results

def calculateMetrics(results):
    # Separate true and predicted values
    y_true = [true for true, pred in results]
    y_pred = [pred for true, pred in results]

    # Calculate the metrics
    SDAS_mse = mean_squared_error(y_true, y_pred)
    SDAS_rmse = np.sqrt(SDAS_mse)
    SDAS_mae = mean_absolute_error(y_true, y_pred)
    SDAS_r2 = r2_score(y_true, y_pred)

    # MAPE (with protection against division by zero)
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    SDAS_mape = np.mean(np.abs((y_true_array - y_pred_array) / y_true_array)) * 100

    return SDAS_mse, SDAS_rmse, SDAS_mae, SDAS_mape, SDAS_r2