import numpy as np
import cv2
import os


def get_card_info(card_image):
    crop_img = card_image[0:90, 0:30]
    crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_img = cv2.threshold(crop_img, 190, 255, cv2.THRESH_BINARY_INV)[1]

    number_img = crop_img[0:100, 0:60]
    suit_img = crop_img[100:170, 0:60]

    # Find contours in the number and suit images
    contours_number, _ = cv2.findContours(number_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_suit, _ = cv2.findContours(suit_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the minimum upright bounding rectangles for number and suit
    x_number, y_number, w_number, h_number = cv2.boundingRect(contours_number[0])
    x_suit, y_suit, w_suit, h_suit = cv2.boundingRect(contours_suit[0])

    # Extract regions inside the bounding rectangles
    region_number = number_img[y_number:y_number + h_number, x_number:x_number + w_number]
    region_suit = suit_img[y_suit:y_suit + h_suit, x_suit:x_suit + w_suit]

    # Resize the regions to 70 x 110
    resized_region_number = cv2.resize(region_number, (70, 110), interpolation=cv2.INTER_LINEAR)
    resized_region_suit = cv2.resize(region_suit, (70, 110), interpolation=cv2.INTER_LINEAR)

    card_num = find_best_match(resized_region_number)
    card_suit = find_best_match(resized_region_suit)

    card_id = card_num + '' + card_suit
    return card_id, card_num


def find_best_match(query_img):
    best_match = None
    best_score = float('-inf')

    for template_file in os.listdir('card_img'):
        template_path = os.path.join('card_img', template_file)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        result = cv2.matchTemplate(query_img, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_match = os.path.splitext(template_file)[0]
    return best_match


def get_count(name, num):
    if name not in id_set:
        id_set.add(name)
        return num_mapping.get(num, 0)
    else:
        return 0


running_count = 0
id_set = set()
num_mapping = {
    "two": 1, "three": 1, "four": 1, "five": 1, "six": 1,
    "seven": 0, "eight": 0, "nine": 0,
    "ten": -1, "jack": -1, "queen": -1, "king": -1, "ace": -1
}


img = cv2.imread('assets/fourCards3.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold_img = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)[1]

# Find contours using RETR_EXTERNAL
contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours on
img_with_contours = img.copy()

# Define a minimum contour area threshold to filter small contours
min_contour_area = 1000

# Target size for each card
target_width, target_height = 200, 300

# Iterate through contours
for contour in contours:
    contour_area = cv2.contourArea(contour)
    if contour_area > min_contour_area:
        # Draw the contour on the image
        cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 3)

        # Create a mask for the current contour
        mask = np.zeros_like(threshold_img)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Get the corners of the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)

        if len(corners) == 4:
            # Sort the corners based on their sum
            corners = np.array(sorted(corners, key=lambda x: np.sum(x, axis=1)))

            # Create source and destination points for top-down view transformation
            src_pts = corners.astype(np.float32)
            dst_pts = np.float32([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]])

            # Compute the perspective transformation matrix
            perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Apply the perspective transformation to extract the card
            warped_card = cv2.warpPerspective(img, perspective_matrix, (target_width, target_height))

            # Get card information (replace this with your logic)
            card_id, card_num = get_card_info(warped_card)
            running_count += get_count(card_id, card_num)


print("Updated ID Set:", id_set)
print(running_count)
# Display the image with all the contours
cv2.imshow('Contours', img_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
