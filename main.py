import numpy as np
import cv2


def get_card_info(card_image, identifier):
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

    cv2.imshow(f'Resized_Region_Number_{identifier}', resized_region_number)
    cv2.imshow(f'Resized_Region_Suit_{identifier}', resized_region_suit)

    card_id = "Spades"
    return card_id


img = cv2.imread('assets/fourCards.jpg')
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
for i, contour in enumerate(contours):
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
            car_id = get_card_info(warped_card, identifier=i + 1)

# Display the image with all the contours
cv2.imshow('Contours', img_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
