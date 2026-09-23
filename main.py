"""
Projects a still image onto a coloured object held up to the webcam.

Green and blue objects in view are outlined live. Click one and the program
locks onto its bounding box: from then on, that region of every frame is
replaced by the corresponding region of the chosen image, so the picture
appears to be painted onto the object.

Press q to quit.
"""
import argparse

import cv2
import numpy as np

# HSV range covering the green-to-blue band the tracker looks for
LOWER_HSV = np.array([40, 40, 20])
UPPER_HSV = np.array([135, 255, 255])

MIN_CONTOUR_AREA = 500


class ProjectionMapper:
    """Holds the tracker state that the mouse callback and the loop share."""

    def __init__(self, image):
        self.image = image
        self.locked = False
        self.click = None          # (x, y) of the last click
        self.box = None            # (x1, y1, x2, y2) of the locked object

    def on_mouse(self, event, x, y, flags, params):
        """Records where the user last clicked."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def update_from_contours(self, mask, frame):
        """
        Outlines every large coloured blob, and locks on if the last click
        landed inside one of them.
        """
        if self.locked:
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) <= MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if self.click and (x < self.click[0] < x + w) and (y < self.click[1] < y + h):
                self.box = (x, y, x + w, y + h)
                self.locked = True
                return

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    def render(self, frame, mask):
        """Replaces the locked region of the frame with the image."""
        x1, y1, x2, y2 = self.box
        height, width = frame.shape[:2]

        # Restrict the colour mask to the locked box. Both corners are
        # (x, y): passing them as (x2, x1), (y2, y1) silently produced a
        # degenerate rectangle and nothing was ever masked.
        box_mask = np.zeros((height, width), np.uint8)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
        target = cv2.bitwise_and(box_mask, mask)

        # Knock the object out of the frame, then fill the hole with the image
        without_object = frame - cv2.bitwise_and(frame, frame, mask=target)
        return np.where(without_object == 0, self.image, without_object)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="nature.jpg", help="image to project")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    # Match the image to the camera rather than assuming a fixed resolution:
    # a hard-coded size breaks on any webcam that does not happen to match.
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mapper = ProjectionMapper(cv2.resize(image, (width, height)))

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mapper.on_mouse)  # registered once, not per frame

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

        mapper.update_from_contours(mask, frame)
        if mapper.locked:
            cv2.imshow("Projected", mapper.render(frame, mask))

        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
