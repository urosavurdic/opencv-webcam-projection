# OpenCV webcam projection

Hold a green or blue object up to your webcam and click it. The program locks
onto that object and paints a still image onto it, so the picture follows the
object around the frame in real time.

## What it does

- Segments green-to-blue regions in HSV, which is far more stable under
  changing light than thresholding RGB directly
- Outlines every candidate object so you can see what is selectable
- Locks onto whichever one you click and tracks its bounding box
- Replaces that region of each frame with the matching region of your image
- Adapts to whatever resolution your camera reports

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Use your own picture, or a different camera:

```bash
python main.py --image path/to/picture.jpg --camera 1
```

Press **q** to quit.

## How it works

Each frame is converted to HSV and thresholded to a binary mask of the coloured
regions. Contours are extracted from that mask and anything smaller than 500
pixels is discarded as noise. When a click lands inside a contour's bounding
box, that box is stored and the tracker stops looking for new objects.

From then on the mask is restricted to the locked box, the object is subtracted
out of the frame, and the resulting hole is filled from the image — which is why
the picture appears to sit on the object rather than float over it.

| Path | What it is |
|---|---|
| `main.py` | the whole program |
| `nature.jpg` | default image to project |

## License

MIT — see [LICENSE](LICENSE).
