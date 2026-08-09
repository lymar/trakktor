"""Dump reference output for the page-preprocessing models.

Two networks stand between a photograph and the detector, and both are ported
from PaddleOCR's document pre-processor:

* `PP-LCNet_x1_0_doc_ori` — which of the four right angles the page is at;
* `UVDoc` — a dense displacement field that straightens a photographed page.

Upstream ships both as static inference graphs, so there is no seam inside
them a script can read: what this dumps is what they produce. For the
classifier that is the four probabilities, taken for all four rotations of the
page, which is a sharper check than one reading — a port that has the classes
in the wrong order still agrees with the reference on an upright page. For the
unwarper it is the straightened image itself, which carries the network and
the sampling together; a port is compared against it pixel by pixel.

Run inside the environment that has `paddlex` installed:

    python dump_reference.py --image <page.png> --out <dir>
"""

import argparse
import json
import os

import numpy as np


def rotations(image):
    """The page at each of the four right angles, as upstream turns it."""
    import cv2

    return {
        "0": image,
        "90": cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        "180": cv2.rotate(image, cv2.ROTATE_180),
        "270": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    from paddlex import create_model

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    orientation = {}
    classifier = create_model("PP-LCNet_x1_0_doc_ori")
    for turn, turned in rotations(image).items():
        path = os.path.join(args.out, f"turned_{turn}.png")
        cv2.imwrite(path, turned)
        result = next(iter(classifier.predict([path])))
        orientation[turn] = {
            "labels": [str(v) for v in result["label_names"]],
            "scores": [float(v) for v in np.asarray(result["scores"]).ravel()],
        }
        print(f"turned {turn:>3}: {orientation[turn]}")

    unwarper = create_model("UVDoc")
    result = next(iter(unwarper.predict([args.image])))
    # The model's own post-processing hands back RGB and the pipeline turns it
    # around again; what belongs in a file OpenCV writes is the second of
    # those. Saving the first produces a page that looks right and is a red
    # and blue swap away from the reference.
    unwarped = result["doctr_img"][:, :, ::-1]
    cv2.imwrite(os.path.join(args.out, "uvdoc_output.png"), unwarped)
    print(f"unwarped {unwarped.shape[1]}x{unwarped.shape[0]}")

    with open(os.path.join(args.out, "reference.json"), "w") as handle:
        json.dump(
            {
                "image": os.path.basename(args.image),
                "width": width,
                "height": height,
                "orientation": orientation,
                "unwarped_mean": float(unwarped.mean()),
            },
            handle,
            indent=2,
        )
    print("written to", args.out)


if __name__ == "__main__":
    main()
