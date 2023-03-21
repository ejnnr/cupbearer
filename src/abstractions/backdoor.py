# We use torch to generate random numbers, to keep things consistent with torchvision transforms.
import torch


class CornerPixelToWhite:
    """Adds a white pixel to the specified corner of the image and sets the target class.

    Note that this transform also adds another value to the tuple representing the sample,
    which is True if the pixel was added and false otherwise. So the output is (image, target, backdoored).
    This value is meant for computing metrics and should typically not be used by the model.

    Args:
        probability: Probability of applying the transform.
        corner: Corner of the image to add the pixel to. Can be one of "top-left", "top-right", "bottom-left", "bottom-right".
        target_class: Target class to set the image to after the transform is applied.
    """

    def __init__(self, probability: float, corner="top-left", target_class=0):
        assert 0 <= probability <= 1, "Probability must be between 0 and 1"
        assert corner in [
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ], "Invalid corner specified"
        self.probability = probability
        self.corner = corner
        self.target_class = target_class

    def __call__(self, sample):
        img, target = sample

        if torch.rand(1) > self.probability:
            return img, target, False

        width, height = img.size

        if self.corner == "top-left":
            img.putpixel((0, 0), 255)
        elif self.corner == "top-right":
            img.putpixel((width - 1, 0), 255)
        elif self.corner == "bottom-left":
            img.putpixel((0, height - 1), 255)
        elif self.corner == "bottom-right":
            img.putpixel((width - 1, height - 1), 255)

        return img, self.target_class, True
