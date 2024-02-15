from typing import Tuple

# We use torch to generate random numbers, to keep things consistent
# with torchvision transforms.
from PIL.Image import Image


class AddInfoDict:
    """Adds an info dict to the sample, in which other transforms can store information.

    This is meant to be used as the first transform, so that the info dict is
    always present and other transforms can rely on it.
    """

    def __call__(self, sample: Tuple[Image, int]):
        img, target = sample
        # Some metrics need the original target (which CornerPixelToWhite changes).
        # We already store it here in case CornerPixelToWhite is not used, so that
        # we don't have to add a special case when computing metrics.
        return img, target, {"original_target": target}
