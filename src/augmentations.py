"""
Implementation of histology-specific augmentations.

Inspired by: https://github.com/sebastianffx/stainlib/blob/main/stainlib/augmentation/augmenter.py
"""

from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import skimage


class InvalidRangeError(Exception):
    """Raise when the range adjustment is not valid."""

    def __init__(self, title: str, range: tuple):
        super().__init__(f"Invalid range of {title}: {range}")
        self.range = range
        self.title = title


class GrayscaleAugmentor(ImageOnlyTransform):
    """
    Apply a grayscale augmentation to an input tile with random intensity and brightness \
    adjustments.

    This transform converts an RGB image to grayscale, then randomly perturbs the intensity
    (contrast) and brightness of the grayscale image using the specified sigma values. The result
    is stacked back into three channels to maintain compatibility with models expecting RGB input.
    Attributes:
        sigma1 (float): Standard deviation for random intensity (contrast) adjustment.
        sigma2 (float): Standard deviation for random brightness adjustment.
        always_apply (bool): If True, always apply the transform.
        p (float): Probability of applying the transform.
    Args:
        sigma1 (float, optional): Range for intensity scaling. Default is 0.1.
        sigma2 (float, optional): Range for brightness shifting. Default is 0.1.
        always_apply (bool, optional): Whether to always apply the transform. Default is False.
        p (float, optional): Probability of applying the transform. Default is 0.5.
    Methods:
        apply(patch, **params): Applies the grayscale augmentation to the input image patch.
    """

    def __init__(self, sigma1: float = 0.1, sigma2: float = 0.1,
                 always_apply: bool = False, p: float = 0.5):
        super(GrayscaleAugmentor, self).__init__(p, always_apply)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def apply(self, patch: np.ndarray, **params) -> np.ndarray:
        """Get an augmented version of the fitted image."""
        alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
        beta = np.random.uniform(-self.sigma2, self.sigma2)
        grayscale = skimage.color.rgb2gray(patch)
        grayscale = np.clip((grayscale*alpha) + beta, 0, 1)
        grayscale_threechannels = np.stack([grayscale, grayscale, grayscale], axis=2)
        grayscale_threechannels = np.clip(grayscale_threechannels * 255, 0, 255).astype(np.uint8)
        return grayscale_threechannels


class HedColorAugmentor(ImageOnlyTransform):
    """
    Apply color correction in HED color space to an RGB patch.

    This augmentation randomly perturbs the Haematoxylin, Eosin, and DAB channels in the HED color
    space by scaling (sigma) and shifting (bias) each channel within configurable ranges. The
    transformation is only applied if the mean intensity of the patch is within a specified cutoff
    range.

    Attributes:
        _sigma_ranges (list of tuple): Adjustment ranges for H, E, and D channels.
        _bias_ranges (list of tuple): Bias ranges for H, E, and D channels.
        _cutoff_range (tuple): Cutoff interval for mean patch intensity.

    Args:
        thresh (float, optional): Range for sigma and bias for all channels. Default is 0.01.
        always_apply (bool, optional): Whether to always apply the transform. Default is False.
        p (float, optional): Probability of applying the transform. Default is 0.5.
    """

    def __init__(
        self,
        thresh: float = 0.01,
        always_apply: bool = False, p: float = 0.5
    ):

        # Initialize base class.
        super(HedColorAugmentor, self).__init__(p, always_apply)
        val = thresh
        bias_val = thresh
        haematoxylin_sigma_range = (-val, val)
        haematoxylin_bias_range = (-bias_val, bias_val)
        eosin_sigma_range = (-val, val)
        eosin_bias_range = (-bias_val, bias_val)
        dab_sigma_range = (-val, val)
        dab_bias_range = (-bias_val, bias_val)
        cutoff_range = (0.05, 0.95)

        # Initialize members.
        self._sigma_ranges = None  # Configured sigma ranges for H, E, and D channels.
        self._bias_ranges = None  # Configured bias ranges for H, E, and D channels.
        self._cutoff_range = None  # Cutoff interval.
        # Save configuration.
        self._setsigmaranges(
            haematoxylin_sigma_range=haematoxylin_sigma_range,
            eosin_sigma_range=eosin_sigma_range,
            dab_sigma_range=dab_sigma_range,
        )
        self._setbiasranges(
            haematoxylin_bias_range=haematoxylin_bias_range,
            eosin_bias_range=eosin_bias_range,
            dab_bias_range=dab_bias_range,
        )
        self._setcutoffrange(cutoff_range=cutoff_range)

    def _setsigmaranges(self, haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range):
        """
        Set the sigma intervals.

        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel.
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel.

        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel
            adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
        """
        # Check the intervals.
        if haematoxylin_sigma_range is not None:
            if (
                len(haematoxylin_sigma_range) != 2
                or haematoxylin_sigma_range[1] < haematoxylin_sigma_range[0]
                or haematoxylin_sigma_range[0] < -1.0
                or 1.0 < haematoxylin_sigma_range[1]
            ):
                raise InvalidRangeError('Haematoxylin Sigma', haematoxylin_sigma_range)

        if eosin_sigma_range is not None:
            if (
                len(eosin_sigma_range) != 2
                or eosin_sigma_range[1] < eosin_sigma_range[0]
                or eosin_sigma_range[0] < -1.0
                or 1.0 < eosin_sigma_range[1]
            ):
                raise InvalidRangeError('Eosin Sigma', eosin_sigma_range)

        if dab_sigma_range is not None:
            if (
                len(dab_sigma_range) != 2
                or dab_sigma_range[1] < dab_sigma_range[0]
                or dab_sigma_range[0] < -1.0
                or 1.0 < dab_sigma_range[1]
            ):
                raise InvalidRangeError('Dab Sigma', dab_sigma_range)

        # Store the settings.
        self._sigma_ranges = [
            haematoxylin_sigma_range,
            eosin_sigma_range,
            dab_sigma_range,
        ]

    def _setbiasranges(self, haematoxylin_bias_range, eosin_bias_range, dab_bias_range):
        """
        Set the bias intervals.

        Args:
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel.
            dab_bias_range (tuple, None): Bias range for the DAB channel.

        Raises:
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel
            adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
        """
        # Check the intervals.
        if haematoxylin_bias_range is not None:
            if (
                len(haematoxylin_bias_range) != 2
                or haematoxylin_bias_range[1] < haematoxylin_bias_range[0]
                or haematoxylin_bias_range[0] < -1.0
                or 1.0 < haematoxylin_bias_range[1]
            ):
                raise InvalidRangeError('Haematoxylin Bias', haematoxylin_bias_range)

        if eosin_bias_range is not None:
            if (
                len(eosin_bias_range) != 2
                or eosin_bias_range[1] < eosin_bias_range[0]
                or eosin_bias_range[0] < -1.0
                or 1.0 < eosin_bias_range[1]
            ):
                raise InvalidRangeError('Eosin Bias', eosin_bias_range)

        if dab_bias_range is not None:
            if (
                len(dab_bias_range) != 2
                or dab_bias_range[1] < dab_bias_range[0]
                or dab_bias_range[0] < -1.0
                or 1.0 < dab_bias_range[1]
            ):
                raise InvalidRangeError('Dab Bias', dab_bias_range)

        # Store the settings.
        self._bias_ranges = [haematoxylin_bias_range, eosin_bias_range, dab_bias_range]

    def _setcutoffrange(self, cutoff_range):
        """
        Set the cutoff value. Patches with mean value outside the cutoff interval \
        will not be augmented.

        Args:
            cutoff_range (tuple, None): Patches with mean value outside the
            cutoff interval will not be augmented.

        Raises:
            InvalidCutoffRangeError: The cutoff range is not valid.
        """
        # Check the interval.
        if cutoff_range is not None:
            if (
                len(cutoff_range) != 2
                or cutoff_range[1] < cutoff_range[0]
                or cutoff_range[0] < 0.0
                or 1.0 < cutoff_range[1]
            ):
                raise InvalidRangeError('Cutoff', cutoff_range)

        # Store the setting.
        self._cutoff_range = cutoff_range if cutoff_range is not None else [0.0, 1.0]

    def apply(self, patch: np.ndarray, **params) -> np.ndarray:
        """
        Apply color deformation on the patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """
        sigmas = [
            np.random.uniform(low=sigma_range[0], high=sigma_range[1],
                              size=None) if sigma_range is not None else 1.0
            for sigma_range in self._sigma_ranges
        ]
        biases = [
            np.random.uniform(low=bias_range[0], high=bias_range[1],
                              size=None) if bias_range is not None else 0.0
            for bias_range in self._bias_ranges
        ]

        # Check if the patch is inside the cutoff values.
        if patch.dtype.kind == "f":
            patch_mean = np.mean(a=patch)
        else:
            patch_mean = np.mean(a=patch.astype(dtype=np.float32)) / 255.0

        if self._cutoff_range[0] <= patch_mean <= self._cutoff_range[1]:
            # Convert the image patch to HED color coding.
            patch_hed = skimage.color.rgb2hed(rgb=patch)

            # Augment the Haematoxylin channel.
            if sigmas[0] != 0.0:
                patch_hed[:, :, 0] *= 1.0 + sigmas[0]

            if biases[0] != 0.0:
                patch_hed[:, :, 0] += biases[0]

            # Augment the Eosin channel.
            if sigmas[1] != 0.0:
                patch_hed[:, :, 1] *= 1.0 + sigmas[1]

            if biases[1] != 0.0:
                patch_hed[:, :, 1] += biases[1]

            # Augment the DAB channel.
            if sigmas[2] != 0.0:
                patch_hed[:, :, 2] *= 1.0 + sigmas[2]

            if biases[2] != 0.0:
                patch_hed[:, :, 2] += biases[2]

            # Convert back to RGB color coding.
            patch_rgb = skimage.color.hed2rgb(hed=patch_hed)
            patch_transformed = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)

            # Convert back to integral data type if the input was also integral.
            if patch.dtype.kind != "f":
                patch_transformed *= 255.0
                patch_transformed = patch_transformed.astype(dtype=np.uint8)

            return patch_transformed

        else:
            # The image patch is outside the cutoff interval.
            return patch
