"""Attention visualization utilities."""

import math
import numpy as np


class Span(object):
    """Span object used to render highlighted text."""

    def __init__(self, text, color):
        self.text = text
        self.color = color


def _rescale(x, min_score=-math.log(2), max_score=math.log(2)):
    """Rescales attention values using the following transformation:

        x' = ln(x * len(x))

    And then maps to the range [0, 1].

    This transformation is useful for putting attention vectors for different
    length sentences on a common scale. If attention is uniform then the score
    for all words is 0. Similarly, if the attention paid to a word is K times
    the uniform value then the score is ln(K).

    Args:
        x: An array of numbers.
        min_score: Scores below this number get mapped to 0.
        max_score: Scores above this number get mapped to 1.

    Returns:
        An np.array of scaled values in x.
    """
    # Scale
    xp = np.log(x * len(x))

    # Truncate
    xp[xp < min_score] = min_score
    xp[xp > max_score] = max_score

    # Normalize
    scores = (xp - min_score) / (max_score - min_score)

    return scores


def _rgb_to_hex(x):
    """Maps an array of RGB values to a hex string.

    Args:
        x: An array of numbers.

    Returns:
        A hex string.
    """
    assert len(x) == 3
    x = x.astype(int)
    hex_vals = tuple('%0.2x' % num for num in x)
    hex_string = '#%s%s%s' % hex_vals
    return hex_string


def generate_spans(title,
                   description,
                   attention,
                   min_rgb=np.array([239, 239, 239]),
                   max_rgb=np.array([255, 161, 0])):
    """Returns span objects used to highlight text.

    Args:
        words: Sequence of words to be highlighted.
        attention: Array of attention weights.
        min_rgb: Color of minimum score.
        max_rgb: Color of maximum score.

    Returns:
        title_spans, description_spans: Lists of Span objects to be used in
            rendering the highlighted text.
    """
    assert len(title) + len(description) == len(attention)

    # Map attention to hex values
    scores = _rescale(attention)
    rgb_vals = [(1 - score) * min_rgb  + score * max_rgb for score in scores]
    hex_vals = [_rgb_to_hex(val) for val in rgb_vals]

    # Assemble HTML
    zipped = zip(title, hex_vals[:len(title)])
    title_spans = [Span(text, color) for text, color in zipped]

    zipped = zip(description, hex_vals[len(title):])
    description_spans = [Span(text, color) for text, color in zipped]

    return title_spans, description_spans


