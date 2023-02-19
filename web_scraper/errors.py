class PageNotFound(Exception):
    """Raised when attempting to extract an
    article from an invalid address"""


class HeaderNotFound(Exception):
    """Raised when there is no h1 element
    for an article but its retrieval is
    attempted nevertheless"""
