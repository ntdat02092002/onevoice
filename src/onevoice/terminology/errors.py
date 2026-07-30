from __future__ import annotations


class TerminologyError(Exception):
    """Base error for terminology data and runtime preparation."""


class BundleValidationError(TerminologyError, ValueError):
    """Raised when a terminology bundle does not satisfy schema version 1."""


class ProfileActivationError(TerminologyError, ValueError):
    """Raised when a bundle cannot cover an activated language/domain profile."""


class TerminologyCoverageError(TerminologyError):
    """Raised when MT loses, duplicates, mutates, or leaks a term placeholder."""
