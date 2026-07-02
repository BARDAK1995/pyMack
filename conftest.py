"""Pytest bootstrap: make the repo checkout importable without installation.

With this file at the repository root, ``pytest`` inserts the root onto
``sys.path`` so ``import pymack`` resolves to the checkout. Installed usage
(``pip install -e .``) does not need it.
"""
