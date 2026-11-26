"""Sphinx configuration for the skgp documentation.

The same build as GPJax and Impulso: MyST markdown pages, the API reference
rendered from the docstrings by autodoc and autosummary, the shibuya theme, and
Cloudflare serving the HTML. Deviations from ../GPJax/docs/conf.py are
commented inline. The largest is that skgp has no notebooks yet, so the pages
are parsed by myst-parser instead of MyST-NB, which would bring a Jupyter
kernel and execution cache for nothing. Switching back is one line in
`extensions` plus the MyST-NB settings from GPJax.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import operator
import os
import types
import typing

from docutils import nodes
from sphinx import addnodes
from sphinx.util.typing import stringify_annotation

import skgp

# -- Project information -----------------------------------------------------
project = "skgp"
author = "Thomas Pinder"
copyright = "2026, Thomas Pinder"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",  # tabs on the installation page
    "sphinx_codeautolink",  # link API names in example code to the reference
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",  # autosummary-generated API stubs
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # In-repo records and notes, not part of the site.
    "adr/**",
    "agents/**",
    "advanced.md",
]

# As in GPJax: skgp's docstrings write names in single backticks, which RST
# would otherwise render as italic <cite> instead of code.
default_role = "literal"

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",  # ::: {note} admonitions
    "deflist",
    "attrs_inline",
]
myst_heading_anchors = 3  # auto-slug headings so in-page [](#anchor) links resolve

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    # DEVIATION from GPJax, which sets `show-inheritance`. Every public skgp
    # class inherits from a private one (`skgp._model._Model`, for one) or from
    # pydantic's BaseModel, so the "Bases:" line would show only internals.
    #
    # The settings that every model shares are declared on that private base
    # class, so they are inherited members. Naming BaseModel
    # here documents them on each public class while leaving out pydantic's own
    # methods (model_dump, model_validate and the rest).
    #
    # BaseException does the same for the two warning classes, which would
    # otherwise list `add_note` and `with_traceback`.
    #
    # scikit-learn's BaseEstimator and mixins do the same for the estimators in
    # `skgp.sklearn`, which would otherwise list get_params, set_params and the
    # rest, with docstrings that link into scikit-learn's own docs.
    "inherited-members": "BaseModel,BaseException,BaseEstimator,RegressorMixin,ClassifierMixin",
    # Public names that only skgp's own code uses. `model_config` is pydantic's
    # configuration of a settings class, `task` tells the data checks whether a
    # model is a regressor or a classifier, and `likelihood` tells the fit how
    # the target is modelled.
    # scikit-learn gives every estimator `get_metadata_routing`, and a
    # `set_<method>_request` for each method that takes more than the data, for
    # its metadata routing; skgp's estimators route no metadata.
    "exclude-members": "fit_kind,require_valid_for,model_config,task,likelihood,get_metadata_routing,set_predict_request,set_score_request",
}
# DEVIATION from GPJax, which uses "description". GPJax's docstrings have
# "Args:" sections that the types merge into; skgp's are prose, so
# "description" would add a list of bare parameter names and types under every
# method. In the signature the types sit next to the names instead.
autodoc_typehints = "signature"
autodoc_preserve_defaults = True
# DEVIATION from GPJax, which lists members alphabetically: here methods come
# first and settings second, each alphabetically, instead of the two mixed.
# ("bysource" would put inherited settings after everything else, because
# autodoc finds no source position for them in the inheriting class.)
autodoc_member_order = "groupwise"
# The page contents list `check()` rather than `SparseGPRegressor.check()`,
# which the theme's narrow right-hand column cuts off.
toc_object_entries_show_parents = "hide"
# DEVIATION from GPJax: no napoleon, which converts "Args:" and "Returns:"
# sections. skgp's docstrings have none.
# DEVIATION from GPJax: names are shown without the `skgp.` prefix. Everything
# public is importable from `skgp` itself (the reference page says so), and the
# global settings class lives in the private module `skgp._config`, whose path
# would otherwise appear on the page.
add_module_names = False

# Classes that skgp makes for you: you get them from `fit`, `predict`, `check`
# and `skgp.config`, never by calling the class, and their constructors take
# skgp's internal state. Their signature is left out so that the reference does
# not suggest calling them. They are the class of `skgp.config` and the public
# dataclasses (fitted models, predictions and checked data), which are found
# here so that a new one is never missed.
_MADE_BY_SKGP = {
    f"skgp.{name}"
    for name in skgp.__all__
    if dataclasses.is_dataclass(getattr(skgp, name))
} | {"skgp._config.Config"}


# Sphinx calls an `autodoc-process-signature` handler with these seven arguments.
def _hide_internal_constructors(  # noqa: PLR0917
    app, what, name, obj, options, signature, return_annotation
):
    """Drop the signature of a class in `_MADE_BY_SKGP`; leave every other one alone."""
    if what == "class" and name in _MADE_BY_SKGP:
        return "", None
    return None


# -- sphinx-codeautolink -----------------------------------------------------
# Adds an "Examples using …" block to each documented object.
codeautolink_autodoc_inject = True

# -- Intersphinx -------------------------------------------------------------
# As in GPJax and Impulso, each mapping lists the upstream inventory first and a
# copy vendored in `_inventories/` second, so an unreachable upstream site
# cannot fail a `-W` build. Refresh the copies by re-downloading each
# `<target_uri>objects.inv` into docs/_inventories/.
_INV = os.path.join(os.path.dirname(__file__), "_inventories")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", (None, f"{_INV}/python.inv")),
    "numpy": ("https://numpy.org/doc/stable/", (None, f"{_INV}/numpy.inv")),
    "pandas": ("https://pandas.pydata.org/docs/", (None, f"{_INV}/pandas.inv")),
    "polars": (
        "https://docs.pola.rs/api/python/stable/",
        (None, f"{_INV}/polars.inv"),
    ),
    # Names such as `gpjax.gps.Prior` in skgp's docs link to GPJax's docs.
    "gpjax": ("https://gpjax.quantclimate.com/", (None, f"{_INV}/gpjax.inv")),
}

# -- HTML output -------------------------------------------------------------
html_theme = "shibuya"
html_title = "skgp"
# Sitemap, canonical link and og:url all derive from this, so it must match the
# custom domain in docs/wrangler.jsonc.
html_baseurl = "https://skgp.quantclimate.com/"
sitemap_url_scheme = "{link}"
html_static_path = ["stylesheets"]
html_css_files = ["extra.css"]
# Cloudflare serves the docs with html_handling "none" (docs/wrangler.jsonc), so
# a directory URL such as / only serves its index.html through the rules in
# _redirects. _headers adds a CORS header so other sites can read objects.inv.
html_extra_path = ["_redirects", "_headers"]
html_theme_options = {
    # Names the token family; stylesheets/extra.css re-tones the crimson scale
    # to the same oxblood as GPJax and Impulso.
    "accent_color": "crimson",
    "color_mode": "auto",  # follow the reader's light/dark preference
    "github_url": "https://github.com/QuantClimate/GPJax/tree/main/skgp",
    "nav_links": [
        {"title": "Installation", "url": "installation"},
        {"title": "API reference", "url": "reference/index"},
        {"title": "PyPI", "url": "https://pypi.org/project/skgp"},
    ],
    # As in GPJax: 0 hands the sidebar depth back to each toctree's `:maxdepth:`.
    "toctree_maxdepth": 0,
    "globaltoc_expand_depth": 1,
}
html_context = {
    "github_user": "QuantClimate",
    "github_repo": "skgp",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Open Graph link previews (sphinxext-opengraph) ---------------------------
ogp_site_url = html_baseurl
ogp_site_name = "skgp"
ogp_enable_meta_description = True
# Off, as in Impulso: with matplotlib installed the extension would otherwise
# generate a text card for every page.
ogp_social_cards = {"enable": False}


# -- Autodoc fixes for skgp's pydantic settings -------------------------------
# skgp's `schema` setting shares its name with pydantic's deprecated
# `BaseModel.schema()` classmethod, and pydantic keeps no class attribute for a
# field, so autodoc finds the classmethod and documents its signature in place
# of the setting. Shadowing the classmethod on every public class with a
# `schema` setting, in this docs build only, makes autodoc document the setting
# with its docstring. A model's own `schema` value is unaffected: pydantic
# stores it on the instance.
for _name in skgp.__all__:
    _object = getattr(skgp, _name)
    if isinstance(_object, type) and "schema" in getattr(_object, "model_fields", {}):
        _object.schema = None


def _without_metadata(hint):
    """`hint` with every `Annotated[...]` in it, including inside a union, unwrapped."""
    if typing.get_origin(hint) is typing.Annotated:
        return _without_metadata(typing.get_args(hint)[0])
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return functools.reduce(
            operator.or_, map(_without_metadata, typing.get_args(hint))
        )
    return hint


def _plain_field_types(app, doctree):
    """Show each pydantic field's type without its validators.

    skgp declares a setting such as `SparseGPRegressor.minibatch_size` as
    `Annotated[int, <validators>] | None`, and autodoc prints the validators in
    full, about 300 characters of pydantic internals. Sphinx has no setting to
    drop them, so this replaces the type after parsing with the declared one
    without the `Annotated` metadata, such as `int | None`.
    """
    for signature in doctree.findall(addnodes.desc_signature):
        if signature.parent.get("objtype") != "attribute":
            continue
        for annotation in signature.findall(addnodes.desc_annotation):
            if "Annotated[" not in annotation.astext():
                continue
            *owner_path, name = signature["fullname"].split(".")
            owner = importlib.import_module(signature["module"])
            for part in owner_path:
                owner = getattr(owner, part)
            # The class that declares the field; a setting may be inherited.
            hint = next(
                inspect.get_annotations(cls)[name]
                for cls in owner.__mro__
                if name in inspect.get_annotations(cls)
            )
            plain = stringify_annotation(_without_metadata(hint), "smart")
            annotation.clear()
            annotation += [
                addnodes.desc_sig_punctuation("", ":"),
                addnodes.desc_sig_space(),
                nodes.Text(plain),
            ]


def setup(app):
    """Register the signature and field-type hooks."""
    app.connect("autodoc-process-signature", _hide_internal_constructors)
    app.connect("doctree-read", _plain_field_types)
