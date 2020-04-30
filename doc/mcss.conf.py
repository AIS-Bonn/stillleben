
PROJECT_TITLE = "stillleben"
PROJECT_SUBTITLE = 'Python'

INPUT_MODULES = ["stillleben"]

PLUGINS = [
    "m.link",
    "m.dox",
    "m.components",
    "m.math",
    "m.code",
    "m.sphinx",
]

INPUT_PAGES = [
    'index.rst',
    'installation.rst',
    'examples/viewer.rst',
    'examples/pbr.rst',
    'examples/examples.rst',
]

LINKS_NAVBAR1 = [
    ('Installation', 'installation', []),
    ('Examples', 'examples', []),
    ('Modules', 'modules', []),
    ('Classes', 'classes', []),
]
LINKS_NAVBAR2 = []

PYBIND11_COMPATIBILITY = True

M_SPHINX_PARSE_DOCSTRINGS = True
