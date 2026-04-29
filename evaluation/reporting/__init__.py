"""Report generation for Z3 SYNAPSE evaluation.

Generates JSON and Markdown reports from evaluation results.

Submodules
----------
json_report
    Machine-readable JSON report with all metrics and statistical tests.
markdown_report
    Human-readable Markdown report for publications and review.
"""

from .json_report import generate_json_report
from .markdown_report import generate_markdown_report

__all__ = [
    "generate_json_report",
    "generate_markdown_report",
]
