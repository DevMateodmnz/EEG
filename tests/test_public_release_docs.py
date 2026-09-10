"""Public-facing documentation contracts without rewriting frozen result artifacts."""

from __future__ import annotations

import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    ROOT / "README.md",
    ROOT / "README_ES.md",
    ROOT / "CITATION.cff",
    ROOT / ".env.example",
    *(ROOT / "docs" / name for name in (
        "project_overview.md", "project_overview_es.md", "methods.md", "methods_es.md",
        "results.md", "results_es.md", "datasets.md", "validation_strategy.md",
        "limitations.md", "external_validation.md", "external_validation_es.md",
        "repository_guide.md", "collaboration.md", "collaboration_es.md", "public_release_audit.md",
    )),
)


class PublicReleaseDocumentationTests(unittest.TestCase):
    def test_public_entry_points_exist_without_local_paths(self) -> None:
        for path in PUBLIC_FILES:
            self.assertTrue(path.is_file(), path)
            self.assertNotIn("/" + "home" + "/md/", path.read_text(encoding="utf-8"), path)

    def test_public_markdown_links_resolve(self) -> None:
        for path in (ROOT / "README.md", ROOT / "README_ES.md", *(item for item in PUBLIC_FILES if item.suffix == ".md")):
            for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", path.read_text(encoding="utf-8")):
                if "://" not in target and not target.startswith("mailto:"):
                    self.assertTrue((path.parent / target).resolve().is_file(), f"{path}: {target}")

    def test_external_study_is_explicitly_pre_outcome(self) -> None:
        text = (ROOT / "docs" / "external_validation.md").read_text(encoding="utf-8")
        self.assertIn("IN PROGRESS", text)
        self.assertIn("no external scientific outcome", text)

    def test_citation_names_the_project_author_without_doi_claim(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("Mateo Gabriel", citation)
        self.assertNotIn("doi:", citation.lower())


if __name__ == "__main__":
    unittest.main()
