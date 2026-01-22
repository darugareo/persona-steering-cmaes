#!/bin/bash

# Script to apply all corrections (①-⑥) to LaTeX files
# Working directory: paper_ieee_access_corrected

BASE_DIR="/data01/nakata/master_thesis/persona2/paper_ieee_access_corrected"

echo "========================================="
echo "Applying corrections to LaTeX files"
echo "========================================="

# Note: Abstract already manually corrected in ieee_access.tex

echo "✓ Abstract corrected (336→280, prompt removed, numbers updated)"
echo ""

echo "Next steps require manual editing of the following files:"
echo "1. sections/introduction.tex"
echo "2. sections/related_work.tex"
echo "3. sections/experimental_setup.tex"
echo "4. sections/results.tex"
echo "5. sections/discussion.tex"
echo "6. sections/conclusion.tex"
echo "7. sections/limitations.tex"
echo ""

echo "All corrections documented in:"
echo "- paper/analysis/section_replacements_corrected.md"
echo "- paper/analysis/correction_03_table_relationship.md"
echo "- paper/analysis/correction_05_remove_prompt.md"
echo "- paper/analysis/correction_06_discussion_tone.md"
echo ""

echo "Recommendation: Apply corrections file by file using Edit tool"
