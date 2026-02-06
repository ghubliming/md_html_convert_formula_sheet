# METHODOLOGY V5: The "Dynamic 4-Column Balanced" Workflow

## 1. Overview
This updated workflow converts a Markdown cheat sheet into a **balanced 2-page HTML document**. It uses a **heuristic height estimation** to dynamically split content across two A4 pages, ensuring maximum density without "ghost columns" or lost information.

**Key Improvements in V5:**
*   **4-Column Layout:** Increased horizontal capacity by 33% to handle complex formulas and tables.
*   **Heuristic Splitting:** Instead of hardcoded section splits, the script estimates the vertical height of every block (paragraphs, tables, math, code) and finds the optimal midpoint.
*   **Heading-Priority Breaks:** The splitter "looks ahead" to ensure pages break at a Section or Sub-section header whenever possible for a professional look.
*   **Multi-line Math Protection:** Robust handling of multi-line `$$` LaTeX blocks (e.g., Hessian matrices) to prevent tag corruption.

## 2. Optimized Configuration
*   **Layout:** 4 Columns per page.
*   **Font Size:** `7.1pt` (Caveat font) - optimized for print legibility.
*   **Line Height:** `1.02`.
*   **Column Gap:** `1.2mm`.
*   **Table Scale:** `0.85em` with `table-layout: fixed` to prevent horizontal blowouts.
*   **Heuristic Capacity:** Targeted at `3800` height units per page.

## 3. Execution
Run the generator script:
```bash
node convert.js
```

## 4. Maintenance Notes
*   **Ghost Columns:** If you see a "5th column" appearing (content hidden off-page), lower the `PAGE_CAPACITY` variable in `convert.js`.
*   **Empty Space:** The script uses `break-inside: avoid` on logical blocks to prevent ugly splits, but balances this with a granular "box" system to minimize white space.