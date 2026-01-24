# METHODOLOGY V3: The "Ultra-Compact Split" Workflow

## 1. Overview
This workflow converts a structured Markdown cheat sheet into a **strictly formatted 2-page HTML document** designed for printing. It solves the problem of "overflowing content" by using a **hardcoded section split** and **ultra-compact styling**.

## 2. The Winning Configuration
To achieve the result where "All Deep Learning (Sec 2-8)" fits on Page 1 and "SVM + Unsupervised (Sec 9-Misc)" fits on Page 2, we use these specific settings:

*   **Layout:** 3 Columns per page.
*   **Page 1 Content:** Sections 2, 3, 4, 5, 6, 7, 8.
*   **Page 2 Content:** Sections 9, 10, 11, 12, Misc.
*   **Font Size:** `6.8pt` (Caveat font).
*   **MathJax Scale:** `0.68` (Prevents horizontal blowout).
*   **ASCII Diagrams:** Reduced to `4pt` font to fit columns.
*   **Scrollbars:** Globally disabled to ensure clean PDF export.

---

## 3. Replication Instruction (The Script)
To replicate this result in a new session, simply **copy and paste the following PowerShell script** into your terminal.

**Pre-requisites:**
1.  Have your markdown file ready (e.g., `Input.md`).
2.  Update the `$input_file` variable in the script below if your filename differs.

### 📜 The Generator Script

```powershell
# --- CONFIGURATION ---
$input_file = "V4_Exam CheatSheet Template Version (WS24_Retake-ML).md" 
$output_file = "ChSt_Gemini.html"

# --- HTML TEMPLATE ---
$html_head = @'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Cheat Sheet (Ultra-Compact)</title>
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap" rel="stylesheet">
    <script>
        window.MathJax = {
            tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$']] },
            chtml: { scale: 0.68, displayAlign: 'left' }, /* 0.68 Scale for Density */
            startup: { pageReady: () => MathJax.startup.defaultPageReady() }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root {
            --hl-y: rgba(255, 230, 0, 0.5); --hl-c: rgba(0, 240, 255, 0.4);
            --hl-g: rgba(50, 255, 50, 0.35); --hl-p: rgba(255, 0, 100, 0.3);
            --text: #111;
        }
        * { box-sizing: border-box; scrollbar-width: none !important; }
        *::-webkit-scrollbar { display: none !important; }
        
        body { 
            font-family: 'Caveat', cursive; margin: 0; padding: 0; background: #888; 
            color: var(--text); font-size: 6.8pt; line-height: 1.02; font-weight: 500;
        }
        
        mjx-container { 
            font-family: inherit; color: #333 !important; margin: 0 !important; 
            max-width: 100% !important; overflow: hidden !important; white-space: nowrap;
        }
        
        /* Prevent Layout Shifts */
        p, li, div, h1, h2, h3 { word-wrap: break-word; overflow-wrap: break-word; }

        .page {
            width: 200mm; height: 292mm; background: white; margin: 10px auto; 
            padding: 3mm; position: relative; overflow: hidden; 
        }
        
        .columns {
            column-count: 3; column-gap: 2mm; column-fill: auto; height: 100%; width: 100%;
        }

        h1 { 
            font-size: 10.5pt; text-align: center; border-bottom: 1.5px solid #444; 
            margin: 0 0 2px 0; font-weight: 700; background-color: #f4f4f4;
            column-span: all; color: #000; padding: 1px;
        }

        .section-header {
            font-size: 8.5pt; text-align: center; border: 1px solid #666;
            margin: 3px 0 1px 0; font-weight: 700; background: #eee;
            break-after: avoid; page-break-inside: avoid; padding: 1px; color: #000;
        }
        
        h2 { 
            font-size: 8pt; border-bottom: 1px solid #888; margin: 1px 0 1px 0; 
            font-weight: 700; padding-left: 2px; break-after: avoid; 
        }
        
        h3 { font-size: 7.5pt; margin: 1px 0 0px 0; font-weight: 700; text-decoration: underline; }
        p, li { margin: 0; }
        
        .box {
            break-inside: avoid; margin-bottom: 2px; padding-bottom: 1px;
            border-bottom: 0.5px dashed #ccc; position: relative; max-width: 100%; 
        }
        
        .hl-y { background: var(--hl-y); padding: 0 2px; border-radius: 2px; }
        .hl-c { background: var(--hl-c); padding: 0 2px; border-radius: 2px; }
        .hl-g { background: var(--hl-g); padding: 0 2px; border-radius: 2px; }
        .hl-p { background: var(--hl-p); padding: 0 2px; border-radius: 2px; }
        
        .b { font-weight: 700; }
        
        table { border-collapse: collapse; width: 100%; font-size: 0.9em; margin: 0; }
        td, th { border: 0.5px solid #666; padding: 0 2px; text-align: center; }
        
        /* Tiny Code Blocks for ASCII Art */
        pre {
            font-family: monospace; font-size: 4pt; line-height: 0.85;
            letter-spacing: -0.1px; white-space: pre-wrap; word-break: break-all;
            margin: 1px 0; background: #f8f8f8; padding: 1px; border: 0.5px solid #eee;
            color: #444; max-width: 100%; overflow: hidden !important; 
        }

        @media print {
            body { background: white; }
            .page { margin: 0 auto; border: none; page-break-after: always; }
        }
    </style>
</head>
<body>
'@

# --- PROCESSING ---
if (-not (Test-Path $input_file)) { Write-Error "Input file not found: $input_file"; exit }
$md_content = Get-Content -Raw $input_file -Encoding UTF8
$lines = $md_content -split "\r?\n"

$pages_content = @(@(), @())
$current_page_idx = 0
$current_box = $null
$colors = @('hl-y', 'hl-c', 'hl-g', 'hl-p')
$color_idx = 0
$in_code_block = $false
$code_content = ""

function Flush-Box {
    if ($current_box) {
        return "<div class='box'><h2 class='$($current_box.color)'>$($current_box.title)</h2>" + ($current_box.content -join "`n") + "</div>"
    }
    return ""
}

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i].TrimEnd()
    
    # Code Blocks
    if ($line.Trim().StartsWith('```')) {
        if ($in_code_block) {
            if ($current_box) {
                $safe_code = $code_content.Replace('<','&lt;').Replace('>','&gt;')
                $current_box.content += "<pre>$safe_code</pre>"
            }
            $in_code_block = $false
            $code_content = ""
        } else { $in_code_block = $true }
        continue
    }
    if ($in_code_block) { $code_content += "$line`n"; continue }
    
    $stripped = $line.Trim()
    
    # --- LOGIC: SPLIT AT "9. SVM" ---
    if ($stripped.StartsWith('# ') -and -not $stripped.StartsWith('##')) {
        $pages_content[$current_page_idx] += Flush-Box
        $current_box = $null
        $title = $stripped.Substring(2).Trim()
        
        # TRIGGER: Switch to Page 2 at Section 9
        if ($title.StartsWith('9.') -or $title.StartsWith('9 ')) {
            $current_page_idx = 1
        }
        $pages_content[$current_page_idx] += "<div class='section-header'>$title</div>"
    }
    elseif ($stripped.StartsWith('## ')) {
        $pages_content[$current_page_idx] += Flush-Box
        $title = $stripped.Substring(3).Trim()
        $current_box = @{ title = $title; color = $colors[$color_idx % 4]; content = @() }
        $color_idx++
    }
    elseif ($stripped.StartsWith('### ')) {
        $text = $stripped.Substring(4).Trim()
        if ($current_box) { $current_box.content += "<h3>$text</h3>" } else { $pages_content[$current_page_idx] += "<h3>$text</h3>" }
    }
    elseif ($stripped.StartsWith('|')) {
        # Simple table handling
        $table_lines = @(); while ($i -lt $lines.Count -and $lines[$i].Trim().StartsWith('|')) { $table_lines += $lines[$i]; $i++ }; $i--
        $rows = @(); foreach ($tl in $table_lines) { $cells = $tl.Trim().Trim('|').Split('|') | ForEach-Object { $_.Trim() }; $rows += ,$cells }
        $header = $rows[0]; $body = $rows[2..($rows.Count-1)]
        $tbl = "<table>"
        if ($header) { $tbl += "<tr>" + ($header | ForEach-Object { "<th>$_</th>" }) + "</tr>" }
        foreach ($r in $body) { $tbl += "<tr>" + ($r | ForEach-Object { "<td>$_</td>" }) + "</tr>" }
        $tbl += "</table>"
        if ($current_box) { $current_box.content += $tbl } else { $pages_content[$current_page_idx] += $tbl }
    }
    elseif ($stripped.Length -gt 0 -and -not $stripped.StartsWith('---')) {
        $c = $stripped -replace '\**(.*?)**', '<span class="b">$1</span>'
        if ($c.StartsWith('- ')) { $c = '&bull; ' + $c.Substring(2) }
        if ($current_box) { $current_box.content += "<p>$c</p>" } else { $pages_content[$current_page_idx] += "<p>$c</p>" }
    }
}
$pages_content[$current_page_idx] += Flush-Box

# --- OUTPUT ---
$final_html = $html_head
$final_html += "<div class='page'><div class='columns'><h1>Page 1: Supervised & All Deep Learning (Sec 2-8)</h1>" + ($pages_content[0] -join "`n") + "</div></div>"
$final_html += "<div class='page'><div class='columns'><h1>Page 2: SVM, Unsupervised & Misc (Sec 9-12)</h1>" + ($pages_content[1] -join "`n") + "</div></div>"
$final_html += "</body></html>"

[System.IO.File]::WriteAllText($output_html, $final_html, [System.Text.Encoding]::UTF8)
Write-Host "Done! Generated $output_file"
```

---

## 4. Final Verification Checklist
1.  **Open** the generated HTML in Chrome/Edge.
2.  **Wait** 1 second for MathJax to render.
3.  **Print (Ctrl + P)**:
    *   **Margins:** Set to **None**.
    *   **Background Graphics:** **Checked**.
    *   **Scale:** **100%**.
4.  **Verify**:
    *   Page 1 should end exactly after the CNN section.
    *   Page 2 should start with SVM.
    *   No content should be cut off on the right edge.
    *   Equations should not have scrollbars.
