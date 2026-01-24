const fs = require('fs');

const inputFile = 'V4_Exam CheatSheet Template Version (WS24_Retake-ML).md';
const outputFile = 'ChSt_Gemini.html';

const htmlHead = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Cheat Sheet (Ultra-Compact)</title>
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&amp;display=swap" rel="stylesheet">
    <script>
        window.MathJax = {
            tex: { inlineMath: [['$', '$'], ['\(', '\)']], displayMath: [['$$', '$$']] },
            chtml: { scale: 0.68, displayAlign: 'left' },
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
        
        table { border-collapse: collapse; width: 100%; font-size: 0.9em; margin: 2px 0; }
        td, th { border: 0.5px solid #666; padding: 0 2px; text-align: center; }
        
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
`

function processInline(text) {
    if (!text) return '';
    
    let placeholders = [];
    
    // Protect Display Math $$...$$
    text = text.replace(/\$\$(.*?)\$\$/g, (match, content) => {
        const placeholder = `___MATH_DISPLAY_${placeholders.length}___`;
        placeholders.push({ placeholder, original: match });
        return placeholder;
    });
    
    // Protect Inline Math $...$
    // Using a more careful regex for inline math to avoid matching across multiple lines if split incorrectly
    text = text.replace(/\$(.*?)\$/g, (match, content) => {
        const placeholder = `___MATH_INLINE_${placeholders.length}___`;
        placeholders.push({ placeholder, original: match });
        return placeholder;
    });

    // Bold: ** -> <span class="b">
    text = text.replace(/\*\*(.*?)\*\*/g, '<span class="b">$1</span>');
    // Italics: * -> <i>
    text = text.replace(/\*(.*?)\*/g, '<i>$1</i>');
    
    // Restore math
    for (let i = placeholders.length - 1; i >= 0; i--) {
        text = text.replace(placeholders[i].placeholder, placeholders[i].original);
    }
    
    return text;
}

const mdContent = fs.readFileSync(inputFile, 'utf-8');
const lines = mdContent.split(/\r?\n/);

let pagesContent = [[], []];
let currentPageIdx = 0;
let currentBox = null;
const colors = ['hl-y', 'hl-c', 'hl-g', 'hl-p'];
let colorIdx = 0;
let inCodeBlock = false;
let codeContent = '';

function flushBox() {
    if (currentBox) {
        return `<div class='box'><h2 class='${currentBox.color}'>${currentBox.title}</h2>${currentBox.content.join('\n')}</div>`;
    }
    return '';
}

for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trimEnd();
    
    if (line.trim().startsWith('```')) {
        if (inCodeBlock) {
            if (currentBox) {
                const safeCode = codeContent.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                currentBox.content.push(`<pre>${safeCode}</pre>`);
            }
            inCodeBlock = false;
            codeContent = '';
        } else {
            inCodeBlock = true;
        }
        continue;
    }
    
    if (inCodeBlock) {
        codeContent += line + '\n';
        continue;
    }
    
    const stripped = line.trim();
    
    if (stripped.startsWith('# ') && !stripped.startsWith('##')) {
        pagesContent[currentPageIdx].push(flushBox());
        currentBox = null;
        const title = stripped.substring(2).trim();
        if (title.startsWith('9.') || title.startsWith('9 ')) {
            currentPageIdx = 1;
        }
        pagesContent[currentPageIdx].push(`<div class='section-header'>${title}</div>`);
    } else if (stripped.startsWith('## ')) {
        pagesContent[currentPageIdx].push(flushBox());
        const title = stripped.substring(3).trim();
        currentBox = { title, color: colors[colorIdx % 4], content: [] };
        colorIdx++;
    } else if (stripped.startsWith('### ')) {
        const text = processInline(stripped.substring(4).trim());
        if (currentBox) currentBox.content.push(`<h3>${text}</h3>`);
        else pagesContent[currentPageIdx].push(`<h3>${text}</h3>`);
    } else if (stripped.startsWith('|')) {
        let tableLines = [];
        while (i < lines.length && lines[i].trim().startsWith('|')) {
            tableLines.push(lines[i].trim());
            i++;
        }
        i--;
        
        let rows = [];
        for (let tl of tableLines) {
            // Improved separator check
            if (/^\|[ :\-\s|]+\|$/.test(tl) && tl.includes('-')) continue;
            
            // Handle escaped pipes
            let t = tl.replace(/\\\|/g, '___ESCAPED_PIPE___');
            
            // Remove outer pipes
            if (t.startsWith('|')) t = t.substring(1);
            if (t.endsWith('|')) t = t.substring(0, t.length - 1);
            
            // Split by |
            const cells = t.split('|').map(c => {
                let val = c.trim().replace(/___ESCAPED_PIPE___/g, '|');
                return processInline(val);
            });
            rows.push(cells);
        }
        
        if (rows.length > 0) {
            let tblHtml = '<table>';
            for (let j = 0; j < rows.length; j++) {
                const tag = (j === 0) ? 'th' : 'td';
                tblHtml += '<tr>';
                for (let cell of rows[j]) {
                    tblHtml += `<${tag}>${cell}</${tag}>`;
                }
                tblHtml += '</tr>';
            }
            tblHtml += '</table>';
            if (currentBox) currentBox.content.push(tblHtml);
            else pagesContent[currentPageIdx].push(tblHtml);
        }
    } else if (stripped.length > 0 && !stripped.startsWith('---')) {
        let c = processInline(stripped);
        if (c.startsWith('- ')) {
            c = '&bull; ' + c.substring(2);
        }
        if (currentBox) currentBox.content.push(`<p>${c}</p>`);
        else pagesContent[currentPageIdx].push(`<p>${c}</p>`);
    }
}

pagesContent[currentPageIdx].push(flushBox());

let finalHtml = htmlHead;
finalHtml += `<div class='page'><div class='columns'><h1>Page 1: Supervised &amp; All Deep Learning (Sec 2-8)</h1>${pagesContent[0].join('\n')}</div></div>`;
finalHtml += `<div class='page'><div class='columns'><h1>Page 2: SVM, Unsupervised &amp; Misc (Sec 9-12)</h1>${pagesContent[1].join('\n')}</div></div>`;
finalHtml += '</body></html>';

fs.writeFileSync(outputFile, finalHtml, 'utf-8');
console.log(`Done! Generated ${outputFile}`);
