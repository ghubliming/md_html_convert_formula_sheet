const fs = require('fs');

const inputFile = 'V5_Exam CheatSheet Template Version (WS24_Retake-ML).md';
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
            tex: { 
                inlineMath: [['$', '$']], 
                displayMath: [['$$', '$$']],
                processEscapes: true
            },
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
            color: var(--text); font-size: 7.1pt; line-height: 1.02; font-weight: 500;
        }
        
        mjx-container { 
            font-family: inherit; color: #333 !important; margin: 0 !important; 
            max-width: 100% !important; overflow-x: auto !important; overflow-y: hidden !important;
        }
        
        p, li, div, h1, h2, h3 { word-wrap: break-word; overflow-wrap: break-word; }

        .page {
            width: 210mm; height: 297mm; background: white; margin: 10px auto; 
            padding: 2mm; position: relative; overflow: hidden; 
        }
        
        .columns {
            column-count: 4; column-gap: 1.2mm; height: 100%; width: 100%;
            column-fill: auto;
        }

        .section-header {
            font-size: 10pt; text-align: center; border: 1.2px solid #333;
            margin: 1.5px 0; font-weight: 700; background: #eee;
            padding: 0.5px; color: #000; break-inside: avoid;
            width: 100%;
        }
        
        .box {
            break-inside: avoid; margin-bottom: 1px;
            width: 100%; max-width: 100%;
        }

        h2 { 
            font-size: 9pt; border-bottom: 1px solid #666; margin: 1px 0; 
            font-weight: 700; padding-left: 2px;
        }
        
        h3 { 
            font-size: 8.2pt; margin: 1px 0 0.2px 0; font-weight: 700; 
            text-decoration: underline;
        }
        
        p, li { margin: 0; }
        
        .hl-y { background: var(--hl-y); padding: 0 1px; border-radius: 1px; }
        .hl-c { background: var(--hl-c); padding: 0 1px; border-radius: 1px; }
        .hl-g { background: var(--hl-g); padding: 0 1px; border-radius: 1px; }
        .hl-p { background: var(--hl-p); padding: 0 1px; border-radius: 1px; }
        
        .b { font-weight: 700; }
        
        table { 
            border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 1px 0; 
            table-layout: fixed; word-break: break-all;
        }
        td, th { border: 0.5px solid #555; padding: 0 0.5px; text-align: center; }
        
        pre {
            font-family: monospace; font-size: 4.5pt; line-height: 0.9;
            letter-spacing: -0.1px; white-space: pre-wrap; word-break: break-all;
            margin: 1px 0; background: #f8f8f8; padding: 1px; border: 0.5px solid #ddd;
            color: #333; max-width: 100%;
        }

        @media print {
            body { background: white; }
            .page { margin: 0; border: none; page-break-after: always; }
        }
    </style>
</head>
<body>
`;

function processInline(text) {
    if (!text) return '';
    let placeholders = [];
    text = text.replace(/\$\$(.*?)\$\$/g, (match) => {
        const placeholder = `___MATH_DISPLAY_${placeholders.length}___`;
        placeholders.push({ placeholder, original: match });
        return placeholder;
    });
    text = text.replace(/\$(.*?)\$/g, (match) => {
        const placeholder = `___MATH_INLINE_${placeholders.length}___`;
        placeholders.push({ placeholder, original: match });
        return placeholder;
    });
    text = text.replace(/&/g, '&amp;');
    text = text.replace(/\*\*(.*?)\*\*/g, '<span class="b">$1</span>');
    text = text.replace(/\*(.*?)\*/g, '<i>$1</i>');
    for (let i = placeholders.length - 1; i >= 0; i--) {
        text = text.replace(placeholders[i].placeholder, () => placeholders[i].original);
    }
    return text;
}

function estimateHeight(html) {
    if (html.includes('section-header')) return 30;
    if (html.includes('<h2')) return 20;
    if (html.includes('<h3')) return 16;
    if (html.includes('<table')) return (html.split('<tr>').length) * 12 + 10;
    if (html.includes('<pre')) return (html.split('\n').length) * 6 + 12;
    if (html.includes('$$')) return (html.split('\n').length) * 20 + 10;
    return 10; // p
}

const mdContent = fs.readFileSync(inputFile, 'utf-8');
const lines = mdContent.split(/\r?\n/);

let allBlocks = [];
const colors = ['hl-y', 'hl-c', 'hl-g', 'hl-p'];
let colorIdx = 0;
let inCodeBlock = false;
let codeContent = '';
let inMathBlock = false;
let mathContent = '';

for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trimEnd();
    const trimmedLine = line.trim();

    if (trimmedLine.startsWith('$$')) {
        if (inMathBlock) {
            mathContent += trimmedLine.length > 2 ? '\n' + line : '$$';
            allBlocks.push(mathContent);
            inMathBlock = false;
            mathContent = '';
        } else {
            if (trimmedLine.endsWith('$$') && trimmedLine.length > 2) {
                allBlocks.push(line);
            } else {
                inMathBlock = true;
                mathContent = line;
            }
        }
        continue;
    }
    if (inMathBlock) { mathContent += '\n' + line; continue; }

    if (trimmedLine.startsWith('```')) {
        if (inCodeBlock) {
            const safeCode = codeContent.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            allBlocks.push(`<pre>${safeCode}</pre>`);
            inCodeBlock = false;
            codeContent = '';
        } else { inCodeBlock = true; }
        continue;
    }
    if (inCodeBlock) { codeContent += line + '\n'; continue; }

    const stripped = trimmedLine;
    if (stripped.startsWith('# ') && !stripped.startsWith('##')) {
        allBlocks.push(`<div class='section-header'>${stripped.substring(2).trim()}</div>`);
    } else if (stripped.startsWith('## ')) {
        const color = colors[colorIdx % 4];
        colorIdx++;
        allBlocks.push(`<h2 class="${color}">${processInline(stripped.substring(3).trim())}</h2>`);
    } else if (stripped.startsWith('### ')) {
        allBlocks.push(`<h3>${processInline(stripped.substring(4).trim())}</h3>`);
    } else if (stripped.startsWith('|')) {
        let tableLines = [];
        while (i < lines.length && lines[i].trim().startsWith('|')) { tableLines.push(lines[i].trim()); i++; }
        i--;
        let rows = [];
        for (let tl of tableLines) {
            if (/^\|[ :-\s|]+\|$/.test(tl) && tl.includes('-')) continue;
            let t = tl.replace(/\\\|/g, '___ESCAPED_PIPE___');
            if (t.startsWith('|')) t = t.substring(1);
            if (t.endsWith('|')) t = t.substring(0, t.length - 1);
            const cells = t.split('|').map(c => processInline(c.trim().replace(/___ESCAPED_PIPE___/g, '|')));
            rows.push(cells);
        }
        if (rows.length > 0) {
            let tblHtml = '<table>';
            for (let j = 0; j < rows.length; j++) {
                const tag = (j === 0) ? 'th' : 'td';
                tblHtml += '<tr>' + rows[j].map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>';
            }
            tblHtml += '</table>';
            allBlocks.push(tblHtml);
        }
    } else if (stripped.length > 0 && !stripped.startsWith('---')) {
        let c = processInline(stripped);
        if (c.startsWith('- ')) c = '&bull; ' + c.substring(2);
        allBlocks.push(`<p>${c}</p>`);
    }
}

let wrappedBlocks = allBlocks.map(b => {
    if (b.includes('section-header')) return b;
    return `<div class="box">${b}</div>`;
});

let totalH = wrappedBlocks.reduce((sum, b) => sum + estimateHeight(b), 0);
let splitTarget = totalH * 0.48; // Pushing more content to Page 2 to avoid overflow
let currentH = 0;
let splitIdx = 0;

for (let i = 0; i < wrappedBlocks.length; i++) {
    currentH += estimateHeight(wrappedBlocks[i]);
    if (currentH >= splitTarget) {
        // Look ahead for a heading to make a cleaner break
        let found = false;
        for (let j = i; j < i + 15 && j < wrappedBlocks.length; j++) {
            if (wrappedBlocks[j].includes('section-header') || wrappedBlocks[j].includes('<h2')) {
                splitIdx = j;
                found = true;
                break;
            }
        }
        if (!found) {
            // Look backward
            for (let j = i; j > i - 15 && j > 0; j--) {
                if (wrappedBlocks[j].includes('section-header') || wrappedBlocks[j].includes('<h2')) {
                    splitIdx = j;
                    found = true;
                    break;
                }
            }
        }
        if (!found) splitIdx = i;
        break;
    }
}

let p1 = wrappedBlocks.slice(0, splitIdx);
let p2 = wrappedBlocks.slice(splitIdx);

// --- OVERFLOW HEURISTIC ---
let p1Height = p1.reduce((sum, b) => sum + estimateHeight(b), 0);
const PAGE_CAPACITY = 3800; // Lowered to ensure no ghost columns
if (p1Height > PAGE_CAPACITY) {
    console.log(`Page 1 might overflow (${p1Height}/${PAGE_CAPACITY}). Shifting split point...`);
    for (let i = splitIdx - 1; i > 0; i--) {
        if (wrappedBlocks[i].includes('section-header') || wrappedBlocks[i].includes('<h2')) {
            splitIdx = i;
            p1 = wrappedBlocks.slice(0, splitIdx);
            p2 = wrappedBlocks.slice(splitIdx);
            p1Height = p1.reduce((sum, b) => sum + estimateHeight(b), 0);
            break;
        }
    }
}

let finalHtml = htmlHead;
console.log(`Final Page 1 Height: ${p1Height} / Target: ${splitTarget.toFixed(0)} / Capacity: ${PAGE_CAPACITY}`);
finalHtml += `<div class='page'><div class='columns'>${p1.join('\n')}</div></div>`;
finalHtml += `<div class='page'><div class='columns'>${p2.join('\n')}</div></div>`;
finalHtml += '</body></html>';

fs.writeFileSync(outputFile, finalHtml, 'utf-8');
console.log(`Done! Aggressive Fill Split at Index ${splitIdx}. Total Heuristic Height: ${totalH}`);
