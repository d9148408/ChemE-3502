import re, json

path = 'D:/MyGit/ChemE-3502/Unit11/Unit11_Example_04.ipynb'
content = open(path, 'r', encoding='utf-8').read()

print('=== COMPREHENSIVE CHECK CYCLES 1-5 ===')
print()

# 1. Simplified Chinese chars
print('--- Check 1: Simplified Chinese ---')
known_simplified = {
    chr(0x9636): 'U+9636 阶→階',
    chr(0x7B80): 'U+7B80 简→簡',
    chr(0x65F6): 'U+65F6 时→時',
    chr(0x9891): 'U+9891 频→頻',
    chr(0x56FE): 'U+56FE 图→圖',
    chr(0x6807): 'U+6807 标→標',
    chr(0x8BE5): 'U+8BE5 该→該',
    chr(0x5BF9): 'U+5BF9 对→對',
    chr(0x7EA7): 'U+7EA7 级→級',
    chr(0x7EDF): 'U+7EDF 统→統',
    chr(0x5355): 'U+5355 单→單',
    chr(0x8F93): 'U+8F93 输→輸',
    chr(0x53C2): 'U+53C2 参→參',
    chr(0x9009): 'U+9009 选→選',
    chr(0x8BAE): 'U+8BAE 议→議',
    chr(0x8BBA): 'U+8BBA 论→論',
}
found = []
for char, desc in known_simplified.items():
    if char in content:
        # find position
        pos = content.index(char)
        ctx = repr(content[max(0,pos-20):pos+20])
        found.append(f'{desc} at {pos}: {ctx}')
if found:
    for f in found:
        print(f'  FAIL: {f}')
else:
    print('  PASS - No simplified Chinese chars found')
print()

# 2. Unicode middle dot
print('--- Check 2: Unicode middle dot U+00B7 ---')
if chr(0x00B7) in content:
    pos = content.index(chr(0x00B7))
    print(f'  FAIL - Found at {pos}: {repr(content[max(0,pos-20):pos+20])}')
else:
    print('  PASS - No U+00B7 found')
print()

# 3. Inline math spacing
print('--- Check 3: Inline math spacing ---')
# Parse actual cell content
nb = json.loads(content)
issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        # Check for CJK/fullwidth char immediately before or after $
        bad = re.findall(r'[\u3000-\u9fff\uff00-\uffef]\$|\$[\u3000-\u9fff\uff00-\uffef]', src)
        if bad:
            issues.append(f'Cell {i+1}: {bad}')
if issues:
    for issue in issues:
        print(f'  WARN: {issue}')
else:
    print('  PASS - Inline math spacing OK')
print()

# 4. Block math blank lines
print('--- Check 4: Block math blank lines ---')
issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        # Check: $$ not preceded by blank line
        if re.search(r'[^\n]\n\$\$', src):
            issues.append(f'Cell {i+1}: $$ not preceded by blank line')
        # Check: $$ not followed by blank line
        if re.search(r'\$\$\n[^\n$]', src):
            issues.append(f'Cell {i+1}: $$ not followed by blank line')
if issues:
    for issue in issues:
        print(f'  WARN: {issue}')
else:
    print('  PASS - Block math blank lines OK')
print()

# 5. English-only plot labels
print('--- Check 5: English-only plot labels ---')
issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        # Find string arguments to plot label functions
        matches = re.findall(r'(?:set_title|set_xlabel|set_ylabel|suptitle)\s*\(\s*[\'\"f](.*?)[\'\"]', src)
        # Also check label= and annotate
        matches += re.findall(r'label\s*=\s*[\'\"](.*?)[\'\"]', src)
        matches += re.findall(r'annotate\s*\(\s*[\'\"](.*?)[\'\"]', src)
        for m in matches:
            if re.search(r'[\u4e00-\u9fff]', m):
                issues.append(f'Cell {i+1}: {repr(m[:50])}')
if issues:
    for issue in issues:
        print(f'  FAIL: {issue}')
else:
    print('  PASS - All plot labels English only')
print()

# 6. Formula formatting
print('--- Check 6: Formula formatting ---')
issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        if r'\cdotp' in src:
            issues.append(f'Cell {i+1}: \\cdotp found')
        # Check \text{} with unescaped underscore
        bad_text = re.findall(r'\\text\{[^}]*_[^}]*\}', src)
        if bad_text:
            issues.append(f'Cell {i+1}: \\text{{}} with _: {bad_text}')
if issues:
    for issue in issues:
        print(f'  FAIL: {issue}')
else:
    print('  PASS - Formula formatting OK')
print()

# 7. Duplicate cells
print('--- Check 7: Duplicate cells ---')
cell_sigs = []
for cell in nb['cells']:
    src = ''.join(cell['source'])[:100]
    cell_sigs.append(src.strip())
dupes = []
seen = set()
for s in cell_sigs:
    if s and s in seen:
        dupes.append(repr(s[:60]))
    seen.add(s)
if dupes:
    for d in dupes:
        print(f'  WARN: Duplicate: {d}')
else:
    print('  PASS - No duplicate cells')
print()

# 8. Cell structure summary
print('--- Check 8: Cell structure ---')
nb_cells = nb['cells']
print(f'  Total cells: {len(nb_cells)}')
md_count = sum(1 for c in nb_cells if c['cell_type'] == 'markdown')
code_count = sum(1 for c in nb_cells if c['cell_type'] == 'code')
print(f'  Markdown: {md_count}, Code: {code_count}')
# Check alternating md/code pattern (standard)
print('  Cell order:', ' '.join(['MD' if c['cell_type']=='markdown' else 'PY' for c in nb_cells]))
print()

# 9. Check figure save paths
print('--- Check 9: Figure save paths ---')
figs = re.findall(r"savefig\([^)]+\)", content)
for f in figs:
    print(f'  {f}')
print()

print('=== ALL CHECKS COMPLETE ===')
