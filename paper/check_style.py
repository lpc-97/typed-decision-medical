"""Writing-style self-check for main.tex: sentence length, cross-reference form, captions, spelling and section structure."""
import re, sys
s = open('main.tex', encoding='utf-8').read()
body = s.split(r'\begin{document}')[1].split(r'\bibliography{refs}')[0]
fails = []

def section(name, nxt):
    m = re.search(r'\\section\{' + re.escape(name) + r'\}(.*?)\\section\{' + re.escape(nxt) + r'\}', body, re.S)
    return m.group(1) if m else ''

def prose(t):
    t = re.sub(r'\\begin\{(figure|table|equation|tikzpicture|itemize)\*?\}.*?\\end\{\1\*?\}', ' ', t, flags=re.S)
    t = re.sub(r'%.*', '', t)
    t = re.sub(r'\\(citep|citet|ref|label|eqref)\{[^}]*\}', 'X', t)
    t = re.sub(r'\$[^$]*\$', 'X', t)
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t); t = re.sub(r'[{}~]', ' ', t)
    return t

# 1. long sentences
txt = prose(body)
sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', re.sub(r'\s+', ' ', txt)) if len(x.split()) > 3]
long = [x for x in sents if len(x.split()) >= 45]
share = len(long) / max(1, len(sents))
print(f"sentences {len(sents)}, >=45 words {len(long)} ({share:.1%})")
if share > 0.05: fails.append('long-sentence share > 5%')
for x in long: print('   LONG:', len(x.split()), x[:110])
# 2. abbreviations in running text
for pat in (r'Sec\.~\\ref', r'Fig\.~\\ref', r'Tab\.~\\ref', r'\bSec\.\s', r'\bTab\.\s'):
    if re.search(pat, body): fails.append(f'abbreviated cross-reference {pat}')
if re.search(r'(?<!Appendix )\bFig\.~', body): fails.append('"Fig.~" in running text')
# 3. bold in captions
for cap in re.findall(r'\\caption\{(.*?)\}\s*\n', body, re.S):
    if r'\textbf' in cap: fails.append('bold in caption: ' + cap[:50])
# 4. British spellings
brit = r'\b\w*(initialis|normalis|standardis|optimis|regularis|stabilis|summaris|recognis|colour|behaviour|favour|centre|labelled|modelling|artefact|analyse)\w*'
bs = sorted(set(m.group(0) for m in re.finditer(brit, body, re.I)))
if bs: fails.append('British spellings: ' + ', '.join(bs))
# 5. headings
for h in re.findall(r'\\(?:sub)*section\*?\{([^}]*)\}', body):
    if '?' in h: fails.append('question mark in heading: ' + h)
# 6. contributions and keywords
intro = section('Introduction', 'Related Work')
items = re.findall(r'\\item (We \w+)', intro)
print('contribution bullets:', items)
if [i.split()[1] for i in items] != ['introduce', 'develop', 'evaluate']: fails.append('contribution bullets not introduce/develop/evaluate')
kw = re.search(r'\\keywords\{(.*?)\}', s).group(1).split(r'\and')
print('keywords:', len(kw))
if not 4 <= len(kw) <= 5: fails.append('keywords count')
# 7. abstract
ab = prose(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S).group(1))
nw = len(ab.split()); print('abstract words:', nw)
if not 200 <= nw <= 280: fails.append(f'abstract length {nw}')
if 'Qwen3.5' not in ab: fails.append('backbone not named in abstract')
# 8. no numbers in introduction paragraphs (model names such as Qwen3.5 excluded)
ip = prose(intro.split(r'\begin{itemize}')[0])
nums = [m.group(0) for m in re.finditer(r'(?<![A-Za-z\-.\d])\d+(\.\d+)?%?', ip)]
print('numbers in introduction:', nums)
if nums: fails.append('numbers in introduction')
# 9. run-in headings in Method and Experiments
meth = section('Method', 'Experiments'); exp = section('Experiments', 'Conclusion')
print('run-in headings: method', meth.count(r'\paragraph{'), 'experiments', exp.count(r'\paragraph{'))
if meth.count(r'\paragraph{') < 4 or exp.count(r'\paragraph{') < 6: fails.append('too few run-in headings')
# 10. related work paragraphs end with positioning sentence
rw = section('Related Work', 'Method')
paras = [p.strip() for p in re.split(r'\n\s*\n', re.sub(r'\\subsection\{[^}]*\}', '\n\n', rw)) if len(p.split()) > 30]
pos = re.compile(r'\b(our|we|rather than|focus|distinction|interest)\b', re.I)
for p in paras:
    last = re.split(r'(?<=[.!?])\s+', prose(p).strip())[-1]
    if not pos.search(last): fails.append('RW paragraph without positioning sentence: ' + last[:70])
print('related-work paragraphs:', len(paras))
# 11. numbers in related work
rwn = [m.group(0) for m in re.finditer(r'(?<![A-Za-z\-.\d])\d+(\.\d+)?', prose(rw))]
if rwn: fails.append(f'numbers in related work: {rwn}')
# 12. conclusion: 2 paragraphs, 180-250 words
con = section('Conclusion', 'Data and Code Availability') if r'\section*{Data' not in body else body.split(r'\section{Conclusion}')[1].split(r'\section*{')[0]
con = re.sub(r'(?m)^%.*\n', '', con)
cp = [p for p in re.split(r'\n\s*\n', con.strip()) if len(p.split()) > 10]
cw = len(prose(con).split()); print('conclusion paragraphs:', len(cp), 'words:', cw)
if len(cp) != 2 or not 180 <= cw <= 250: fails.append(f'conclusion shape ({len(cp)} paragraphs, {cw} words)')
if re.search(r'\\section\{Limitations', body) or re.search(r'\\paragraph\{Limitations', body.split(r'\appendix')[0] if r'\appendix' in body else body): fails.append('separate Limitations section')
# 13. unresolved refs
log = open('main.log', encoding='utf-8', errors='ignore').read()
if 'undefined' in log: fails.append('undefined references in log')
print('\nRESULT:', 'PASS' if not fails else 'FAIL'); [print(' -', f) for f in fails]
sys.exit(1 if fails else 0)
