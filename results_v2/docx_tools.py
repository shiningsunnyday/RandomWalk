"""
Helpers to edit an unpacked .docx (word/document.xml) in place with lxml, preserving formatting.
Text edits replace the text of the first run of a cell/paragraph and drop its other runs, so the first run's
formatting carries over. Never pretty-prints.
"""
import re
from copy import deepcopy
from lxml import etree

WNS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W = '{%s}' % WNS
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'


def load(path):
    return etree.parse(path, etree.XMLParser(remove_blank_text=False, huge_tree=True))


def save(tree, path):
    tree.write(path, xml_declaration=True, encoding='UTF-8', standalone=True)


def text(e):
    return ''.join(t.text or '' for t in e.iter(W + 't'))


def body(tree):
    return tree.getroot().find(W + 'body')


def find_par(tree, prefix):
    """first top-level paragraph whose text starts with prefix"""
    for e in body(tree):
        if e.tag == W + 'p' and text(e).strip().startswith(prefix):
            return e
    raise KeyError(prefix)


def tables_after(tree, caption_prefix, stop=r'^(Table|Figure) S\d+'):
    """all top-level tables after the caption paragraph, up to the next caption paragraph"""
    kids = list(body(tree)); cap = find_par(tree, caption_prefix); i = kids.index(cap) + 1; out = []
    while i < len(kids):
        e = kids[i]
        if e.tag == W + 'p' and re.match(stop, text(e).strip()):
            break
        if e.tag == W + 'tbl':
            out.append(e)
        i += 1
    return out


def set_runs_text(p, s):
    """set paragraph p's text to s in its first text-bearing run (keeping that run's formatting)"""
    runs = [r for r in p.findall(W + 'r')]
    keep = next((r for r in runs if r.find(W + 't') is not None), runs[0] if runs else None)
    if keep is None:
        keep = etree.SubElement(p, W + 'r')
    for r in runs:
        if r is not keep:
            p.remove(r)
    for child in list(keep):
        if child.tag in (W + 't', W + 'tab', W + 'br', W + 'drawing'):
            keep.remove(child)
    t = etree.SubElement(keep, W + 't'); t.text = s; t.set(XML_SPACE, 'preserve')


def set_cell_text(tc, s):
    ps = tc.findall(W + 'p')
    for p in ps[1:]:
        tc.remove(p)
    set_runs_text(ps[0], s)


def fill_table_rows(tables, rows, n_header=1, template_row=None):
    """write rows (list of lists of str) into the data rows of a (possibly fragmented) table, in order.
    Header rows of the first fragment are kept; rows are cloned/removed so the data-row count equals len(rows)."""
    slots = []
    for k, tbl in enumerate(tables):
        trs = tbl.findall(W + 'tr')
        slots += trs[n_header:] if k == 0 else trs
    ncols = max(len(r) for r in rows)
    good = template_row if template_row is not None else next(tr for tr in slots if len(tr.findall(W + 'tc')) == ncols)
    good = deepcopy(good)
    while len(slots) < len(rows):                       # append clones to the last fragment
        new = deepcopy(good); tables[-1].append(new); slots.append(new)
    for tr in slots[len(rows):]:                          # drop surplus rows
        tr.getparent().remove(tr)
    for tr, vals in zip(slots, rows):
        if len(tr.findall(W + 'tc')) != len(vals):        # malformed (merged) row: replace with a clean clone
            new = deepcopy(good); tr.addprevious(new); tr.getparent().remove(tr); tr = new
        for tc, v in zip(tr.findall(W + 'tc'), vals):
            set_cell_text(tc, v)


def add_par_after(p, s):
    """Insert a new paragraph holding `s` directly after `p`, cloning `p`'s paragraph and run formatting."""
    from copy import deepcopy
    new = deepcopy(p)
    set_runs_text(new, s)
    p.addnext(new)
    return new


def append_rich(p, s, normal, sub):
    """Append text to a paragraph, cloning `normal`/`sub` (two w:r elements) for formatting. In `s`, text between
    tildes is set in the subscript run, so 'CO~2~ and H~2~O' renders the digits as subscripts."""
    from copy import deepcopy
    for i, chunk in enumerate(s.split('~')):
        if not chunk:
            continue
        r = deepcopy(sub if i % 2 else normal)
        ts = r.findall(W + 't')
        assert ts, 'template run has no text node'
        for extra in ts[1:]:
            r.remove(extra)
        ts[0].text = chunk
        ts[0].set(XML_SPACE, 'preserve')
        p.append(r)


def sub_run(p):
    """A run of `p` that is set as subscript, to use as a template for append_rich."""
    hits = p.xpath('.//*[local-name()="r"][.//*[local-name()="vertAlign"][@*[local-name()="val"]="subscript"]]')
    assert hits, 'no subscript run in this paragraph'
    return hits[0]


def replace_in_par(p, old, new, required=True):
    """replace text old->new inside paragraph p; requires old to sit inside one w:t (after merge_runs).

    Raises by default when `old` is not found: a citation or a field splits a sentence into separate runs even after
    merge_runs, so a phrase that is plainly visible in the rendered document may not exist as one string in the XML,
    and a silent no-op there means an edit the build log claims to have made is simply missing."""
    for t in p.iter(W + 't'):
        if t.text and old in t.text:
            t.text = t.text.replace(old, new); t.set(XML_SPACE, 'preserve'); return True
    if required:
        raise ValueError(f'replace_in_par: {old!r} not found in a single run of {text(p)[:120]!r}')
    return False


def replace_everywhere(tree, old, new, expect=None):
    n = 0
    for t in tree.getroot().iter(W + 't'):
        if t.text and old in t.text:
            n += t.text.count(old); t.text = t.text.replace(old, new); t.set(XML_SPACE, 'preserve')
    if expect is not None and n != expect:
        raise ValueError(f'{old!r}: expected {expect} replacements, made {n}')
    return n


def set_col_widths(tbl, widths):
    """set column widths (dxa) in the grid, the table width and every cell of rows that have len(widths) cells"""
    grid = tbl.find(W + 'tblGrid')
    assert len(grid) == len(widths), (len(grid), widths)
    for g, w in zip(grid, widths):
        g.set(W + 'w', str(w))
    tw = tbl.find(W + 'tblPr').find(W + 'tblW')
    if tw is not None and tw.get(W + 'type') == 'dxa':
        tw.set(W + 'w', str(sum(widths)))
    for tr in tbl.findall(W + 'tr'):
        tcs = tr.findall(W + 'tc')
        if len(tcs) == len(widths):
            for tc, w in zip(tcs, widths):
                tcw = tc.find(W + 'tcPr/' + W + 'tcW')
                if tcw is not None:
                    tcw.set(W + 'w', str(w)); tcw.set(W + 'type', 'dxa')
