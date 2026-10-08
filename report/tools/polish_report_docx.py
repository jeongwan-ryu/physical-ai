"""Claude 문서에서 내보낸 Word(docx)를 한글 보고서 서식으로 다듬는다.

적용 내용: 맑은 고딕 10pt · 줄 간격 1.25 · A4 2cm 여백 · 색 있는 제목과 구분선 ·
한국식 날짜 · 내용 길이에 맞춘 표 열 너비 · 연회색 테두리와 칸 여백 · 그림 가운데 정렬과 회색 캡션

사용법 (표준 라이브러리만 사용):
    python polish_report_docx.py <내보낸.docx> <결과.docx>
"""
import io
import re
import sys
import zipfile

src, out = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
files = {n: zin.read(n) for n in zin.namelist()}

FONT = "맑은 고딕"
ACCENT = "1F4E79"
PAGE_W, MARGIN = 11906, 1134            # A4, 2cm 여백
TEXT_W = PAGE_W - 2 * MARGIN            # 9638 dxa
EMU_PER_DXA = 635

# ---------------- styles.xml ----------------
st = files["word/styles.xml"].decode("utf-8")
st = re.sub(r'<w:rFonts w:ascii="Calibri"[^/]*/>',
            f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>', st, count=1)
st = st.replace('<w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-US"/>',
                '<w:sz w:val="20"/><w:szCs w:val="20"/><w:lang w:val="en-US" w:eastAsia="ko-KR"/>')
st = st.replace('<w:spacing w:after="160" w:line="259" w:lineRule="auto"/>',
                '<w:spacing w:after="140" w:line="300" w:lineRule="auto"/>')


def heading(level, size, before, after, color, border):
    bdr = (f'<w:pBdr><w:bottom w:val="single" w:sz="{border}" w:space="4" w:color="{color}"/></w:pBdr>'
           if border else "")
    return (f'<w:style w:type="paragraph" w:styleId="Heading{level}"><w:name w:val="heading {level}"/>'
            f'<w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/>{bdr}'
            f'<w:spacing w:before="{before}" w:after="{after}"/><w:outlineLvl w:val="{level - 1}"/></w:pPr>'
            f'<w:rPr><w:b/><w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr></w:style>')


for lvl, args in {1: (40, 0, 120, "1F3864", 0), 2: (28, 440, 160, ACCENT, 6), 3: (23, 280, 100, "2E75B6", 0)}.items():
    st = re.sub(rf'<w:style w:type="paragraph" w:styleId="Heading{lvl}">.*?</w:style>', heading(lvl, *args), st, count=1)

# 코드 글꼴은 한글이 섞여도 깨지지 않게
st = st.replace('w:eastAsia="Consolas"', f'w:eastAsia="{FONT}"')
# 표: 연한 회색 테두리, 칸 안쪽 여백
st = st.replace('w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
                'w:color="BFBFBF"/><w:left w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>')
st = re.sub(r'(<w:style w:type="table" w:styleId="TableGrid">.*?</w:style>)',
            lambda m: m.group(1).replace('w:color="auto"', 'w:color="BFBFBF"'), st, flags=re.S)
st = st.replace('<w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/>',
                '<w:top w:w="70" w:type="dxa"/><w:left w:w="110" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/>')
files["word/styles.xml"] = st.encode("utf-8")

# ---------------- document.xml ----------------
doc = files["word/document.xml"].decode("utf-8")

# A4 + 2cm 여백
doc = re.sub(r'<w:pgSz [^/]*/>', f'<w:pgSz w:w="{PAGE_W}" w:h="16838"/>', doc)
doc = re.sub(r'<w:pgMar [^/]*/>',
             f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" '
             'w:header="567" w:footer="567" w:gutter="0"/>', doc)

# 바이라인: 영어 날짜 → 한국식, 회색 작은 글씨
doc = re.sub(r'<w:p><w:pPr></w:pPr><w:r><w:t xml:space="preserve">([A-Z][a-z]{2}) (\d+), (\d{4})</w:t></w:r>'
             r'<w:r><w:t xml:space="preserve"> · </w:t></w:r><w:r><w:t xml:space="preserve">@([^<]+)</w:t></w:r></w:p>',
             lambda m: ('<w:p><w:pPr><w:spacing w:after="320"/></w:pPr><w:r><w:rPr><w:color w:val="7F7F7F"/>'
                        f'<w:sz w:val="19"/></w:rPr><w:t xml:space="preserve">작성일 {m.group(3)}-'
                        f'{["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].index(m.group(1)) + 1:02d}-'
                        f'{int(m.group(2)):02d} · 작성자 {m.group(4)}</w:t></w:r></w:p>'), doc, count=1)


def text_len(xml):
    """칸 안 글자의 화면상 폭 근사: 한글 2, 그 외 1."""
    s = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml))
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)


def polish_table(m):
    tbl = m.group(0)
    rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, flags=re.S)
    cells = [re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S) for r in rows]
    ncol = max(len(c) for c in cells)
    # 열 너비: 열마다 가장 긴 칸(상한 60)의 비율, 최소 12%
    need = [max(min(text_len(r[i]), 60) if i < len(r) else 0 for r in cells) + 6 for i in range(ncol)]
    share = [n / sum(need) for n in need]
    floor = 0.12
    share = [max(s, floor) for s in share]
    widths = [int(TEXT_W * s / sum(share)) for s in share]
    widths[-1] = TEXT_W - sum(widths[:-1])

    tbl = re.sub(r'<w:tblW [^/]*/>', f'<w:tblW w:w="{TEXT_W}" w:type="dxa"/>', tbl, count=1)
    tbl = re.sub(r'<w:tblGrid>.*?</w:tblGrid>',
                 '<w:tblGrid>' + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>', tbl, count=1)

    def fix_row(rm):
        row = rm.group(0)
        is_head = "<w:tblHeader/>" in row
        idx = iter(range(ncol))

        def fix_cell(cm):
            cell = cm.group(0)
            i = next(idx, ncol - 1)
            cell = re.sub(r'<w:tcW [^/]*/>', f'<w:tcW w:w="{widths[i]}" w:type="dxa"/>', cell, count=1)
            if is_head:
                cell = cell.replace('w:fill="F2F2F2"', 'w:fill="DEEAF6"')
            # 칸 안 문단: 간격 줄이고 글씨 9pt
            cell = cell.replace('<w:pPr></w:pPr>', '<w:pPr><w:spacing w:before="0" w:after="0" w:line="264" w:lineRule="auto"/></w:pPr>')
            cell = re.sub(r'<w:r>(?!<w:rPr>)', '<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>', cell)
            cell = re.sub(r'<w:r><w:rPr>(?!<w:sz )', '<w:r><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/>', cell)
            if is_head:
                cell = cell.replace('<w:rPr><w:sz w:val="18"/>', '<w:rPr><w:b/><w:color w:val="1F3864"/><w:sz w:val="18"/>')
            return cell

        return re.sub(r"<w:tc>.*?</w:tc>", fix_cell, row, flags=re.S)

    tbl = re.sub(r"<w:tr>.*?</w:tr>", fix_row, tbl, flags=re.S)
    # 표 뒤에 숨 쉴 공간
    return tbl


doc = re.sub(r"<w:tbl>.*?</w:tbl>", polish_table, doc, flags=re.S)
doc = doc.replace("</w:tbl>", '</w:tbl><w:p><w:pPr><w:spacing w:before="0" w:after="60"/></w:pPr></w:p>')

# 그림: 본문 폭에 맞게 확대
max_cx = TEXT_W * EMU_PER_DXA


def scale(m):
    cx, cy = int(m.group(1)), int(m.group(2))
    ncy = int(cy * max_cx / cx)
    return f'cx="{max_cx}" cy="{ncy}"'


doc = re.sub(r'(?<=<wp:extent )cx="(\d+)" cy="(\d+)"', scale, doc)
doc = re.sub(r'(?<=<a:ext )cx="(\d+)" cy="(\d+)"', scale, doc)
# 그림은 가운데, 그림 바로 아래 문단은 작은 회색 캡션
doc = re.sub(r'<w:p><w:pPr></w:pPr>(<w:r>(?:(?!</w:p>).)*?<w:drawing>)',
             r'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="0"/></w:pPr>\1', doc, flags=re.S)
doc = re.sub(r'(</w:drawing></w:r></w:p>)<w:p><w:pPr></w:pPr><w:r>(<w:t[^>]*>[^<]*</w:t></w:r></w:p>)',
             r'\1<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="240"/></w:pPr>'
             r'<w:r><w:rPr><w:color w:val="7F7F7F"/><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr>\2', doc)
files["word/document.xml"] = doc.encode("utf-8")

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for info in zin.infolist():
        z.writestr(info, files[info.filename])
open(out, "wb").write(buf.getvalue())
print("saved", out, len(buf.getvalue()))
