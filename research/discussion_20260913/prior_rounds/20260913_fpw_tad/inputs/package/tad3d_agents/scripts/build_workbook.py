from pathlib import Path
import json,csv,re
from artifact_tool import Workbook, SpreadsheetFile
R=Path(__file__).resolve().parents[1]
wb=Workbook.create()

def format_sheet(s,lastcol,n):
    s.freeze_panes.freeze_rows(1)
    s.get_range(f'A1:{lastcol}1').format={'fill':'#202020','font':{'bold':True,'color':'#FFFFFF'},'row_height':28,'wrap_text':True}
    s.get_range(f'A2:{lastcol}{n}').format.row_height=36
    s.get_range(f'A1:{lastcol}{n}').format.column_width=17
    s.get_range(f'A1:{lastcol}{n}').format.wrap_text=True

s=wb.worksheets.add('Historical')
rows=json.loads((R/'data/historical_results.json').read_text())
headers=['Backbone','Method','mAP@0.3','mAP@0.4','mAP@0.5','mAP@0.6','mAP@0.7','Average mAP','GFLOPs (2MAC)','Model mean ms','Model median ms','Selection','Source','Measurement scope']
values=[headers]
for row in rows:
    values.append([row['backbone'],row['method']]+[row[f'map_{i}_ratio'] for i in ['03','04','05','06','07']]+[None,row['gflops_2mac'],row['model_mean_ms'],row['model_median_ms'],row['selection'],row['source_url'],row['profile_scope']])
s.get_range('A1:N7').values=values
s.get_range('H2:H7').formulas=[[f'=AVERAGE(C{i}:G{i})'] for i in range(2,8)]
format_sheet(s,'N',7)
s.get_range('A1:A7').format.column_width=10;s.get_range('B1:B7').format.column_width=20
s.get_range('C2:H7').set_number_format('0.00%');s.get_range('I2:K7').set_number_format('0.00')
s.get_range('M1:N7').format.column_width=58
s.get_range('A9:N10').merge();s.get_range('A9').values=[['Historical fixed-snapshot records only. Official/new models use different training. Latency is one fixed GPU-resident window, not E2E, and nodes may differ. No new model performance has been measured.']]
s.get_range('A9:N10').format.wrap_text=True
p=json.loads((R/'plans/experiments.json').read_text())
s=wb.worksheets.add('Experiments')
head=['ID','Tier','Stage','Dependencies','Intervention','Matched factors','Interpretation','Falsifying result','State','New mAP']
v=[head]
for e in p['experiments']:
    v.append([e['id'],e['tier'],e['stage'],','.join(e['depends_on']),e['intervention'],e['matched_factors'],e['interpretation'],e['falsifying_outcome'],'planned',None])
s.get_range(f'A1:J{len(v)}').values=v;format_sheet(s,'J',len(v))
s.get_range(f'E1:H{len(v)}').format.column_width=46
s.get_range(f'A2:J{len(v)}').format.row_height=64
s=wb.worksheets.add('Figures')
with (R/'plans/figures.csv').open(encoding='utf-8-sig') as f:v=list(csv.reader(f))
s.get_range(f'A1:I{len(v)}').values=v;format_sheet(s,'I',len(v));s.get_range(f'F1:I{len(v)}').format.column_width=46
s.get_range(f'A2:I{len(v)}').format.row_height=64
s=wb.worksheets.add('Claims')
with (R/'plans/claims.csv').open(encoding='utf-8-sig') as f:v=list(csv.reader(f))
s.get_range(f'A1:E{len(v)}').values=v;format_sheet(s,'E',len(v));s.get_range(f'B1:E{len(v)}').format.column_width=48
s=wb.worksheets.add('Sources')
text=(R/'SOURCES.md').read_text()
v=[['Reference','Title / access scope','Primary URL']]
for block in text.split('\n\n'):
    if re.match(r'\[(P?\d+)\]',block):
        label=re.match(r'\[([^]]+)\]',block)[1]
        urls=re.findall(r'https?://[^\s;；]+',block)
        for url in urls:v.append([label,re.sub(r'https?://[^\s;；]+','',block),url])
s.get_range(f'A1:C{len(v)}').values=v;format_sheet(s,'C',len(v));s.get_range(f'B1:C{len(v)}').format.column_width=85
s.get_range(f'A2:C{len(v)}').format.row_height=70
# Inspect formulas and rendered historical data before export.
inspection=wb.inspect({'kind':'region','sheet_id':'Historical','range':'A1:K7','max_chars':12000})
(R/'validation/workbook_inspection.txt').write_text(inspection.ndjson)
if any(err in inspection.ndjson for err in ['#REF!','#DIV/0!','#VALUE!','#NAME?','#NUM!']):raise ValueError('Formula error in workbook')
wb.render({'sheetName':'Historical','range':'A1:K7','format':'png','scale':1.2}).save(str(R/'validation/workbook_preview.png'))
SpreadsheetFile.export_xlsx(wb).save(str(R/'TAD_experiments_figures_data.xlsx'))
print('Workbook exported; historical average-mAP formula checked.')
