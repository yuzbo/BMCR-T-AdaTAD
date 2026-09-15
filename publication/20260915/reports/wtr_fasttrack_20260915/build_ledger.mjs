import fs from 'node:fs/promises';
import { Workbook } from '@oai/artifact-tool';
const out = new URL('.', import.meta.url);
const wb=Workbook.create();
const sheet=wb.worksheets.add('Final model ledger');
const rows=[
 ['Mechanism','Gate','Evidence','Status','Final'],
 ['T Value','Value > Uniform/H65','Pending measured evidence','WAITING','no'],
 ['D Value','Policy-level Value > Uniform; A75/F100 diagnostic','Pending measured evidence','WAITING','no'],
 ['S Value','Value > Uniform; A100/F75 diagnostic','Pending measured evidence','WAITING','no'],
 ['Nested DS','Joint > best single; single-axis gate first','Pending measured evidence','WAITING','no'],
 ['Graph Context','Regret improves + downstream improves; matched MLP','Pending measured evidence','WAITING','no'],
 ['Graph Recovery','Improvement over matched Cross_fresh','Pending measured evidence','WAITING','no'],
 ['FVD','Actual drift + function forecast + downstream gain','Pending measured evidence','WAITING','no'],
 ['DB','Trained-tier oracle gap + learned planner gain','Pending measured evidence','WAITING','no'],
 ['Raw','Domain headroom + learned value + matched retraining','Pending measured evidence','WAITING','no'],
];
sheet.getRange('A1:E10').values=rows;
sheet.showGridLines=false;
sheet.getRange('A1:E10').format.font={name:'Arial',size:10};
sheet.getRange('A1:E1').format.fill='#334155';
sheet.getRange('A1:E1').format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
sheet.getRange('A1:A10').format.columnWidth=23;
sheet.getRange('B1:B10').format.columnWidth=70;
sheet.getRange('C1:C10').format.columnWidth=33;
sheet.getRange('D1:E10').format.columnWidth=13;
sheet.getRange('A1:E10').format.rowHeight=23;
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:"'Final model ledger'!A1:E10",include:'values',tableMaxRows:10,tableMaxCols:5,maxChars:4500})).ndjson);
const values=sheet.getRange('A1:E10').values;
if(values.length!==10 || values.slice(1).some(r=>r[3]!=='WAITING'||r[4]!=='no')) throw new Error('Unmeasured mechanism marked accepted');
const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
await fs.writeFile(new URL('FINAL_MODEL_LEDGER.csv',out), values.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n','utf8');
const png=await wb.render({sheetName:sheet.name,range:'A1:E10',scale:1,format:'png'});
await fs.writeFile(new URL('ledger_preview.png',out),new Uint8Array(await png.arrayBuffer()));
