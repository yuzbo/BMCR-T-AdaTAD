import fs from 'node:fs/promises';
import { Workbook } from '@oai/artifact-tool';
const out = new URL('.', import.meta.url);
const wb=await Workbook.fromCSV(await fs.readFile(new URL('FINAL_MODEL_LEDGER.csv',out),'utf8'),{sheetName:'Final model ledger'});
const sheet=wb.worksheets.getItemAt(0);
const rows=[
 ['Mechanism','Research level','Implementation','Evidence / gate','Task status','Scope / limits','Source','Final'],
 ['Atlas S allocation','DIAGNOSTIC','COMPLETE','T/D/S finite group headroom','NOT_A_TRAINED_METHOD','GT-assisted; extra query cost','a50b84d / 65e14a8','no'],
 ['Atlas interaction','DIAGNOSTIC','COMPLETE','INTERACTION_INCONCLUSIVE','NOT_A_TRAINED_METHOD','AP interaction CI crosses0','49f74bd / 5d2402a','no'],
 ['T local opportunity','DIAGNOSTIC','HEADROOM_PASS','cal20/46: +0.528pp [+.082,+.715]','LOCAL_FAMILY_ONLY','Not a global subset ceiling','2ca4d4b / dca6564','no'],
 ['Plain Value-R1','DIAGNOSTIC','TECH_PASS','LEARNABILITY_FAIL','LOCAL_NOT_UNLOCKED','18 heads; original407 and8/8','b644d87','no'],
 ['Reverse P1/P1-bi/P2','DIAGNOSTIC','TECH_PASS','STRUCTURE_SIGNAL_FAIL','LOCAL_NOT_UNLOCKED','6 heads; packing-family OOD','800bcd1 / 4aa1ca2','no'],
 ['Local Graph Context','DIAGNOSTIC','TECH_PASS','FAIL_OLD / WAITING_V3','LOCAL_NOT_UNLOCKED','Matched-MLP CI unstable','b679ae0 / b644d87','no'],
 ['Historical RISE','DIAGNOSTIC','COMPLETE','DRIFT_YES / FORECAST_FAIL','LOCAL_FVD_NOT_UNLOCKED','held30:0 final choice changes','3a7e93f / 800bcd1','no'],
 ['D/S route + clipping','DIAGNOSTIC','REVIEW_PASS','clip ratio>=0.998567','NOT_A_TASK_CAUSAL_PROOF','raw40;4 windows/axis;0 updates','7245972','no'],
 ['Raw mini','DIAGNOSTIC','TECH_PASS','DOMAIN_COVERAGE_INSUFFICIENT','NO_MATCHED_RETRAIN','R+ offgrid queries24/180','27d557e','no'],
 ['D Value / Uniform','FULL_COMPONENT_COURSE','TRAINING','D60 V64.3688% / U64.4652%','TASK_FAIL_SO_FAR','T fixed; quotas;80 endpoint pending','4055294','no'],
 ['S Value / Uniform','FULL_COMPONENT_COURSE','TRAINING','S40 V64.0616% / U64.2459%','TASK_FAIL_SO_FAR','T fixed; quotas;80 endpoint pending','4055294','no'],
 ['Historical BMCR/H65','HISTORICAL_FULL_MODEL','RECORDED','Own historical course evidence','NOT_NEW_GLOBAL_CONTROL','Different recipe/coordinate history','Historical source refs','no'],
 ['Global-Task T','FORMAL_CANDIDATE','PRIMITIVES_EXIST','OWN_TECHNICAL_GATE_PENDING','NOT_STARTED','BMCR sampler/transport; recipe gap','Formal design','no'],
 ['Global-Task+Value','FORMAL_CANDIDATE','DESIGN_PENDING','OWN_ABLATION_PENDING','NOT_STARTED','Same policy; actual Value auxiliary','Formal design','no'],
 ['Global S/D routing','FORMAL_CANDIDATE','PARTIAL_EXECUTOR','TASK_GRADIENT_PENDING','NOT_STARTED','Cross-time/layer budget not wired','Formal design','no'],
 ['Global TSD Core','FORMAL_CANDIDATE','DESIGN_PENDING','JOINT_TASK_EVIDENCE_PENDING','NOT_STARTED','Compare with strongest single axis','Formal design','no'],
 ['Graph Recovery','FORMAL_CANDIDATE','PRIMITIVES_EXIST','OWN_TASK_GATE_PENDING','NOT_PROVEN','Atlas recovery is a different probe','Paper Graph modules','no'],
 ['Global Graph/RISE','FORMAL_CANDIDATE','DESIGN_PENDING','OWN_MECHANISM_GATES','NOT_STARTED','Do not inherit local gate verdicts','Formal design','no'],
 ['Variable budget planner','FORMAL_CANDIDATE','PRIMITIVES_EXIST','TRAINED_TIERS_REQUIRED','NOT_STARTED','Separate from free fixed-K identities','Paper budget router','no'],
 ['Raw frame/ROI access','FORMAL_CANDIDATE','PARTIAL_READER','SPATIAL_EVIDENCE_PENDING','NOT_STARTED','ROI/front-end cost not implemented','Raw / formal design','no'],
 ['Final validated WTR','FINAL_MODEL','NONE','NO_FINAL_EVIDENCE','NOT_PROVEN','No complete validated joint model','Publication status','no'],
];
const last=rows.length, range='A1:H'+last;
sheet.getRange(range).values=rows;
sheet.showGridLines=false;
sheet.getRange(range).format.font={name:'Arial',size:10};
sheet.getRange('A1:H1').format.fill='#334155';
sheet.getRange('A1:H1').format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
sheet.getRange('A1:A'+last).format.columnWidth=25;
sheet.getRange('B1:B'+last).format.columnWidth=29;
sheet.getRange('C1:C'+last).format.columnWidth=24;
sheet.getRange('D1:D'+last).format.columnWidth=42;
sheet.getRange('E1:E'+last).format.columnWidth=34;
sheet.getRange('F1:F'+last).format.columnWidth=43;
sheet.getRange('G1:G'+last).format.columnWidth=25;
sheet.getRange('H1:H'+last).format.columnWidth=8;
sheet.getRange(range).format.rowHeight=24;
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:"'Final model ledger'!"+range,include:'values',tableMaxRows:last,tableMaxCols:8,maxChars:7000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
const values=sheet.getRange(range).values;
if(values.length!==last || values.slice(1).some(r=>r[7]!=='no'||r[4]==='TASK_PASS')) throw new Error('Unmeasured mechanism marked accepted');
const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
await fs.writeFile(new URL('FINAL_MODEL_LEDGER.csv',out), values.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n','utf8');
const png=await wb.render({sheetName:sheet.name,range,scale:1,format:'png'});
await fs.writeFile(new URL('ledger_preview.png',out),new Uint8Array(await png.arrayBuffer()));
