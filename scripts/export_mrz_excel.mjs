import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const inputRoot = path.resolve(process.argv[2]);
const outputPath = path.resolve(process.argv[3]);
const artifactModule = process.argv[4];

if (!artifactModule) throw new Error("缺少 artifact-tool 模块路径");
const { SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactModule).href);

const MRZ_TO_ALPHA2 = {
  ARE: "AE", AFG: "AF", ALB: "AL", AND: "AD", AGO: "AO", ARG: "AR", ARM: "AM",
  AUS: "AU", AUT: "AT", AZE: "AZ", BDI: "BI", BEL: "BE", BGD: "BD", BGR: "BG",
  BHR: "BH", BHS: "BS", BLR: "BY", BLZ: "BZ", BOL: "BO", BRA: "BR", BRN: "BN",
  BTN: "BT", BWA: "BW", CAF: "CF", CAN: "CA", CHE: "CH", CHL: "CL", CHN: "CN",
  CIV: "CI", CMR: "CM", COD: "CD", COG: "CG", COL: "CO", CRI: "CR", CUB: "CU",
  CYP: "CY", CZE: "CZ", DEU: "DE", DJI: "DJ", DNK: "DK", DOM: "DO", DZA: "DZ",
  ECU: "EC", EGY: "EG", ERI: "ER", ESP: "ES", EST: "EE", ETH: "ET", FIN: "FI",
  FJI: "FJ", FRA: "FR", GBR: "GB", GEO: "GE", GHA: "GH", GRC: "GR", GTM: "GT",
  GUY: "GY", HND: "HN", HRV: "HR", HTI: "HT", HUN: "HU", IDN: "ID", IND: "IN",
  IRL: "IE", IRN: "IR", IRQ: "IQ", ISL: "IS", ISR: "IL", ITA: "IT", JAM: "JM",
  JOR: "JO", JPN: "JP", KAZ: "KZ", KEN: "KE", KGZ: "KG", KHM: "KH", KOR: "KR",
  KWT: "KW", LAO: "LA", LBN: "LB", LBY: "LY", LKA: "LK", LTU: "LT", LUX: "LU",
  LVA: "LV", MAR: "MA", MDA: "MD", MDG: "MG", MEX: "MX", MKD: "MK", MLI: "ML",
  MLT: "MT", MMR: "MM", MNG: "MN", MOZ: "MZ", MRT: "MR", MUS: "MU", MWI: "MW",
  MYS: "MY", NAM: "NA", NER: "NE", NGA: "NG", NIC: "NI", NLD: "NL", NOR: "NO",
  NPL: "NP", NZL: "NZ", OMN: "OM", PAK: "PK", PAN: "PA", PER: "PE", PHL: "PH",
  PNG: "PG", POL: "PL", PRK: "KP", PRT: "PT", PRY: "PY", QAT: "QA", ROU: "RO",
  RUS: "RU", RWA: "RW", SAU: "SA", SDN: "SD", SEN: "SN", SGP: "SG", SLB: "SB",
  SLE: "SL", SLV: "SV", SMR: "SM", SOM: "SO", SRB: "RS", SSD: "SS", SUR: "SR",
  SVK: "SK", SVN: "SI", SWE: "SE", SWZ: "SZ", SYR: "SY", TCD: "TD", TGO: "TG",
  THA: "TH", TJK: "TJ", TKM: "TM", TLS: "TL", TTO: "TT", TUN: "TN", TUR: "TR",
  TZA: "TZ", UGA: "UG", UKR: "UA", URY: "UY", USA: "US", UZB: "UZ", VAT: "VA",
  VEN: "VE", VNM: "VN", VUT: "VU", WSM: "WS", YEM: "YE", ZAF: "ZA", ZMB: "ZM",
  ZWE: "ZW",
};

const NATIONALITY_NAMES = {
  AE: "阿联酋", AF: "阿富汗", AL: "阿尔巴尼亚", AD: "安道尔", AO: "安哥拉", AR: "阿根廷",
  AM: "亚美尼亚", AU: "澳大利亚", AT: "奥地利", AZ: "阿塞拜疆", BD: "孟加拉国", BE: "比利时",
  BG: "保加利亚", BH: "巴林", BY: "白俄罗斯", BO: "玻利维亚", BR: "巴西", BN: "文莱",
  CA: "加拿大", CH: "瑞士", CL: "智利", CN: "中国", CI: "科特迪瓦", CM: "喀麦隆",
  CO: "哥伦比亚", CR: "哥斯达黎加", HR: "克罗地亚", CU: "古巴", CY: "塞浦路斯", CZ: "捷克",
  DE: "德国", DK: "丹麦", DO: "多米尼加", DZ: "阿尔及利亚", EC: "厄瓜多尔", EG: "埃及",
  EE: "爱沙尼亚", ET: "埃塞俄比亚", FI: "芬兰", FR: "法国", GB: "英国", GE: "格鲁吉亚",
  GR: "希腊", GT: "危地马拉", HN: "洪都拉斯", HU: "匈牙利", ID: "印度尼西亚", IN: "印度",
  IE: "爱尔兰", IR: "伊朗", IQ: "伊拉克", IS: "冰岛", IL: "以色列", IT: "意大利",
  JP: "日本", JO: "约旦", KZ: "哈萨克斯坦", KE: "肯尼亚", KH: "柬埔寨", KR: "韩国",
  KW: "科威特", LA: "老挝", LV: "拉脱维亚", LB: "黎巴嫩", LK: "斯里兰卡", LT: "立陶宛",
  LU: "卢森堡", MY: "马来西亚", MX: "墨西哥", MN: "蒙古", MM: "缅甸", NP: "尼泊尔",
  NL: "荷兰", NZ: "新西兰", NG: "尼日利亚", NO: "挪威", OM: "阿曼", PK: "巴基斯坦",
  PA: "巴拿马", PE: "秘鲁", PH: "菲律宾", PL: "波兰", PT: "葡萄牙", QA: "卡塔尔",
  RO: "罗马尼亚", RU: "俄罗斯", SA: "沙特阿拉伯", SG: "新加坡", SK: "斯洛伐克", SI: "斯洛文尼亚",
  ZA: "南非", ES: "西班牙", SE: "瑞典", TH: "泰国", TR: "土耳其",
  UA: "乌克兰", US: "美国", UY: "乌拉圭", UZ: "乌兹别克斯坦", VE: "委内瑞拉", VN: "越南",
};

const sourceFiles = await findFiles(path.join(inputRoot, "results"), "mrz.json");
const sourceResults = [];
for (const file of sourceFiles) sourceResults.push(JSON.parse(await fs.readFile(file, "utf8")));
sourceResults.sort((a, b) =>
  `${a.document}`.localeCompare(`${b.document}`, "zh-CN") || Number(a.page_number) - Number(b.page_number),
);

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Sheet1");
const records = sourceResults.map((result, index) => toTemplateRow(result, index + 1));
const headers = ["序号", "船员姓名", "性别", "船员国藉", "船员职务", "出生日期", "出生地", "证件类别", "证书号", "备注", "申请登陆", "上船日期", "登船口岸"];
const lastRow = Math.max(2, records.length + 1);
sheet.getRange(`A1:M${records.length + 1}`).values = [headers, ...records];
sheet.showGridLines = false;
sheet.getRange("A1:M1").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
sheet.getRange(`A2:M${lastRow}`).format = { horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
sheet.getRange(`F2:F${lastRow}`).format.numberFormat = "yyyy/m/d";
sheet.getRange("A1:M1").format.rowHeight = 28;
sheet.getRange(`A2:M${lastRow}`).format.rowHeight = 24;
setWidths(sheet, { A: 8, B: 28, C: 10, D: 16, E: 14, F: 14, G: 14, H: 16, I: 18, J: 14, K: 14, L: 14, M: 14 }, lastRow);
if (records.length) {
  const table = sheet.tables.add(`A1:M${lastRow}`, true, "CrewList");
  table.style = "TableStyleMedium2";
}
sheet.freezePanes.freezeRows(1);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: records.length }));

function toTemplateRow(result, sequence) {
  const parsed = result.mrz_parse || {};
  const fields = ["valid", "partial"].includes(parsed.status) ? (parsed.fields || {}) : {};
  const documentCode = String(fields.document_code || "").toUpperCase();
  return [
    sequence,
    [fields.surname, fields.given_names].filter(Boolean).join(" "),
    translateSex(fields.sex_code),
    translateNationality(fields.nationality),
    null,
    parseMrzDate(fields.date_of_birth),
    null,
    documentCode === "P<" ? "14-普通护照" : documentCode === "PM" ? "17-海员证" : null,
    fields.passport_number || null,
    parsed.status === "partial" ? "MRZ校验需复核" : null,
    null,
    null,
    null,
  ];
}

function parseMrzDate(value) {
  const text = String(value || "");
  if (!/^\d{6}$/.test(text)) return null;
  const yy = Number(text.slice(0, 2));
  const month = Number(text.slice(2, 4));
  const day = Number(text.slice(4, 6));
  const year = yy >= 50 ? 1900 + yy : 2000 + yy;
  const date = new Date(year, month - 1, day);
  if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day) return null;
  return (Date.UTC(year, month - 1, day) - Date.UTC(1899, 11, 30)) / 86400000;
}

function templateNationalityCode(value) {
  const code = String(value || "").toUpperCase();
  const normalized = code.replaceAll("0", "O").replaceAll("1", "I");
  return MRZ_TO_ALPHA2[normalized] || normalized;
}

function translateSex(value) {
  return { M: "1-男", F: "0-女" }[String(value || "").toUpperCase()] || null;
}

function translateNationality(value) {
  const alpha2 = templateNationalityCode(value);
  return alpha2 && NATIONALITY_NAMES[alpha2] ? `${alpha2}-${NATIONALITY_NAMES[alpha2]}` : alpha2 || null;
}

function setWidths(sheet, widths, lastRow) {
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}1:${column}${lastRow}`).format.columnWidth = width;
  }
}

async function findFiles(root, fileName) {
  const entries = await fs.readdir(root, { withFileTypes: true });
  const found = [];
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) found.push(...(await findFiles(full, fileName)));
    else if (entry.isFile() && entry.name === fileName) found.push(full);
  }
  return found;
}
