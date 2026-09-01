import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, "..", "..");
const rawDir = path.join(projectRoot, "数据", "原始数据");
const cleanedDir = path.join(projectRoot, "数据", "清洗后数据");
const outputDir = path.join(projectRoot, "outputs", "data_cleaning_20260830");
const qaDir = path.join(projectRoot, "logs", "data_cleaning_qa");

const sourcePaths = {
  attachment1: path.join(rawDir, "附件1.xlsx"),
  attachment2: path.join(rawDir, "附件2.xlsx"),
  resultTemplate: path.join(rawDir, "附件3", "result1_1.xlsx"),
};

const cleanText = (value) =>
  value === null || value === undefined ? "" : String(value).trim();

const normalizeSeason = (value) => {
  const text = cleanText(value).replace(/\s+/g, "");
  const mapping = {
    单季: "single",
    第一季: "first",
    第二季: "second",
    "第一季、第二季": "first_and_second",
  };
  return mapping[text] ?? text;
};

const normalizeCategory = (cropType) => {
  if (cropType.includes("粮食")) return "grain";
  if (cropType.includes("蔬菜")) return "vegetable";
  if (cropType.includes("食用菌")) return "fungus";
  return "unknown";
};

const round = (value, digits = 6) => {
  const factor = 10 ** digits;
  return Math.round((Number(value) + Number.EPSILON) * factor) / factor;
};

const parsePriceRange = (value) => {
  const text = cleanText(value).replace(/\s+/g, "");
  const match = text.match(/^([0-9]+(?:\.[0-9]+)?)[-–—~～至]([0-9]+(?:\.[0-9]+)?)$/);
  if (!match) return null;
  const low = Number(match[1]);
  const high = Number(match[2]);
  if (!Number.isFinite(low) || !Number.isFinite(high) || high < low) return null;
  return { low, high, mid: (low + high) / 2 };
};

const csvEscape = (value) => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

const writeCsv = async (filePath, columns, rows) => {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  await fs.writeFile(filePath, `\uFEFF${lines.join("\r\n")}\r\n`, "utf8");
};

const columnLetter = (oneBasedColumn) => {
  let value = oneBasedColumn;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
};

const sha256 = async (filePath) => {
  const data = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(data).digest("hex");
};

const loadWorkbook = async (filePath) => {
  const input = await FileBlob.load(filePath);
  return SpreadsheetFile.importXlsx(input);
};

const getSheetValues = (workbook, sheetName) => {
  const sheet = workbook.worksheets.getItem(sheetName);
  const usedRange = sheet.getUsedRange();
  if (!usedRange) throw new Error(`工作表 ${sheetName} 为空`);
  return usedRange.values;
};

await fs.mkdir(cleanedDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });

const [attachment1, attachment2, resultTemplate] = await Promise.all([
  loadWorkbook(sourcePaths.attachment1),
  loadWorkbook(sourcePaths.attachment2),
  loadWorkbook(sourcePaths.resultTemplate),
]);

const issues = [];
const addIssue = (severity, issueType, table, sourceRow, key, detail) => {
  issues.push({
    severity,
    issue_type: issueType,
    table,
    source_row: sourceRow,
    key,
    detail,
  });
};

// 1. 地块表
const plotValues = getSheetValues(attachment1, "乡村的现有耕地");
const plots = plotValues
  .slice(1)
  .filter((row) => cleanText(row[0]) !== "")
  .map((row, index) => ({
    plot_id: cleanText(row[0]),
    land_type: cleanText(row[1]),
    area_mu: Number(row[2]),
    source_row: index + 2,
  }));
const plotById = new Map(plots.map((row) => [row.plot_id, row]));

for (const plot of plots) {
  if (!Number.isFinite(plot.area_mu) || plot.area_mu <= 0) {
    addIssue("error", "invalid_plot_area", "plots", plot.source_row, plot.plot_id, `面积=${plot.area_mu}`);
  }
}

// 2. 作物表；“种植耕地”一列有纵向合并单元格，因此向下继承规则。
const cropValues = getSheetValues(attachment1, "乡村种植的农作物");
let lastPlantingRule = "";
const crops = [];
for (let index = 1; index < cropValues.length; index += 1) {
  const row = cropValues[index];
  const cropId = Number(row[0]);
  if (cleanText(row[0]) === "" || !Number.isInteger(cropId) || cropId < 1 || cropId > 41) continue;
  if (cleanText(row[3])) lastPlantingRule = cleanText(row[3]);
  const cropType = cleanText(row[2]);
  crops.push({
    crop_id: cropId,
    crop_name: cleanText(row[1]),
    crop_type: cropType,
    crop_category: normalizeCategory(cropType),
    is_legume: cropType.includes("豆类") ? 1 : 0,
    planting_rule_text: lastPlantingRule.replace(/\s+/g, " "),
    source_row: index + 1,
  });
}
const cropById = new Map(crops.map((row) => [row.crop_id, row]));
const cropIdByName = new Map(crops.map((row) => [row.crop_name, row.crop_id]));

// 3. 2023 年种植记录；“种植地块”空白是合并单元格造成的，必须向下填充。
const plantingValues = getSheetValues(attachment2, "2023年的农作物种植情况");
let lastPlotId = "";
let plantingBlankPlotCellsFilled = 0;
const planting2023 = [];
for (let index = 1; index < plantingValues.length; index += 1) {
  const row = plantingValues[index];
  const cropId = Number(row[1]);
  if (!Number.isInteger(cropId)) continue;
  const rawPlotId = cleanText(row[0]);
  if (rawPlotId) lastPlotId = rawPlotId;
  else plantingBlankPlotCellsFilled += 1;
  const plotId = lastPlotId;
  const crop = cropById.get(cropId);
  const plot = plotById.get(plotId);
  const cropName = cleanText(row[2]);
  const cropType = cleanText(row[3]);
  const areaMu = Number(row[4]);
  const season = normalizeSeason(row[5]);
  const sourceRow = index + 1;

  if (!plot) addIssue("error", "plot_join_failed", "planting_2023", sourceRow, plotId, "找不到对应地块");
  if (!crop) addIssue("error", "crop_join_failed", "planting_2023", sourceRow, String(cropId), "找不到对应作物");
  if (crop && crop.crop_name !== cropName) {
    addIssue("error", "crop_name_mismatch", "planting_2023", sourceRow, String(cropId), `${cropName} != ${crop.crop_name}`);
  }
  if (!Number.isFinite(areaMu) || areaMu <= 0) {
    addIssue("error", "invalid_planting_area", "planting_2023", sourceRow, `${plotId}/${cropId}`, `面积=${areaMu}`);
  }
  if (!["single", "first", "second"].includes(season)) {
    addIssue("error", "invalid_season", "planting_2023", sourceRow, `${plotId}/${cropId}`, String(row[5]));
  }

  planting2023.push({
    plot_id: plotId,
    crop_id: cropId,
    crop_name: cropName,
    crop_type: cropType,
    area_mu: areaMu,
    season,
    source_row: sourceRow,
  });
}

// 4. 2023 年参数表：只保留序号和作物编号均为数值的记录。
const parameterValues = getSheetValues(attachment2, "2023年统计的相关数据");
const observedParameters = [];
for (let index = 1; index < parameterValues.length; index += 1) {
  const row = parameterValues[index];
  const sequence = Number(row[0]);
  const cropId = Number(row[1]);
  if (
    cleanText(row[0]) === "" ||
    cleanText(row[1]) === "" ||
    !Number.isInteger(sequence) ||
    !Number.isInteger(cropId) ||
    sequence < 1 ||
    cropId < 1 ||
    cropId > 41
  ) continue;
  const parsedPrice = parsePriceRange(row[7]);
  const sourceRow = index + 1;
  if (!parsedPrice) {
    addIssue("error", "price_parse_failed", "crop_parameters_2023", sourceRow, `${cropId}`, String(row[7]));
    continue;
  }
  const cropName = cleanText(row[2]);
  const crop = cropById.get(cropId);
  if (!crop) addIssue("error", "crop_join_failed", "crop_parameters_2023", sourceRow, String(cropId), "找不到对应作物");
  if (crop && crop.crop_name !== cropName) {
    addIssue("error", "crop_name_mismatch", "crop_parameters_2023", sourceRow, String(cropId), `${cropName} != ${crop.crop_name}`);
  }
  observedParameters.push({
    crop_id: cropId,
    crop_name: cropName,
    land_type: cleanText(row[3]),
    season: normalizeSeason(row[4]),
    yield_jin_per_mu: Number(row[5]),
    cost_yuan_per_mu: Number(row[6]),
    price_low: parsedPrice.low,
    price_high: parsedPrice.high,
    price_mid: parsedPrice.mid,
    parameter_source: "observed_2023",
    source_row: sourceRow,
  });
}

const parameterKey = (row) => `${row.crop_id}|${row.land_type}|${row.season}`;
const observedParameterCounts = new Map();
for (const row of observedParameters) {
  const key = parameterKey(row);
  observedParameterCounts.set(key, (observedParameterCounts.get(key) ?? 0) + 1);
}
const observedDuplicateParameterKeys = [...observedParameterCounts.values()].filter((count) => count > 1).length;
for (const [key, count] of observedParameterCounts) {
  if (count > 1) addIssue("error", "duplicate_parameter_key", "crop_parameters_2023", "", key, `重复${count}次`);
}

// 题目脚注明确说明：智慧大棚第一季参数与普通大棚第一季相同。
const fallbackParameters = observedParameters
  .filter((row) => row.land_type === "普通大棚" && row.season === "first" && row.crop_id >= 17 && row.crop_id <= 34)
  .map((row) => ({
    ...row,
    land_type: "智慧大棚",
    parameter_source: "fallback_ordinary_greenhouse_first_season",
    source_row: row.source_row,
  }));
const cropParameters2023 = [...observedParameters, ...fallbackParameters].sort(
  (a, b) => a.crop_id - b.crop_id || a.land_type.localeCompare(b.land_type, "zh-CN") || a.season.localeCompare(b.season),
);
const parameterByKey = new Map(cropParameters2023.map((row) => [parameterKey(row), row]));

// 5. 用 2023 年实际产量近似基准销量。
const demandGroups = new Map();
let smartGreenhouseFirstSeasonPlantingFallbackCount = 0;
for (const planting of planting2023) {
  const plot = plotById.get(planting.plot_id);
  if (!plot) continue;
  const key = `${planting.crop_id}|${plot.land_type}|${planting.season}`;
  const parameter = parameterByKey.get(key);
  if (!parameter) {
    addIssue("error", "parameter_join_failed", "demand_2023", planting.source_row, key, "无法计算该种植记录的产量");
    continue;
  }
  if (parameter.parameter_source === "fallback_ordinary_greenhouse_first_season") {
    smartGreenhouseFirstSeasonPlantingFallbackCount += 1;
    addIssue(
      "warning",
      "fallback_parameter_used",
      "demand_2023",
      planting.source_row,
      key,
      "按题目脚注，智慧大棚第一季沿用普通大棚第一季参数",
    );
  }
  const demandKey = `${planting.crop_id}|${planting.season}`;
  const current = demandGroups.get(demandKey) ?? {
    crop_id: planting.crop_id,
    crop_name: cropById.get(planting.crop_id)?.crop_name ?? planting.crop_name,
    season: planting.season,
    expected_sales_jin: 0,
    source_record_count: 0,
  };
  current.expected_sales_jin += planting.area_mu * parameter.yield_jin_per_mu;
  current.source_record_count += 1;
  demandGroups.set(demandKey, current);
}
const demand2023 = [...demandGroups.values()]
  .map((row) => ({ ...row, expected_sales_jin: round(row.expected_sales_jin, 3) }))
  .sort((a, b) => a.crop_id - b.crop_id || a.season.localeCompare(b.season));

// 6. 完整适宜性矩阵：为每个真实可用的地块-季次槽位列出 41 种作物是否允许。
const slotsForLandType = (landType) => {
  if (["平旱地", "梯田", "山坡地"].includes(landType)) return ["single"];
  if (landType === "水浇地") return ["single", "first", "second"];
  if (["普通大棚", "智慧大棚"].includes(landType)) return ["first", "second"];
  return [];
};

const eligibilityRule = (landType, season, cropId) => {
  if (["平旱地", "梯田", "山坡地"].includes(landType) && season === "single") {
    return { eligible: cropId >= 1 && cropId <= 15, source: "附件1规则：露天非水浇地单季粮食（不含水稻）" };
  }
  if (landType === "水浇地" && season === "single") {
    return { eligible: cropId === 16, source: "附件1规则：水浇地单季水稻" };
  }
  if (landType === "水浇地" && season === "first") {
    return { eligible: cropId >= 17 && cropId <= 34, source: "附件1规则：水浇地第一季蔬菜" };
  }
  if (landType === "水浇地" && season === "second") {
    return { eligible: cropId >= 35 && cropId <= 37, source: "附件1规则：水浇地第二季三种根菜" };
  }
  if (landType === "普通大棚" && season === "first") {
    return { eligible: cropId >= 17 && cropId <= 34, source: "附件1规则：普通大棚第一季蔬菜" };
  }
  if (landType === "普通大棚" && season === "second") {
    return { eligible: cropId >= 38 && cropId <= 41, source: "附件1规则：普通大棚第二季食用菌" };
  }
  if (landType === "智慧大棚" && ["first", "second"].includes(season)) {
    return { eligible: cropId >= 17 && cropId <= 34, source: "附件1规则：智慧大棚两季蔬菜" };
  }
  return { eligible: false, source: "附件1规则：不适宜组合" };
};

const eligibility = [];
for (const plot of plots) {
  for (const season of slotsForLandType(plot.land_type)) {
    for (const crop of crops) {
      const rule = eligibilityRule(plot.land_type, season, crop.crop_id);
      eligibility.push({
        plot_id: plot.plot_id,
        land_type: plot.land_type,
        season,
        crop_id: crop.crop_id,
        crop_name: crop.crop_name,
        eligible: rule.eligible ? 1 : 0,
        rule_source: rule.source,
      });
    }
  }
}

// 7. 官方模板映射：把每个“年份-地块-季次-作物”定位到具体单元格。
const templateMapping = [];
const templateYears = resultTemplate.worksheets.items.map((sheet) => sheet.name).filter((name) => /^20\d{2}$/.test(name));
let templateCropColumnCount = 0;
let templateFirstSeasonPlotRows = 0;
let templateSecondSeasonPlotRows = 0;

for (const year of templateYears) {
  const sheet = resultTemplate.worksheets.getItem(year);
  const values = sheet.getUsedRange().values;
  const header = values[0];
  const cropColumns = [];
  for (let columnIndex = 2; columnIndex < header.length; columnIndex += 1) {
    const cropName = cleanText(header[columnIndex]);
    if (!cropName || !cropIdByName.has(cropName)) continue;
    cropColumns.push({ columnIndex, cropId: cropIdByName.get(cropName), cropName });
  }
  if (year === templateYears[0]) templateCropColumnCount = cropColumns.length;

  let currentTemplateSeason = "";
  let firstCount = 0;
  let secondCount = 0;
  for (let rowIndex = 1; rowIndex < values.length; rowIndex += 1) {
    const seasonLabel = cleanText(values[rowIndex][0]).replace(/\s+/g, "");
    if (seasonLabel.includes("第一季")) currentTemplateSeason = "first";
    if (seasonLabel.includes("第二季")) currentTemplateSeason = "second";
    const plotId = cleanText(values[rowIndex][1]);
    const plot = plotById.get(plotId);
    if (!plot || !["first", "second"].includes(currentTemplateSeason)) continue;
    if (currentTemplateSeason === "first") firstCount += 1;
    else secondCount += 1;

    for (const cropColumn of cropColumns) {
      let modelSeason = currentTemplateSeason;
      if (currentTemplateSeason === "first" && ["平旱地", "梯田", "山坡地"].includes(plot.land_type)) {
        modelSeason = "single";
      } else if (currentTemplateSeason === "first" && plot.land_type === "水浇地" && cropColumn.cropId === 16) {
        modelSeason = "single";
      }
      const oneBasedRow = rowIndex + 1;
      const oneBasedColumn = cropColumn.columnIndex + 1;
      const colLetter = columnLetter(oneBasedColumn);
      templateMapping.push({
        year: Number(year),
        season: modelSeason,
        template_season: currentTemplateSeason,
        plot_id: plotId,
        crop_id: cropColumn.cropId,
        crop_name: cropColumn.cropName,
        sheet: year,
        row: oneBasedRow,
        column: oneBasedColumn,
        column_letter: colLetter,
        cell: `${colLetter}${oneBasedRow}`,
      });
    }
  }
  if (year === templateYears[0]) {
    templateFirstSeasonPlotRows = firstCount;
    templateSecondSeasonPlotRows = secondCount;
  }
}

const sourceHashes = Object.fromEntries(
  await Promise.all(Object.entries(sourcePaths).map(async ([name, filePath]) => [name, await sha256(filePath)])),
);

const sumArea = (landTypes) => round(plots.filter((row) => landTypes.includes(row.land_type)).reduce((sum, row) => sum + row.area_mu, 0), 3);
const countLandType = (landType) => plots.filter((row) => row.land_type === landType).length;
const fatalIssueCount = issues.filter((row) => row.severity === "error").length;
const metrics = [
  { id: "plots_total", label: "地块总数", expected: 54, actual: plots.length, explanation: "每块土地只保留一行" },
  { id: "outdoor_plots", label: "露天耕地数量（含水浇地）", expected: 34, actual: plots.filter((row) => ["平旱地", "梯田", "山坡地", "水浇地"].includes(row.land_type)).length, explanation: "A-D 类地块" },
  { id: "outdoor_area_mu", label: "露天耕地面积/亩", expected: 1201, actual: sumArea(["平旱地", "梯田", "山坡地", "水浇地"]), explanation: "A-D 类面积合计" },
  { id: "ordinary_greenhouse_count", label: "普通大棚数量", expected: 16, actual: countLandType("普通大棚"), explanation: "E1-E16" },
  { id: "smart_greenhouse_count", label: "智慧大棚数量", expected: 4, actual: countLandType("智慧大棚"), explanation: "F1-F4" },
  { id: "total_area_mu", label: "全部地块面积/亩", expected: 1213, actual: sumArea(["平旱地", "梯田", "山坡地", "水浇地", "普通大棚", "智慧大棚"]), explanation: "含两类大棚" },
  { id: "crops_total", label: "作物总数", expected: 41, actual: crops.length, explanation: "编号1-41" },
  { id: "legume_crops", label: "豆类作物数", expected: 8, actual: crops.filter((row) => row.is_legume === 1).length, explanation: "粮食豆类5种、蔬菜豆类3种" },
  { id: "planting_records_2023", label: "2023有效种植记录", expected: 87, actual: planting2023.length, explanation: "合并单元格向下填充后保留" },
  { id: "planting_plots_covered", label: "2023覆盖地块数", expected: 54, actual: new Set(planting2023.map((row) => row.plot_id)).size, explanation: "54块地均有记录" },
  { id: "observed_parameter_records", label: "原始有效参数记录", expected: 107, actual: observedParameters.length, explanation: "排除底部说明行" },
  { id: "observed_parameter_duplicate_keys", label: "原始参数重复主键数", expected: 0, actual: observedDuplicateParameterKeys, explanation: "主键为作物-地类-季次" },
  { id: "smart_first_planting_fallback_records", label: "2023智慧大棚第一季回填记录", expected: 6, actual: smartGreenhouseFirstSeasonPlantingFallbackCount, explanation: "按题目脚注沿用普通大棚参数" },
  { id: "template_year_sheets", label: "模板年份工作表数", expected: 7, actual: templateYears.length, explanation: "2024-2030" },
  { id: "template_crop_columns", label: "模板作物列数", expected: 41, actual: templateCropColumnCount, explanation: "黄豆至羊肚菌" },
  { id: "template_first_season_rows", label: "模板第一季地块行数", expected: 54, actual: templateFirstSeasonPlotRows, explanation: "单季作物也写入第一季区" },
  { id: "template_second_season_rows", label: "模板第二季地块行数", expected: 28, actual: templateSecondSeasonPlotRows, explanation: "D、E、F类地块" },
  { id: "fatal_data_issues", label: "致命数据问题数", expected: 0, actual: fatalIssueCount, explanation: "连接失败、价格解析失败等" },
];

const auditPassed = metrics.every((metric) => metric.expected === metric.actual);
const audit = {
  generated_at: new Date().toISOString(),
  project: "2024年C题：农作物的种植策略",
  seed: 2024,
  audit_passed: auditPassed,
  metrics: Object.fromEntries(metrics.map((metric) => [metric.id, { expected: metric.expected, actual: metric.actual, passed: metric.expected === metric.actual, explanation: metric.explanation }])),
  derived_counts: {
    planting_blank_plot_cells_filled: plantingBlankPlotCellsFilled,
    generated_smart_greenhouse_first_season_parameters: fallbackParameters.length,
    total_parameter_records_after_fallback: cropParameters2023.length,
    demand_rows: demand2023.length,
    eligibility_rows: eligibility.length,
    eligible_combinations: eligibility.filter((row) => row.eligible === 1).length,
    template_mapping_rows: templateMapping.length,
    warning_count: issues.filter((row) => row.severity === "warning").length,
  },
  source_sha256: sourceHashes,
};

const csvOutputs = [
  ["plots.csv", ["plot_id", "land_type", "area_mu", "source_row"], plots],
  ["crops.csv", ["crop_id", "crop_name", "crop_type", "crop_category", "is_legume", "planting_rule_text", "source_row"], crops],
  ["planting_2023.csv", ["plot_id", "crop_id", "crop_name", "crop_type", "area_mu", "season", "source_row"], planting2023],
  ["crop_parameters_2023.csv", ["crop_id", "crop_name", "land_type", "season", "yield_jin_per_mu", "cost_yuan_per_mu", "price_low", "price_high", "price_mid", "parameter_source", "source_row"], cropParameters2023],
  ["eligibility.csv", ["plot_id", "land_type", "season", "crop_id", "crop_name", "eligible", "rule_source"], eligibility],
  ["demand_2023.csv", ["crop_id", "crop_name", "season", "expected_sales_jin", "source_record_count"], demand2023],
  ["template_mapping.csv", ["year", "season", "template_season", "plot_id", "crop_id", "crop_name", "sheet", "row", "column", "column_letter", "cell"], templateMapping],
  ["data_issues.csv", ["severity", "issue_type", "table", "source_row", "key", "detail"], issues],
];

await Promise.all(csvOutputs.map(([fileName, columns, rows]) => writeCsv(path.join(cleanedDir, fileName), columns, rows)));
await fs.writeFile(path.join(cleanedDir, "data_audit.json"), `${JSON.stringify(audit, null, 2)}\n`, "utf8");

// 8. 生成便于人工查看的审计工作簿。CSV/JSON 是程序使用，xlsx 是给队员快速核对。
const auditWorkbook = Workbook.create();
const summarySheet = auditWorkbook.worksheets.add("审计汇总");
const issueSheet = auditWorkbook.worksheets.add("数据异常");
const guideSheet = auditWorkbook.worksheets.add("输出说明");

summarySheet.showGridLines = false;
summarySheet.getRange("A1:F1").values = [["指标编号", "检查项", "期望值", "实际值", "状态", "说明"]];
summarySheet.getRange(`A2:D${metrics.length + 1}`).values = metrics.map((metric) => [metric.id, metric.label, metric.expected, metric.actual]);
summarySheet.getRange(`F2:F${metrics.length + 1}`).values = metrics.map((metric) => [metric.explanation]);
summarySheet.getRange("E2").formulas = [["=IF(C2=D2,\"通过\",\"检查\")"]];
summarySheet.getRange(`E2:E${metrics.length + 1}`).fillDown();
summarySheet.getRange(`A1:F${metrics.length + 1}`).format = {
  font: { name: "Microsoft YaHei", size: 10 },
  verticalAlignment: "center",
};
summarySheet.getRange("A1:F1").format = {
  fill: "#1F4E78",
  font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summarySheet.getRange(`C2:D${metrics.length + 1}`).format.numberFormat = "#,##0.###";
summarySheet.getRange(`E2:E${metrics.length + 1}`).conditionalFormats.add("containsText", {
  text: "通过",
  format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } },
});
summarySheet.getRange(`E2:E${metrics.length + 1}`).conditionalFormats.add("containsText", {
  text: "检查",
  format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } },
});
summarySheet.freezePanes.freezeRows(1);
summarySheet.getRange(`A1:A${metrics.length + 1}`).format.columnWidth = 34;
summarySheet.getRange(`B1:B${metrics.length + 1}`).format.columnWidth = 26;
summarySheet.getRange(`C1:E${metrics.length + 1}`).format.columnWidth = 14;
summarySheet.getRange(`F1:F${metrics.length + 1}`).format.columnWidth = 42;
summarySheet.getRange(`A1:F${metrics.length + 1}`).format.autofitRows();

issueSheet.showGridLines = false;
issueSheet.getRange("A1:F1").values = [["级别", "问题类型", "数据表", "源行号", "键", "说明"]];
const issueRows = issues.length > 0
  ? issues.map((row) => [row.severity, row.issue_type, row.table, row.source_row, row.key, row.detail])
  : [["info", "none", "-", "-", "-", "未发现问题"]];
issueSheet.getRange(`A2:F${issueRows.length + 1}`).values = issueRows;
issueSheet.getRange(`A1:F${issueRows.length + 1}`).format = { font: { name: "Microsoft YaHei", size: 10 }, verticalAlignment: "center" };
issueSheet.getRange("A1:F1").format = { fill: "#1F4E78", font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
issueSheet.freezePanes.freezeRows(1);
issueSheet.getRange(`A1:A${issueRows.length + 1}`).format.columnWidth = 12;
issueSheet.getRange(`B1:B${issueRows.length + 1}`).format.columnWidth = 30;
issueSheet.getRange(`C1:C${issueRows.length + 1}`).format.columnWidth = 22;
issueSheet.getRange(`D1:D${issueRows.length + 1}`).format.columnWidth = 12;
issueSheet.getRange(`E1:E${issueRows.length + 1}`).format.columnWidth = 34;
issueSheet.getRange(`F1:F${issueRows.length + 1}`).format.columnWidth = 54;
issueSheet.getRange(`A1:F${issueRows.length + 1}`).format.autofitRows();

guideSheet.showGridLines = false;
const outputDescriptions = [
  ["plots.csv", "地块清单", plots.length],
  ["crops.csv", "作物清单及豆类标记", crops.length],
  ["planting_2023.csv", "清洗后的2023年种植记录", planting2023.length],
  ["crop_parameters_2023.csv", "产量、成本、价格及参数来源", cropParameters2023.length],
  ["eligibility.csv", "地块-季次-作物适宜性矩阵", eligibility.length],
  ["demand_2023.csv", "由2023年产量推算的预期销量", demand2023.length],
  ["template_mapping.csv", "模型结果到官方模板单元格的映射", templateMapping.length],
  ["data_issues.csv", "异常和参数回填记录", issues.length],
  ["data_audit.json", "供程序读取的审计结果", metrics.length],
];
guideSheet.getRange("A1:C1").values = [["输出文件", "用途", "记录数/指标数"]];
guideSheet.getRange(`A2:C${outputDescriptions.length + 1}`).values = outputDescriptions;
guideSheet.getRange(`A1:C${outputDescriptions.length + 1}`).format = { font: { name: "Microsoft YaHei", size: 10 }, verticalAlignment: "center" };
guideSheet.getRange("A1:C1").format = { fill: "#1F4E78", font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
guideSheet.getRange(`C2:C${outputDescriptions.length + 1}`).format.numberFormat = "#,##0";
guideSheet.freezePanes.freezeRows(1);
guideSheet.getRange(`A1:A${outputDescriptions.length + 1}`).format.columnWidth = 34;
guideSheet.getRange(`B1:B${outputDescriptions.length + 1}`).format.columnWidth = 48;
guideSheet.getRange(`C1:C${outputDescriptions.length + 1}`).format.columnWidth = 18;
guideSheet.getRange(`A1:C${outputDescriptions.length + 1}`).format.autofitRows();

const auditInspect = await auditWorkbook.inspect({
  kind: "table",
  sheetId: "审计汇总",
  range: `A1:F${metrics.length + 1}`,
  include: "values,formulas",
  tableMaxRows: metrics.length + 1,
  tableMaxCols: 6,
  maxChars: 12000,
});
const formulaErrors = await auditWorkbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
  maxChars: 4000,
});

for (const sheetName of ["审计汇总", "数据异常", "输出说明"]) {
  const preview = await auditWorkbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(qaDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const auditXlsx = await SpreadsheetFile.exportXlsx(auditWorkbook);
const auditWorkbookPath = path.join(outputDir, "data_audit.xlsx");
await auditXlsx.save(auditWorkbookPath);

await fs.writeFile(path.join(qaDir, "audit_inspect.ndjson"), `${auditInspect.ndjson}\n`, "utf8");
await fs.writeFile(path.join(qaDir, "formula_error_scan.ndjson"), `${formulaErrors.ndjson}\n`, "utf8");

process.stdout.write(`${JSON.stringify({
  audit_passed: auditPassed,
  cleaned_dir: cleanedDir,
  audit_workbook: auditWorkbookPath,
  metrics: Object.fromEntries(metrics.map((metric) => [metric.id, metric.actual])),
  derived_counts: audit.derived_counts,
}, null, 2)}\n`);

if (!auditPassed) process.exitCode = 2;
