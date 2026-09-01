import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const files = process.argv.slice(2);

if (files.length === 0) {
  throw new Error("请至少传入一个 xlsx 文件路径");
}

for (const filePath of files) {
  const input = await FileBlob.load(filePath);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheetSummary = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 5000,
  });
  process.stdout.write(`\n### FILE ${filePath}\n`);
  process.stdout.write(`${sheetSummary.ndjson}\n`);

  for (const sheet of workbook.worksheets.items) {
    const used = sheet.getUsedRange();
    if (!used) {
      process.stdout.write(`SHEET ${sheet.name}: EMPTY\n`);
      continue;
    }
    const requestedRows = Number.parseInt(process.env.INSPECT_ROWS ?? "20", 10);
    const requestedCols = Number.parseInt(process.env.INSPECT_COLS ?? "16", 10);
    const rowCount = Math.min(used.rowCount, requestedRows);
    const colCount = Math.min(used.columnCount, requestedCols);
    const preview = sheet.getRangeByIndexes(0, 0, rowCount, colCount);
    process.stdout.write(
      `${JSON.stringify({
        sheet: sheet.name,
        usedRange: used.address,
        rowCount: used.rowCount,
        columnCount: used.columnCount,
        preview: preview.values,
      })}\n`,
    );
  }
}
