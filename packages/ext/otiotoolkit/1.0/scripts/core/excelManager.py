#coding:utf-8
import xlwt
import xlrd2

from Define import Column2

class ExcelMng():
    def __init__(self):
        self.excelList = {'scan_list':[{}], 'omit_list': [{}]} # ROW { "ColumnName", "Data" }
        for column in Column2:
            self.excelList['scan_list'][0][column.name] = column.name.lower()
            self.excelList['omit_list'][0][column.name] = column.name.lower()

    def getRow(self, rowIndex, key, sheet='scan_list'):
        if len(self.excelList[sheet]) > rowIndex:
            if self.excelList[sheet][rowIndex].has_key(key):
                return self.excelList[sheet][rowIndex][key]
            else:
                return ""
        print "# WARNING : ExcelMng() - getRow() - index out of range [%s][%d <- %d]" % (key, self.count(sheet), rowIndex)
        return ""

    def setRow(self, rowIndex, key, value, sheet='scan_list'):
        if len(self.excelList[sheet]) <= rowIndex:
            for gap in range(len(self.excelList[sheet]), rowIndex + 1):
                self.excelList[sheet].append({})
        self.excelList[sheet][rowIndex][key] = value

    def mergeRow(self, rowIndex, key, value, sheet='scan_list'):
        if len(self.excelList[sheet]) > rowIndex:
            self.excelList[sheet][rowIndex][key] += value
        else:
            print "# ERROR : ExcelMng() - mergeRow() - index out of range [%d <- %d]" % (len(self.excelList[sheet]), rowIndex)

    # def __len__(self):
    #     return len(self.excelList)

    def count(self, sheet='scan_list'):
        if self.excelList.has_key(sheet):
            return len(self.excelList[sheet])
        else:
            return 0

    def load(self, filename):
        self.filename = filename
        plateListExcel = xlrd2.open_workbook(self.filename)
        self.excelList = {}
        for sheetName in plateListExcel.sheet_names():
            sheet = plateListExcel.sheet_by_name(sheetName)
            self.excelList[sheetName] = []
            for row in range(sheet.nrows):
                data = sheet.row_values(row)
                tmp = {}
                for col, columnData in enumerate(data):
                    try:
                        tmp[Column2(col).name] = columnData
                    except Exception as e:
                        print e.message
                self.excelList[sheetName].append(tmp)

            print sheetName, self.excelList[sheetName]

    def save(self, filename):
        excelData = xlwt.Workbook(encoding='utf-8')

        for sheetName in self.excelList.keys():
            excelSheet = excelData.add_sheet(sheetName)
            for index, celData in enumerate(self.excelList[sheetName]):
                if celData:
                    for columnName in celData:
                        excelSheet.write(index, eval('Column2.%s.value' % columnName), celData[columnName])

                    issueRow = self.getRow(index, Column2.ISSUE.name, sheet=sheetName).strip().count('\n')
                    editIssueRow = self.getRow(index, Column2.EDIT_ISSUE.name, sheet=sheetName).strip().count('\n')

                    rowCount = max(issueRow, editIssueRow)
                    if rowCount != 0:
                        excelSheet.row(index).height_mismatch = True
                        excelSheet.row(index).height = 256 + (256 * rowCount)

            print sheetName, self.excelList[sheetName]

        excelData.save(filename)