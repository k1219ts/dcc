import xlwt # Excel Write
import xlrd2 # Excel Read

def writeXLS(filename):
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('EDL')

    # Excel Cell StyleSheet
    st = xlwt.easyxf('pattern: pattern solid;')
    st.pattern.pattern_fore_colour = 1
    st.pattern.pattern_back_colour = 0

    for row in range(5):
        for col in range(0, 5, 2):
            sheet.write(row, col, '{ROW}_{COL}'.format(ROW=row, COL=col * 2), st)

    book.save(filename)
    print '# Write XLS :', filename

def readXLS(filename):
    book = xlrd2.open_workbook(filename)
    print '# Read XLS :', filename
    sheet = book.sheet_by_name('EDL')

    for row in range(sheet.nrows):
        data = sheet.row_values(row)
        print 'row : {ROW} - data : {DATA}'.format(ROW=row, DATA=data)

if __name__ == '__main__':
    readXLS('test.xls')
    writeXLS('test.xls')
