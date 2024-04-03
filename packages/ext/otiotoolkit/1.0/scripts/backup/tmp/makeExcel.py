import xlwt # Excel Write
import os

targetDir = '/prod_nas/__DD_PROD/PRAT2/edit/210127_onset_edit/test2'
book = xlwt.Workbook(encoding='utf-8')
sheet = book.add_sheet('EDL')

cleanup = {}
for fileName in os.listdir(targetDir):
    splitFileName = fileName.split('-')
    index = splitFileName[0]
    frame = splitFileName[1]

    if not cleanup.has_key(index):
        cleanup[index] = {'index':int(index),
                          'frame':frame,
                          'clipName': []}
    cleanup[index]['clipName'] += splitFileName[2:]

for index in cleanup:
    sheet.write(int(index), 0, int(index))
    sheet.write(int(index), 1, cleanup[index]['frame'])
    if len(cleanup[index]['clipName']) > 1 and "empty" in cleanup[index]['clipName']:
        cleanup[index]['clipName'].remove('empty')
    sheet.write(int(index), 2, '\n'.join(cleanup[index]['clipName']))
    sheet.row(int(index)).height_mismatch = True
    sheet.row(int(index)).height = 256 + (256 * (len(cleanup[index]['clipName'])))

book.save(os.path.join(os.path.dirname(targetDir), 'test2.xls'))