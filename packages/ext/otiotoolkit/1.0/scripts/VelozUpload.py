#coding:utf-8
from tactic_client_lib import TacticServerStub
import os, argparse, sys
import dxConfig
import datetime
TACTIC_IP = dxConfig.getConf("TACTIC_IP")

login = 'cgsup'
passwd = 'dexter'

def uploadVelozMov(args):
    server = TacticServerStub(login='cgsup', password='dexter', server=TACTIC_IP, project=args.showCode)
    category = ''
    task_code = ''
    context = ''
    if args.shotName:
        category = 'shot'
        task_code = args.shotName
        context = '%s' % args.context
    elif args.seqName:
        category = 'sequence'
        task_code = args.seqName
        context = args.context
    else:
        assert False, '# ERROR: Not input seq OR shot'

    search_type = '%s/%s' % (args.showCode, category)
    # context = 'publish/%s' % args.context
    search_key = server.build_search_key(search_type, task_code)
    print search_key
    server.start()

    if not os.path.exists(args.movFile):
        assert False, "# ERROR: Not Found Mov File : %s" % args.movFile

    try:
        description = args.description
        if not args.description:
            description = '%s 편집본' % datetime.datetime.now().strftime('%Y/%m/%d')
        snapShot = server.simple_checkin(search_key, context, args.movFile, description=description, mode="copy")

        if args.sync == "True":
            server.set_project('dexter_studios')

            filters = []
            filters.append(('snapshot_code', snapShot['code']))
            file_item = server.query('sthpw/file', filters, single=True)

            syncData = {
                'snapshot_code': snapShot['code'],
                'snapshot_type': snapShot['snapshot_type'],
                'file_name': file_item['file_name'],
                'file_size': file_item['st_size'],
                'location': 'korea',
                'relative_dir': file_item['relative_dir'],
                'status': 'queue',
                'login': login,
                'project_code': file_item['project_code']
            }
            sync_item = server.insert('dexter_studios/sync_data', syncData)
    except Exception as e:
        print "ERROR:", e.message
        server.abort()
    else:
        server.finish()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # mov file name
    argparser.add_argument('-sc', '--showCode', dest='showCode', type=str, required=True, help='show code query.')
    argparser.add_argument('-shn', '--shotName', dest='shotName', type=str, default='', help='upload to shot Name')
    argparser.add_argument('-sen', '--seqName', dest='seqName', type=str, default='', help='upload to seq Name')
    argparser.add_argument('-m', '--movFile', dest='movFile', type=str, required=True, help='upload file path')
    argparser.add_argument('-c', '--context', dest='context', type=str, default="publish/edit", help="publish type (previz', 'onset', 'edit', 'source', 'reference', 'plate')")
    argparser.add_argument('-d', '--description', dest='description', type=str, default='', help='custom description')
    argparser.add_argument('-sy', '--sync', dest='sync', type=str, default='False', help='Tactic Sync')

    args, unknown = argparser.parse_known_args(sys.argv)

    uploadVelozMov(args)
