import site
site.addsitedir('/backstage/libs/python_lib')
import nuke
import pika
if nuke.NUKE_VERSION_MAJOR >= 9:
    from pymongo import MongoClient
import json, sys


def test_sender(exchange='', exchange_name='topic_test', body={}):

    try:
        fullPath = body['body']
        if fullPath.split('/')[1].startswith('show'):
            prj = fullPath.split('/')[2]
            body['project'] = prj

            # NORMAL /show/prj/shot/seq/shot/ PATH CASE
            if '_' in fullPath.split('/')[5]:
                if fullPath.split('/')[4] == fullPath.split('/')[5].split('_')[0]:
                    seq = fullPath.split('/')[4]

                else:
                    seq = fullPath.split('/')[5].split('_')[0]
                shot = fullPath.split('/')[5]

                body['sequence'] = seq
                body['shot'] = shot

                routing_key = '%s.%s.%s' % (prj, seq, shot)
            # CASE NO '_' IN SHOT CASE
            else:
                routing_key = '%s.etc.etc' % prj
                body['sequence'] = 'etc'
                body['shot'] = 'etc'

        else:
            routing_key = 'etc.etc.etc'
            body['project'] = 'etc'
            body['sequence'] = 'etc'
            body['shot'] = 'etc'

        print(routing_key)


#        # SEND MESSAGE
#        #------------------------------------------------------------------------------
#        connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.0.11.104'))
#        channel = connection.channel()
#
#        channel.exchange_declare(exchange=exchange_name,
#                                 type='topic')
#
#        channel.basic_publish(exchange=exchange_name,
#                              routing_key=routing_key,
#                              body=json.dumps(body)
#                              )
#        connection.close()
#        #------------------------------------------------------------------------------
#        # SEND DB
#        if nuke.NUKE_VERSION_MAJOR >= 9:
#            client = MongoClient('10.0.11.104', 27017)
#            db = client.test
#            posts = db['topic_test']
#            postid = posts.insert_one(body)
    except:
        print("ERROR EXCEPT")
        print(sys.exc_info())
