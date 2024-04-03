#coding:utf-8
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import traceback
import sys
from rocketchat.api import RocketChatAPI

try:
    import maya.cmds as cmds
    isMaya = True
except:
    isMaya = False

def sendMsg(message, showName="", shotName="", artist=""):
    # print showName, shotName, artist
    if showName != "" and shotName != "" and artist != "":
        sendMail(message, showName, shotName, artist)
        sendRocketChatFromBad(message, showName, shotName, artist)
    try:
        raise AssertionError(message)
    except:
        if isMaya:
            isBatch = cmds.about(batch=True)
            if not isBatch:
                cmds.confirmDialog(m=message, icon="warning", button="OK", title="error")
                assert False, message
        traceback.print_exc()
        sys.exit(-9)

def sendMail(mailMsg, showName = "", shotName = "", artist = ""):
    receiverEmail = "%s@tactic.com"

    sender = receiverEmail % artist
    receiver = receiverEmail % artist

    msg = MIMEMultipart()
    msg["Subject"] = "[UsdCacheExport] %s:%s %s Error!" % (showName, shotName, datetime.date.today().isoformat())
    msg["From"] = sender
    msg['To'] = receiver

    msg.attach(MIMEText(unicode(mailMsg).encode('utf-8'), _charset='utf-8'))

    s = smtplib.SMTP('10.0.0.221')

    needReceiver = ['taehoon.kim@tactic.com', 'taeseob.kim@tactic.com', 'sungoh.moon@tactic.com',       # Part Sup
                    'sanghun.kim@tactic.com', 'wonchul.kang@tactic.com', 'seockhee.joung@tactic.com',   # CG Sup
                    "huisu.jeong@tactic.com", "donghyuk.kang@tactic.com", "junghoon.park@tactic.com",   # ANI LEADER
                    'daeseok.chae@tactic.com', receiver]

    s.sendmail(sender, needReceiver, msg.as_string())
    s.close()

def sendRocketChatFromBad(mailMsg, showName = "", shotName = "", artist = ""):
    # username : @daeseok.chae
    HOST = 'http://10.10.10.232:61015'
    ADMIN_USERNAME = "BadBot"
    ADMIN_PASSWORD = "dexterbot123"

    try:
        api = RocketChatAPI(settings={'username': ADMIN_USERNAME, 'password': ADMIN_PASSWORD, 'domain': HOST})
        api.send_message(mailMsg, '@%s' % artist)
        api.send_message(mailMsg, 'CacheErrorRoom')
    except Exception as e:
        print e.message

def sendRocketChatFromGood(mailMsg, artist=""):
    # username : @daeseok.chae
    HOST = 'http://10.10.10.232:61015'
    ADMIN_USERNAME = "GoodBot"
    ADMIN_PASSWORD = "dexterbot123"

    try:
        api = RocketChatAPI(settings={'username': ADMIN_USERNAME, 'password': ADMIN_PASSWORD, 'domain': HOST})
        print mailMsg, '@%s' % artist
        api.send_message(mailMsg, '@%s' % artist)
        # api.send_message(mailMsg, 'CacheOkRoom')
    except Exception as e:
        print e.message

if __name__ == '__main__':
    showName = sys.argv[1]
    titleInfo = '%s %s' % (sys.argv[2], sys.argv[3])
    artist = sys.argv[4]
    print sys.argv

    sendRocketChatFromGood(mailMsg='%s : %s 작업 끝' % (showName, titleInfo), artist=artist)
