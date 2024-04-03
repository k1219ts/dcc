# -*- coding: utf-8 -*-

from tactic_client_lib import TacticServerStub
import getpass
#import user
import os
import json

from PySide2 import QtWidgets, QtCore, QtGui

class tactic_checkin(QtWidgets.QMainWindow):
    #login = getpass.getuser()
    #password = user.login[login]

    def tacticLogin(self, user, password):
        try:
            self.tactic = TacticServerStub(login=user, password=password,
                                           server='10.0.0.51', project="dexter_studios")
            return True
        except Exception, e:
            QtWidgets.QMessageBox.warning(self, "Warning!", e.faultString)
            return False

    def getProjectCode(self, project):
        #shot_exp = "@SOBJECT(sthpw/project['category','Active'])"
        tactic = TacticServerStub(login="test.creator", password="eprtmxj",
                         server='10.0.0.51', project="dexter_studios")
        shot_exp = "@SOBJECT(sthpw/project['name','{project}'])".format(project=project)
        info = tactic.eval(shot_exp, single=True)

        project_code = info.get('code')
        return project_code
#        for i in info:
#            if i["code"] == project:
#                project_code = i['name']
#                return project_code

    def getTaskInfo(self, user, shot):
        task_filters = [('extra_code', shot),
                        ('assigned', user)]

        task = self.tactic.query("sthpw/task", filters=task_filters)

        return task

    def changeStatus(self, snapshot, task, status):
        snapshot_code = snapshot.get('code')
        snapshot_expr = self.tactic.eval("@SOBJECT(sthpw/snapshot['code', '{}'])".format(snapshot_code), single=True)

        task_code = task.get('code')
        status_old = task.get('status')

        task_status_data = {'status': str(status)}
        task_search_key = self.tactic.build_search_key('sthpw/task', task_code)
        self.tactic.update(task_search_key, task_status_data)
        print '  --> Status Changed : %s --> %s' % (status_old, status)

        keywords = "publish"

        try:
            snapshot_search_key = snapshot_expr.get("__search_key__")
            snapshot_data = {'keywords': keywords, 'status': str(status), 'status_confirm': str(status)}
            self.tactic.update(snapshot_search_key, snapshot_data)
            print '  --> Updated Snapshot Info (Status: %s, Keyword: %s)' % (status, keywords)
        except:
            print "Status와 Keywords 입력에 에러가 발생하였습니다.\n관리자에게 문의하여 주세요."
            pass

    def createNote(self, infoDict):
        exceptionList = ["TASK", "start", "end", "SHOW", "SHOT",
                         "cameras", "render_camera", "reference"]

        infofile = infoDict["jsonPath"]

        with open(infofile, 'r') as f:
            info = json.load(f)

        startFrame = info["AlembicCache"]["start"]
        endFrame = info["AlembicCache"]["end"]

        note = "# Animation Publish\n\n"
        note += "# duration\n"
        note += "{start} - {end}\n\n".format(start=startFrame, end=endFrame)
        note += "# json path\n"
        note += "{}\n\n".format(infofile)

        for key in info["AlembicCache"].keys():
            if key not in exceptionList:
                note += "# {}\n".format(key)
                note += "{}\n\n".format(info["AlembicCache"][key])

        return note

    def insertNote(self, user, project_code, shot_id, context, note):
        search_type = 'sthpw/note'
        note_data = {
            'project_code': project_code,
            'search_type': '%(project_code)s/shot?project=%(project_code)s' % {
                'project_code': project_code
            },
            'search_id': shot_id,
            'login': user,
            #'process': "animation",
            'context': context,
            'note': note
        }

        # insert note
        self.tactic.start()
        try:
            self.tactic.insert(search_type, note_data)
        except Exception, e:
            print str(e)
            self.tactic.abort()
        else:
            self.tactic.finish('Insert Note')

    def checkin(self, user, project_code, shotName, checkin_file, note, status):
        ls_context = list()
        filters = [('code', shotName)] # ie. AAA_0010
        columns = ['id', 'code', 'status']

        shot = self.tactic.query("{}/shot".format(project_code), filters, columns, single=True)
        search_key = shot.get('__search_key__')
        extra_code = shot.get('code')
        shot_id    = shot.get('id')

        description = "({previewfile})\n{note}".format(previewfile=checkin_file.split(os.sep)[-1],
                                                       note=note)

        ls_task = self.getTaskInfo(user, shotName)
        #context = task.get('context') # ie. "animation"
        for t in ls_task:
            ls_context.append(t.get('context'))

        context, ok = QtWidgets.QInputDialog.getItem(self, "Select Context",
                                                 "Context List", ls_context, 0, False)
        if ok and context:
            snapshot = self.tactic.simple_checkin(search_key, context, checkin_file,
                                                  description=description, mode='upload')
            for tk in ls_task:
                if tk["context"] == context:
                    task = tk
            self.changeStatus(snapshot=snapshot, task=task, status=status)
            self.insertNote(user=user, project_code=extra_code, shot_id=shot_id, context=context, note=note)

