# coding:utf-8

# make preview item (thumnail, label)

# relative direction (Arrow)

# relative level
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore
import os
import math

import AnimBrowser.Pipeline.dbConfig as dbConfig

class RelativeGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent = None):
        QtWidgets.QGraphicsView.__init__(self, parent)

        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setAcceptDrops(True)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(120,120,120)))

        self.graphicsScene = QtWidgets.QGraphicsScene()
        self.setScene(self.graphicsScene)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            for item in self.graphicsScene.items():
                if item.contains(QtCore.QPointF(event.localPos().x() - (self.graphicsScene.width()/2), event.localPos().y())):
                    menu = QtWidgets.QMenu(self)
                    menu.addAction(QtGui.QIcon(), u"폴더 열기", lambda: os.system("nautilus %s" % os.path.dirname(item.getMovFile())))
                    menu.popup(QtGui.QCursor.pos())
        event.accept()

    def resizeEvent(self, event):
        # left = -(event.size().width() / 2)
        # top = -(event.size().height() / 2)
        # right = (event.size().width() / 2)
        # bottom = (event.size().height() / 2)

        left = 0
        top = 0
        right = event.size().width()
        bottom = event.size().height()
        self.graphicsScene.setSceneRect(left, top, right, bottom)

        for item in self.graphicsScene.items():
            if type(item) == GraphicsNodeItem:
                itemPosX = self.graphicsScene.width() / item.interval * item.position
                itemPosY = (self.graphicsScene.height() / 3 * item.getDepth())
                item.setPos(QtCore.QPointF(itemPosX, itemPosY))#20 + (item.getDepth() * 130)))
        for arrowItem in self.graphicsScene.items():
            if type(arrowItem) == Arrow:
                arrowItem.updatePosition()

    def dropEvent(self, event):
        # Qt::ItemSelectionMode mode = Qt::IntersectsItemShape

        graphicsNodeItemList = []
        for item in self.items(event.pos().x(), event.pos().y()):
            if type(item) == GraphicsNodeItem:
                graphicsNodeItemList.append(item)

        selectedItem = None
        if len(graphicsNodeItemList) >= 2:
            # selectItem
            pass
        elif len(graphicsNodeItemList) == 1:
            selectedItem = graphicsNodeItemList[0]
        else:
            # Error selectFailed
            return

        if selectedItem.getDepth() == 2:
            # fail not append child
            return

        insertTier = selectedItem.getDepth() + 1

        if selectedItem.getDepth() == 0:
            #select child depth 1 or 2
            pass

        tierItemId = selectedItem.tierItemId
        if selectedItem.tierItemId == "":
            tierItemId = dbConfig.addRelativeInfo(selectedItem.myObjId,
                                     insertTier,
                                     event.source().currentItem().contentInfo['_id']).inserted_id
        else:
            dbConfig.appendTierItem(updateObjId = selectedItem.tierItemId,
                                    selfTier = insertTier,
                                    selfObjId = event.source().currentItem().contentInfo['_id'],
                                    parentObjId = selectedItem.myObjId)

        # refresh UI
        self.refreshUI(tierItemId)
        event.accept()

    def refreshUI(self, objId):
        self.graphicsScene.clear()

        resultTier = dbConfig.getRelativeInfo(objId)

        # prevItem = None
        # # treeDepthDic = {0:}

        if not resultTier:
            item = GraphicsNodeItem(tierItemId="",
                                    myObjId=objId,
                                    parentObjId="")
            item.setDepth(0)
            # iconPath

            curInfo = dbConfig.getItemForObjID(objId)

            iconPath = curInfo['files']['preview']
            item.setPixmapPath(iconPath)

            title = "%s%s" % (curInfo['tag3tier'], curInfo['fileNum'])
            item.setTextitemName(title)

            if curInfo['files'].has_key('mov'):
                item.setMovFile(curInfo['files']['mov'])
            else:
                item.setMovFile(curInfo['files']['preview'])

            tempFont = item.textItem.font()
            # textWidth = QtGui.QFontMetrics(tempFont).width("Anim Data")
            textHeight = QtGui.QFontMetrics(tempFont).height()
            item.setPolygonPoint(165, 125 + textHeight)
            # item.setPolygonPoint(textWidth + 60, textHeight + 85)

            itemPosX = self.graphicsScene.width() / float(2) * 1
            itemPosY = self.graphicsScene.height() / 3 * 0
            item.setNeedResizeData(interval=float(2),
                                   position=1)

            item.setPos(QtCore.QPointF(itemPosX, itemPosY))  # 20 + (index * 130)))
            # item.setMovFile(iconPath)
            #
            self.graphicsScene.addItem(item)
        else:
            tierDic = {0:[], 1:[], 2:[]}
            tierItemId = ""
            for info in resultTier.keys():
                if info == '_id':
                    tierItemId = resultTier[info]
                    continue
                tierDic[int(info)] = resultTier[info]

            relativeItemDic = {}
            for index in range(3):
                if not (tierDic[index]):
                    pass
                else:
                    for j in tierDic[index]:
                        # make item
                        parentId = ""
                        if j.has_key('parent'):
                            parentId = j['parent']
                        item = GraphicsNodeItem(tierItemId = tierItemId,
                                                myObjId = j['_id'],
                                                parentObjId = parentId)
                        item.setDepth(index)
                        # iconPath

                        curInfo = dbConfig.getItemForObjID(j['_id'])

                        iconPath = curInfo['files']['preview']
                        item.setPixmapPath(iconPath)

                        if curInfo['files'].has_key('mov'):
                            item.setMovFile(curInfo['files']['mov'])
                        else:
                            item.setMovFile(curInfo['files']['preview'])

                        title = "%s%s" % (curInfo['tag3tier'], curInfo['fileNum'])
                        item.setTextitemName(title)

                        tempFont = item.textItem.font()
                        # textWidth = QtGui.QFontMetrics(tempFont).width("Anim Data")
                        textHeight = QtGui.QFontMetrics(tempFont).height()
                        item.setPolygonPoint(165, 125 + textHeight)
                        # item.setPolygonPoint(160, textHeight + 85)

                        interval = float(len(tierDic[index]) + 1)
                        position = (tierDic[index].index(j) + 1)

                        itemPosX = (self.graphicsScene.width() / interval * position)
                        itemPosY = (self.graphicsScene.height() / 3 * index)
                        item.setNeedResizeData(interval = interval,
                                               position = position)

                        item.setPos(QtCore.QPointF(itemPosX, itemPosY))# 20 + (index * 130)))
                        # item.setMovFile(iconPath)
                        #
                        self.graphicsScene.addItem(item)
                        relativeItemDic[j['_id']] = item

                        if j.has_key('parent'):
                            self.connectNodes(relativeItemDic[j['parent']], item, "Solid", QtCore.Qt.black, 2)

    def connectNodes(self, srcNode, destNode, type, color, thick):
        arrow = Arrow(srcNode, destNode, type, color, thick)
        arrow.setZValue(-1000.0)
        self.graphicsScene.addItem(arrow)
        arrow.updatePosition()

class GraphicsNodeItem(QtWidgets.QGraphicsPolygonItem):
    def __init__(self, parent = None, depth = 1, tierItemId = "", myObjId = "", parentObjId = ""):
        QtWidgets.QGraphicsPolygonItem.__init__(self, parent)

        self.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable)

        self.tierItemId = tierItemId
        self.myObjId = myObjId
        self.parentObjId = parentObjId

        self.movFilePath = ""
        self.setDepth(depth)

        #
        self.defaultBrush = QtGui.QBrush(QtGui.QColor(200, 200, 200))
        self.highlightBrush = QtGui.QBrush(QtGui.QColor(247.095, 146.88, 30.09))
        # # rgb(247.095,146.88,30.09)
        self.setBrush(self.defaultBrush)
        #

        self.textItem = QtWidgets.QGraphicsSimpleTextItem(self)
        self.textItem.setBrush(QtGui.QBrush(QtCore.Qt.darkRed))
        # tempPen = self.textItem.font()
        # tempPen.setBold(True)
        # self.textItem.setFont(tempPen)

        self.pixmapItem = QtWidgets.QGraphicsPixmapItem(self)

        # ------------------------------------------------------------------------------
        # self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.RightButton)
        self.setAcceptDrops(True)
        self.posX = self.posY = 0

    def setNeedResizeData(self, interval, position):
        self.interval = interval
        self.position = position

    def getDepth(self):
        return self.depth

    def setDepth(self, depth):
        self.depth = depth

    def setPixmapPath(self, pixmap):
        width = 150
        height = 117
        # self.pixmapItem.setOffset(-width / 2.0, -height / 2.0)
        self.pixmapItem.setOffset(-width / 2.0, 5)
        if type(pixmap) == str or type(pixmap) == unicode:
            pixmapImage = QtGui.QPixmap(pixmap)
            self.pixmapItem.setPixmap(pixmapImage.scaled(width, height,
                                                         QtCore.Qt.IgnoreAspectRatio,
                                                         QtCore.Qt.SmoothTransformation))
        else:
            self.pixmapItem.setPixmap(pixmap.scaled(width, height,
                                                    QtCore.Qt.IgnoreAspectRatio,
                                                    QtCore.Qt.SmoothTransformation))

    # ------------------------------------------------------------------------------
    def setMovFile(self, path):
        self.movFilePath = path

    def getMovFile(self):
        return self.movFilePath
    # ------------------------------------------------------------------------------

    def setPolygonPoint(self, width, height):
        topLeft = QtCore.QPointF(-(width / 2.0), 0)
        topRight = QtCore.QPointF((width / 2.0), 0)
        bottomRight = QtCore.QPointF(width / 2.0, height)
        bottomLeft = QtCore.QPointF(-(width / 2.0), height)

        self.setPolygon(QtGui.QPolygonF([topLeft, topRight, bottomRight, bottomLeft, topLeft]))

    def setTextitemName(self, name):
        self.textItem.setText(name)

        textWidthOffset = self.textItem.boundingRect().width() / 2.0
        # if self.depth == 0:
        self.textItem.setPos(QtCore.QPointF(-textWidthOffset, 120))
        # else:
        #     self.textItem.setPos(QtCore.QPointF(-textWidthOffset, 132))

    def mouseDoubleClickEvent(self, e):
        QtGui.QDesktopServices.openUrl(self.getMovFile())
        e.accept()

class Arrow(QtWidgets.QGraphicsLineItem):
    def __init__(self, startItem, endItem, type=None, color=None, thick=2, parent=None):
        QtWidgets.QGraphicsLineItem.__init__(self, parent)

        if not (thick):
            thick = 2

        self.arrowHead = QtGui.QPolygonF()
        self.myStartItem = startItem
        self.myEndItem = endItem
        # self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        self.myColor = color

        self.typeDic = {'Dash': QtCore.Qt.DashLine, 'Solid': QtCore.Qt.SolidLine, 'Dot': QtCore.Qt.DotLine,
                        None: QtCore.Qt.DashLine}

        self.setPen(QtGui.QPen(self.myColor, thick, self.typeDic[type],
                               QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))

    def setColor(self, color):
        self.myColor = color

    def startItem(self):
        return self.myStartItem

    def endItem(self):
        return self.myEndItem

    def boundingRect(self):
        extra = (self.pen().width() + 20) / 2.0
        p1 = self.line().p1()
        p2 = self.line().p2()
        return QtCore.QRectF(p1, QtCore.QSizeF(p2.x() - p1.x(), p2.y() - p1.y())).normalized().adjusted(-extra, -extra,
                                                                                                        extra, extra)

    def shape(self):
        path = super(Arrow, self).shape()
        path.addPolygon(self.arrowHead)
        return path

    def updatePosition(self):
        line = QtCore.QLineF(self.mapFromItem(self.myStartItem, 0, 0), self.mapFromItem(self.myEndItem, 0, 0))
        self.setLine(line)

    def paint(self, painter, option, widget=None):

        if (self.myStartItem.collidesWithItem(self.myEndItem)):
            return
        myStartItem = self.myStartItem
        myEndItem = self.myEndItem
        myColor = self.myColor
        myPen = self.pen()
        myPen.setColor(self.myColor)
        # arrowSize = 14.0
        arrowSize = 8.0
        painter.setPen(myPen)
        painter.setBrush(self.myColor)

        # centerLine = QtCore.QLineF(myStartItem.pos() + QtCore.QPointF(0,25 + myStartItem.boundingRect().height() / 2.0), myEndItem.pos() + QtCore.QPointF(0,25 + myEndItem.boundingRect().height() / 2.0))
        centerLine = QtCore.QLineF(myStartItem.pos() + QtCore.QPointF(0, myStartItem.boundingRect().height() / 2.0),
                                   myEndItem.pos() + QtCore.QPointF(0, myEndItem.boundingRect().height() / 2.0))
        # centerLine = QtCore.QLineF(myStartItem.scenePos(), myEndItem.scenePos())

        endPolygon = myEndItem.polygon()
        p1 = endPolygon.first() + myEndItem.pos()
        intersectPoint = QtCore.QPointF()

        for i in endPolygon:
            p2 = i + myEndItem.pos()
            polyLine = QtCore.QLineF(p1, p2)
            intersectType , intersectPoint = polyLine.intersect(centerLine)
            if intersectType == QtCore.QLineF.BoundedIntersection:
                break
            p1 = p2

        # self.setLine(QtCore.QLineF(intersectPoint, myStartItem.pos()))
        # middleOffset = (myStartItem.boundingRect().height() / 2.0) - 50
        middleOffset = (myStartItem.boundingRect().height() / 2.0) - 25

        # startPoint = QtCore.QPointF(myStartItem.pos().x(), myStartItem.pos().y())
        startPoint = QtCore.QPointF(myStartItem.pos().x(), myStartItem.pos().y() + middleOffset + 25)
        # self.setLine(QtCore.QLineF(intersectPoint, startPoint ))

        # self.setLine(QtCore.QLineF(centerLine.p1(), intersectPoint))
        self.setLine(QtCore.QLineF(intersectPoint, centerLine.p1()))
        #
        line = self.line()
        angle = math.acos(line.dx() / line.length())

        if line.dy() >= 0:
            angle = (math.pi * 2.0) - angle

        arrowP1 = line.p1() + QtCore.QPointF(math.sin(angle + math.pi / 3.0) * arrowSize,
                                             math.cos(angle + math.pi / 3.0) * arrowSize)

        arrowP2 = line.p1() + QtCore.QPointF(math.sin(angle + math.pi - math.pi / 3.0) * arrowSize,
                                             math.cos(angle + math.pi - math.pi / 3.0) * arrowSize)

        self.arrowHead.clear()

        for point in [line.p1(), arrowP1, arrowP2]:
            self.arrowHead.append(point)

        painter.drawLine(line)

        painter.setPen(QtGui.QPen(myColor, 3, QtCore.Qt.SolidLine,
                                  QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))

        painter.drawPolygon(self.arrowHead)