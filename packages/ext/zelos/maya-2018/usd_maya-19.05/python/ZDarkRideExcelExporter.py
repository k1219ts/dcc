#-*- coding: utf-8 -*-

import os, time
import xlwt
import maya.cmds as mc
import maya.mel as mel
import math

class ExcelExporter():
    def __init__(self):
        self.red = xlwt.easyxf('font: name Arial, color-index black; pattern: pattern solid, fore_color red;')#pattern: back_color yellow;
        self.black = xlwt.easyxf('font: name Arial, color-index black')

        self.kRailSpeed                     = 3.0   #m / sec
        self.kRailAccSpeed                  = 0.3   #g

        self.kSpindleAngularSpeed           = 72.0    #degree / sec
        self.kSpindleAngularAccSpeed        = 100.0   #degree / sec

        self.kMotionPlatformPos             = 150.0   #mm / sec
        self.kMotionPlatformLinearSpeed     = 0.3     #m / sec
        self.kMotionPlatformLinearAccSpeed  = 0.3     #g

        self.kMotionPlatformAngle           = 12.0    #degree
        self.kMotionPlatformAngularSpeed    = 30.0    #degree / sec^2
        self.kMotionPlatformAngularAccSpeed = 200.0   #degree / sec^2

        self.frames          = []
        self.times           = []
        self.totalDists      = []

        # Rail
        self.railVelSpeeds   = []
        self.railAccSpeeds   = []

        # Spindle (Rx)
        self.spindleAngleXs    = []
        self.spindleAngularSpeedXs    = []
        self.spindleAngularAccSpeedXs = []
        # Spindle (Ry)
        self.spindleAngleYs    = []
        self.spindleAngularSpeedYs    = []
        self.spindleAngularAccSpeedYs = []
        # Spindle (Rz)
        self.spindleAngleZs    = []
        self.spindleAngularSpeedZs    = []
        self.spindleAngularAccSpeedZs = []

        # Motion Platform (Rx)
        self.motionPlatformAngleXs = []
        self.motionPlatformAngularSpeedXs = []
        self.motionPlatformAngularAccSpeedXs = []
        # Motion Platform (Ry)
        self.motionPlatformAngleYs = []
        self.motionPlatformAngularSpeedYs = []
        self.motionPlatformAngularAccSpeedYs = []
        # Motion Platform (Rz)
        self.motionPlatformAngleZs = []
        self.motionPlatformAngularSpeedZs = []
        self.motionPlatformAngularAccSpeedZs = []


        # Motion Platform (Tx)
        self.motionPlatformDispXs = []
        self.motionPlatformLinearSpeedXs = []
        self.motionPlatformLinearAccSpeedXs = []
        # Motion Platform (Ty)
        self.motionPlatformDispYs = []
        self.motionPlatformLinearSpeedYs = []
        self.motionPlatformLinearAccSpeedYs = []
        # Motion Platform (Tz)
        self.motionPlatformDispZs = []
        self.motionPlatformLinearSpeedZs = []
        self.motionPlatformLinearAccSpeedZs = []

        # motion platform check list */
        self.motionPlatformCheck = []
        self.motionPlatformCheckVel = []
        self.motionPlatformCheckAcc = []

    def setFrame(self, frames):
        self.frames = frames

    def setTime(self, times):
        self.times = times

    # for rail
    def setRailData(self, totalDists, speeds, accSpeeds):
        self.totalDists = totalDists
        self.railVelSpeeds = speeds
        self.railAccSpeeds = accSpeeds

    def setSpindleRxData(self, angles, angularSpeeds, angularAccSpeeds):
        self.spindleAngleXs = angles;
        self.spindleAngularSpeedXs = angularSpeeds
        self.spindleAngularAccSpeedXs = angularAccSpeeds

    def setSpindleRyData(self, angles, angularSpeeds, angularAccSpeeds):
        self.spindleAngleYs = angles;
        self.spindleAngularSpeedYs = angularSpeeds
        self.spindleAngularAccSpeedYs = angularAccSpeeds

    def setSpindleRzData(self, angles, angularSpeeds, angularAccSpeeds):
        self.spindleAngleZs = angles;
        self.spindleAngularSpeedZs = angularSpeeds
        self.spindleAngularAccSpeedZs = angularAccSpeeds

    def setMotionPlatformRxData(self, angles, angularSpeeds, angularAccSpeeds):
        self.motionPlatformAngleXs = angles
        self.motionPlatformAngularSpeedXs = angularSpeeds
        self.motionPlatformAngularAccSpeedXs = angularAccSpeeds

    def setMotionPlatformRyData(self, angles, angularSpeeds, angularAccSpeeds):
        self.motionPlatformAngleYs = angles
        self.motionPlatformAngularSpeedYs = angularSpeeds
        self.motionPlatformAngularAccSpeedYs = angularAccSpeeds

    def setMotionPlatformRzData(self, angles, angularSpeeds, angularAccSpeeds):
        self.motionPlatformAngleZs = angles
        self.motionPlatformAngularSpeedZs = angularSpeeds
        self.motionPlatformAngularAccSpeedZs = angularAccSpeeds

    def setMotionPlatformTxData(self, disps, speeds, accSpeeds):
        self.motionPlatformDispXs = disps
        self.motionPlatformLinearSpeedXs = speeds
        self.motionPlatformLinearAccSpeedXs = accSpeeds

    def setMotionPlatformTyData(self, disps, speeds, accSpeeds):
        self.motionPlatformDispYs = disps
        self.motionPlatformLinearSpeedYs = speeds
        self.motionPlatformLinearAccSpeedYs = accSpeeds

    def setMotionPlatformTzData(self, disps, speeds, accSpeeds):
        self.motionPlatformDispZs = disps
        self.motionPlatformLinearSpeedZs = speeds
        self.motionPlatformLinearAccSpeedZs = accSpeeds

    def setMotionPlatformCheckData(self, disp, vel, acc):
        self.motionPlatformCheck = disp
        self.motionPlatformCheckVel = vel
        self.motionPlatformCheckAcc = acc


    ################################################  Excel   ##########################################################
    def excelWrite(self, path):
        self.rideExcel(path)
        #self.motionExcel(excelPath)
        #self.additionalExcel(excelPath)

    def rideExcel(self, excelPath):
        rideTitle               = ['t(s)', 'Rail Distance (m)', 'Rail Velocity (m/s)', 'Rail Acceleration (g)']
        spindleTitle            = ['t(s)', 'Spindle Rz (degree)', 'Spindle Angular Velocity (degree/s)', 'Spindle Angular Acceleration(degree/s^2)']
        motionPlatformRxTitle   = ['t(s)', 'Motion Platform Rx(degree)', 'Motion Platform Rx Angular Velocity (degree/s)', 'Motion Platform Rx Angular Acceleration(degree/s^2)']
        motionPlatformRyTitle   = ['t(s)', 'Motion Platform Ry(degree)', 'Motion Platform Ry Angular Velocity (degree/s)', 'Motion Platform Ry Angular Acceleration(degree/s^2)']
        motionPlatformRzTitle   = ['t(s)', 'Motion Platform Rz(degree)', 'Motion Platform Rz Angular Velocity (degree/s)', 'Motion Platform Rz Angular Acceleration(degree/s^2)']
        motionPlatformTxTitle   = ['t(s)', 'Motion Platform Tx(mm)', 'Motion Platform Tx Linear Velocity(m/s)', 'Motion Platform Tx Linear Acceleration(g)']
        motionPlatformTyTitle   = ['t(s)', 'Motion Platform Ty(mm)', 'Motion Platform Ty Linear Velocity(m/s)', 'Motion Platform Ty Linear Acceleration(g)']
        motionPlatformTzTitle   = ['t(s)', 'Motion Platform Tz(mm)', 'Motion Platform Tz Linear Velocity(m/s)', 'Motion Platform Tz Linear Acceleration(g)']

        motionPlatformCheckTitle        = ['t(s)',
                                           'Motion Platform Rx(degree)' , 'Motion Platform Ry (degree)',
                                           'Motion Platform Tx (mm)', 'Motion Platform Ty (mm)', 'Motion Platform  Tz (mm)',
                                           '<=100%']
        motionPlatformCheckVelTitle     = ['t(s)',
                                           'Motion Platform Rx Angular Velocity(degree/s)', 'Motion Platform Ry Angular Velocity (degree/s)',
                                           'Motion Platform Tx Linear Velocity(m/s)', 'Motion Platform Ty Linear Velocity(m/s)', 'Motion Platform Tz Linear Velocity(m/s)',
                                           '<=100%']
        motionPlatformCheckAccTitle     = ['t(s)',
                                           'Rail Acceleration (m/s^2)',
                                           'Spindle Rz Angular Acceleration (degree/s^2)',
                                           'Motion Platform Rx Angular Acceleration(degree/s^2)', 'Motion Platform Ry Angular Acceleration(degree/s^2)',
                                           'Motion Platform Tx Linear Acceleration (g)', 'Motion Platform Ty Linear Acceleration (g)', 'Motion Platform Tz Linear Acceleration(g)',
                                           '<=100%']

        wbk = xlwt.Workbook()
        ##wrtie rail sheet
        railSheet = self.createSheet(wbk, 'Rail', rideTitle)

        #wrtie spindle sheet
        spindleSheet = self.createSheet(wbk, 'Spindle', spindleTitle)

        #wrtie motion platform sheet
        motionPlatformRxSheet = self.createSheet(wbk, 'Motion Platform Rx', motionPlatformRxTitle)
        motionPlatformRySheet = self.createSheet(wbk, 'Motion Platform Ry', motionPlatformRyTitle)
        motionPlatformRzSheet = self.createSheet(wbk, 'Motion Platform Rz', motionPlatformRzTitle)
        motionPlatformTxSheet = self.createSheet(wbk, 'Motion Platform Tx', motionPlatformTxTitle)
        motionPlatformTySheet = self.createSheet(wbk, 'Motion Platform Ty', motionPlatformTyTitle)
        motionPlatformTzSheet = self.createSheet(wbk, 'Motion Platform Tz', motionPlatformTzTitle)

        motionPlatformCheckSheet    = self.createSheet(wbk, u'변동 폭 대조', motionPlatformCheckTitle)
        motionPlatformCheckVelSheet = self.createSheet(wbk, u'속도 대조', motionPlatformCheckVelTitle)
        motionPlatformCheckAccSheet = self.createSheet(wbk, u'가속도 대조', motionPlatformCheckAccTitle)


        count = len(self.times)
        for i in range(count):
            #write rail sheet
            railSheet.write      (i+1, 0, self.times[i])
            railSheet.write      (i+1, 1, self.totalDists[i])
            self.write(railSheet, i+1, 2, self.railVelSpeeds[i], self.kRailSpeed)
            self.write(railSheet, i+1, 3, self.railAccSpeeds[i], self.kRailAccSpeed)
            #railSheet.write      (i+1, 4, self.frames[i])

            #write spindle sheet
            spindleSheet.write      (i+1, 0, self.times[i])
            spindleSheet.write      (i+1, 1, self.spindleAngleZs[i])
            self.write(spindleSheet, i+1, 2, self.spindleAngularSpeedZs[i], self.kSpindleAngularSpeed)
            self.write(spindleSheet, i+1, 3, self.spindleAngularAccSpeedZs[i], self.kSpindleAngularAccSpeed)
            #spindleSheet.write      (i+1, 0, self.frames[i])

            #write motion platform rx sheet
            motionPlatformRxSheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformRxSheet, i+1, 1, self.motionPlatformAngleXs[i], self.kMotionPlatformAngle)
            self.write(motionPlatformRxSheet, i+1, 2, self.motionPlatformAngularSpeedXs[i], self.kMotionPlatformAngularSpeed)
            self.write(motionPlatformRxSheet, i+1, 3, self.motionPlatformAngularAccSpeedXs[i], self.kMotionPlatformAngularAccSpeed)
            #motionPlatformRxSheet.write      (i+1, 0, self.frames[i])

            #write motion platform ry sheet
            motionPlatformRySheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformRySheet, i+1, 1, self.motionPlatformAngleYs[i], self.kMotionPlatformAngle)
            self.write(motionPlatformRySheet, i+1, 2, self.motionPlatformAngularSpeedYs[i], self.kMotionPlatformAngularSpeed)
            self.write(motionPlatformRySheet, i+1, 3, self.motionPlatformAngularAccSpeedYs[i], self.kMotionPlatformAngularAccSpeed)
            #motionPlatformRySheet.write      (i+1, 0, self.frames[i])

            #write motion platform rz sheet
            motionPlatformRzSheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformRzSheet, i+1, 1, self.motionPlatformAngleZs[i], self.kMotionPlatformAngle)
            self.write(motionPlatformRzSheet, i+1, 2, self.motionPlatformAngularSpeedZs[i], self.kMotionPlatformAngularSpeed)
            self.write(motionPlatformRzSheet, i+1, 3, self.motionPlatformAngularAccSpeedZs[i], self.kMotionPlatformAngularAccSpeed)
            #motionPlatformRzSheet.write      (i+1, 0, self.frames[i])

            #write motion platform tx sheet
            motionPlatformTxSheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformTxSheet, i+1, 1, self.motionPlatformDispXs[i], self.kMotionPlatformPos)
            self.write(motionPlatformTxSheet, i+1, 2, self.motionPlatformLinearSpeedXs[i], self.kMotionPlatformLinearSpeed)
            self.write(motionPlatformTxSheet, i+1, 3, self.motionPlatformLinearAccSpeedXs[i], self.kMotionPlatformLinearAccSpeed)
            #motionPlatformTxSheet.write      (i+1, 0, self.frames[i])

            #write motion platform ty sheet
            motionPlatformTySheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformTySheet, i+1, 1, self.motionPlatformDispYs[i], self.kMotionPlatformPos)
            self.write(motionPlatformTySheet, i+1, 2, self.motionPlatformLinearSpeedYs[i], self.kMotionPlatformLinearSpeed)
            self.write(motionPlatformTySheet, i+1, 3, self.motionPlatformLinearAccSpeedYs[i], self.kMotionPlatformLinearAccSpeed)
            #motionPlatformTySheet.write      (i+1, 0, self.frames[i])

            #write motion platform tz sheet
            motionPlatformTzSheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformTzSheet, i+1, 1, self.motionPlatformDispZs[i], self.kMotionPlatformPos)
            self.write(motionPlatformTzSheet, i+1, 2, self.motionPlatformLinearSpeedZs[i], self.kMotionPlatformLinearSpeed)
            self.write(motionPlatformTzSheet, i+1, 3, self.motionPlatformLinearAccSpeedZs[i], self.kMotionPlatformLinearAccSpeed)
            #motionPlatformTzSheet.write      (i+1, 0, self.frames[i])

            # write motionPlatformCheck Sheet
            motionPlatformCheckSheet.      write(i+1, 0, self.times[i])
            self.write(motionPlatformCheckSheet, i+1, 1, self.motionPlatformAngleXs[i], self.kMotionPlatformAngle)
            self.write(motionPlatformCheckSheet, i+1, 2, self.motionPlatformAngleYs[i], self.kMotionPlatformAngle)
            self.write(motionPlatformCheckSheet, i+1, 3, self.motionPlatformDispXs[i],  self.kMotionPlatformPos)
            self.write(motionPlatformCheckSheet, i+1, 4, self.motionPlatformDispYs[i],  self.kMotionPlatformPos)
            self.write(motionPlatformCheckSheet, i+1, 5, self.motionPlatformDispZs[i],  self.kMotionPlatformPos)
            self.write(motionPlatformCheckSheet, i+1, 6, self.motionPlatformCheck[i], 1.0)
            #motionPlatformCheckSheet.      write(i+1, 0, self.frames[i])

            # write motionPlatformCheckVel Sheet
            motionPlatformCheckVelSheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformCheckVelSheet, i+1, 1, self.motionPlatformAngularSpeedXs[i], self.kMotionPlatformAngularSpeed)
            self.write(motionPlatformCheckVelSheet, i+1, 2, self.motionPlatformAngularSpeedYs[i], self.kMotionPlatformAngularSpeed)
            self.write(motionPlatformCheckVelSheet, i+1, 3, self.motionPlatformLinearSpeedXs[i],  self.kMotionPlatformLinearSpeed)
            self.write(motionPlatformCheckVelSheet, i+1, 4, self.motionPlatformLinearSpeedYs[i],  self.kMotionPlatformLinearSpeed)
            self.write(motionPlatformCheckVelSheet, i+1, 5, self.motionPlatformLinearSpeedZs[i],  self.kMotionPlatformLinearSpeed)
            self.write(motionPlatformCheckVelSheet, i+1, 6, self.motionPlatformCheckVel[i], 1.0)
            #motionPlatformCheckVelSheet.write      (i+1, 0, self.frames[i])

            # write motionPlatformCheckAcc Sheet
            motionPlatformCheckAccSheet.write      (i+1, 0, self.times[i])
            self.write(motionPlatformCheckAccSheet, i+1, 1, self.railAccSpeeds[i],                      self.kRailAccSpeed)
            self.write(motionPlatformCheckAccSheet, i+1, 2, self.spindleAngularAccSpeedZs[i],           self.kSpindleAngularAccSpeed)
            self.write(motionPlatformCheckAccSheet, i+1, 3, self.motionPlatformAngularAccSpeedXs[i],    self.kMotionPlatformAngularAccSpeed)
            self.write(motionPlatformCheckAccSheet, i+1, 4, self.motionPlatformAngularAccSpeedYs[i],    self.kMotionPlatformAngularAccSpeed)
            self.write(motionPlatformCheckAccSheet, i+1, 5, self.motionPlatformLinearAccSpeedXs[i],     self.kMotionPlatformLinearAccSpeed)
            self.write(motionPlatformCheckAccSheet, i+1, 6, self.motionPlatformLinearAccSpeedYs[i],     self.kMotionPlatformLinearAccSpeed)
            self.write(motionPlatformCheckAccSheet, i+1, 7, self.motionPlatformLinearAccSpeedZs[i],     self.kMotionPlatformLinearAccSpeed)
            self.write(motionPlatformCheckAccSheet, i+1, 8, self.motionPlatformCheckAcc[i], 1.0)
            #motionPlatformCheckAccSheet.write      (i+1, 0, self.frames[i])

        wbk.save(excelPath)

    def createSheet(self, wbk=None, sheetname=None, titles=None):
        sheets = wbk.add_sheet(sheetname)
        self.titleSet(sheets, titles, 5000)
        return sheets

    def titleSet(self, sheet, title, value=7000):
        for i, j in enumerate(title):
            sheet.write(0, i, j, self.black)
            sheet.col(i).width = value
    def write0(self, sheet, row, col, val):
        sheet.write(row, col, val)

    def write(self, sheet, row, col, val, limit):
        if abs(val) > limit:
            sheet.write(row, col, val, self.red)
        else:
            sheet.write(row, col, val)





#run = Wanda_cal(0) # 0 = 24fps, 1 = 100fps

