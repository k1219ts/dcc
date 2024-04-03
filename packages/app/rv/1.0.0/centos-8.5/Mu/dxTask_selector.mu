
//
//  Widget to display/select versions
//

module: dxTask_selector
{

  use rvtypes;
  use commands;
  use rvui;
  use gl;
  use glyph;
  use app_utils;
  use math;
  use math_util;
  use extra_commands;
  use glu;
  require python;
  require glyph;
  require runtime;

  \: deb(void; string s)  { if (false) print("vwid: " + s + "\n"); }
  \: rdeb(void; string s) { if (false) print("vwid: " + s + "\n"); }

  \: sourceName(string; string name)
  {
      let s = name.split(".");
      return s[0];
  }

  class: TaskSelector : Widget
  {

      int     _activeTaskIndex;
      int     _selectTaskIndex;
      int     _activeVersionIndex;
      int     _selectVersionIndex;
      bool    _set;
      bool    _in;
      float   _th;
      int     _nSources;
      Vec2    _tbox;
      Vec2    _taskTbox;
      Vec2    _nameTbox;
      bool    _drawInMargin;

      python.PyObject _pyGetTaskDataFromSource;
      python.PyObject _pySourceHasTaskData;
      python.PyObject _pyResetSourceFrame;
      python.PyObject _pyStaticSetTask;
      python.PyObject _pyStaticSetVersion;

      class: IntWrapper
      {
          int _int;
          method: IntWrapper (IntWrapper; int i) { _int = i; }
      }

      class: TaskDataMu
      {
          string[] _task;
          string[] _name;
          int      _current;
          int      _last;
          int      _currentTask;
          string   _source;

          rvtypes.Color[]  _color;

          method: TaskDataMu (TaskDataMu;)
          {
              _task        = string[]();
              _name        = string[]();
              _color       = rvtypes.Color[]();
              _current     = 0;
              _last        = 0;
              _currentTask = 0;
              _source      = "";
          }

          method: taskSize (int; )
          {
              return _task.size();
          }

          method: nameSize (int; )
          {
              return _name.size();
          }

          method: empty (bool; )
          {
              return (taskSize() == 0);
          }
      }

      method: getTaskDataMuFromSource (TaskDataMu; string sourceName=nil)
      {
          //  deb ("getTaskDataMuFromSource");
          let s = sourceName,
              vd = TaskDataMu();

          if (s eq nil || s == "")
          {
              extra_commands.updatePixelInfo(nil);

              State state = data();

              let hasVD = false;

              if (state.pixelInfo neq nil && state.pixelInfo.size() > 0)
              {
                  s = nodeGroup(state.pixelInfo[0].node) + "_source";
                  hasVD = propertyExists (s + ".versioning.task");
              }
          }

          if (s eq nil || s == "") return vd;

          let propPrefix = s + ".versioning.";

          if (propertyExists(propPrefix + "task")) vd._task       = getStringProperty (propPrefix + "task");
          if (propertyExists(propPrefix + "name")) vd._name       = getStringProperty (propPrefix + "name");

          // if (vd._task.size() != vd._task.size())
          // {
          //     require io;
          //
          //     vd._task.clear();
          //     for_each (m; vd._task)
          //     {
          //         vd._task.push_back (io.path.basename (m));
          //     }
          // }

          if (propertyExists(propPrefix + "currentIndex")) vd._current     = getIntProperty    (propPrefix + "currentIndex").front();
          else
          {
              newProperty    (propPrefix + "currentIndex", IntType, 1);
              setIntProperty (propPrefix + "currentIndex", int[] { 0 }, true);
          }

          if (propertyExists (propPrefix + "lastIndex"))   vd._last        = getIntProperty    (propPrefix + "lastIndex").front();
          else
          {
              newProperty    (propPrefix + "lastIndex", IntType, 1);
              setIntProperty (propPrefix + "lastIndex", int[] { 0 }, true);
          }

          if (propertyExists(propPrefix + "currentTask")) vd._currentTask     = getIntProperty    (propPrefix + "currentTask").front();
          else
          {
              newProperty    (propPrefix + "currentTask", IntType, 1);
              setIntProperty (propPrefix + "currentTask", int[] { 0 }, true);
          }

          //  deb ("    color");
          if (propertyExists(propPrefix + "color"))
          {
              float[] p = getFloatProperty (propPrefix + "color");
              if (p.size() == 3 * vd.taskSize())
              {
                  vd._color.resize(vd.taskSize());
                  for_index (i; vd._task) vd._color[i] = rvtypes.Color(p[3*i], p[3*i+1], p[3*i+2], 1);
              }
          }

          vd._source = s;

          //  deb ("    vd %s" % vd);
          return vd;
      }

      class: PyTaskData
      {
          python.PyObject _pySetVersion;

          string[] _task;
          string[] _name;
          Color[]  _color;
          int      _last;
          int      _current;
          int      _currentTask;
          string   _source;

          method: PyTaskData (PyTaskData; python.PyObject vd)
          {
              if (python.is_nil (vd)) return nil;

              _name = string[]();
              let pyname = python.PyObject_GetAttr (vd, "_name");
              for (int i = 0; i < python.PyTuple_Size(pyname); ++i)
              {
                  _name.push_back (to_string (python.PyTuple_GetItem (pyname, i)));
              }

              _task = string[]();
              let pytask = python.PyObject_GetAttr (vd, "_task");
              for (int i = 0; i < python.PyTuple_Size(pytask); ++i)
              {
                  _task.push_back (to_string (python.PyTuple_GetItem (pytask, i)));
              }

              _color = Color[]();
              let pycolor = python.PyObject_GetAttr (vd, "_color");
              for (int i = 0; i < python.PyTuple_Size(pycolor); ++i)
              {
                  let c = python.PyTuple_GetItem (pycolor, i);

                  _color.push_back (Color(
                      to_float  (python.PyTuple_GetItem (c, 0)),
                      to_float  (python.PyTuple_GetItem (c, 1)),
                      to_float  (python.PyTuple_GetItem (c, 2)),
                      1.0));
              }

              _last        = to_int    (python.PyObject_GetAttr (vd, "_last"));
              _currentTask = to_int    (python.PyObject_GetAttr (vd, "_currentTask"));
              _source      = to_string (python.PyObject_GetAttr (vd, "_source"));

              _pySetVersion= python.PyObject_GetAttr (vd, "setVersion");
          }

          method: nameSize (int; )
          {
              return _name.size();
          }

          method: taskSize (int; )
          {
              return _task.size();
          }

          method: setVersion (void; int index)
          {
              python.PyObject_CallObject (_pySetVersion, IntWrapper(index));
          }

          method: empty (bool; )
          {
              return (taskSize() == 0);
          }

      } //  class PyTaskData

      method: sourceHasTaskData (bool; string source="")
      {
  	     return to_bool(python.PyObject_CallObject (_pySourceHasTaskData, source));
      }

      method: getTaskDataFromSource (PyTaskData; string source="")
      {
          let vd = python.PyObject_CallObject (_pyGetTaskDataFromSource, source);

          return PyTaskData(vd);
      }

      method: resetSourceFrame (void; string source)
      {
          python.PyObject_CallObject (_pyResetSourceFrame, source);
      }

      method: staticSetTask (void; int task)
      {
          python.PyObject_CallObject (_pyStaticSetTask, IntWrapper(task));
      }

      method: staticSetVersion (void; int version)
      {
          python.PyObject_CallObject (_pyStaticSetVersion, IntWrapper(version));
      }

      method: selectSource(void; Event event, int incr)
      {
          /*
          if (_getVersions eq nil) return;

          let domain  = event.domain(),
              (currentIndex, lastIndex, versions) = _getVersions();

          let idx = _selectVersionIndex - incr;

          if(idx >= 0 && idx < _nSources) _selectVersionIndex = idx;
          */
          redraw();

      }

      method: setSelectedSource(void; Event event)
      {
          State state = data();
          let rx = event.relativePointer().x,
              margin  = state.config.bevelMargin;

          print("rx: %s\n" % rx);
          if(margin < rx && rx < _taskTbox.x + margin)
          {
            // print("active Task!!\n");
            _activeTaskIndex = _selectTaskIndex;
          }
          else
          {
            // print("active Name!!\n");
            _activeVersionIndex = _selectVersionIndex;
          }

          // _activeVersionIndex = _selectVersionIndex;
          /*
          State state = data();

          let pinfo   = state.pixelInfo,
              iname   = if (pinfo neq nil && !pinfo.empty())
                           then pinfo.front().name
                           else nil;

          let s = sources();

          _activeVersionIndex = _selectVersionIndex;
          let frames = getIntProperty("sequence.edl.frame");

          if (getSessionType() == StackSession)
          {
              setInPoint(1);
              setFrame(frames[1]);
              setOutPoint(frames[_selectVersionIndex + 1] - frames[_selectVersionIndex]);
          }
          else
          {
              setInPoint(frames[_selectVersionIndex]);
              setFrame(frames[_selectVersionIndex]);
              setOutPoint(frames[_selectVersionIndex + 1] - 1);
          }
          _set = true;

          redraw();
          */
      }

      method: eventToIndex(int; Point p)
      {
          State state = data();
          let margin  = state.config.bevelMargin;

          let vd = getTaskDataMuFromSource();

          if (vd eq nil || vd.empty()) return 0;

          if(margin < p.x && p.x < _taskTbox.x + margin)
          {
            // print ("    calling eventToIndex Task\n");
            return vd.taskSize() - int(((p.y - _y + margin ) / _th )) + 1;
          }
          else
          // if(margin < p.x && p.x < _nameTbox.x + margin && p.x > _taskTbox.x + margin)
          {
            // print ("    calling eventToIndex Name\n");
            // print("idx: %s\n" % (vd.nameSize() - int(((p.y - _y + margin ) / _th )) + 1));
            return vd.nameSize() - int(((p.y - _y + margin ) / _th )) + 1;
          }

          // return vd.taskSize() - int(((p.y - _y + margin ) / _th )) + 1;
      }

      method: releaseSelect(void; Event event)
      {
          // print ("releaseSelect\n");
          State state = data();
          let margin  = state.config.bevelMargin;

          let rx = event.relativePointer().x;
          if(margin < rx && rx < _taskTbox.x + margin)
          {
              let ti = eventToIndex(_downPoint),
                  vd = getTaskDataMuFromSource();

              if (vd eq nil || vd.empty()) return;

              if(ti < vd.taskSize() && ti >= 0 && ti != vd._currentTask)
              {
                  // print ("    task ti %s\n" % ti);
                  _selectTaskIndex = ti;
                  setSelectedSource(event);
                  // print ("    calling setTask %s\n" % ti);
                  staticSetTask (ti);
                  // print ("    done\n");
              }
          }
          else
          {
              let ni = eventToIndex(_downPoint),
                  vd = getTaskDataMuFromSource();

              if (vd eq nil || vd.empty()) return;

              if(ni < vd.nameSize() && ni >= 0 && ni != vd._current)
              {
                  // print ("    name ni %s\n" % ni);
                  _selectVersionIndex = ni;
                  setSelectedSource(event);
                  // print ("    calling setVersion %s\n" % ni);
                  staticSetVersion (ni);
                  // print ("    done\n");
              }
          }

          if (!_drawInMargin) release(this, event, nil);
      }

      method: handleMotion(void; Event event)
      {
          // print("handleMotion\n");
          let gp = event.pointer();

          if (!this.contains (gp))
          {
              _selectVersionIndex = _activeVersionIndex ;
              _in = false;
          }
          else
          {
              _in = true;
              let di = eventToIndex(event.pointer()),
                  vd = getTaskDataMuFromSource();

              // print("di: %s\n" % di);

              if (vd eq nil || vd.empty()) return;

              State state = data();
              let margin  = state.config.bevelMargin;

              let rx = event.relativePointer().x;
              if(margin < rx && rx < _taskTbox.x + margin)
              {
                if(di < vd.taskSize() && di >= 0) _selectTaskIndex = di;
                // print("_selectTaskIndex: %s\n" % _selectTaskIndex);
              }
              else
              {
                if(di < vd.nameSize() && di >= 0) _selectVersionIndex = di;
                // print("_selectVersionIndex: %s\n" % _selectVersionIndex);
              }
          }

          State state = data();

          let domain = event.subDomain(),
              p      = event.relativePointer(),
              tl     = vec2f(0, domain.y),
              pc     = p - tl,
              d      = mag(pc),
              m      = state.config.bevelMargin,
              lc     = this._inCloseArea,
              near   = d < m;

          if (near != lc) redraw();
          this._inCloseArea = near;

          redraw();
      }

      method: optFloatingVersioningSelector (void; Event event)
      {
          _drawInMargin = !_drawInMargin;
          writeSetting ("TaskSelector", "selectorIsDocked", SettingsValue.Bool(_drawInMargin));

          if (_drawInMargin) drawInMargin (1);
          else
          {
              drawInMargin (-1);
              vec4f m = vec4f{-1.0, -1.0, -1.0, -1.0};
              m[1] = 0;
              setMargins (m, true);
          }
          redraw();
      }

      method: isFloatingVersioningSelector (int;)
      {
          if _drawInMargin then UncheckedMenuState else CheckedMenuState;
      }

      method: popupOpts (void; Event event)
      {
          popupMenu (event, Menu {
              {"Versions", nil, nil, \: (int;) { DisabledMenuState; }},
              {"_", nil},
              {"Floating Versions Selector", optFloatingVersioningSelector, nil, isFloatingVersioningSelector},
          });
      }

      method: invalidateSetState(void; Event ev)
      {
          _set = false;
          ev.reject();
      }

      method: TaskSelector (TaskSelector; string name)
      {
          deb ("** constructor");
          init(name,
               [ ("pointer-1--push", storeDownPoint(this,), ""),
                 ("pointer--move", handleMotion, ""),
                 ("pointer-1--release", releaseSelect, ""),
                 // ("pointer-1--drag", drag(this,), "Move Selector"),
                 ("pointer--wheelup", selectSource(,1), "Choose Previous Source"),
                 ("pointer--wheeldown", selectSource(,-1), "Choose Next Source"),
                 ("pointer-2--push", setSelectedSource, "Set Selected Source") ,
                 ("pointer-3--push", popupOpts, "Popup Selector Options") ,
                 ("new-in-point",  invalidateSetState, "Invalidate Set State"),
                 ("new-out-point", invalidateSetState, "Invalidate Set State")
                 ],
               false);

          _x = 40;
          _y = 60;
          _activeTaskIndex = _selectTaskIndex = 0;
          _activeVersionIndex = _selectVersionIndex = 0;
          _in = false;

          let SettingsValue.Bool b1 = readSetting ("TaskSelector", "selectorIsDocked", SettingsValue.Bool(false));
          _drawInMargin = b1;

          let pymodule = python.PyImport_Import ("dxVersioning_api");

          _pyGetTaskDataFromSource = python.PyObject_GetAttr (pymodule, "getVersionDataFromSource");
          _pySourceHasTaskData = python.PyObject_GetAttr (pymodule, "sourceHasTaskData");
          _pyResetSourceFrame = python.PyObject_GetAttr (pymodule, "resetSourceFrame");
          _pyStaticSetTask = python.PyObject_GetAttr (pymodule, "staticSetTask");
          _pyStaticSetVersion = python.PyObject_GetAttr (pymodule, "staticSetVersion");

          this.toggle();
          deb ("** constructor complete");
      }

      \: drawNameValuePairsColors (NameValueBounds;
                          StringPair[] tasks,
                          StringPair[] names,
                          int currentTask,
                          Color fg, Color bg,
                          int x, int y, int margin,
                          int maxw=0, int maxh=0,
                          int minw=0, int minh=0,
                          bool nobox=false,
                          Color[] tColors=nil,
                          Color[] nColors=nil)
      {
          m := margin;    // alias

          let (taskTbox, nbounds, vbounds, nw) = nameValuePairBounds(tasks, m);
          let (nameTbox, nameNbounds, nameVbounds, nameNW) = nameValuePairBounds(names, m);

          // print("taskTbox: %s\n" % taskTbox);
          // print("nameTbox: %s\n" % nameTbox);

          let vw      = 0,
              h       = 0,
              a       = gltext.ascenderHeight(),
              d       = gltext.descenderDepth(),
              th      = a - d;

          float
              x0      = x - d,
              y0      = y - m,
              x1      = taskTbox.x + x0,
              y1      = taskTbox.y + y0;

          let xs = x1 - x0,
              ys = y1 - y0;

          if (minw > 0 && xs < minw) x1 = x0 + minw;
          if (minh > 0 && ys < minh) y1 = y0 + minh;
          if (maxw > 0 && xs > maxw ) x1 = x0 + maxw;
          if (maxh > 0 && ys > maxh ) y1 = y0 + maxh;

          taskTbox.x = x1 - x0;   // adjust
          taskTbox.y = y1 - y0;

          glEnable(GL_BLEND);
          glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

          int rooty = y;

          float
              rectx0 = x - d,
              recty0 = y - m,
              rectx1 = 85 + nameTbox.x + rectx0 + 45,
              recty1 = max(taskTbox.y  + recty0, nameTbox.y  + recty0);

          // print("rectx0: %s\n" % rectx0);
          // print("recty0: %s\n" % recty0);
          // print("rectx1: %s\n" % rectx1);
          // print("recty1: %s\n" % recty1);

          if (!nobox) drawRoundedBox(rectx0, recty0, rectx1+13, recty1, m-13, bg, fg * Color(.5,.5,.5,.5));

          glBegin(GL_LINES);
          glVertex(85+x + nw + m/4, y0 + m/2);
          glVertex(85+x + nw + m/4, y1 - m/2);
          glEnd();

          for_index (i; tasks)
          {
              let (n, v)  = tasks[i],
                  bn      = nbounds[i],
                  bv      = vbounds[i],
                  tw      = bn[2] + bn[0];

              let c1 = (if (tColors neq nil) then tColors[i] else fg),
                  c2 = c1 - Color(0,0,0,.25);

              if (tasks.size()-currentTask-1 == i) gltext.color(0,1,1,1);
              else gltext.color(c1);
              gltext.writeAt(x + m/2, y, v);
              gltext.color(fg);

              y += th;
              //if (i == s - 3) y+= m/2;
          }

          y = rooty;

          for_index (i; names)
          {
              let (n, v)  = names[i],
                  bn      = nameNbounds[i],
                  bv      = nameVbounds[i],
                  tw      = bn[2] + bn[0];

              let c1 = (if (nColors neq nil) then nColors[i] else fg),
                  c2 = c1 - Color(0,0,0,.25);

              gltext.color(c2);
              gltext.writeAt(90+x + nw + m/4 + (nameNW - tw), y, n);
              gltext.color(c1);
              gltext.writeAt(90+x + nw + m/4 + nameNW + m/2, y, v);
              gltext.color(fg);

              y += th;
              //if (i == s - 3) y+= m/2;
          }

          glDisable(GL_BLEND);

          // _taskTbox = taskTbox;
          // _nameTbox = nameTbox;

          (taskTbox, nbounds, vbounds, nw);
      }

      method: cleanup (void; )
      {
          this.updateBounds(vec2f {_x,_y}, vec2f{_x,_y});
          drawInMargin(-1);
          runtime.gc.pop_api();
      }

      method: render (void; Event event)
      {
          event.reject();

          extra_commands.updatePixelInfo(nil);

          runtime.gc.push_api(3);
          State state = data();

          let hasVD = false,
          source = "";

        	if (state.pixelInfo neq nil && state.pixelInfo.size() > 0)
        	{
        	    source = nodeGroup(state.pixelInfo[0].node) + "_source";
        	    hasVD = propertyExists (source + ".versioning.task");
        	}
          // print ("source %s hasVD %s\n" % (source, hasVD));

        	if (! hasVD) { cleanup(); return; }

        	rdeb ("render");

        	let vd = getTaskDataMuFromSource(source);

        	if (vd eq nil || vd.empty()) { cleanup(); return; }

        	rdeb ("selIndex %s" % _selectVersionIndex);

        	let domain  = event.domain(),
        	    bg      = state.config.bg,
        	    fg      = state.config.fg;

        	if (_drawInMargin) drawInMargin(1);

        	rdeb ("    resetSourceFrame");
        	rdeb ("    resetSourceFrame %s" % vd._source);
          runtime.gc.push_api(0);
        	resetSourceFrame (vd._source);
          runtime.gc.pop_api();
        	rdeb ("    resetSourceFrame done");

        	rdeb ("    attrs, colors");
        	(string,string)[] attrs;
          (string,string)[] names;
          Color[] tColors = Color[]();
          Color[] nColors = Color[]();
        	// Color[] tColors = if (vd.taskSize() == vd._color.size()) then Color[]() else nil;
          // Color[] nColors = if (vd.nameSize() == vd._color.size()) then Color[]() else nil;

        	for (int i = vd.taskSize()-1; i >= 0; --i)
        	{
        	    attrs.push_back(("     ", vd._task[i]));
              if (i == _selectTaskIndex) tColors.push_back (1.2*(fg));
              else                       tColors.push_back (0.8*(fg));
        	}

          for (int i = vd.nameSize()-1; i >= 0; --i)
        	{
              names.push_back(("     ", vd._name[i]));
    		      if (i == _selectVersionIndex) nColors.push_back (1.2*(fg));
    	        else                          nColors.push_back (0.8*(fg));
        	}

        	gltext.size(state.config.infoTextSize);
        	setupProjection(domain.x, domain.y, event.domainVerticalFlip());

           let nvbT     = nameValuePairBounds(expandNameValuePairs(attrs), margin),
               nvbN     = nameValuePairBounds(expandNameValuePairs(names), margin);
          Vec2 taskTbox = nvbT._0,
               nameTbox = nvbN._0,
               tbox     = vec2f(85+nvbN._0.x+50, nvbT._0.y + nvbN._0.y);
          // print("taskTbox tbox: %s\n" % taskTbox);
          // print("taskTbox tbox: %s\n" % nameTbox);
          // print("total tbox: %s\n" % tbox);

        	let margin  = state.config.bevelMargin,
        	    vs      = viewSize(),
        	    vMargins= margins(),
              targetW = tbox[0] + 1.25*margin,
        	    x       = if (_drawInMargin) then vs[0] - targetW else _x + margin,
        	    yspace  = vs[1]-vMargins[3]-vMargins[2],
        	    midy    = vMargins[3] + yspace/2,
              adjy    = midy - (tbox[1])/2,
        	    targetY = max(vMargins[3] + margin, adjy + margin);

          // print("-------------------------\n");
          // print("nvb1._0: %s\n" % nvb1._0);
          // print("margin: %s\n" % margin);
          // print("vs: %s\n" % vs);
          // print("vMargins: %s\n" % vMargins);
          // print("targetW: %s\n" % targetW);
          // print("targetY: %s\n" % targetY);
          // // let w = max (vMargins[1], targetW);
          // // print("w: %s\n" % w);
          // print("x: %s\n" % x);
          // print("yspace: %s\n" % yspace);
          // print("midy: %s\n" % midy);
          // print("adjy: %s\n" % adjy);
          // print("_drawInMargin: %s\n" % _drawInMargin);

        	if (_drawInMargin)
        	{
        	    _y = targetY - margin;
        	    let w = max (vMargins[2], targetW);
        	    glColor(Color(0,0,0,1));
        	    glBegin(GL_QUADS);
        	    glVertex(vs[0]-w, vs[1]-vMargins[2]);
        	    glVertex(vs[0], vs[1]-vMargins[2]);
        	    glVertex(vs[0], vMargins[3]);
        	    glVertex(vs[0]-w, vMargins[3]);
        	    glEnd();
        	}

        	rdeb ("    draw pairs");
        	let y       = _y + margin,
        	    nvb     = drawNameValuePairsColors(expandNameValuePairs(attrs), expandNameValuePairs(names), vd._currentTask, fg, bg, x, y, margin,
        		    0, 0, 0, 0, _drawInMargin, tColors, nColors),
        	    emin    = vec2f(if (_drawInMargin) then vs[0] - targetW else _x, _y),
        	    emax    = emin + tbox + vec2f(margin + (if (_drawInMargin) then margin/4 else margin), 0.0);

        	let fa = int(gltext.ascenderHeight()),
        	    fd = int(gltext.descenderDepth()),
        	    th = fa - fd,
        	    gx = x + 112 + margin/2;

        	_th = th;
          _taskTbox = vec2f(taskTbox.x + 40, taskTbox.y);
          _nameTbox = nameTbox;

        	glEnable(GL_POINT_SMOOTH);
        	glPointSize(6.0);
        	glBegin(GL_POINTS);

        	rdeb ("    draw points");
        	if (vd._current != vd._last)
        	{
        	    glColor(Color(.40,.40,.09,1));
        	    let gy = y -3 + th * (vd.nameSize() - vd._last -1 ) + fd + th/2 + 2.0;
        	    glVertex(gx, gy);
        	}

        	glColor(Color(.75,.75,.15,1));
        	let gy = y -3 + th * (vd.nameSize() - vd._current - 1) + fd + th/2 + 2.0;
        	glVertex(gx, gy);
        	glEnd();

        	if (_inCloseArea && !_drawInMargin)
        	{
        	    drawCloseButton(x - margin/2,
        			    tbox.y + y - margin - margin/4,
        			    margin/2, bg, fg);
        	}
        	if (!_in)
          {
            _selectVersionIndex = vd._current;
            _selectTaskIndex = vd._currentTask;
          }

        	// print ("update bounds %s %s\n" % (emin, emax));
        	this.updateBounds(emin, emax);
        	runtime.gc.pop_api();
        	rdeb ("    done");

          // let rx = event.pointer();
          // print("rx: %s\n" % rx);
      }
  }

  function: createMode (Mode;)
  {
      return TaskSelector("task-selector");
  }

  function: theMode (TaskSelector; )
  {
      TaskSelector m = rvui.minorModeFromName("task-selector");

      return m;
  }

  function: selectorIsActive (bool; )
  {
      let m = theMode();

      return (m neq nil && m._active);
  }

  function: toggleSelector (void; )
  {
      deb ("toggleSelector");
      theMode().toggle();
      deb ("toggleSelector complete");
  }

}
