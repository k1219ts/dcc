#coding:utf-8
from __future__ import print_function
import DXRulebook.Rulebook as Rulebook
import os

# ------------------------------------------------------------------------------
# load rulebook

# find DXRulebook.yaml
_RBYAML = os.path.dirname(os.path.realpath(__file__))
_RBYAML = os.path.join(_RBYAML, 'resources/DXRulebook.yaml')
# rulebook instance
_RBROOT = Rulebook.Coder()

def __LinearizeDict(d, res, key=''):
    updatable = dict()
    for k, child in d.items():
        if isinstance(child, dict):
            __LinearizeDict(child, res, '%s %s'%(key, k))
        else:
            updatable[k] = d.pop(k)
    res.append((key.strip(), updatable))


def Reload(addition=None):
    global _RBROOT, _CATEGORY, _DCCS, _FLAGS, _TAGS
    _RBROOT = Rulebook.Coder()

    yamls = []
    if addition and os.path.exists(addition):
        yamls.append(addition)

    for f in (os.getenv('DXRULEBOOKFILE') or '').split(':'):
        if f and os.path.exists(f):
            yamls.append(f)

    datas = _RBROOT.load_yaml(_RBYAML)
    for f in yamls:
        extra = _RBROOT.load_yaml(f)

        res = []
        __LinearizeDict(extra, res)
        res.reverse()

        for key, data in res:
            if not data: continue
            if not key:
                data1.update(data)
            else:
                dst = datas
                for k in key.split(' '):
                    if not dst.has_key(k):
                        dst[k] = dict()
                    dst = dst[k]
                dst.update(data)

    _RBROOT.update_rule(datas, recursive=True)

    _CATEGORY = _RBROOT.tag['CATEGORY'].value.split('|')
    _DCCS     = _RBROOT.tag['DCC'].value.split('|')
    _FLAGS    = _RBROOT.allFlags
    _TAGS     = _RBROOT.allTags

Reload()




# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 만약 파이프라인에서 특정 플래그나 값이 있을 때, 변경이 필요하다면, 여기 함수 변경.
def CustomChangingFlags(res, rb):
    # ----------------------------------------------------------------------
    # TODO: for ZN, should be removed
    if res.has_key('task') and res['task'] == 'ZN':
        res['task'] = rb.tag['GROOM'].value
    # ----------------------------------------------------------------------

    if res.has_key('set'):
        res['task'] = rb.tag['MODEL'].value

    if res.has_key('root'):
        root = _RBROOT.D.OTHER.decode(res['root'], 'EXCLUSIVEROOT')
        res['root'] = root['root']

    return res
# ------------------------------------------------------------------------------

def FindDccFromFile(f):
    ext = f.split('.')[-1]
    for child in _RBROOT.F.children:
        child = _RBROOT.F.child[child]
        try:
            child.decode(ext, 'CHECKEXT')
            return child.name
        except:
            continue
    return None


def MatchFlag(f, v, search=False):
    if not f in _FLAGS.keys():
        return None
    res = _FLAGS[f].is_valid_value(v)
    if res:
        if search:
            return res.group()
        else:
            return v == res.group()
    else:
        return False


class _BaseFlagDict(dict):
    def __setattr__(self, k, v):
        if k in _FLAGS.keys():
            dict.__setitem__(self, k, v)
        else:
            self.__dict__[k] = v

    def __setitem__(self, k, v):
        self.__setattr__(k, v)

    def __getattr__(self, k):
        if k in _FLAGS.keys():
            if self.has_key(k):
                return dict.__getitem__(self, k)
            else:
                return
        else:
            return self.__dict__[k]


class _BaseTagDict(dict):
    def __setattr__(self, k, v):
        if k in _TAGS.keys():
            self[k] = v
        else:
            self.__dict__[k] = v

    def __getattr__(self, k):
        if k in _TAGS.keys():
            return dict.__getitem__(self, k)
        else:
            return self.__dict__[k]


class _Products(object):
    def __init__(self, category, flag={}, dcc=None):
        '''
        [Arguments]
        flag (dict)      :
        defaultDCC (str) :
        '''
        self._category = category
        self._flag     = flag
        self._products = list()

        self._rb       = _RBROOT.child[category]
        self._dcc      = dcc if dcc in self._rb.children else None

        self._initialize(self, self._rb)

    def _initialize(self, product, rb, products=[]):
        for child in rb.children:
            _prodList = products + [child]

            newProduct = _Products.Product(self, _prodList)
            product.__dict__[child] = newProduct

            self._initialize(newProduct, rb.child[child], _prodList)

    class Product:
        def __init__(self, root, products):
            self._root     = root
            self._products = products

        def __getattr__(self, k):
            if self.__dict__.has_key(k):
                return self.__dict__[k]
            else:
                self._root._products = list(self.__dict__['_products'])
                return getattr(self._root, k)

        def __getitem__(self, k):
            return getattr(self, k)

    def __getattr__(self, k):
        if self.__dict__.has_key(k):
            return self.__dict__[k]
        elif self._dcc and self[self._dcc].__dict__.has_key(k):
            return self[self._dcc][k]
        else:
            # ------------------------------------------------------------------
            # Encode
            # ------------------------------------------------------------------
            rb = self._get_rulebook()

            _Products.SetFlags(rb, self._flag)
            return rb.product[k]

    def __getitem__(self, k):
        return self.__getattr__(k)

    def _get_rulebook(self):
        dccs = self._rb.children
        # add default dcc if nessecially
        if self._dcc:
            if not self._products or self._products[0] not in dccs:
                self._products.insert(0, self._dcc)
        # find product in rulebook
        rb = self._rb
        rb.resetFlags()
        for child in self._products:
            # if it has reference flag, then set the value to flag.
            if rb.ref:
                self._flag[rb.ref] = child
            rb = rb.child[child]
            rb.resetFlags()

        # reset self._products
        self._products = list()
        return rb

    @staticmethod
    def SetFlags(rb, kwargs):
        # reset rulebook
        # rb.resetFlags()

        # set flags to rulebook
        for k in kwargs.keys():
            if k in rb.flags:
                if isinstance(kwargs[k], (unicode, str)) and \
                   rb.flag[k].is_valid_value(kwargs[k]):
                    rb.flag[k] = kwargs[k]

        # treat abname
        if 'abname' in rb.flags:
            if isinstance(kwargs, Flags):
                if kwargs.IsBranch():
                    rb.flag['abname'].value = rb.flag['branch'].value
                else:
                    rb.flag['abname'].value = rb.flag['asset'].value
            else:
                if rb.flag['branch'].value != Rulebook.RBNONE:
                    rb.flag['abname'].value = rb.flag['branch'].value
                else:
                    rb.flag['abname'].value = rb.flag['asset'].value

    @staticmethod
    def TreatABName(res, isBranch):
        if res.has_key('abname'):
            if isBranch:
                res['branch'] = res.pop('abname')
                res['__isbranch__'] = True
            else:
                res['asset'] = res.pop('abname')
                res['__isbranch__'] = False
        else:
            res['__isbranch__'] = res.has_key('branch')

        return Parsed(res)

    def Decode(self, src, product=None, err=True, set=False):
        # get rulebook
        rb  = self._get_rulebook()
        _Products.SetFlags(rb, self._flag)
        # decode from rulebook
        try:
            res = rb.decode(src, product)
        except ValueError:
            res = dict()
        except Exception as E:
            raise(E)

        # 결과 res에서 abname이 있으면, asset 또는 branch로 변경해야 한다. 만약,
        # self._flag가 Flags 인스턴스이면, IsBranch()로 알 수 있지만, 그렇지 않는
        # 경우는 일반적인 dictionary이므로 branch 키의 여부로 branch임을 확인한다.
        if isinstance(self._flag, Flags):
            isBranch = self._flag.IsBranch()
        else:
            isBranch = self._flag.has_key('branch')

        res = _Products.TreatABName(res, isBranch)
        res = CustomChangingFlags(res, rb)

        # check result
        if err and not res.Success():
            msg = 'Decode@_Products in DXRulebook :' + \
                  'Cannot decode given value.\n' + \
                  'Input    >> %s\n'%src + \
                  'Rulebook >> %s\n'%(rb.name+'.'.join(self._products)) + \
                  'Product  >> %s\n'%(product if product else res.product) + \
                  'Decode   >> %s\n'%res + \
                  'Rest     >> %s'%res.rest
            raise ValueError(msg)

        # update self's argument dictionary
        if set:
            self._flag.update(res)

        if self._flag.has_key('ext'):
            self._flag.pop('ext')

        return res

    def SetDecode(self, src, product=None, err=True):
        return self.Decode(src, product, True, err)


class Tags:
    def __init__(self, dcc=None, _parent=None):
        '''
        [Arguments]
        dcc (str)      : Default dcc
        _parent (Tags) : Parent Tags (internally use)
        '''
        self._dcc      = dcc
        self._parent   = _parent
        self._tags     = _BaseTagDict()

        if not isinstance(_parent, Tags):
            self.__update(_RBROOT)

    @property
    def tags(self):
        return self._tags.keys()

    @property
    def tag(self):
        return self._tags

    def __update(self, rb=None):
        if self._dcc and self._dcc not in _DCCS:
            raise KeyError('__update@Tags :',
                           'This default dcc must be one of belows.\n',
                           '%s'%str(_DCCS))

        def setTags(rb, attr=None):
            tags = self
            if attr:
                if self.__dict__.has_key(attr):
                    tags = self.__dict__[attr]
                else:
                    tags = Tags(_parent=self)
                    self.__dict__[attr] = tags
            # update tag
            if rb:
                for k in rb.myTags:
                    val = rb.tag[k].value
                    if '|' in val:
                        val = val.split('|')
                    tags._tags[k] = val

        # set root tag
        setTags(rb)
        # set child tags
        for c in _CATEGORY:
            rb_c = rb.child[c]
            setTags(rb_c, c)
            for d in _DCCS:
                rb_d = rb_c.child[d] if d in rb_c.children else None
                setTags(rb_d, d)

    def __getattr__(self, k):
        if self.__dict__.has_key(k):
            return self.__dict__[k]
        elif self._dcc and self.__dict__[self._dcc]._tags.has_key(k):
            return getattr(self.__dict__[self._dcc], k)
        elif self._tags.has_key(k):
            return self._tags[k]
        else:
            raise KeyError('__getattr__@Tags : Given tag is %s'%k)

    def __getitem__(self, k):
        return self.__getattr__(k)


class Parsed(_BaseFlagDict):
    def __init__(self, d={}):
        self.__product__  = None
        self.__rest__     = ''
        self.__isbranch__ = None

        for k in d.keys():
            if k.startswith('__'):
                self.__dict__[k] = d.pop(k)

        self.update(d)

    @property
    def product(self):
        return self.__product__

    @property
    def rest(self):
        return self.__rest__

    def ABName(self):
        if self.has_key('branch') and self.IsBranch():
            return self.branch
        elif self.has_key('asset'):
            return self.asset
        else:
            return ''

    def Success(self):
        if self.product:
            return True
        return False

    def IsBranch(self):
        return self.__isbranch__

    def AsBranch(self):
        self.__isbranch__ = True

    def IsShot(self):
        if self.has_key('shot'):
            return not self.has_key('asset')
        else:
            return False

    def Copy(self):
        return Parsed(self)

class Coder:
    def __init__(self, category=None, dcc=None, pub=None, **kwargs):
        self._category = category
        self._dcc = dcc
        self._pub = pub
        self._product = None
        self._path = list()
        self._preserved = kwargs

    def __getattr__(self, k):
        if k.startswith('_'):
            return self.__dict__[k]
        else:
            self._path.append(k)
            return self

    def __getitem__(self, k):
        return self.__getattr__(k)

    def _getRulebook(self, args={}):
        # find category and dcc
        rb  = _RBROOT
        rb.resetFlags()
        for k in [self._category, self._dcc]:
            if self._path and self._path[0] in rb.children:
                rb = rb.child[self._path.pop(0)]
                rb.resetFlags()
            elif k and k in rb.children:
                rb = rb.child[k]
                rb.resetFlags()

        # find task
        for t in self._path:

            if t in rb.children:
                rb = rb.child[t]
            elif t in rb.products:
                self._product = t
                break
            elif rb.combiner:
                self._product = t
            else:
                raise KeyError('%s not in %s'%(t, rb.name))
        # reset path
        self._path = list()
        return rb

    def Decode(self, val, isBranch=False):
        rb  = self._getRulebook()
        res = rb.decode(val, self._product)
        res = CustomChangingFlags(res, rb)

        self._product = None

        return _Products.TreatABName(res, isBranch)

    def Encode(self, **kwargs):
        rb  = self._getRulebook(kwargs)
        if 'pub' in rb.flags and not kwargs.has_key('pub'):
            kwargs['pub'] = self._pub

        flags = dict(self._preserved)
        flags.update(kwargs)
        _Products.SetFlags(rb, flags)
        res = None
        if self._product:
            res = rb.product[self._product]
        elif rb.combiner:
            res = rb.combine(self._product)
        else:
            raise ValueError('This rulebook(%s) needs product name'%rb.name)
        # reset all
        self._product = None
        return res

    def Rulebook(self):
        return self._getRulebook()


class Flags(_BaseFlagDict):
    def __init__(self, dcc=None, category=_CATEGORY, **kwargs):
        # find pre-defined flags
        self.__predefined = self.keys()
        # check it has branch
        self.__dict__['__isbranch'] = False

        # add category rule books
        for c in category:
            self.__dict__['__%s'%c] = _Products(c, self, dcc)

        # update kwargs to this
        self.update(kwargs)

    def __setattr__(self, k, v):
        if k == 'branch' and v:
            self.AsBranch()
        _BaseFlagDict.__setattr__(self, k, v)

    def __setitem__(self, k, v):
        if k == 'branch' and v:
            self.AsBranch()
        _BaseFlagDict.__setitem__(self, k, v)

    def __getattr__(self, k):
        if k in _CATEGORY:
            return self.__dict__['__%s'%k]
        else:
            return _BaseFlagDict.__getattr__(self, k)

    def get(self, k, dv=None):
        try:
            return self.__getattr__(k)
        except:
            return dv

    def update(self, d):
        for k, v in d.items():
            _BaseFlagDict.__setitem__(self, k, v)

    def has_flag(self, k):
        return self.has_key(k)

    def has_attr(self, k):
        return self.__dict__.has_key(k)

    def pop(self, k):
        if self.has_key(k):
            return _BaseFlagDict.pop(self, k)
        return None

    def Set(self, **kwargs):
        self.update(kwargs)

    def Switch(self, **kwargs):
        res = Parsed(self)
        for k, v in kwargs.items():
            res[k] = v
        return res

    def Reset(self, remove=True, predefined=False):
        # reset branch
        self.AsAsset()
        # reset flags
        for k, v in self.items():
            if not predefined and k in self.__predefined:
                continue
            if remove:
                self.pop(k)

    def HasBranch(self):
        return self.has_key('branch') and self.branch

    def HasAsset(self):
        return self.has_key('asset') and self.asset

    def IsBranch(self):
        return self.__dict__['__isbranch']

    def IsAsset(self):
        return not self.__dict__['__isbranch']

    def AsBranch(self, v=True):
        self.__dict__['__isbranch'] = v

    def AsAsset(self, v=True):
        self.__dict__['__isbranch'] = not v

    def IsShot(self):
        if self.shot:
            return not self.HasAsset()
        else:
            return False

    def ABName(self):
        if self.IsBranch():
            return self.branch
        else:
            return self.asset
