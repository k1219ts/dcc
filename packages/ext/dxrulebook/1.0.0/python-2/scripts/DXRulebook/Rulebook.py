#coding:utf-8
'''
    Copyright & Author: 2016, Sehwi Park <sehwida@gmail.com>

    Description:
        Names and Paths generator class
'''
from __future__ import print_function
import os
import re
import yaml
from copy import deepcopy
from collections import defaultdict
from .Utilities import libpath
from .Parser import substitute_keys

RBNONE        = '!!none'
RBPRODUCTNAME = '__product__'
RBRESTNAME    = '__rest__'

class Coder(object):
    '''
    Long Name Generator
    '''

    def __init__(self, family='Global', env=os.environ, parent=None):
        '''
        Args:
            family (str): family name of this Coder instance
            env (dict): dictionary of environment variables
            parent (Coder): parent instance to inherit attributes
        '''
        if parent and not isinstance(parent, Coder):
            msg = 'Parent instance should be the same type: %s' % str(parent)
            raise TypeError(msg)

        self._family = family
        self._env = env
        self._flag = defaultdict(Coder.Flag)  # flag instances
        self._myFlags = []
        self._flag_raw_data = {}  # configuration data for flags
        self._product = {}  # product rules
        self._myProduct = {}  # don't have parent product rules
        self._child = {}  # child instances
        self._rule_raw_data = {}  # raw data of naming rules
        self._tag = defaultdict(Coder.Tag)
        self._myTags = []
        self._tag_raw_data = {}  # configuration data for flags
        self._combiner = None
        self._childRefFlag = None

        self._parent = parent
        if parent:
            self._inherit(parent)

        self._default_keys = self.__dict__.keys() + Coder.__dict__.keys()
        return

    def _inherit(self, parent):
        '''
        Inherit naming rules from parent
        Args:
            parent (Coder): parent instance to inherit naming rules
        '''
        for key, flag in parent._flag.iteritems():
            self._flag[key] = flag
            flag._users.append(self)

        for key, tag in parent._tag.iteritems():
            self._tag[key] = tag
            tag._users.append(self)

        self._product = deepcopy(parent._product)

        return

    @property
    def name(self):
        return self._family

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        '''
        Returns:
            (list) children names
        '''
        return self._child.keys()

    @property
    def ref(self):
        '''
        Returns:
            (str) reference flag
        '''
        return self._childRefFlag

    @property
    def child(self):
        '''
        Returns:
            (dict) dictionary of child instances
        '''
        return dict(**self._child)

    def add_child(self, name, rule={}):
        '''
        Add Coder instance as achild

        Args:
            name (str): new family member's name
            rule (dict): new naming rule
        Returns:
            (Coder) a new coder instance
        '''
        if name in self._default_keys:
            msg = 'Reserved keyword [%s]' % ', '.join(self._default_keys)
            raise NameError(msg)

        if name == '__ref__':
            self._childRefFlag = rule
            return

        child = self.__class__(name, parent=self)
        child.update_rule(rule, recursive=True)
        self._child[name] = child

        # register as class attribute (direct access)
        self.__dict__[name] = child

        return child

    @property
    def flags(self):
        '''
        Returns:
            (list) flag names
        '''
        return self._flag.keys()

    @property
    def myFlags(self):
        '''
        Returns:
            (list) flags only this has
        '''
        return list(self._myFlags)

    @property
    def allFlags(self):
        '''
        Returns:
            (dict) all flags including children
        '''
        res = {}
        children = [self]
        while children:
            child = children.pop(0)
            res.update({_:child.flag[_] for _ in child.myFlags})
            children.extend([child.child[_] for _ in child.children])
        return res

    @property
    def flag(self):
        '''
        Returns:
            (FlagDict) flag setter dictionary
        '''
        return Coder.FlagDict(self._flag)

    def resetFlags(self):
        for flag in self.flags:
            self.flag[flag].reset()


    @property
    def tags(self):
        '''
        Returns:
            (list) tag names
        '''
        return self._tag.keys()

    @property
    def tag(self):
        '''
        Returns:
            (dict) tag
        '''
        return Coder.TagDict(self._tag)

    @property
    def myTags(self):
        '''
        Returns:
            (list) my tag names
        '''
        return list(self._myTags)

    @property
    def allTags(self):
        '''
        Returns:
            (dict) all flags including children
        '''
        res = {}
        children = [self]
        while children:
            child = children.pop(0)
            res.update({_:child.tag[_] for _ in child.myTags})
            children.extend([child.child[_] for _ in child.children])
        return res

    @property
    def products(self):
        '''
        Returns:
            (list) product names
        '''
        return self._product.keys()

    @property
    def myProducts(self):
        '''
        :return:
         (list) product names but don't return parent products
        '''
        return self._myProduct.keys()

    @property
    def product(self):
        '''
        Returns:
            (ProductDict) product value generator dictionary
        '''
        rulebook = self
        while rulebook.ref:
            child = rulebook.flag[rulebook.ref].value
            if child in rulebook.children:
                rulebook = rulebook._child[child]
            else:
                break
        return Coder.ProductDict(rulebook._product, rulebook)

    @property
    def combiner(self):
        '''
        Returns:
            (tuple) combiner list
        '''
        return self._combiner

    def _get_product_value(self, product_name='path'):
        '''
        Args:
            product_name (str): product name to generate (default: 'path')
        Returns:
            (str) generate name by flag and product expressions
        '''
        if self.combiner:
            return self.combine(product_name)
        else:
            return self.encode(product_name)

    @property
    def environ(self):
        '''
        Returns:
            (dict) current Coder's environment variables
        '''
        return deepcopy(self._env)

    @environ.setter
    def environ(self, environ_dict):
        if not isinstance(environ_dict, dict):
            raise TypeError('Environment variable accepts only a dictionary')
        self._env = environ_dict

    def _encode(self, expression):
        '''
        Directly encode an expression

        Args:
            expression (str): dxname name expression
        Returns:
            (str) generated name via flags and environment variables
        '''
        prod_extended   = self._substitute_product(expression)
        tag_expression  = self._substitute_tag(prod_extended)
        flag_expression = self._substitute_flag(tag_expression)
        return self._substitute_environment(flag_expression)

    def encode(self, product_name=None):
        expressions = self._product.get(product_name, '')

        if not expressions:
            msg = 'Undefined product "%s" ' % product_name
            raise KeyError(msg)
        elif not isinstance(expressions, list):
            expressions = [expressions]

        for expression in expressions:
            try:
                return self._encode(expression)
            except ValueError:
                continue
            except Exception as E:
                raise(E)
        else:
            msg = 'Cannot encode : %s\n    (%s)'%(product_name, str(expression))
            raise ValueError(msg)

    def combine(self, product_name=None):
        target = None
        combiner = self.combiner.combines

        if product_name:
            if product_name in combiner:
                combiner = combiner[:combiner.index(product_name)+1]
            else:
                for i, k in enumerate(combiner):
                    for prod in self.combiner.products(k):
                        if product_name in prod:
                            combiner = combiner[:i+1]
                exp = self._product.get(product_name, '')
                if not exp:
                    msg = 'Undefined product "%s" ' % product_name
                    raise KeyError(msg)
                else:
                    target = self._encode(exp)

        res = ''
        for i, k in enumerate(combiner):
            for exp in self.combiner.products(k):
                if i == len(combiner) - 1 and k != product_name:
                    if product_name and not exp.endswith('<%s>'%product_name):
                        continue
                try:
                    val = self._encode(exp)
                except ValueError:
                    continue
                except Exception as E:
                    raise(E)

                if target:
                    # regex = re.search(self.combiner.addSeper(target),
                    #                   self.combiner.addSeper(val))
                    # if regex:
                    if target == val:
                        val = val.split(target)[0] + target

                        return self.combiner.addString(res, val)

                res = self.combiner.addString(res, val)
                break
        return res

    def decode(self, product=None, product_name=None):
        '''
        Extract flag values from product
        Args:
            product (str): string in format of the product rule
            product_name (str): product name for naming rule
        Returs:
            (dict) flag names and values
        '''

        rulebook = self
        while rulebook.ref:
            child = rulebook.flag[rulebook.ref].value
            if child in rulebook.children:
                rulebook = rulebook._child[child]
            else:
                break

        if rulebook.combiner:
            return rulebook._combiner_decode(product, product_name)
        else:
            if not product_name:
                prods  = rulebook.myProducts
                prods += list(set(rulebook.products) - set(prods))
                for p in prods:
                    try:
                        return rulebook._decode(product, p)
                    except ValueError:
                        continue
                    except Exception as E:
                        raise(E)
                else:
                    msg = 'Value does not match with the product expression: \n'
                    msg += '    (%s) ' % product
                    raise ValueError(msg)
            else:
                return rulebook._decode(product, product_name)

    def _check_decodeResult(self, result, product_name):
        ret_val = dict()

        for key, val in result.iteritems():
            pattern = Coder.DecoderPatternDict.suffix + '[0-9]+'
            key_clean = re.sub(pattern, '', key)

            if not key == key_clean:
                if not result.get(key_clean) == val:
                    msg = 'Unmatching values for one flag. \n'
                    msg += '    product (%s): ' % product_name
                    msg += '\n    flag (%s): ' % (key_clean)
                    msg += '%s <-> %s' % (val, result.get(key_clean))

                    raise ValueError(msg)
            else:
                ret_val[key] = val
                ret_val[RBPRODUCTNAME] = product_name
        return ret_val

    def _combiner_decode(self, product, product_name, customdir=None):
        _orgproduct = product
        res = {RBRESTNAME:''}
        pname = []
        doubleCheck = None
        lastIdx = len(self.combiner.combines) - 1

        def _search(exp, product, res):
            regex = re.search(exp, product)
            # print('>>>>> rigex :', regex.groupdict() if regex else 'None')
            if not regex:
                raise ValueError()

            # search로 찾은 문자열은 반드시 combiner의 구분자와 딱 맞게 잘려야 한다.
            # 예를 들어. /abc/def/ghi 에서 def/ghi를 찾으면 구분자(/)에 맞게 찾은거지만
            # 찾는 문자열이 df/ghi여서 남는 문자열이 /abc/d 가 된다면, 잘못 찾은
            # 경우가 된다. 이를 아래 구문에서 걸러주는 역활을 한다.
            splits = product.split(regex.group())
            if splits[0] and not splits[0].endswith(self.combiner.sep):
                raise ValueError()

            res.update(self._check_decodeResult(regex.groupdict(),
                                                product_name))
            return regex.group()

        # decode conbiner
        combines = self.combiner.combines
        combine_name = None

        if customdir:
            combines = combines[1:]
            res.update(self._check_decodeResult({'customdir':customdir},
                                                product_name))
            pname.append('CUSTOMDIR')
            doubleCheck = '^(?P<customdir>/[a-z.A-Z0-9_/]*)/'

        if product_name in combines:
            idx = combines.index(product_name)
            combines = combines[:idx+1]
            combine_name = product_name

        if product_name and self.combiner.sep in product_name:
            product_name = product_name.split(self.combiner.sep)
            product_name = ['<%s>'%_ for _ in product_name]
            product_name = self.combiner.sep.join(product_name)
        else:
            product_name = '<%s>'%product_name

        for k in combines:
            products = self.combiner.products(k)

            if product_name in self.combiner.products(k):
                products = [product_name]

            for p in products:
                exp, f = self._decode_expression(exps=p,
                                                 s='' if doubleCheck else '^',
                                                  e=self.combiner.sep)[0]

                # add seperator to product words( /a/b > /a/b/)
                product = self.combiner.addSeper(product, s=False)

                # print('-------------------------------------------------------')
                # print('>>>>>', p)
                try:
                    search = _search(exp, product, res)
                except ValueError:
                    continue
                except Exception as E:
                    raise(E)

                # print('>>>>>', res)

                if self.combiner.combines.index(k) < lastIdx and\
                   product == search:
                    doubleCheck = exp
                else:
                    splits = product.split(search)
                    if splits[0] and not splits[0].endswith(self.combiner.sep):
                        continue

                    if doubleCheck:
                        _search(doubleCheck, splits[0], res)

                    product = splits[-1]
                    doubleCheck = None

                pname.append(p.replace('<', '').replace('>', ''))

                # found something and break
                if  p == product_name:
                    res[RBPRODUCTNAME] = self.combiner.sep.join(pname)
                    return res
                else:
                    break
            else:
                # not match in combin's products, but if doubleCheck has value,
                # it means already found all keywords
                if doubleCheck:
                    res[RBPRODUCTNAME] = self.combiner.sep.join(pname)
                    return res

            if not product or k == combine_name:
                res[RBPRODUCTNAME] = self.combiner.sep.join(pname)
                return res

        # left some words in product
        rest = [v for v in product.split(self.combiner.sep) if v]

        # if rest exists, one more find with customdir
        if rest and not customdir and len(pname) == 1:
            return self._combiner_decode(_orgproduct, product_name, _orgproduct)

        res[RBRESTNAME] = self.combiner.sep.join(rest)
        return res


    def _decode(self, product, product_name):
        for decode_exp, flag_exp in self._decode_expression(product_name):
            regex = re.match(decode_exp, product)

            if not regex:
                continue

            try:
                return self._check_decodeResult(regex.groupdict(),
                                                product_name)
            except ValueError:
                continue
            except Exception as E:
                raise(E)
        else:
            msg = 'Value does not match with the product expression: \n'
            msg += '    (%s) ' % product
            raise ValueError(msg)

    def _decode_expression(self, product_name='path', exps=None, s='^', e='$'):
        '''
        Args:
            product_name (str): product key to make decode pattern
        Returns:
            (str, str) decode expression, and original flag expression
        '''
        if exps:
            exps = exps
        else:
            exps = self._product.get(product_name, '')

        if not isinstance(exps, list):
            exps = [exps]

        res = []
        for exp in exps:
            prod_extended = self._substitute_product(exp)
            tag_expression = self._substitute_tag(prod_extended)
            flag_expression = self._substitute_environment(tag_expression)

            for char in ('\.',):
                # escape wild characters
                flag_expression = re.sub(char, char, flag_expression)

            flags_decoder = Coder.DecoderPatternDict()
            for key, flag in self._flag.iteritems():
                flags_decoder[key] = flag.decode_pattern

            decode_exp = substitute_keys('\(([a-zA-Z0-9-_]+)\)',
                                         flags_decoder, flag_expression,
                                         once=True)
            res.append((s + decode_exp + e, flag_expression))

        return res

    class DecoderPatternDict(dict):
        '''
        Special key iterator to avoid duplication of flag keyword for decoding
        '''

        suffix = '_' * 10

        def __init__(self):
            self._key_count = defaultdict(int)

            dict.__init__(self)

        def __getitem__(self, key):
            '''
            Returns:
                (str) added suffix at multiple calls for each keyword
            '''
            val = dict.__getitem__(self, key)

            count = self._key_count[key]
            count += 1
            if count > 1:
                val = re.sub(key, key + self.suffix + '%d' % count, val)

            self._key_count[key] = count

            return val

    class ProductDict(dict):
        '''
        Convenience class to get product output easier
            * use get() method to get the raw expression of a product
        '''

        def __init__(self, source_dict, parent):
            self.update(source_dict)
            self._parent = parent

        def __getitem__(self, key):
            return self._parent._get_product_value(key)

        def __setitem__(self, key, value):
            self._parent._product.__setitem__(key, value)
            dict.__setitem__(key, value)

    class FlagDict(dict):
        '''
        Convenience class to set flag values easier
        '''

        def __init__(self, source_dict):
            self.update(source_dict)

        def __setitem__(self, key, value):
            self.__getitem__(key).value = value

    class TagDict(dict):
        '''
        Convenience class to set tag values easier
        '''

        def __init__(self, source_dict):
            self.update(source_dict)

        def __setitem__(self, key, value):
            self.__getitem__(key).value = value

    class Combiner(object):
        def __init__(self):
            self._container = None
            self.sep  = None
            self.combines   = list()
            self._products  = dict()
            self._name      = None

        def products(self, k):
            return self._products[k]

        def initialize(self, container, data, name):
            if not data:
                return

            self._name = name
            self._container = container

            if data.has_key('seperator'):
                self.sep = data.pop('seperator')
            else:
                msg = 'Must set sep in __combiner__(%s)'%self._name
                raise ValueError(msg)

            if data.has_key('combine'):
                self.combines = data.pop('combine').split(self.sep)
            else:
                msg = 'Must set combine in __combiner__(%s)'%self._name
                raise ValueError(msg)

            self._products = data
            for k in self.combines:
                if not data.has_key(k):
                    msg = '__combiner__(%s) has no %s attr.'%(self._name, k)
                    raise KeyError(msg)
            return

        def addString(self, res, val):
            if not val.startswith(self.sep):
                val = self.sep + val
            if val[-1] == self.sep:
                val = val[:-1]
            return res + val

        def addSeper(self, val, s=True, e=True):
            if s and not val.startswith(self.sep):
                val = self.sep + val
            if e and not val.endswith(self.sep):
                val += self.sep
            return val

        def __len__(self):
            return len(self.combines)

        def __iter__(self):
            for combine in self.combines:
                yield combine

    class Tag(object):
        name_pattern = re.compile('[A-Z_0-9]+$')

        def __init__(self):
            self._container = None
            self._users = list()
            self._name = None
            self._val = None
            self._rule = None

            return

        def __deepcopy__(self, memo={}):
            '''
            deepcopy constructor
            '''
            inst = self.__class__()
            inst.initialize(self._container, self._name, self._val)

            return inst

        def __str__(self):
            '''
            print or str casting
            '''
            return self.value

        def initialize(self, container, name, val):
            '''
            Initialize or Update tag instance
            '''

            if not self.name_pattern.match(name):
                msg = 'Invalid Flag name: ' + str(name)
                msg += '    Flag name should match to '
                msg += str(self.name_pattern.pattern[:-1])

                raise ValueError(msg)

            self._container = container
            self._name = name
            self._val = val

            return

        @property
        def name(self):
            return self._name

        @property
        def value(self):
            '''
            Returns:
                (str) current flag value
            '''
            return self._val

        @value.setter
        def value(self, new_value):
            '''
            Args:
                new_value (str): new flag value
            '''
            self._val = new_value

            return

        @property
        def rule(self):
            return self._rule

        @rule.setter
        def rule(self, new_rule):
            self._rule = new_rule
            return

        pass  # end of class Flag


    class Flag(object):
        '''
        Data class for flags
        Properties:
            default (str): default value or expression
            pattern (str): regular expression for validity
            value (str): current flag value
            name (str): flag name (accepts only [a-zA-Z0-9_]+)
        '''

        name_pattern = re.compile('[a-zA-Z0-9_]+$')

        def __init__(self):
            self._container = None
            self._users = list()
            self._name = None
            self._default = None
            self._pattern_rex = None
            self._default_val = None
            self._val = None

            return

        def __deepcopy__(self, memo={}):
            '''
            deepcopy constructor
            '''
            inst = self.__class__()
            inst.initialize(self._container, self._name,
                            self._default, self.pattern)

            return inst

        def __str__(self):
            '''
            print or str casting
            '''
            return self.value

        def initialize(self, container, name, default, pattern='.+'):
            '''
            Initialize or Update flag instance
            Args:
                container (Coder): class instance holding this flag
                name (str): flag name
                default (str): default value expression
                pattern (str): flag validity pattern
            '''

            if not self.name_pattern.match(name):
                msg = 'Invalid Flag name: ' + str(name)
                msg += '    Flag name should match to '
                msg += str(self.name_pattern.pattern[:-1])

                raise ValueError(msg)

            self._container = container
            self._name = name

            self.default = default
            self.pattern = pattern

            return

        @property
        def name(self):
            return self._name

        @property
        def default(self):
            '''
            Returns:
                (str) default value or expression
            '''
            return self._default

        @default.setter
        def default(self, default_value):
            '''
            Args:
                default_value (str): value or simple expression
                    e.g. test -> test, {USER} -> os.environ['USER']
            '''

            if self.has_default_value():
                self._default = default_value
                self._val = self._get_default_value()
            else:
                self._default = default_value

            return

        @property
        def pattern(self):
            '''
            Returns:
                (str) pattern expression
            '''
            return self._pattern_rex.pattern

        @pattern.setter
        def pattern(self, new_pattern):
            '''
            Args:
                new_pattern (str): regular expression for validity of value
            '''
            try:
                self._pattern_rex = re.compile(new_pattern)
            except Exception, _:
                raise ValueError('Invalid pattern for a flag: %s (flag: %s)'
                                 '' % (new_pattern, self._name))

            return

        @property
        def decode_pattern(self):
            '''
            Returns:
                (str) decode pattern expression
            '''
            return '(?P<%s>%s)' % (self.name, self.pattern)

        @property
        def value(self):
            '''
            Returns:
                (str) current flag value
            '''
            return self._val

        def reset(self):
            self._val = self._get_default_value()

        @value.setter
        def value(self, new_value, ignore_pattern=False):
            '''
            Args:
                new_value (str): new flag value
                ignore_pattern (bool): bypass pattern check (default: False)
            '''
            if new_value == RBNONE:
                self.reset()
                return

            if not ignore_pattern and not self.is_valid_value(new_value):
                raise ValueError('Invalid value for a pattern:'
                                 ' "%s" -> "%s" (flag: %s)'
                                 % (new_value, self.pattern, self._name))

            self._val = new_value
            return

        def is_valid_value(self, value):
            '''
            Returns:
                (None or SRE_Match) validity of input value
            Args:
                value (str): string to check validity to this flag value
            '''
            return self._pattern_rex.match(value)

        def has_valid_value(self):
            '''
            Returns:
                (None or SRE_Match) validity of current value
            '''
            return self.is_valid_value(self._val)

        def has_default_value(self):
            '''
            Returns:
                (bool) True if current value is the same as default value
            '''
            return self._default_val == self.value

        env_extractor = re.compile('{([a-zA-Z0-9-_]+)}')

        def _get_default_value(self):
            '''
            replace default value pattern to actual value
            '''
            default = self._default

            rex = self.env_extractor.match(default)
            if not rex:
                return default

            key = rex.groups()[0]

            env = self._container._env
            if key not in env:
                raise KeyError('Cannot find an environment variable:'
                               ' %s (expression: %s)' % (self._name, default))
            self._default_val = env.get(key, default)

            return self._default_val

        pass  # end of class Flag

    def _substitute_product(self, expression):
        '''
        Substitute product variables to values in expression
            e.g. <path> -> (USERNAME)_(VER)
        Args:
            expression (str): target expression to try match
        '''
        prod_dict = self._product

        return substitute_keys('<([a-zA-Z0-9-_]+)>', prod_dict, expression, True)

    def _substitute_flag(self, expression):
        '''
        Substitute flag variables to values in expression
            e.g. (USERNAME) -> {USER}
        Args:
            expression (str): target expression to try match
        '''
        flags = self._flag

        return substitute_keys('\(([a-zA-Z0-9-_]+)\)', flags, expression)

    def _substitute_tag(self, expression, tags=None):
        '''
        Substitute tag variables to values in expression
            e.g. ()
        Args:
            expression (str): target expression to try match
        '''
        tags = tags if tags else self._tag

        return substitute_keys('@([a-zA-Z0-9-_]+)@', tags, expression)

    def _substitute_environment(self, expression):
        '''
        Substitute environment variables to values in expression
            e.g. {USER} -> guest
        Args:
            expression (str): target expression to try match
        '''
        env_dict = self._env
        return substitute_keys('{([a-zA-Z0-9-_]+)}', env_dict, expression)

    def update_rule(self, rule_data, recursive=False):
        '''
        Update current naming rules
        Args:
            rule_data (dict): naming rule data
            recursive (bool): create and update children if True

        '''
        self._rule_raw_data = rule_data

        self._myFlags = rule_data.get('__flag__', {}).keys()
        for key, attr in rule_data.get('__flag__', {}).iteritems():
            if key not in self._flag_raw_data:
                self._flag_raw_data[key] = {}
            self._flag_raw_data[key].update(attr)

        self._myTags = rule_data.get('__tag__', {}).keys()
        for key, val in rule_data.get('__tag__', {}).iteritems():
            self._tag_raw_data[key] = val

        self._initialize_tags()
        self._initialize_flags()

        self._myProduct = rule_data.get('__product__', {})
        self._product.update(rule_data.get('__product__', {}))

        self._initialize_combiner(rule_data.get('__combiner__', {}))

        if recursive:
            children_data = rule_data.get('__child__', {})
            self._child = dict()

            for key, val in children_data.iteritems():
                self.add_child(key, val)
            pass

        return

    def _initialize_tags(self):
        for key, attr in self._tag_raw_data.iteritems():
            tag = Coder.Tag()
            tag.initialize(self, key, attr)

            if self._tag[key]._container == self:
                # update children instances
                for child in self._tag[key]._users:
                    child._tag[key] = tag
                    tag._users.append(child)

            self._tag[key] = tag

        # substitute tags
        roopCnt = 5
        keys = list()
        for k, tag in self._tag.iteritems():
            if '@' in tag.value:
                keys.append(k)
                if '|' in tag.value:
                    tag.rule = tag.value
        keys.append(0)

        while keys:
            k = keys.pop(0)
            if isinstance(k, int):
                if k < 5:
                    keys.append(k+1)
                else:
                    break
            else:
                try:
                    if '@%s@'%k in self._tag[k].value:
                        v = self._substitute_tag(self._tag[k].value,
                                                 self._parent._tag)
                    else:
                        v = self._substitute_tag(self._tag[k].value)
                    self._tag[k].value = v
                except Exception as E:
                    keys.append(k)
        if keys:
            raise(KeyError('Cannot find tags (%s)'%str(keys)))

        return

    def _initialize_flags(self):
        '''
        Initialize or Update flag instances
        '''

        for key, attr in self._flag_raw_data.iteritems():
            flag = Coder.Flag()
            # substitute tags
            default = self._substitute_tag(attr.get('__default__', RBNONE))
            pattern = self._substitute_tag(attr.get('__pattern__', '.+'))
            flag.initialize(self, key, default, pattern)

            if self._flag[key]._container == self:
                # update children instances
                for child in self._flag[key]._users:
                    child._flag[key] = flag
                    flag._users.append(child)

            self._flag[key] = flag

        return

    def _initialize_combiner(self, data):
        if data:
            self._combiner = Coder.Combiner()
            self._combiner.initialize(self, data, self.name)
        return

    def load_rulebook(self, file_name='dxname.yaml'):
        '''
        Search a configuration from paths in an environment variable,
            DX_NAME_PATH, and then build naming rule encoder/decoder.
        Args:
            file_name (str): naming file name in search directories
        '''
        config_paths = os.environ.get('DX_NAME_PATH', '').split(':')
        # config_paths.sort(reverse=True) # lower hierarchy first

        # package sample as default
        yml_path = libpath('preset', 'dxname.yaml')

        for directory in reversed(config_paths):
            file_path = os.path.join(directory, file_name)
            if os.access(file_path, os.R_OK):
                yml_path = file_path

                break
            pass

        data = self.load_yaml(yml_path)

        return self.update_rule(data, recursive=True)

    @staticmethod
    def load_yaml(yml_path):
        '''
        Returns:
            (dict) naming rule data dictionary
        Args:
            file_name (str): file name to read (default='dexter.yaml')
        '''

        with open(yml_path, 'r') as yml_file:
            data = yaml.load(yml_file.read())

        return data
