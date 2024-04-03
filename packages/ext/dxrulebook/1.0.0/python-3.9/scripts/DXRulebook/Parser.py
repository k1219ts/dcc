#coding:utf-8
'''
    Copyright & Author: 2016, Sehwi Park <sehwida@gmail.com>

    Description:
        keyword parsing utility functions
'''

from __future__ import print_function
import re

from .Utilities import memoized


class CycleError(Exception):
    pass


def match_iter(match, dict_map, recursive=True, *history):
    '''
    Substitute match result based on dictionary key
    Args:
        match (re.match): match object
        dict_map (dict): dictionary data for keyword replacement
        recursive (bool): execute this function recursively
        history (tuple): match keyword history for recursion cycling check
    Returns:
        (str) string after replaced keywords to dictionary values
    '''

    buf = ''
    for key in match.groups():
        val = str(dict_map[key]) # can raise KeyError

        if val == '!!none':
            raise ValueError('Undefined product (%s)'%key)

        if recursive:
            if key in history:
                print('cycling detected', history)
                msg = key + ' <- '
                msg += ' <- '.join(history)

                raise CycleError('Rotation of definition exists "%s" ' % msg)

            regex = match.re
            func = lambda x: match_iter(x, dict_map, True, key, *history)

            val = regex.sub(func, val)

        buf += val

    return buf


@memoized
def pattern_generator(pattern):
    '''
    Cached pattern instance generator
    Returns:
        (re.Pattern) compiled regular expression instance
    Args:
        pattern (str): pattern regular expression
    '''
    return re.compile(pattern)


def substitute_keys(pattern, dict_data, expression, recursive=False, once=False):
    '''
    Substitute an expression based on dictionary and a pattern
        e.g. {USER} -> guest
    Returns:
        (str) character string having keywords replaced to values
    Args:
        pattern (str): pattern to match in expression e.g. {(\w+)}
        dict_data (dict): dictionary map having values to replace keys
        expression (str): target expression to try match
        recursive (bool): execute recursively (cycling check will happen)
    '''
    regex = pattern_generator(pattern)

    def _prod_recursion(prod_exp):

        try:
            result = regex.sub(lambda x: match_iter(x, dict_data, recursive),
                               prod_exp)
        except CycleError as e:
            msg = e.message + ' in the expression "%s"' % prod_exp
            raise CycleError(msg)
        except KeyError as e:
            msg = 'Not defined product "%s" in %s' % (e.args[0], prod_exp)
            raise KeyError(msg)

        if regex.search(result) and not once:
            result = _prod_recursion(result)

        return result

    return _prod_recursion(expression)
