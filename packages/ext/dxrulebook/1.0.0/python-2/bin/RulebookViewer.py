#!/usr/bin/python

from __future__ import print_function
import os, re, pprint

from flask import Flask
import webbrowser

import DXRulebook.Interface as rbi

app = Flask(__name__)
RB  = rbi._RBROOT

class RBViewer:
    def __init__(self):
        self.tags     = dict()
        self.products = dict()
        self.paddings = [15, 40, 65, 90]

        self.GetTags(RB)
        self.GetProducts(RB, self.products)

    def GetTags(self, rb, names=None):
        key = 'Global'
        if names != None:
            for name in names:
                if name in rbi._DCCS:
                    key = name; break

        if not self.tags.has_key(key):
            self.tags[key] = dict()

        # for the first find options
        myTags = rb.myTags
        for tag in rb.myTags:
            if rb.tag[tag].rule:
                myTags.pop(myTags.index(tag))
                rule = rb.tag[tag].rule
                self.tags[key][tag] = dict()

                for t in rule.split('|'):
                    t = t[1:-1] # @name@
                    self.tags[key][tag][t] = rb.tag[t].value
                    if t in myTags:
                        myTags.pop(myTags.index(t))

        for tag in myTags:
            self.tags[key].update({tag:rb.tag[tag].value})

        for child in rb.children:
            if names == None:
                self.GetTags(rb.child[child], list())
            else:
                self.GetTags(rb.child[child], names + [rb.child[child].name])

    def GetProducts(self, rb, data):
        for p in rb.myProducts:
            data[p] = rb._product.get(p, '')

        if rb.ref:
            data['__ref__'] = '(%s)'%rb.ref

        if rb.combiner:
            data['__combiner__'] = dict()
            data['__combiner__']['sep'] = rb.combiner.sep
            data['__combiner__']['combines'] = rb.combiner.combines
            data['__combiner__']['products'] = rb.combiner._products

            for k, v in data['__combiner__']['products'].items():
                for i in range(len(v)):
                    v[i] = v[i].replace('<', '< ')
                    v[i] = v[i].replace('>', ' >')

        for child in rb.children:
            data[child] = dict()
            self.GetProducts(rb.child[child], data[child])


    def ProductHtml(self, products):
        html = list()

        def RecursiveHtml(html, name, product, padding=0):
            combiner = None
            ref      = None
            children = dict()
            if product.has_key('__combiner__'):
                combiner = product.pop('__combiner__')
                padding += 1
                html.append('''
                <tr>
                    <td colspan="3" style="padding-left:%d" bgcolor="#AAAAAA">
                    <table>
                        <tr>
                            <td align="left" valign="top" rawspan="3">%s</td>
                            <td valign="top"  style="padding-top:0px"><table>
                '''%(self.paddings[0],
                     "Combiner [ %s ] : "%combiner['sep']))

                for comb in combiner['combines']:
                    values = '<br>'.join(combiner['products'][comb])
                    values = '''%s'''%values

                    html.append('''
                        <tr>
                            <td align="left" valign="top" style="padding-right:10px">%s</td>
                            <td aligh="left">%s</td>
                        </tr>
                    '''%(comb, values))


                html.append('''
                            </table></td>
                        <tr>
                    </table></td>
                </tr>
                ''')

            if product.has_key('__ref__'):
                ref = product.pop('__ref__')

            for k in product.keys():
                if isinstance(product[k], dict):
                    children[k] = product.pop(k)

            keys = product.keys()
            keys.sort()
            for k in keys:
                vs = product[k]
                if not isinstance(vs, list):
                    vs = [vs]
                html.append('''
                <tr>
                    <td align="left" valign="top" style="padding-left: %d;padding-right:15px">%s<td>
                    <td align="left" width="350">'''%(self.paddings[padding], k))

                for v in vs:
                    html.append('''
                    %s</br>
                    '''%v)

                html.append('''</td>
                </tr>
                <tr>
                    <td colspan="3" class="line"></td>
                </tr>
                ''')

            keys = children.keys()
            keys.sort()
            for k in keys:
                subTitle = k
                if name:
                    subTitle = ('<font color="777777">%s. </font>'%name) + k
                html.append('''
                <tr>
                    <td style="padding-left:%d" colspan="3" bgcolor="#AAAAAA">%s</td>
                </tr>
                '''%(self.paddings[0], subTitle))
                RecursiveHtml(html, k, children[k], padding + 1)

        RecursiveHtml(html, '', products)
        return ''.join(html)


    def html_tag(self):
        # reorder keys
        keys = self.tags.keys()
        keys.sort()
        # keys.insert(0, keys.pop(keys.index('USD')))
        keys.insert(0, keys.pop(keys.index('Global')))

        html = '''
        <div>
        <table aligh="left">
            <tr><td valign="top">
            <h1>Tags</h1>
            <table align="left">
        '''
        for k in keys:
            html += '''
            <tr>
                <td colspan="3" class="t1">%s</td>
            </tr>
            '''%k
            itemKeys = self.tags[k].keys()
            itemKeys.sort()
            for name in itemKeys:
                padding = self.paddings[0]
                items   = {name:self.tags[k][name]}
                if isinstance(self.tags[k][name], dict):
                    items   = self.tags[k][name]
                    html   += '''
                    <tr>
                        <td style="padding-left:%d" colspan="3">%s</td>
                    </tr>
                    <tr>
                        <td colspan="3" class="line"></td>
                    </tr>
                    '''%(padding, name)
                    padding = self.paddings[1]

                subKeys = items.keys()
                subKeys.sort()
                for sub in subKeys:
                    value = items[sub]
                    if '|' in value:
                        value = value.replace('|', ', ')
                    html += '''
                    <tr>
                        <td align="left" valign="top" style="padding-left: %d;padding-right:15px">%s<td>
                        <td align="left" width="250">%s<td>
                    </tr>
                    <tr>
                        <td colspan="3" class="line"></td>
                    </tr>
                    '''%(padding, sub, value)


        html += '''
        </table>
        </td>
        <td width="20"></td>
        <td valign="top">
        <h1>Products</h1>
        <table aligh="left">
        '''

        # reorder keys
        keys = self.products.keys()
        keys.sort()
        for c in keys:
            title = '%s : %s'
            if c == 'D':
                title = title%(c, 'Directory')
            elif c == 'F':
                title = title%(c, 'File')
            elif c == 'N':
                title = title%(c, 'Name')
            html += '''
            <tr>
                <td colspan="3" class="t1">%s</td>
            </tr>
            '''%title

            html += self.ProductHtml(self.products[c])

        html += '''
        </table>
        </td></tr>
        </table>
        '''
        return html



@app.route('/')
def htmlCode():
    html = '''
    <html>
    <style>
    table{
        padding:0px;
    }
    td{
        padding:0px;
        padding-top:2px;
    }
    td.line{
        height:1px;
        padding:0px;
        background-color:D0D0D0;
    }
    .t1{
        color: white;
        font-size:20px;
        valign:center;
        background-color:333333;
        padding-left:5px;
    };
    </style>
    <body bgcolor="E0E0E0">
    <font size="4">Dx Rule Book</font>
    <hr style="border:0.5px solid"></hr>
    '''

    rbviewer = RBViewer()
    html += rbviewer.html_tag()

    html += '''
    </body>
    </html>
    '''
    return html


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    webbrowser.open_new(url)
    app.run()
