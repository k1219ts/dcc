import os

def partition(list, l, e, g):
    while list != []:
        head = list.pop(0)
        if head < e[0]:
            l = [head] + l
        elif head > e[0]:
            g = [head] + g
        else:
            e = [head] + e
    return (l, e, g)

def qsort1(list):
    """
    Quicksort using list comprehensions
    >>> qsort1<<docstring test numeric input>>
    <<docstring test numeric output>>
    >>> qsort1<<docstring test string input>>
    <<docstring test string output>>
    """
    if list == []:
        return []
    else:
        pivot = list[0]
        lesser = qsort1([x for x in list[1:] if x < pivot])
        greater = qsort1([x for x in list[1:] if x >= pivot])
        return lesser + [pivot] + greater


from random import randrange
def qsort1a(path, list):
    """
    Quicksort using list comprehensions and randomized pivot
    >>> qsort1a<<docstring test numeric input>>
    <<docstring test numeric output>>
    >>> qsort1a<<docstring test string input>>
    <<docstring test string output>>
    """
    temp2 = {}

    def qsort(list):
        if list == []:
            return []
        else:
            pivot = list.pop(randrange(len(list)))
            lesser = qsort([l for l in list if l < pivot])
            greater = qsort([l for l in list if l >= pivot])
            try:
                statinfo = os.stat(os.path.join(path, pivot))
                n = statinfo.st_size
                basename, frame, ext = pivot.split('.')
                _key = basename + ':' + ext
                try:
                    start = temp2[_key][0]
                    end = temp2[_key][1]
                    if frame < start:
                        temp2[_key][0] = frame
                    if frame > end:
                        temp2[_key][1] = frame
                except:
                    temp2[_key] = [frame, frame, 0, 0]

                temp2[_key][2] += 1
                temp2[_key][3] += n
            except:
                temp2[ pivot] = None

            return lesser + [pivot] + greater
    qsort(list[:])
    return temp2

def qsort1aN(list):
    """
    Quicksort using list comprehensions and randomized pivot
    >>> qsort1a<<docstring test numeric input>>
    <<docstring test numeric output>>
    >>> qsort1a<<docstring test string input>>
    <<docstring test string output>>
    """
    temp2 = {}

    def qsort(list):
        if list == []:
            return []
        else:
            pivot = list.pop(randrange(len(list))).fileName()
            lesser = qsort([l for l in list if l.fileName() < pivot])
            greater = qsort([l for l in list if l.fileName() >= pivot])
            try:
                basename, frame, ext = pivot.split('.')
                _key = basename + ':' + ext
                try:
                    start = temp2[_key][0]
                    end = temp2[_key][1]
                    if frame < start:
                        temp2[_key][0] = frame
                    if frame > end:
                        temp2[_key][1] = frame

                except:
                    temp2[_key] = [frame, frame, 0]

                temp2[_key][2] += 1
            except:
                temp2[ pivot] = None
                pass

            return lesser + [pivot] + greater
    q = qsort(list[:])
    return temp2


def qsort2(list):
    """
    Quicksort using a partitioning function
    >>> qsort2<<docstring test numeric input>>
    <<docstring test numeric output>>
    >>> qsort2<<docstring test string input>>
    <<docstring test string output>>
    """

    if list == []:
        return []
    else:
        pivot = list[0]
        lesser, equal, greater = partition(list[1:], [], [pivot], [])
        return qsort2(lesser) + equal + qsort2(greater)

if __name__ == '__main__':
    import os

    files = os.listdir( '/show/dragonGate/seq/AA/AA02/comp')
    q = qsort1a( files)
    print q
