"""
position format:

64 characters at with {p,n,b,r,q,k} as black pieces and uppercase as white pieces, X between rows, z as space
ex:
rnbqkbnrXppppppppXzzzzzzzzXzzzzzzzzXzzzzzzzzXzzzzzzzzXPPPPPPPPXRNBQKBNR
"""

"""
input: the previous and current positions to diff (in the format specified above)
output/return: the two positions where the position has changed
"""


def diff(last_pos: str, cur_pos: str):
    d1 = (-1, -1)
    d1complete = False
    d2 = (-1, -1)
    def ind(i, j):
        return i*8 + j
    for i in range(8):
        for j in range(8):
            if last_pos[ind(i, j)] != cur_pos[ind(i, j)]:
                if d1complete:
                    d2 = (i, j)
                else:
                    d1complete = True
                    d1 = (i, j)

    if not d1complete:
        return "--"
    pos1 = "%s%s" % (chr(ord("a")+d1[0]), d1[1]+1)
    pos2 = "%s%s" % (chr(ord("a")+d2[0]), d2[1]+1)
    return pos1, pos2
