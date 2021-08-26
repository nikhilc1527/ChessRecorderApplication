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
    rows1 = last_pos.split("X")
    rows2 = cur_pos.split("X")
    d1 = (-1, -1)
    d1complete = False
    d2 = (-1, -1)
    for i in range(8):
        if not(rows1[i] == rows2[i]):
            for j in range(8):
                if not(rows1[i][j] == rows2[i][j]):
                    if not d1complete:
                        d1 = (i+1, j+1)
                        d1complete = True
                    else:
                        d2 = (i+1, j+1)

    return d1, d2
