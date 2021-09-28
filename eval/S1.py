# a, b are two lists, e.g, a: [1,2,3], b: [3,4,5]
# a is pred, b is ans
def _s1_entry(a, b):
    if len(b) == 0:
        if len(a) == 0:
            s1 =  1.0
        else:
            s1 = 0.0
    elif len(a) == 0: # empty pred but non-empty answer
        s1 = 0.0
    else:
        sa = set(a)
        sb = set(b)
        c = len(sa.intersection(sb))
        p = 1.0 * c / len(sa)
        r = 1.0 * c / len(sb)

        if p == 0 or r == 0: 
            s1 = 0.0
        else:
            s1 = 2.0 * (p * r) / (p + r)

    return s1
    
# ================== Compute S1 ========================
# ans and pred are list of lists (set of sets in nature)
def eval_S1(pred, ans):
    assert len(pred) == len(ans), 'Size mismatch!'

    N = len(pred)
    s1 = 0
    for i in range(N):
        s1 += _s1_entry(pred[i], ans[i])

    s1 = s1 / N

    return s1

if __name__ == '__main__':
    # test
    pred = [['a', 'b'],
            [],
            ['c'],
            ['d','e']
            ]
    ans  = [['b', 'c', 'd'],
            [],
            [],
            ['f','g']
            ]

    S1 = eval_S1(pred,ans)
    print ('Pred:', pred)
    print ('Ans:', ans)
    print ('Avg S1:', S1)
    assert S1==0.35, 'Test failed'
    
