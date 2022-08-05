"""
给定一个字符串 s 和一个整数 k 。你可以从 s 的前 k 个字母中选择一个，并把它加到字符串的末尾。

返回 在应用上述步骤的任意数量的移动后，字典上最小的字符串 。

输入：s = "cba", k = 1
输出："acb"
解释：
在第一步中，我们将第一个字符（“c”）移动到最后，获得字符串 “bac”。
在第二步中，我们将第一个字符（“b”）移动到最后，获得最终结果 “acb”。

"""

from sympy import Predicate


class Solution:
    def __init__(self) -> None:
        remap = {}
        for i  in range(26):
            remap[chr(ord('a')+i)] = i
        self.remap = remap

    def orderlyQueue(self, s: str, k: int) -> str:
        n = len(s)
        tonum = [self.remap[x] for x in s]
        min_ = s[0]
        idx  = 0

        for i in range(n):
            if self.remap[s[i]]<= self.remap[min_]:
                min_ = s[i]
                idx = i
        min_idx = idx
        res = []
        while (idx!=0):
            if k<idx:
                pre = min(tonum[:k])
                pre_idx = tonum.index(pre)
                res.append(chr(ord('a')+tonum.pop(pre_idx)))
                idx-=1
                # x x x x x x x x x x x x x x x x x x x x x x x x x x x x 
            else:
                # - - - - | - - - k - -- - - -
                pre = min(tonum[:idx])
                pre_idx= tonum.index(pre)
                res.append(chr(ord('a')+tonum.pop(pre_idx)))
                idx -= 1
        res = [chr(ord('a') + x) for x in tonum] + res
        return ''.join(res)
        